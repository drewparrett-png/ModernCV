"""Studio API — YOLO26 / YOLO11 image datasets, prompting, training, testing, Pal/DePal.

All routes are project-scoped under /projects/{pid}/studio except the
global model catalog.

    GET    /studio/catalog
    GET    /projects/{pid}/studio                         classes + images + stats
    PUT    /projects/{pid}/studio/classes
    POST   /projects/{pid}/studio/images                  multipart upload (files[])
    POST   /projects/{pid}/studio/images/from_video
    POST   /projects/{pid}/studio/images/synthetic        rendered pallets with exact depth
    PATCH  /projects/{pid}/studio/images/{iid}
    DELETE /projects/{pid}/studio/images/{iid}
    GET    /projects/{pid}/studio/images/{iid}/file | /thumb | /depth.png
    PUT    /projects/{pid}/studio/images/{iid}/depth      multipart (16-bit PNG mm / .npy m)
    DELETE /projects/{pid}/studio/images/{iid}/depth
    GET|PUT /projects/{pid}/studio/images/{iid}/annotations
    GET    /projects/{pid}/studio/images/{iid}/suggestions
    POST   /projects/{pid}/studio/images/{iid}/suggestions/accept | /reject
    POST   /projects/{pid}/studio/prompt/sam | /paint | /visual | /detect
    POST   /projects/{pid}/studio/predict
    GET|POST /projects/{pid}/studio/models
    GET|DELETE /projects/{pid}/studio/models/{mid}
    POST   /projects/{pid}/studio/models/{mid}/cancel | /val | /export | /benchmark
    GET    /projects/{pid}/studio/models/{mid}/plots/{name} | /exports/{file} | /weights
    POST   /projects/{pid}/studio/pallet/{iid}
    GET|POST /projects/{pid}/studio/tracks
    GET|DELETE /projects/{pid}/studio/tracks/{tid}
    GET    /projects/{pid}/studio/tracks/{tid}/overlay.mp4
    GET    /projects/{pid}/studio/dataset.zip?task=detect|segment|obb
"""

from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from pipeline import runs as runs_mod
from pipeline.studio import engines, pallet, prompting, synth, track, training
from pipeline.studio.export import export_yolo_dataset
from pipeline.studio.infer import run_predict
from pipeline.studio.store import Invalid, NotFound, StudioStore

log = logging.getLogger(__name__)

router = APIRouter()


def _store(project_id: str) -> StudioStore:
    if not (runs_mod.project_dir(project_id) / "project.json").exists():
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return StudioStore.for_project(project_id)


def _errors(fn: Callable) -> Callable:
    """Map store/engine exceptions onto HTTP status codes."""

    def translate(e: Exception) -> HTTPException:
        if isinstance(e, NotFound):
            return HTTPException(status_code=404, detail=f"not found: {e.args[0] if e.args else e}")
        return HTTPException(status_code=422, detail=str(e))

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except HTTPException:
                raise
            except (NotFound, Invalid, ValueError) as e:
                raise translate(e) from e

        return async_wrapper

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise
        except (NotFound, Invalid, ValueError) as e:
            raise translate(e) from e

    return wrapper


# ---- Request models -------------------------------------------------------


class ClassesBody(BaseModel):
    classes: list[dict]


class ImagePatch(BaseModel):
    split: Optional[str] = None
    negative: Optional[bool] = None
    intrinsics: Optional[dict] = None


class FromVideoBody(BaseModel):
    video_path: str
    stride: int = Field(30, ge=1)
    max_frames: int = Field(20, ge=1, le=500)
    start_frame: int = Field(0, ge=0)


class SyntheticBody(BaseModel):
    count: int = Field(6, ge=1, le=60)
    with_labels: bool = True
    seed: Optional[int] = None
    width: int = Field(960, ge=320, le=1920)
    height: int = Field(720, ge=240, le=1440)


class AnnotationsBody(BaseModel):
    annotations: list[dict]


class AcceptBody(BaseModel):
    ids: Optional[list[str]] = None
    min_score: float = 0.0
    class_id: Optional[int] = None


class RejectBody(BaseModel):
    ids: Optional[list[str]] = None


class SamBody(BaseModel):
    image_id: str
    prompts: list[dict]
    sam_model: str = "sam2.1_t"
    granularity: str = "auto"


class PaintBody(BaseModel):
    image_id: str
    prompts: list[dict]


class VisualBody(BaseModel):
    refs: list[dict]
    scope: str = "image"
    image_id: Optional[str] = None
    image_ids: Optional[list[str]] = None
    family: str = "11"
    size: str = "s"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    refine_with_sam: bool = False
    sam_model: str = "sam2.1_t"


class DetectBody(BaseModel):
    model: dict
    scope: str = "image"
    image_id: Optional[str] = None
    image_ids: Optional[list[str]] = None
    params: dict = Field(default_factory=dict)
    refine_with_sam: bool = False
    sam_model: str = "sam2.1_t"
    assign_class_id: Optional[int] = None


class PredictBody(BaseModel):
    model: dict
    image_id: str
    params: dict = Field(default_factory=dict)


class TrainBody(BaseModel):
    task: str = "detect"
    size: str = "n"
    family: str = "26"
    base: str = "pretrained"
    name: Optional[str] = None
    config: dict = Field(default_factory=dict)


class ValBody(BaseModel):
    split: str = "val"
    conf: Optional[float] = None
    iou: float = 0.7


class ExportBody(BaseModel):
    format: str = "onnx"
    imgsz: Optional[int] = None
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    nms: bool = False


class BenchBody(BaseModel):
    n_images: int = Field(20, ge=2, le=200)


class TrackBody(BaseModel):
    video_path: str
    model: dict
    tracker: str = "bytetrack"
    params: dict = Field(default_factory=dict)
    stride: int = Field(1, ge=1)
    max_frames: int = Field(300, ge=0)
    count_line: Optional[dict] = None


# ---- Catalog --------------------------------------------------------------


@router.get("/studio/catalog")
def catalog() -> dict:
    cat = engines.catalog()
    cat["export_formats"] = training.export_formats()
    cat["trackers"] = list(track.TRACKERS)
    cat["train_tasks"] = list(training.TRAIN_TASKS)
    cat["augment_keys"] = sorted(training.AUGMENT_KEYS)
    cat["default_train_config"] = training.DEFAULT_CONFIG
    return cat


# ---- Dataset --------------------------------------------------------------


@router.get("/projects/{project_id}/studio")
@_errors
def studio_state(project_id: str) -> dict:
    st = _store(project_id)
    return {"classes": st.classes(), "images": st.list_images(), "stats": st.stats()}


@router.put("/projects/{project_id}/studio/classes")
@_errors
def put_classes(project_id: str, body: ClassesBody) -> dict:
    st = _store(project_id)
    return {"classes": st.set_classes(body.classes), "images": st.list_images()}


@router.post("/projects/{project_id}/studio/images")
@_errors
async def upload_images(project_id: str, files: list[UploadFile] = File(...)) -> dict:
    st = _store(project_id)
    added, errors = [], []
    for f in files:
        data = await f.read()
        try:
            added.append(st.add_image_bytes(data, f.filename or "upload", "upload"))
        except Invalid as e:
            errors.append(str(e))
    return {"images": added, "errors": errors}


@router.post("/projects/{project_id}/studio/images/from_video")
@_errors
def images_from_video(project_id: str, body: FromVideoBody) -> dict:
    st = _store(project_id)
    path = track.resolve_video(body.video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise Invalid(f"cannot open {body.video_path}")
    added = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, body.start_frame)
        idx = body.start_frame
        while len(added) < body.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - body.start_frame) % body.stride == 0:
                added.append(st.add_image_array(frame, f"{path.stem}_f{idx:06d}.jpg", f"video:{body.video_path}#{idx}"))
            idx += 1
    finally:
        cap.release()
    return {"images": added}


@router.post("/projects/{project_id}/studio/images/synthetic")
@_errors
def images_synthetic(project_id: str, body: SyntheticBody) -> dict:
    st = _store(project_id)
    carton = st.ensure_class("carton") if body.with_labels else None
    base_seed = body.seed if body.seed is not None else int(np.random.default_rng().integers(0, 1_000_000))
    added = []
    for k in range(body.count):
        seed = base_seed + k
        sc = synth.render_pallet_scene(seed=seed, width=body.width, height=body.height)
        gt = [
            {"bbox": i["bbox"], "top_height_m": i["top_height_m"], "height_above_deck_m": i["height_above_deck_m"],
             "layer": i["layer"], "size_m": i["size_m"]}
            for i in sc.instances
        ]
        rec = st.add_image_array(
            sc.bgr, f"synthetic_pallet_{seed}.jpg", "synthetic",
            extra={"synthetic": {**sc.meta, "gt": gt}},
        )
        st.set_depth(rec["id"], sc.depth_mm, sc.intrinsics)
        if carton is not None:
            st.set_annotations(
                rec["id"],
                [{"class_id": carton["id"], "polygon": i["polygon"], "source": "synthetic-gt"} for i in sc.instances],
            )
        added.append(st.get_image(rec["id"]))
    return {"images": added, "classes": st.classes()}


@router.patch("/projects/{project_id}/studio/images/{image_id}")
@_errors
def patch_image(project_id: str, image_id: str, body: ImagePatch) -> dict:
    st = _store(project_id)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    return st.update_image(image_id, **fields)


@router.delete("/projects/{project_id}/studio/images/{image_id}")
@_errors
def delete_image(project_id: str, image_id: str) -> dict:
    _store(project_id).delete_image(image_id)
    return {"deleted": image_id}


@router.get("/projects/{project_id}/studio/images/{image_id}/file")
@_errors
def image_file(project_id: str, image_id: str) -> FileResponse:
    st = _store(project_id)
    return FileResponse(st.image_path(image_id), headers={"Cache-Control": "max-age=86400"})


@router.get("/projects/{project_id}/studio/images/{image_id}/thumb")
@_errors
def image_thumb(project_id: str, image_id: str, size: int = Query(160)) -> FileResponse:
    st = _store(project_id)
    return FileResponse(st.thumbnail(image_id, size), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@router.put("/projects/{project_id}/studio/images/{image_id}/depth")
@_errors
async def put_depth(
    project_id: str, image_id: str, file: UploadFile = File(...), scale_to_mm: Optional[float] = Form(None),
    fx: Optional[float] = Form(None), fy: Optional[float] = Form(None),
    cx: Optional[float] = Form(None), cy: Optional[float] = Form(None),
) -> dict:
    st = _store(project_id)
    rec = st.set_depth_bytes(image_id, await file.read(), file.filename or "depth.png", scale_to_mm)
    if fx and fy and cx is not None and cy is not None:
        rec = st.update_image(image_id, intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy})
    return rec


@router.delete("/projects/{project_id}/studio/images/{image_id}/depth")
@_errors
def delete_depth(project_id: str, image_id: str) -> dict:
    return _store(project_id).delete_depth(image_id)


@router.get("/projects/{project_id}/studio/images/{image_id}/depth.png")
@_errors
def depth_preview(project_id: str, image_id: str) -> Response:
    st = _store(project_id)
    d = st.read_depth_m(image_id)
    if d is None:
        raise NotFound(f"{image_id}/depth")
    valid = d > 0.05
    lo, hi = np.percentile(d[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    img = pallet.colorize(np.where(valid, d, np.nan), float(hi), float(lo))
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png")


@router.get("/projects/{project_id}/studio/images/{image_id}/annotations")
@_errors
def get_annotations(project_id: str, image_id: str) -> dict:
    st = _store(project_id)
    st.get_image(image_id)
    return {"annotations": st.annotations(image_id)}


@router.put("/projects/{project_id}/studio/images/{image_id}/annotations")
@_errors
def put_annotations(project_id: str, image_id: str, body: AnnotationsBody) -> dict:
    st = _store(project_id)
    return {"annotations": st.set_annotations(image_id, body.annotations), "image": st.get_image(image_id)}


@router.get("/projects/{project_id}/studio/images/{image_id}/suggestions")
@_errors
def get_suggestions(project_id: str, image_id: str) -> dict:
    st = _store(project_id)
    st.get_image(image_id)
    return st.suggestions(image_id)


@router.post("/projects/{project_id}/studio/images/{image_id}/suggestions/accept")
@_errors
def accept_suggestions(project_id: str, image_id: str, body: AcceptBody) -> dict:
    st = _store(project_id)
    out = prompting.accept_suggestions(st, image_id, body.ids, body.min_score, body.class_id)
    out["image"] = st.get_image(image_id)
    out["classes"] = st.classes()
    return out


@router.post("/projects/{project_id}/studio/images/{image_id}/suggestions/reject")
@_errors
def reject_suggestions(project_id: str, image_id: str, body: RejectBody) -> dict:
    st = _store(project_id)
    data = st.clear_suggestions(image_id, body.ids)
    return {"suggestions": data, "image": st.get_image(image_id)}


# ---- Prompting ------------------------------------------------------------


@router.post("/projects/{project_id}/studio/prompt/sam")
@_errors
def prompt_sam(project_id: str, body: SamBody) -> dict:
    st = _store(project_id)
    return {"result": prompting.sam_segment(st, body.image_id, body.prompts, body.sam_model, body.granularity)}


@router.post("/projects/{project_id}/studio/prompt/paint")
@_errors
def prompt_paint(project_id: str, body: PaintBody) -> dict:
    st = _store(project_id)
    new = prompting.paint_to_annotations(st, body.image_id, body.prompts)
    if not new:
        raise Invalid("nothing to add — the painted area is empty")
    anns = st.add_annotations(body.image_id, new)
    return {"annotations": anns, "added": len(new), "image": st.get_image(body.image_id)}


@router.post("/projects/{project_id}/studio/prompt/visual")
@_errors
def prompt_visual(project_id: str, body: VisualBody) -> dict:
    st = _store(project_id)
    out = prompting.visual_prompt_detect(st, **body.model_dump())
    out["images"] = st.list_images()
    return out


@router.post("/projects/{project_id}/studio/prompt/detect")
@_errors
def prompt_detect(project_id: str, body: DetectBody) -> dict:
    st = _store(project_id)
    out = prompting.detect_to_suggestions(
        st, body.model, body.scope, body.image_id, body.image_ids, body.params,
        body.refine_with_sam, body.sam_model, body.assign_class_id,
    )
    out["images"] = st.list_images()
    return out


@router.post("/projects/{project_id}/studio/predict")
@_errors
def predict(project_id: str, body: PredictBody) -> dict:
    st = _store(project_id)
    img = st.read_image(body.image_id)
    return run_predict(st, body.model, img, body.params)


# ---- Models ---------------------------------------------------------------


@router.get("/projects/{project_id}/studio/models")
@_errors
def list_models(project_id: str) -> dict:
    return {"models": training.list_models(_store(project_id))}


@router.post("/projects/{project_id}/studio/models")
@_errors
def train_model(project_id: str, body: TrainBody) -> dict:
    return training.start_training(_store(project_id), body.model_dump())


@router.get("/projects/{project_id}/studio/models/{model_id}")
@_errors
def get_model(project_id: str, model_id: str) -> dict:
    return training.model_detail(_store(project_id), model_id)


@router.delete("/projects/{project_id}/studio/models/{model_id}")
@_errors
def delete_model(project_id: str, model_id: str) -> dict:
    training.delete_model(_store(project_id), model_id)
    return {"deleted": model_id}


@router.post("/projects/{project_id}/studio/models/{model_id}/cancel")
@_errors
def cancel_model(project_id: str, model_id: str) -> dict:
    return training.cancel(_store(project_id), model_id)


@router.get("/projects/{project_id}/studio/models/{model_id}/plots/{name}")
@_errors
def model_plot(project_id: str, model_id: str, name: str) -> FileResponse:
    st = _store(project_id)
    p = (st.models_dir / model_id / "ultralytics" / "train" / name).resolve()
    if p.suffix != ".png" or (st.models_dir / model_id).resolve() not in p.parents or not p.exists():
        raise NotFound(name)
    return FileResponse(p, media_type="image/png")


@router.post("/projects/{project_id}/studio/models/{model_id}/val")
@_errors
def validate_model(project_id: str, model_id: str, body: ValBody) -> dict:
    return training.validate(_store(project_id), model_id, body.split, body.conf, body.iou)


@router.post("/projects/{project_id}/studio/models/{model_id}/export")
@_errors
def export_model(project_id: str, model_id: str, body: ExportBody) -> dict:
    return training.export_model(_store(project_id), model_id, body.format, body.imgsz, body.half,
                                 body.int8, body.dynamic, body.nms)


@router.get("/projects/{project_id}/studio/models/{model_id}/exports/{file}")
@_errors
def download_export(project_id: str, model_id: str, file: str) -> FileResponse:
    p = training.export_download_path(_store(project_id), model_id, file)
    return FileResponse(p, filename=p.name)


@router.get("/projects/{project_id}/studio/models/{model_id}/weights")
@_errors
def download_weights(project_id: str, model_id: str) -> FileResponse:
    st = _store(project_id)
    p = st.models_dir / model_id / "best.pt"
    if not p.exists():
        raise NotFound(f"{model_id}/best.pt")
    return FileResponse(p, filename=f"{model_id}.pt")


@router.post("/projects/{project_id}/studio/models/{model_id}/benchmark")
@_errors
def benchmark_model(project_id: str, model_id: str, body: BenchBody) -> dict:
    return {"rows": training.benchmark(_store(project_id), model_id, body.n_images)}


# ---- Pal / DePal ----------------------------------------------------------


@router.post("/projects/{project_id}/studio/pallet/{image_id}")
@_errors
def pallet_analysis(project_id: str, image_id: str, body: dict) -> dict:
    st = _store(project_id)
    res = pallet.run_analysis(st, image_id, body)
    syn = st.get_image(image_id).get("synthetic")
    if syn and syn.get("gt"):
        res["ground_truth"] = _match_ground_truth(res["boxes"], syn["gt"])
    return res


def _match_ground_truth(boxes: list[dict], gt: list[dict]) -> dict:
    """Synthetic images only: compare measured heights with the renderer's."""
    from pipeline.studio.geometry import box_iou

    errs = []
    for b in boxes:
        poly = b.get("polygon")
        if not poly or b.get("height_m") is None:
            continue
        arr = np.asarray(poly)
        bb = [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]
        best = max(gt, key=lambda g: box_iou(bb, g["bbox"]), default=None)
        if best is None or box_iou(bb, best["bbox"]) < 0.5:
            continue
        b["gt_height_m"] = best["top_height_m"]
        if not b.get("height_is_lower_bound"):
            errs.append(abs(b["height_m"] - best["top_height_m"]))
    return {
        "n_matched": len(errs),
        "mean_abs_err_m": round(float(np.mean(errs)), 4) if errs else None,
        "max_abs_err_m": round(float(np.max(errs)), 4) if errs else None,
    }


# ---- Tracking -------------------------------------------------------------


@router.get("/projects/{project_id}/studio/tracks")
@_errors
def list_tracks(project_id: str) -> dict:
    return {"tracks": track.list_tracks(_store(project_id))}


@router.post("/projects/{project_id}/studio/tracks")
@_errors
def start_track(project_id: str, body: TrackBody) -> dict:
    return track.start_track(_store(project_id), body.model_dump())


@router.get("/projects/{project_id}/studio/tracks/{track_id}")
@_errors
def get_track(project_id: str, track_id: str) -> dict:
    return track.track_detail(_store(project_id), track_id)


@router.delete("/projects/{project_id}/studio/tracks/{track_id}")
@_errors
def delete_track(project_id: str, track_id: str) -> dict:
    track.delete_track(_store(project_id), track_id)
    return {"deleted": track_id}


@router.get("/projects/{project_id}/studio/tracks/{track_id}/overlay.mp4")
@_errors
def track_video(project_id: str, track_id: str) -> FileResponse:
    st = _store(project_id)
    p = st.tracks_dir / track_id / "overlay.mp4"
    if not p.exists():
        raise NotFound(f"{track_id}/overlay.mp4")
    return FileResponse(p, media_type="video/mp4")


# ---- Dataset download -----------------------------------------------------


@router.get("/projects/{project_id}/studio/dataset.zip")
@_errors
def download_dataset(project_id: str, task: str = Query("segment")) -> FileResponse:
    """YOLO-format zip of the current labels (Roboflow-style dataset export)."""
    st = _store(project_id)
    tmp = Path(tempfile.mkdtemp(prefix="studio_export_"))
    try:
        summary = export_yolo_dataset(st, tmp / "dataset", task)
        # Portable yaml: with no `path:` key Ultralytics resolves train/val
        # relative to the yaml's own folder, wherever the zip is unpacked.
        lines = summary.data_yaml.read_text().splitlines()
        summary.data_yaml.write_text("\n".join(ln for ln in lines if not ln.startswith("path:")) + "\n")
        zip_base = tmp / f"{project_id}_{task}"
        shutil.make_archive(str(zip_base), "zip", root_dir=str(tmp / "dataset"))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return FileResponse(
        f"{zip_base}.zip", filename=f"{project_id}_{task}_yolo.zip",
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True),
    )
