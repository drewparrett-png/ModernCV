"""Generic predict for every model Studio can run → JSON the GUI can draw.

Model specs (what the GUI sends):

    {"kind": "yolo",    "family": "26|11", "task": "detect|segment|classify|pose|obb", "size": "n|s|m|l|x"}
                        ("yolo26" without a family is accepted as YOLO26)
    {"kind": "trained", "model_id": "...", "artifact": "best.pt" | "<export file>"}
    {"kind": "yoloe-text", "family": "26|11", "size": "s|m|l", "classes": ["carton", …]}
    {"kind": "yoloe-pf",   "family": "26|11", "size": "s|m|l"}          # prompt-free

Prediction JSON:

    {"task", "model_label", "names": {id: name}, "image": {"width", "height"},
     "speed": {"preprocess", "inference", "postprocess"},   # ms, from Ultralytics
     "detections": [{"class_id", "class_name", "score", "bbox",
                     "polygon"?, "keypoints"?, "obb"?}],
     "classification": {"top": [{"class_id", "class_name", "score"}]} | null}
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.studio import engines
from pipeline.studio.geometry import simplify_polygon
from pipeline.studio.store import Invalid, NotFound, StudioStore, check_id

PREDICT_PARAM_KEYS = (
    "conf", "iou", "imgsz", "max_det", "agnostic_nms", "augment", "classes", "retina_masks", "half", "end2end",
)


def trained_model_dir(store: StudioStore, model_id: str) -> Path:
    d = store.models_dir / check_id(model_id, "model id")
    if not (d / "model.json").exists():
        raise NotFound(model_id)
    return d


def resolve_model(store: StudioStore, spec: dict, nms_head: bool = False) -> tuple[Any, str, bool]:
    """→ (Ultralytics model, human label, is_yolo26_family).

    `nms_head` selects the one-to-many + NMS head of YOLO26 checkpoints
    (see `engines.load_yolo`); it is ignored for other families, which only
    have that head.
    """
    kind = spec.get("kind")
    if kind in ("yolo", "yolo26"):
        fam = engines.spec_family(spec)
        name = engines.yolo_name(fam, spec.get("task", "detect"), spec.get("size", "n"))
        is26 = fam == "26"
        return engines.load_yolo(engines.ensure_weights(name), nms_head and is26), name.removesuffix(".pt"), is26
    if kind == "trained":
        mdir = trained_model_dir(store, str(spec.get("model_id")))
        artifact = check_id(str(spec.get("artifact") or "best.pt"), "artifact")
        path = (mdir / artifact) if artifact == "best.pt" else (mdir / "exports" / artifact)
        if not path.exists():
            raise NotFound(f"{spec.get('model_id')}/{artifact}")
        import json

        meta = json.loads((mdir / "model.json").read_text())
        is26 = str(meta.get("family") or "26") == "26"
        return (
            engines.load_yolo(path, nms_head and is26 and artifact == "best.pt"),
            f"{meta.get('name', mdir.name)} ({artifact})",
            is26,
        )
    if kind in ("yoloe-text", "yoloe-pf"):
        fam, size = str(spec.get("family", "26")), str(spec.get("size", "s"))
        model = engines.load_yoloe(fam, size, prompt_free=(kind == "yoloe-pf"))
        return model, engines.yoloe_name(fam, size, kind == "yoloe-pf").removesuffix(".pt"), False
    raise Invalid(f"unknown model kind {kind!r}")


def predict_kwargs(params: dict, allow_end2end: bool) -> dict:
    kw: dict[str, Any] = {"device": engines.device(), "verbose": False}
    for k in PREDICT_PARAM_KEYS:
        v = params.get(k)
        if v is None or (k == "classes" and not v):
            continue
        if k == "end2end" and not allow_end2end:
            continue
        kw[k] = v
    return kw


def configure_head(model: Any, params: dict) -> Optional[bool]:
    """Keep a cached YOLO26 model's head settings in sync; returns end2end.

    Ultralytics copies `max_det` / `agnostic_nms` into the end-to-end head,
    and `half` into the backend, only when a predictor is first built, so on
    a cached model later changes would be silently ignored — drop the
    predictor when they change.
    """
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "end2end"):
        return None
    cfg = (params.get("max_det"), bool(params.get("agnostic_nms")), bool(params.get("half")))
    if getattr(model, "_studio_head_cfg", None) != cfg:
        model.predictor = None
        model._studio_head_cfg = cfg
    return bool(inner.end2end)


def run_predict(store: StudioStore, spec: dict, image_bgr: np.ndarray, params: Optional[dict] = None) -> dict:
    params = params or {}
    with engines.INFER_LOCK:
        model, label, is26 = resolve_model(store, spec, nms_head=params.get("end2end") is False)
        if spec.get("kind") == "yoloe-text":
            classes = [str(c).strip() for c in spec.get("classes") or [] if str(c).strip()]
            if not classes:
                raise Invalid("text prompt needs at least one class name")
            hint = engines.missing_dep("clip")
            if hint:
                raise Invalid(hint)
            engines.yoloe_set_text_classes(model, classes)
        # YOLO26's end2end head is NMS-free; `end2end=False` switches to the
        # one-to-many head + NMS. Other families reject the flag.
        kw = predict_kwargs(params, allow_end2end=is26)
        e2e = configure_head(model, params) if is26 else None
        if e2e is not None:
            kw["end2end"] = e2e
        res = model.predict(image_bgr, **kw)[0]
        out = results_to_json(res)
    out["model_label"] = label
    out["end2end"] = e2e
    return out


def results_to_json(r: Any, polygons: bool = True) -> dict:
    """Ultralytics `Results` → plain JSON (pixel coords of the original image)."""
    from ultralytics.utils import ops

    names = {int(k): str(v) for k, v in (r.names or {}).items()}
    h, w = r.orig_shape[:2]
    out: dict[str, Any] = {
        "task": None,
        "names": names,
        "image": {"width": int(w), "height": int(h)},
        "speed": {k: round(float(v), 2) for k, v in (r.speed or {}).items() if v is not None},
        "detections": [],
        "classification": None,
    }

    if r.probs is not None:
        out["task"] = "classify"
        top = []
        for cid, conf in zip(r.probs.top5, r.probs.top5conf.tolist()):
            top.append({"class_id": int(cid), "class_name": names.get(int(cid), str(cid)), "score": round(float(conf), 4)})
        out["classification"] = {"top": top}
        return out

    if getattr(r, "obb", None) is not None:
        out["task"] = "obb"
        obb = r.obb
        pts = obb.xyxyxyxy.cpu().numpy()
        xywhr = obb.xywhr.cpu().numpy()
        conf = obb.conf.cpu().numpy()
        cls = obb.cls.cpu().numpy()
        obb_ids = obb.id.cpu().numpy().astype(int) if getattr(obb, "id", None) is not None else None
        for i in range(len(conf)):
            p = pts[i]
            cid = int(cls[i])
            cx, cy, bw, bh, rot = (float(v) for v in xywhr[i])
            det: dict[str, Any] = {
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "score": round(float(conf[i]), 4),
                "bbox": [round(float(p[:, 0].min()), 1), round(float(p[:, 1].min()), 1),
                         round(float(p[:, 0].max()), 1), round(float(p[:, 1].max()), 1)],
                "obb": {
                    "cx": round(cx, 1), "cy": round(cy, 1), "w": round(bw, 1), "h": round(bh, 1),
                    "angle_deg": round(math.degrees(rot), 2),
                    "points": [[round(float(x), 1), round(float(y), 1)] for x, y in p],
                },
            }
            if obb_ids is not None:
                det["track_id"] = int(obb_ids[i])
            out["detections"].append(det)
        return out

    boxes = r.boxes
    if boxes is None:
        return out
    out["task"] = "segment" if r.masks is not None else ("pose" if r.keypoints is not None else "detect")
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

    segs: list[Optional[np.ndarray]] = [None] * len(conf)
    if polygons and r.masks is not None and len(conf):
        raw = ops.masks2segments(r.masks.data, strategy="largest")
        segs = [
            ops.scale_coords(r.masks.data.shape[1:], s, r.orig_shape, normalize=False) if len(s) else None
            for s in raw
        ]
    kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

    for i in range(len(conf)):
        cid = int(cls[i])
        det: dict[str, Any] = {
            "class_id": cid,
            "class_name": names.get(cid, str(cid)),
            "score": round(float(conf[i]), 4),
            "bbox": [round(float(v), 1) for v in xyxy[i]],
        }
        if segs[i] is not None:
            poly = simplify_polygon(segs[i])
            if poly:
                det["polygon"] = poly
        if kpts is not None:
            det["keypoints"] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kpts[i]]
        if track_ids is not None:
            det["track_id"] = int(track_ids[i])
        out["detections"].append(det)
    return out
