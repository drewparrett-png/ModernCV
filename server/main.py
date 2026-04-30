"""FastAPI app — bridges the React Flow GUI to the Python runner.

Project-scoped surface (Phase 1 rearchitecture).

Endpoints:
    GET  /health
    GET  /blocks
    GET  /videos
    GET  /models/cache_status
    POST /run                                     — graph editor (legacy)

    POST /projects                                — create
    GET  /projects                                — list (with summary counters)
    GET  /projects/{pid}                          — full project record
    PATCH /projects/{pid}                         — rename only
    DELETE /projects/{pid}                        — recursive delete

    POST /projects/{pid}/learn                    — kick a Teacher run
    GET  /projects/{pid}/runs                     — list Teacher runs in this project
    GET  /projects/{pid}/runs/{rid}               — manifest + stats + progress
    GET  /projects/{pid}/runs/{rid}/labels        — per_frame.jsonl
    GET  /projects/{pid}/runs/{rid}/frame/{idx}   — JPEG of frame idx
    GET  /projects/{pid}/runs/{rid}/overlay.mp4
    GET  /projects/{pid}/runs/{rid}/frame_states
    PUT  /projects/{pid}/runs/{rid}/frame_states/{frame_idx}
    DELETE /projects/{pid}/runs/{rid}/frame_states/{frame_idx}
    GET  /projects/{pid}/runs/{rid}/detections           — flat list, score_asc
    GET  /projects/{pid}/runs/{rid}/detection_crop/{fi}/{di}.jpg
    DELETE /projects/{pid}/runs/{rid}

    POST /projects/{pid}/optimize                 — kick a Student run
    GET  /projects/{pid}/students                 — list Students in this project
    GET  /projects/{pid}/students/{sid}           — manifest + stats + progress
    GET  /projects/{pid}/students/architectures   — registered trainer names
    POST /projects/{pid}/students/preview-buckets — live bucket preview
    DELETE /projects/{pid}/students/{sid}

    POST /projects/{pid}/students/{sid}/run         — kick a Student-run
    GET  /projects/{pid}/students/{sid}/runs        — list runs
    GET  /projects/{pid}/students/{sid}/runs/{rid}  — manifest+stats+progress
    GET  /projects/{pid}/students/{sid}/runs/{rid}/overlay.mp4
    GET  /projects/{pid}/students/{sid}/runs/{rid}/frame/{idx}
    GET  /projects/{pid}/students/{sid}/runs/{rid}/labels
    DELETE /projects/{pid}/students/{sid}/runs/{rid}
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from pipeline import distill, model_cache, runs as runs_mod
from pipeline.graph import GraphSpec
from pipeline.learn import (
    ensure_learn_worker_started,
    run_learn_in_background,
    stop_learn_worker,
)
from pipeline.models.registry import REGISTRY
from pipeline.optimize import (
    ensure_optimize_worker_started,
    run_optimize_in_background,
    stop_optimize_worker,
)
from pipeline.runner import run as run_graph
from pipeline.student_run import (
    ensure_student_run_worker_started,
    run_student_run_in_background,
    stop_student_run_worker,
)
from pipeline.students import list_trainers
from server.schemas import (
    ArchitecturesResponse,
    BlockKindInfo,
    BlocksResponse,
    CacheStatusModel,
    CacheStatusResponse,
    DetectionRowModel,
    DetectionsResponse,
    FrameStateEntry,
    FrameStatesResponse,
    LearnRequest,
    OptimizeRequest,
    PreviewBucketsAggregate,
    PreviewBucketsPerTeacher,
    PreviewBucketsRequest,
    PreviewBucketsResponse,
    ProjectCreateRequest,
    ProjectModel,
    ProjectPatchRequest,
    ProjectsResponse,
    ProjectSummaryModel,
    PutFrameStateRequest,
    RunDetail,
    RunManifestModel,
    RunPatchRequest,
    RunProgressModel,
    RunRequest,
    RunsResponse,
    RunStatsModel,
    StudentDetail,
    StudentManifestModel,
    StudentRunDetail,
    StudentRunModel,
    StudentRunRequest,
    StudentRunStatsModel,
    StudentRunsResponse,
    StudentStatsModel,
    StudentsResponse,
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DATA_DIR = Path("data")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    n = runs_mod.mark_stale_runs_failed()
    if n:
        log.info("startup: marked %d stale runs as failed", n)
    ensure_learn_worker_started()
    ensure_optimize_worker_started()
    ensure_student_run_worker_started()
    try:
        yield
    finally:
        stop_student_run_worker()
        stop_optimize_worker()
        stop_learn_worker()


app = FastAPI(title="ModernCV", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Health / discovery ---------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/videos")
def videos() -> dict:
    if not DATA_DIR.exists():
        return {"videos": [], "data_dir": str(DATA_DIR.resolve()), "count": 0}
    found = sorted(
        str(p)
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return {"videos": found, "data_dir": str(DATA_DIR.resolve()), "count": len(found)}


@app.get("/videos/info")
def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"cannot open: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return {"frame_count": frame_count, "fps": fps, "duration_seconds": duration}


@app.get("/models/cache_status", response_model=CacheStatusResponse)
def cache_status_endpoint(impls: str = "") -> CacheStatusResponse:
    impl_list = [s.strip() for s in impls.split(",") if s.strip()]
    return CacheStatusResponse(
        impls=[
            CacheStatusModel(**model_cache.cache_status(i).__dict__)
            for i in impl_list
        ]
    )


@app.get("/blocks", response_model=BlocksResponse)
def blocks() -> BlocksResponse:
    return BlocksResponse(
        blocks=[
            BlockKindInfo(kind=kind.value, impls=impls)
            for kind, impls in REGISTRY.items()
        ]
    )


@app.post("/run")
def run_endpoint(req: RunRequest) -> dict:
    """Graph-editor passthrough — kept for the dev tools page."""
    graph = GraphSpec.from_dict(req.graph.model_dump())
    n_frames = 0
    last_error: str | None = None
    try:
        for _batch in run_graph(graph):
            n_frames += 1
    except NotImplementedError as e:
        last_error = str(e)
    return {
        "frames_processed": n_frames,
        "error": last_error,
        "graph_node_count": len(graph.nodes),
    }


# ---- Projects -------------------------------------------------------------


def _require_project(project_id: str) -> runs_mod.Project:
    pdir = runs_mod.project_dir(project_id)
    if not (pdir / runs_mod.PROJECT_FILE).exists():
        raise HTTPException(status_code=404, detail=f"no such project: {project_id}")
    return runs_mod.read_project(pdir)


@app.post("/projects", response_model=ProjectModel)
def create_project_endpoint(req: ProjectCreateRequest) -> ProjectModel:
    try:
        project = runs_mod.create_project(
            name=req.name,
            task=req.task,
            prompts=req.prompts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ProjectModel(**asdict(project))


@app.get("/projects", response_model=ProjectsResponse)
def list_projects_endpoint() -> ProjectsResponse:
    out: list[ProjectSummaryModel] = []
    for p in runs_mod.list_projects():
        counts = runs_mod.project_summary_counts(p.id)
        out.append(
            ProjectSummaryModel(
                id=p.id,
                name=p.name,
                task=p.task,  # type: ignore[arg-type]
                prompts=p.prompts,
                created_at=p.created_at,
                **counts,
            )
        )
    return ProjectsResponse(projects=out)


@app.get("/projects/{project_id}", response_model=ProjectModel)
def get_project_endpoint(project_id: str) -> ProjectModel:
    project = _require_project(project_id)
    return ProjectModel(**asdict(project))


@app.patch("/projects/{project_id}", response_model=ProjectModel)
def patch_project_endpoint(project_id: str, req: ProjectPatchRequest) -> ProjectModel:
    _require_project(project_id)
    project = runs_mod.rename_project(project_id, req.name)
    return ProjectModel(**asdict(project))


@app.delete("/projects/{project_id}")
def delete_project_endpoint(project_id: str) -> dict:
    _require_project(project_id)
    runs_mod.delete_project(project_id)
    return {"deleted": project_id}


# ---- Per-project: Teacher runs --------------------------------------------


def _manifest_to_model(
    project_id: str, manifest: runs_mod.RunManifest, rdir: Path
) -> RunManifestModel:
    """Wrap a RunManifest as the API model with `project_id` and the
    review_status + frame counts derived from disk."""
    status, n_reviewed, n_total = runs_mod.derive_review_progress(rdir)
    return RunManifestModel(
        project_id=project_id,
        review_status=status,  # type: ignore[arg-type]
        n_frames_reviewed=n_reviewed,
        n_frames_total=n_total,
        **manifest.__dict__,
    )


def _run_detail(
    project_id: str,
    run_id: str,
    threshold_override: Optional[float] = None,
) -> RunDetail:
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    manifest = runs_mod.read_manifest(rdir)
    threshold = (
        threshold_override
        if threshold_override is not None
        else manifest.display_threshold
    )
    stats = runs_mod.compute_stats_at_threshold(rdir, threshold)
    progress = runs_mod.read_progress(rdir)
    return RunDetail(
        manifest=_manifest_to_model(project_id, manifest, rdir),
        stats=RunStatsModel(**stats.__dict__) if stats else None,
        progress=RunProgressModel(**progress.__dict__) if progress else None,
    )


@app.post("/projects/{project_id}/learn", response_model=RunDetail)
def learn_endpoint(project_id: str, req: LearnRequest) -> RunDetail:
    project = _require_project(project_id)
    try:
        manifest = run_learn_in_background(
            project_id=project_id,
            task=project.task,
            prompts=project.prompts,
            video_path=req.video_path,
            detect_impl=req.detect_impl,
            segment_impl=req.segment_impl,
            reid_impl=req.reid_impl,
            track_impl=req.track_impl,
            max_frames=req.max_frames,
            frame_stride=req.frame_stride,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _run_detail(project_id, manifest.id)


@app.get("/projects/{project_id}/runs", response_model=RunsResponse)
def runs_list(project_id: str) -> RunsResponse:
    _require_project(project_id)
    runs: list[RunManifestModel] = []
    for m in runs_mod.list_runs(project_id):
        rdir = runs_mod.run_dir(project_id, m.id)
        runs.append(_manifest_to_model(project_id, m, rdir))
    return RunsResponse(runs=runs)


@app.get("/projects/{project_id}/runs/{run_id}", response_model=RunDetail)
def run_detail(
    project_id: str,
    run_id: str,
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Preview override for the post-hoc display threshold. When "
            "omitted the manifest's `display_threshold` is used. Does "
            "not persist — pass via PATCH to save."
        ),
    ),
) -> RunDetail:
    _require_project(project_id)
    return _run_detail(project_id, run_id, threshold_override=threshold)


@app.patch("/projects/{project_id}/runs/{run_id}", response_model=RunDetail)
def patch_run(
    project_id: str, run_id: str, req: RunPatchRequest
) -> RunDetail:
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    runs_mod.set_display_threshold(rdir, req.display_threshold)
    return _run_detail(project_id, run_id)


@app.get("/projects/{project_id}/runs/{run_id}/labels")
def run_labels(project_id: str, run_id: str) -> Response:
    _require_project(project_id)
    p = (
        runs_mod.run_dir(project_id, run_id)
        / runs_mod.LABELS_DIR
        / runs_mod.PER_FRAME_NAME
    )
    if not p.exists():
        raise HTTPException(status_code=404, detail="no per-frame labels yet")
    return Response(content=p.read_bytes(), media_type="application/x-ndjson")


@app.get("/projects/{project_id}/runs/{run_id}/frame/{idx}")
def run_frame(
    project_id: str,
    run_id: str,
    idx: int,
    source: str = Query("overlay", pattern="^(raw|overlay)$"),
) -> Response:
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")

    if source == "overlay":
        video_path = str(rdir / runs_mod.OVERLAY_NAME)
    else:
        manifest = runs_mod.read_manifest(rdir)
        video_path = manifest.video_path

    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail=f"cannot open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=404, detail=f"frame {idx} unavailable")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(status_code=500, detail="jpeg encode failed")
        return Response(content=bytes(buf), media_type="image/jpeg")
    finally:
        cap.release()


@app.get("/projects/{project_id}/runs/{run_id}/overlay.mp4")
def run_overlay(project_id: str, run_id: str) -> FileResponse:
    _require_project(project_id)
    p = runs_mod.run_dir(project_id, run_id) / runs_mod.OVERLAY_NAME
    if not p.exists():
        raise HTTPException(status_code=404, detail="overlay not yet written")
    return FileResponse(p, media_type="video/mp4")


@app.get(
    "/projects/{project_id}/runs/{run_id}/frame_states",
    response_model=FrameStatesResponse,
)
def get_frame_states(project_id: str, run_id: str) -> FrameStatesResponse:
    """Return all per-frame review entries for a run.

    Phase 3 source-of-truth: each frame can carry one of `curated`,
    `confirmed_empty`, `marked_missed`. Frames absent from the response
    are unreviewed.
    """
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    raw = runs_mod.read_frame_states(rdir)
    return FrameStatesResponse(
        frame_states={
            str(k): FrameStateEntry(
                state=v["state"],
                rejected_dets=v.get("rejected_dets", []),
            )
            for k, v in raw.items()
        }
    )


@app.put(
    "/projects/{project_id}/runs/{run_id}/frame_states/{frame_idx}",
    response_model=FrameStateEntry,
)
def put_frame_state(
    project_id: str,
    run_id: str,
    frame_idx: int,
    req: PutFrameStateRequest,
) -> FrameStateEntry:
    """Set or update one frame's review state.

    Validation lives in `runs_mod.set_frame_state` and surfaces as 400 on
    `ValueError` (invalid state, rejected_dets with non-curated state, or
    out-of-range det index). Side effect: stamps `manifest.approved_at`
    when this PUT brings the run to full coverage.
    """
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    if req.rejected_dets is not None and req.state != "curated":
        raise HTTPException(
            status_code=400,
            detail="rejected_dets only allowed with state='curated'",
        )
    try:
        entry = runs_mod.set_frame_state(
            rdir,
            frame_idx,
            req.state,
            rejected_dets=req.rejected_dets,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return FrameStateEntry(
        state=entry["state"],
        rejected_dets=entry.get("rejected_dets", []),
    )


@app.delete(
    "/projects/{project_id}/runs/{run_id}/frame_states/{frame_idx}",
    status_code=204,
)
def delete_frame_state(project_id: str, run_id: str, frame_idx: int) -> Response:
    """Unset one frame's review state.

    No-op if the frame had no entry. Clears `approved_at` if removing the
    entry takes the run below full coverage.
    """
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    runs_mod.unset_frame_state(rdir, frame_idx)
    return Response(status_code=204)


@app.get(
    "/projects/{project_id}/runs/{run_id}/detections",
    response_model=DetectionsResponse,
)
def list_detections(
    project_id: str,
    run_id: str,
    sort: str = Query("score_asc", pattern="^score_asc$"),
    limit: Optional[int] = Query(None, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> DetectionsResponse:
    """Run-level flat list of detections for crop-flip review.

    Default sort is `score_asc` — lowest-confidence first, the order the
    user most wants to walk because borderline boxes are where curation
    pays off. Pagination is optional; for a few hundred detections the GUI
    is happy to hold the full list and the `total` field doubles as the
    "n / N" position indicator.
    """
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")

    rows = runs_mod.iter_detection_rows(rdir)
    if sort == "score_asc":
        # Stable secondary key keeps the order deterministic for tied
        # scores (a 0.05 floor produces lots of 0.05 detections).
        rows.sort(key=lambda r: (r.score, r.frame_idx, r.det_idx))

    total = len(rows)
    end = offset + limit if limit is not None else total
    page = rows[offset:end]
    return DetectionsResponse(
        detections=[DetectionRowModel(**r.__dict__) for r in page],
        total=total,
    )


@app.get("/projects/{project_id}/runs/{run_id}/detection_crop/{frame_idx}/{det_idx}.jpg")
def detection_crop(
    project_id: str,
    run_id: str,
    frame_idx: int,
    det_idx: int,
    pad: int = Query(24, ge=0, le=512),
) -> Response:
    """Cropped JPEG of a single detection, with the box outlined.

    The crop is cached on disk under `<run>/crops/<fi>_<di>_p<pad>.jpg`;
    `pad` is in the filename so the slider effectively cache-busts. Crops
    are derived from the *raw* video (the manifest's `video_path`) so the
    overlay's drawn boxes don't double-render with our outline.
    """
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")

    cache_path = runs_mod.detection_crop_path(rdir, frame_idx, det_idx, pad)
    if cache_path.exists():
        return Response(content=cache_path.read_bytes(), media_type="image/jpeg")

    bbox = runs_mod.detection_bbox_xyxy(rdir, frame_idx, det_idx)
    if bbox is None:
        raise HTTPException(
            status_code=404,
            detail=f"no detection at frame={frame_idx} det={det_idx}",
        )

    manifest = runs_mod.read_manifest(rdir)
    video_path = manifest.video_path
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail=f"cannot open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(
                status_code=404, detail=f"frame {frame_idx} unavailable"
            )
    finally:
        cap.release()

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # Clip the box itself first (bad coords = misordered or beyond frame),
    # then the padded crop window. Outline coords are in *crop space*.
    bx1 = max(0, min(int(round(x1)), w - 1))
    by1 = max(0, min(int(round(y1)), h - 1))
    bx2 = max(0, min(int(round(x2)), w - 1))
    by2 = max(0, min(int(round(y2)), h - 1))
    if bx2 <= bx1 or by2 <= by1:
        raise HTTPException(status_code=422, detail="degenerate bbox")

    cx1 = max(0, bx1 - pad)
    cy1 = max(0, by1 - pad)
    cx2 = min(w, bx2 + pad)
    cy2 = min(h, by2 + pad)
    crop = frame[cy1:cy2, cx1:cx2].copy()

    # Outline relative to crop origin. Use a 2px green box for the same
    # reason the overlay does — visible on most natural backgrounds.
    cv2.rectangle(
        crop,
        (bx1 - cx1, by1 - cy1),
        (bx2 - cx1, by2 - cy1),
        (0, 255, 0),
        2,
    )

    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="jpeg encode failed")
    data = bytes(buf)

    # Persist cache on first hit. Best-effort: a write failure (full disk,
    # permissions) shouldn't fail the request.
    try:
        runs_mod.crops_dir(rdir).mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    except OSError as e:
        log.warning("failed to cache crop %s: %s", cache_path, e)

    return Response(content=data, media_type="image/jpeg")


@app.delete("/projects/{project_id}/runs/{run_id}")
def delete_run(project_id: str, run_id: str) -> dict:
    _require_project(project_id)
    rdir = runs_mod.run_dir(project_id, run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    runs_mod.delete_run(rdir)
    return {"deleted": run_id}


# ---- Per-project: Students ------------------------------------------------


def _student_manifest_to_model(
    project_id: str, manifest: runs_mod.StudentManifest
) -> StudentManifestModel:
    return StudentManifestModel(project_id=project_id, **manifest.__dict__)


def _student_detail(project_id: str, student_id: str) -> StudentDetail:
    rdir = runs_mod.student_dir(project_id, student_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    manifest = runs_mod.read_student_manifest(rdir)
    stats = runs_mod.read_student_stats(rdir)
    progress = runs_mod.read_progress(rdir)
    return StudentDetail(
        manifest=_student_manifest_to_model(project_id, manifest),
        stats=StudentStatsModel(**stats.__dict__) if stats else None,
        progress=RunProgressModel(**progress.__dict__) if progress else None,
    )


@app.post("/projects/{project_id}/optimize", response_model=StudentDetail)
def optimize_endpoint(project_id: str, req: OptimizeRequest) -> StudentDetail:
    project = _require_project(project_id)
    if not req.train_teacher_ids:
        raise HTTPException(
            status_code=400,
            detail="pick at least one Train teacher — a Student needs training data",
        )

    def _load_teacher(tid: str) -> runs_mod.RunManifest:
        tdir = runs_mod.run_dir(project_id, tid)
        if not (tdir / runs_mod.MANIFEST_NAME).exists():
            raise HTTPException(status_code=404, detail=f"no such teacher: {tid}")
        m = runs_mod.read_manifest(tdir)
        if m.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"teacher {tid!r} status is {m.status!r}; "
                    "wait for the Learn run to finish before optimizing"
                ),
            )
        return m

    train_manifests = [_load_teacher(tid) for tid in req.train_teacher_ids]
    eval_manifests = [_load_teacher(tid) for tid in req.eval_teacher_ids]

    tasks = {m.task for m in train_manifests}
    if len(tasks) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Train teachers mix tasks ({}); pick teachers with the same task"
                .format(sorted(tasks))
            ),
        )
    task = train_manifests[0].task
    if task != project.task:
        # Defensive — teachers should always inherit project.task, but a stale
        # manifest could drift. Surface as a 400.
        raise HTTPException(
            status_code=400,
            detail=(
                f"teacher task {task!r} disagrees with project task "
                f"{project.task!r}"
            ),
        )
    bad_eval = [m.id for m in eval_manifests if m.task != task]
    if bad_eval:
        log.warning(
            "eval teachers with task != %s: %s — mAP will likely be 0", task, bad_eval
        )

    try:
        manifest = run_optimize_in_background(
            project_id=project_id,
            train_teacher_ids=req.train_teacher_ids,
            eval_teacher_ids=req.eval_teacher_ids,
            task=task,
            detect_impl=req.detect_impl,
            segment_impl=req.segment_impl,
            track_impl=req.track_impl,
            epochs=req.epochs,
            export_threshold=req.export_threshold,
            t_low=req.t_low,
            treat_empty_as_negative=req.treat_empty_as_negative,
            architecture=req.architecture,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _student_detail(project_id, manifest.id)


@app.get(
    "/projects/{project_id}/students/architectures",
    response_model=ArchitecturesResponse,
)
def students_architectures(project_id: str) -> ArchitecturesResponse:
    _require_project(project_id)
    return ArchitecturesResponse(architectures=list_trainers())


@app.get("/projects/{project_id}/students", response_model=StudentsResponse)
def students_list(project_id: str) -> StudentsResponse:
    _require_project(project_id)
    return StudentsResponse(
        students=[
            _student_manifest_to_model(project_id, m)
            for m in runs_mod.list_students(project_id)
        ]
    )


@app.get(
    "/projects/{project_id}/students/{student_id}", response_model=StudentDetail
)
def student_detail(project_id: str, student_id: str) -> StudentDetail:
    _require_project(project_id)
    return _student_detail(project_id, student_id)


@app.post(
    "/projects/{project_id}/students/preview-buckets",
    response_model=PreviewBucketsResponse,
)
def preview_buckets(
    project_id: str, req: PreviewBucketsRequest
) -> PreviewBucketsResponse:
    _require_project(project_id)
    per_teacher: list[PreviewBucketsPerTeacher] = []
    agg_pos = 0
    agg_unc = 0
    agg_neg = 0
    class_names_union: list[str] = []
    seen_classes: set[str] = set()

    for tid in req.teacher_ids:
        tdir = runs_mod.run_dir(project_id, tid)
        if not (tdir / runs_mod.MANIFEST_NAME).exists():
            raise HTTPException(status_code=404, detail=f"no such teacher: {tid}")
        coco_path = tdir / runs_mod.LABELS_DIR / runs_mod.COCO_NAME
        if not coco_path.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"teacher {tid!r} has no coco.json yet — finish a Learn run first"
                ),
            )
        try:
            coco = json.loads(coco_path.read_text())
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"teacher {tid!r} coco.json is malformed: {e}",
            )

        frame_states = runs_mod.read_frame_states(tdir)
        distill._apply_frame_state_overrides(coco, frame_states)

        buckets = distill.classify_frames(
            coco, export_threshold=req.export_threshold, t_low=req.t_low,
        )
        positive = len(buckets.positive)
        uncertain = len(buckets.uncertain)
        true_negative = len(buckets.true_negative)

        if req.treat_empty_as_negative:
            true_negative += uncertain
            uncertain = 0

        per_teacher.append(
            PreviewBucketsPerTeacher(
                teacher_id=tid,
                positive=positive,
                uncertain=uncertain,
                true_negative=true_negative,
            )
        )
        agg_pos += positive
        agg_unc += uncertain
        agg_neg += true_negative

        for cat in coco.get("categories", []):
            cname = cat.get("name")
            if isinstance(cname, str) and cname not in seen_classes:
                seen_classes.add(cname)
                class_names_union.append(cname)

    return PreviewBucketsResponse(
        aggregate=PreviewBucketsAggregate(
            positive=agg_pos,
            uncertain=agg_unc,
            true_negative=agg_neg,
            n_classes=len(class_names_union),
            class_names=class_names_union,
        ),
        per_teacher=per_teacher,
    )


@app.get("/projects/{project_id}/students/{student_id}/training_curve")
def training_curve(project_id: str, student_id: str) -> dict:
    """Read the Ultralytics-emitted `results.csv` for a Student and
    return per-epoch loss + val mAP as parallel arrays.

    Useful for the GUI's training-curve charts. Returns 404 if the
    file isn't there yet — happens during the prep / first epoch
    window. The GUI renders an empty-state placeholder in that case.
    """
    _require_project(project_id)
    rdir = runs_mod.student_dir(project_id, student_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    csv_path = rdir / "ultralytics" / "train" / "results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="no results.csv yet")

    epochs: list[int] = []
    train_loss: list[float] = []
    val_map50: list[float] = []
    val_map50_95: list[float] = []
    try:
        import csv as _csv

        with csv_path.open() as f:
            reader = _csv.DictReader(f)
            # Ultralytics' CSV column names have shifted across versions —
            # tolerate "epoch" with or without leading whitespace, and
            # accept either of two naming conventions for box loss / mAP.
            for raw in reader:
                row = {k.strip(): v for k, v in raw.items() if k}
                # Pick the first matching key from each candidate list.
                def pick(keys: list[str]) -> Optional[str]:
                    for k in keys:
                        if k in row and row[k] not in ("", None):
                            return row[k]
                    return None

                ep = pick(["epoch"])
                tl = pick(["train/box_loss", "train/loss"])
                m50 = pick(["metrics/mAP50(B)", "metrics/mAP_0.5", "val/mAP50"])
                m5095 = pick(
                    ["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "val/mAP50-95"]
                )
                if ep is None:
                    continue
                try:
                    epochs.append(int(float(ep)))
                    train_loss.append(float(tl) if tl is not None else 0.0)
                    val_map50.append(float(m50) if m50 is not None else 0.0)
                    val_map50_95.append(float(m5095) if m5095 is not None else 0.0)
                except (TypeError, ValueError):
                    continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse results.csv: {e}")

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_map50": val_map50,
        "val_map50_95": val_map50_95,
    }


@app.get("/projects/{project_id}/students/{student_id}/samples")
def list_student_samples(project_id: str, student_id: str) -> dict:
    """List rendered eval-prediction-comparison images, grouped by
    eval teacher. Returns {teacher_id: [filename, ...]} sorted by
    filename. Empty dict when nothing has been rendered yet (run
    didn't complete, or pre-Phase-7 student).
    """
    _require_project(project_id)
    rdir = runs_mod.student_dir(project_id, student_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    samples_root = rdir / "samples"
    out: dict[str, list[str]] = {}
    if samples_root.exists():
        for tdir in sorted(samples_root.iterdir()):
            if not tdir.is_dir():
                continue
            jpgs = sorted(p.name for p in tdir.glob("*.jpg"))
            if jpgs:
                out[tdir.name] = jpgs
    return {"samples": out}


@app.get(
    "/projects/{project_id}/students/{student_id}/samples/{teacher_id}/{name}"
)
def get_student_sample(
    project_id: str, student_id: str, teacher_id: str, name: str
) -> FileResponse:
    """Serve one rendered comparison image. `name` must end in `.jpg`
    and contain no path separators — guards against directory traversal."""
    _require_project(project_id)
    if "/" in name or "\\" in name or not name.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="invalid sample name")
    rdir = runs_mod.student_dir(project_id, student_id)
    path = rdir / "samples" / teacher_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="sample not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.delete("/projects/{project_id}/students/{student_id}")
def delete_student(project_id: str, student_id: str) -> dict:
    _require_project(project_id)
    rdir = runs_mod.student_dir(project_id, student_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    runs_mod.delete_run(rdir)
    return {"deleted": student_id}


# ---- Per-student: run inference (Phase 5) ---------------------------------


def _require_student(project_id: str, student_id: str) -> Path:
    """Validate the student exists and return its dir.

    Used by every student-run endpoint to keep the 404 path uniform.
    """
    sdir = runs_mod.student_dir(project_id, student_id)
    if not (sdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(
            status_code=404, detail=f"no such student: {student_id}"
        )
    return sdir


def _student_run_to_model(m: runs_mod.StudentRunManifest) -> StudentRunModel:
    return StudentRunModel(**m.__dict__)


def _student_run_detail(
    project_id: str, student_id: str, run_id: str
) -> StudentRunDetail:
    rdir = runs_mod.student_run_dir(project_id, student_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(
            status_code=404, detail=f"no such student-run: {run_id}"
        )
    manifest = runs_mod.read_student_run_manifest(rdir)
    stats = runs_mod.read_student_run_stats(rdir)
    progress = runs_mod.read_progress(rdir)
    return StudentRunDetail(
        manifest=_student_run_to_model(manifest),
        stats=StudentRunStatsModel(**stats.__dict__) if stats else None,
        progress=RunProgressModel(**progress.__dict__) if progress else None,
    )


@app.post(
    "/projects/{project_id}/students/{student_id}/run",
    response_model=StudentRunDetail,
)
def start_student_run(
    project_id: str, student_id: str, req: StudentRunRequest
) -> StudentRunDetail:
    """Kick off a Student running inference against an arbitrary input.

    Validates the input synchronously (so the user sees a 400 immediately
    rather than a queued-then-failed state for typos) before enqueueing.
    """
    _require_project(project_id)
    _require_student(project_id, student_id)
    try:
        manifest = run_student_run_in_background(
            project_id=project_id,
            student_id=student_id,
            input_kind=req.input_kind,
            input_ref=req.input_ref,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _student_run_detail(project_id, student_id, manifest.id)


@app.get(
    "/projects/{project_id}/students/{student_id}/runs",
    response_model=StudentRunsResponse,
)
def list_student_runs_endpoint(
    project_id: str, student_id: str
) -> StudentRunsResponse:
    _require_project(project_id)
    _require_student(project_id, student_id)
    runs = runs_mod.list_student_runs(project_id, student_id)
    return StudentRunsResponse(runs=[_student_run_to_model(m) for m in runs])


@app.get(
    "/projects/{project_id}/students/{student_id}/runs/{run_id}",
    response_model=StudentRunDetail,
)
def get_student_run_endpoint(
    project_id: str, student_id: str, run_id: str
) -> StudentRunDetail:
    _require_project(project_id)
    _require_student(project_id, student_id)
    return _student_run_detail(project_id, student_id, run_id)


@app.get(
    "/projects/{project_id}/students/{student_id}/runs/{run_id}/overlay.mp4"
)
def student_run_overlay(
    project_id: str, student_id: str, run_id: str
) -> FileResponse:
    _require_project(project_id)
    _require_student(project_id, student_id)
    rdir = runs_mod.student_run_dir(project_id, student_id, run_id)
    p = rdir / runs_mod.OVERLAY_NAME
    if not p.exists():
        raise HTTPException(status_code=404, detail="overlay not yet written")
    return FileResponse(p, media_type="video/mp4")


@app.get(
    "/projects/{project_id}/students/{student_id}/runs/{run_id}/labels"
)
def student_run_labels(
    project_id: str, student_id: str, run_id: str
) -> Response:
    _require_project(project_id)
    _require_student(project_id, student_id)
    rdir = runs_mod.student_run_dir(project_id, student_id, run_id)
    p = rdir / runs_mod.PREDICTIONS_DIR / runs_mod.PER_FRAME_NAME
    if not p.exists():
        raise HTTPException(status_code=404, detail="no per-frame predictions yet")
    return Response(content=p.read_bytes(), media_type="application/x-ndjson")


@app.get(
    "/projects/{project_id}/students/{student_id}/runs/{run_id}/frame/{idx}"
)
def student_run_frame(
    project_id: str,
    student_id: str,
    run_id: str,
    idx: int,
    source: str = Query("overlay", pattern="^(raw|overlay)$"),
) -> Response:
    """Single frame from a student-run, raw or overlay.

    `raw` reads the source video the student-run was driven against
    (the `input_ref` for kind=video, or the teacher's video_path for
    kind=teacher_dataset). `overlay` reads the rendered overlay.mp4.
    """
    _require_project(project_id)
    _require_student(project_id, student_id)
    rdir = runs_mod.student_run_dir(project_id, student_id, run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such student-run: {run_id}")

    if source == "overlay":
        video_path = str(rdir / runs_mod.OVERLAY_NAME)
    else:
        manifest = runs_mod.read_student_run_manifest(rdir)
        if manifest.input_kind == "video":
            video_path = manifest.input_ref
        else:
            teacher_dir = runs_mod.run_dir(project_id, manifest.input_ref)
            try:
                tm = runs_mod.read_manifest(teacher_dir)
                video_path = tm.video_path
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail=f"teacher source video missing: {manifest.input_ref}",
                )

    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail=f"cannot open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=404, detail=f"frame {idx} unavailable")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(status_code=500, detail="jpeg encode failed")
        return Response(content=bytes(buf), media_type="image/jpeg")
    finally:
        cap.release()


@app.delete(
    "/projects/{project_id}/students/{student_id}/runs/{run_id}"
)
def delete_student_run_endpoint(
    project_id: str, student_id: str, run_id: str
) -> dict:
    _require_project(project_id)
    _require_student(project_id, student_id)
    rdir = runs_mod.student_run_dir(project_id, student_id, run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such student-run: {run_id}")
    runs_mod.delete_student_run(rdir)
    return {"deleted": run_id}
