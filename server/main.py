"""FastAPI app — bridges the React Flow GUI to the Python runner.

Endpoints:
    GET  /health                          — liveness
    GET  /blocks                          — what blocks exist + impls each supports
    GET  /videos                          — videos found in data/
    POST /run                             — execute an arbitrary graph (Graph Editor mode)
    POST /learn                           — execute a Learn-mode pipeline (writes a run dir)
    GET  /runs                            — list Teacher run manifests
    GET  /runs/{id}                       — manifest + stats for one run
    GET  /runs/{id}/labels                — per_frame.jsonl content (for Inspector)
    GET  /runs/{id}/frame/{idx}           — JPEG of frame idx (?source=raw|overlay)
    GET  /runs/{id}/overlay.mp4           — the overlay video file
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
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
from pipeline.optimize import run_optimize_in_background
# Importing the students package triggers each trainer module's
# `@register(...)` side effect — we read `list_trainers()` for the
# `/students/architectures` endpoint so the dropdown is populated by
# whatever's actually wired today, not a hardcoded list.
from pipeline.students import list_trainers
from pipeline.runner import run as run_graph
from server.schemas import (
    ApproveResponse,
    ArchitecturesResponse,
    BlockKindInfo,
    BlocksResponse,
    CacheStatusModel,
    CacheStatusResponse,
    LearnRequest,
    OptimizeRequest,
    PreviewBucketsAggregate,
    PreviewBucketsPerTeacher,
    PreviewBucketsRequest,
    PreviewBucketsResponse,
    RejectionsResponse,
    RejectToggleRequest,
    RunDetail,
    RunManifestModel,
    RunProgressModel,
    RunRequest,
    RunsResponse,
    RunStatsModel,
    StudentDetail,
    StudentManifestModel,
    StudentStatsModel,
    StudentsResponse,
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DATA_DIR = Path("data")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks.

    Startup:
      1. Sweep the runs directory for any run still flagged 'running' or
         'queued' from a prior process — workers are daemon threads + the
         queue is in-memory, so neither survives a restart. Without this
         sweep, the UI shows a forever-spinning row that never updates.
      2. Spawn the singleton Teacher queue worker so the first /learn
         request doesn't pay the worker-startup latency.

    Shutdown:
      • Push a sentinel onto the Learn queue and join briefly. The worker
        is daemon-flagged so it'll be killed on hard exit anyway, but
        graceful drain on a clean shutdown means the next-startup sweep
        has less to do.
    """
    n = runs_mod.mark_stale_runs_failed()
    if n:
        log.info("startup: marked %d stale runs as failed", n)
    ensure_learn_worker_started()
    try:
        yield
    finally:
        stop_learn_worker()


app = FastAPI(title="ModernCV", version="0.1.0", lifespan=lifespan)

# Permissive in dev — tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/videos")
def videos() -> dict:
    """List video files in data/ — used by the InputNode dropdown.

    Paths are returned relative to the project root so they can be passed
    straight to the OpenCV reader as-is. Sorted alphabetically; nested
    directories (e.g. data/dfl/clip_001.mp4) are walked recursively.
    """
    if not DATA_DIR.exists():
        return {"videos": [], "data_dir": str(DATA_DIR.resolve()), "count": 0}
    found = sorted(
        str(p)
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return {"videos": found, "data_dir": str(DATA_DIR.resolve()), "count": len(found)}


@app.get("/models/cache_status", response_model=CacheStatusResponse)
def cache_status_endpoint(impls: str = "") -> CacheStatusResponse:
    """Per-impl cache status: cached + estimated download size.

    Lets the GUI ask "are you OK with a ~700MB download?" before kicking
    off a Learn run when the user is on a fresh machine. `impls` is a
    comma-separated list, e.g. "groundingdino,sam2-tiny,dinov3-vits16".
    Unknown impls are reported as already-cached / 0 bytes.
    """
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
    graph = GraphSpec.from_dict(req.graph.model_dump())
    n_frames = 0
    last_error: str | None = None
    try:
        for _batch in run_graph(graph):
            n_frames += 1
    except NotImplementedError as e:
        # Expected during scaffold — surface which block stub bit us.
        last_error = str(e)
    return {
        "frames_processed": n_frames,
        "error": last_error,
        "graph_node_count": len(graph.nodes),
    }


# ---- Learn / Optimize / Inspector ------------------------------------------


@app.post("/learn", response_model=RunDetail)
def learn_endpoint(req: LearnRequest) -> RunDetail:
    """Kick off a Learn-mode run in a background thread; return the
    initial manifest immediately so the client can poll /runs/{id} for
    progress.

    With real models wired (GroundingDINO weight download on first run,
    multi-second per-frame inference) a synchronous endpoint would time
    out browsers. The polling endpoint is /runs/{id}.
    """
    try:
        manifest = run_learn_in_background(
            task=req.task,
            prompt=req.prompt,
            prompts=req.prompts,
            video_path=req.video_path,
            detect_impl=req.detect_impl,
            segment_impl=req.segment_impl,
            reid_impl=req.reid_impl,
            track_impl=req.track_impl,
            max_frames=req.max_frames,
            box_threshold=req.box_threshold,
            text_threshold=req.text_threshold,
            full_resolution=req.full_resolution,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _run_detail(manifest.id)


@app.get("/runs", response_model=RunsResponse)
def runs_list() -> RunsResponse:
    return RunsResponse(
        runs=[RunManifestModel(**m.__dict__) for m in runs_mod.list_runs()]
    )


@app.get("/runs/{run_id}", response_model=RunDetail)
def run_detail(run_id: str) -> RunDetail:
    return _run_detail(run_id)


def _run_detail(run_id: str) -> RunDetail:
    rdir = runs_mod.run_dir(run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    manifest = runs_mod.read_manifest(rdir)
    stats = runs_mod.read_stats(rdir)
    progress = runs_mod.read_progress(rdir)
    return RunDetail(
        manifest=RunManifestModel(**manifest.__dict__),
        stats=RunStatsModel(**stats.__dict__) if stats else None,
        progress=RunProgressModel(**progress.__dict__) if progress else None,
    )


@app.get("/runs/{run_id}/labels")
def run_labels(run_id: str) -> Response:
    """Serve labels/per_frame.jsonl as plain text. Frontend slices client-side."""
    p = runs_mod.run_dir(run_id) / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    if not p.exists():
        raise HTTPException(status_code=404, detail="no per-frame labels yet")
    return Response(content=p.read_bytes(), media_type="application/x-ndjson")


@app.get("/runs/{run_id}/frame/{idx}")
def run_frame(
    run_id: str,
    idx: int,
    source: str = Query("overlay", pattern="^(raw|overlay)$"),
) -> Response:
    """Seek to frame `idx` in the source or overlay video, return JPEG bytes.

    Doing the seek server-side is a deliberate choice — it avoids pre-extracting
    every frame as a jpg on disk (which would balloon to hundreds of MB per
    run). cv2.VideoCapture's seek is O(frame_idx) on some codecs, so this isn't
    free; if Inspector scrubbing feels slow we'll add an LRU cache here.
    """
    rdir = runs_mod.run_dir(run_id)
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


@app.get("/runs/{run_id}/overlay.mp4")
def run_overlay(run_id: str) -> FileResponse:
    p = runs_mod.run_dir(run_id) / runs_mod.OVERLAY_NAME
    if not p.exists():
        raise HTTPException(status_code=404, detail="overlay not yet written")
    return FileResponse(p, media_type="video/mp4")


# ---- Rejections / Delete ---------------------------------------------------


@app.get("/runs/{run_id}/rejections", response_model=RejectionsResponse)
def get_rejections(run_id: str) -> RejectionsResponse:
    rdir = runs_mod.run_dir(run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    state = runs_mod.read_rejections(rdir)
    return RejectionsResponse(rejections={str(k): v for k, v in state.items()})


@app.post("/runs/{run_id}/rejections/toggle", response_model=RejectionsResponse)
def toggle_rejection(run_id: str, req: RejectToggleRequest) -> RejectionsResponse:
    rdir = runs_mod.run_dir(run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    state = runs_mod.toggle_rejection(rdir, req.frame_idx, req.det_idx)
    return RejectionsResponse(rejections={str(k): v for k, v in state.items()})


@app.post("/runs/{run_id}/approve", response_model=ApproveResponse)
def approve_run_endpoint(run_id: str) -> ApproveResponse:
    """Mark a completed Teacher run as "Approved as ground truth".

    Idempotent: re-approving an already-approved run returns the existing
    manifest unchanged (the timestamp does NOT shift on a second call —
    that's a deliberate no-op so a double-click doesn't drift the date).

    Errors:
      404 — run does not exist.
      400 — run's status is not "completed". A still-running or failed
            run has nothing meaningful to approve.
    """
    rdir = runs_mod.run_dir(run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    manifest = runs_mod.read_manifest(rdir)
    if manifest.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"run {run_id!r} status is {manifest.status!r}; only "
                "completed runs can be approved as ground truth"
            ),
        )
    updated = runs_mod.approve_run(rdir)
    return ApproveResponse(manifest=RunManifestModel(**updated.__dict__))


@app.post("/runs/{run_id}/unapprove", response_model=ApproveResponse)
def unapprove_run_endpoint(run_id: str) -> ApproveResponse:
    """Clear the approval stamp.

    Idempotent: unapproving an unreviewed/reviewed run is a no-op
    returning the manifest unchanged. Status check still applies — the
    spec disallows touching `approved_at` on non-completed runs in either
    direction so the wire field never lies about state.
    """
    rdir = runs_mod.run_dir(run_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    manifest = runs_mod.read_manifest(rdir)
    if manifest.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"run {run_id!r} status is {manifest.status!r}; only "
                "completed runs can have their approval changed"
            ),
        )
    updated = runs_mod.unapprove_run(rdir)
    return ApproveResponse(manifest=RunManifestModel(**updated.__dict__))


@app.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    rdir = runs_mod.run_dir(run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    # Best-effort: if the run is still flagged "running" we could refuse, but
    # background workers are daemon threads — the user explicitly asked us to
    # delete, so we honor it. The worker will hit a write error and exit.
    runs_mod.delete_run(rdir)
    return {"deleted": run_id}


# ---- Students / Optimize ---------------------------------------------------


@app.post("/optimize", response_model=StudentDetail)
def optimize_endpoint(req: OptimizeRequest) -> StudentDetail:
    """Kick off a Student training run from one or more Teachers' COCO labels.

    `train_teacher_ids` (required, ≥1): merged into the training set.
    `eval_teacher_ids` (optional): held-out Teachers used to score
    transferability — never seen at train time.

    All train teachers must be `completed` and share the same `task`
    (mixing detection + segmentation in one Student doesn't make sense).
    Eval teachers must also be completed; a task mismatch is logged but
    not refused — the trainer surfaces it as a poor mAP, which is more
    informative than an opaque 400.

    Runs in a daemon thread; the client polls /students/{id} for progress.
    """
    if not req.train_teacher_ids:
        raise HTTPException(
            status_code=400,
            detail="pick at least one Train teacher — a Student needs training data",
        )

    # Validate every referenced teacher exists + is completed. Doing this
    # synchronously (not in the worker) means the user sees the error
    # immediately rather than as a "Student failed" row a few seconds later.
    def _load_teacher(tid: str) -> runs_mod.RunManifest:
        tdir = runs_mod.run_dir(tid)
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
    bad_eval = [m.id for m in eval_manifests if m.task != task]
    if bad_eval:
        log.warning(
            "eval teachers with task != %s: %s — mAP will likely be 0", task, bad_eval
        )

    try:
        manifest = run_optimize_in_background(
            train_teacher_ids=req.train_teacher_ids,
            eval_teacher_ids=req.eval_teacher_ids,
            task=task,
            detect_impl=req.detect_impl,
            segment_impl=req.segment_impl,
            track_impl=req.track_impl,
            epochs=req.epochs,
            t_high=req.t_high,
            t_low=req.t_low,
            treat_empty_as_negative=req.treat_empty_as_negative,
            architecture=req.architecture,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _student_detail(manifest.id)


@app.get("/students/architectures", response_model=ArchitecturesResponse)
def students_architectures() -> ArchitecturesResponse:
    """List every registered Student-trainer architecture (Phase 1.4).

    Reads `pipeline.students.list_trainers()` live so newly-wired
    architectures show up in the GUI without any server-side schema
    change. The GUI fetches this once on mount and caches it in the
    Zustand store.
    """
    return ArchitecturesResponse(architectures=list_trainers())


@app.get("/students", response_model=StudentsResponse)
def students_list() -> StudentsResponse:
    return StudentsResponse(
        students=[StudentManifestModel(**m.__dict__) for m in runs_mod.list_students()]
    )


@app.get("/students/{student_id}", response_model=StudentDetail)
def student_detail(student_id: str) -> StudentDetail:
    return _student_detail(student_id)


def _student_detail(student_id: str) -> StudentDetail:
    rdir = runs_mod.run_dir(student_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    manifest = runs_mod.read_student_manifest(rdir)
    stats = runs_mod.read_student_stats(rdir)
    progress = runs_mod.read_progress(rdir)
    return StudentDetail(
        manifest=StudentManifestModel(**manifest.__dict__),
        stats=StudentStatsModel(**stats.__dict__) if stats else None,
        progress=RunProgressModel(**progress.__dict__) if progress else None,
    )


@app.post("/students/preview-buckets", response_model=PreviewBucketsResponse)
def preview_buckets(req: PreviewBucketsRequest) -> PreviewBucketsResponse:
    """Live frame-bucket preview for the New Student form (Phase 0.4).

    Cheap to call — reads each teacher's `coco.json` once, runs the same
    `classify_frames` pass `prepare_yolo_dataset` would run, and returns
    the bucket counts. No frame extraction, no I/O beyond the COCO read.

    Used by the GUI to show "you're about to train on N positive frames,
    drop M uncertain ones" while the user is still tweaking thresholds.
    The 250ms client-side debounce + this O(annotations) server pass is
    fast enough to feel live on every checkbox toggle and slider drag.

    Errors:
      • t_low > t_high  → 422 via Pydantic validator on PreviewBucketsRequest.
      • teacher_id missing on disk     → 404 with the offending id.
      • teacher has no coco.json yet   → 400 (Learn run never finished).
    """
    per_teacher: list[PreviewBucketsPerTeacher] = []
    agg_pos = 0
    agg_unc = 0
    agg_neg = 0
    class_names_union: list[str] = []
    seen_classes: set[str] = set()

    for tid in req.teacher_ids:
        tdir = runs_mod.run_dir(tid)
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

        # Same bucket logic the trainer uses, so the preview is honest.
        # We've already validated t_low <= t_high in the Pydantic model;
        # classify_frames raises ValueError on inversion which would be
        # an internal bug at this point.
        buckets = distill.classify_frames(
            coco, t_high=req.t_high, t_low=req.t_low,
        )
        positive = len(buckets.positive)
        uncertain = len(buckets.uncertain)
        true_negative = len(buckets.true_negative)

        # Escape hatch: reclassify uncertain frames as true_negative *before*
        # reporting counts, to mirror what `prepare_yolo_dataset` would do
        # at training time. The user-visible "uncertain" then drops to 0
        # and the negative count grows — exactly what the trainer would see.
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


@app.delete("/students/{student_id}")
def delete_student(student_id: str) -> dict:
    rdir = runs_mod.run_dir(student_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail=f"no such student: {student_id}")
    runs_mod.delete_run(rdir)
    return {"deleted": student_id}
