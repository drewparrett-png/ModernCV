"""Learn-mode runner.

Wraps `pipeline.runner.run` with the bookkeeping that turns a one-shot
pipeline run into a persisted "Teacher run" — a run directory with a
manifest, an overlay video, per-frame labels, and timing stats.

The actual work is still done by Blocks and Adapters; this module is just
the glue that turns a `LearnRequest` into the right `GraphSpec` and
captures the side products (timing, labels, progress) as frames stream by.

Background execution: the function `run_learn_in_background` returns
immediately after creating the run directory, then runs the pipeline on
a worker thread that updates `progress.json` periodically. The server's
`POST /learn` endpoint uses this; CLI/test callers can use the synchronous
`run_learn` directly.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from pipeline import model_cache, runs as runs_mod
from pipeline.blocks.base import BlockKind
from pipeline.graph import GraphSpec, NodeSpec
from pipeline.runner import run as run_graph

log = logging.getLogger(__name__)


# ---- Sequential job queue --------------------------------------------------
#
# Earlier we spawned a daemon thread per /learn request. With real models
# that meant two parallel runs both held a ~700MB GroundingDINO instance and
# fought for the same compute, dragging per-frame latency from ~1s to ~2s.
# The latency numbers are *the demo* — we need them clean — so Teachers are
# now strictly sequential: one persistent worker, FIFO queue, the rest wait.
#
# `queue.Queue` is thread-safe; the sentinel `None` is a soft shutdown signal
# that the lifespan handler can push when the server stops.


@dataclass
class _LearnJob:
    """Frozen request payload + the run dir/manifest the worker needs."""

    rdir: Path
    project_id: str
    manifest_id: str
    task: str
    # `prompts` is the canonical class list — one user chip per element.
    # `prompt` is a human-readable joined string we keep around for the
    # manifest/UI ("soccer ball, player"). The detector adapter consumes
    # `prompts`; everything user-facing reads `prompt`.
    prompts: list[str]
    prompt: str
    video_path: str
    total_frames: int
    started_at: str
    detect_impl: Optional[str]
    segment_impl: Optional[str]
    reid_impl: Optional[str]
    track_impl: Optional[str]
    max_frames: Optional[int]


_LEARN_QUEUE: "queue.Queue[Optional[_LearnJob]]" = queue.Queue()
_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: Optional[threading.Thread] = None


def ensure_learn_worker_started() -> None:
    """Idempotent — spawn the singleton Teacher worker if not already alive.

    Called by the FastAPI lifespan startup hook (so the worker is ready for
    the first /learn request) and as a safety net inside
    `run_learn_in_background` (in case anything ever bypasses the lifespan
    setup).
    """
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _WORKER_THREAD = threading.Thread(
            target=_learn_worker_loop,
            name="moderncv-learn-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()
        log.info("Learn queue worker thread started")


def stop_learn_worker(timeout: float = 2.0) -> None:
    """Push a sentinel and wait briefly for the worker to drain. Called by
    the lifespan shutdown hook so the worker doesn't get killed mid-batch
    on a clean shutdown (it still gets killed on Ctrl-C since it's a
    daemon — that's fine, the next startup sweeps stale runs)."""
    global _WORKER_THREAD
    with _WORKER_LOCK:
        th = _WORKER_THREAD
        if th is None or not th.is_alive():
            return
    try:
        _LEARN_QUEUE.put_nowait(None)
    except Exception:
        pass
    th.join(timeout=timeout)


def _learn_worker_loop() -> None:
    """Pull jobs one at a time, run them, repeat. Runs forever until a
    sentinel `None` is pushed onto the queue."""
    while True:
        job = _LEARN_QUEUE.get()
        if job is None:
            log.info("Learn worker received shutdown sentinel")
            _LEARN_QUEUE.task_done()
            break
        try:
            _execute_queued_job(job)
        except Exception:
            log.exception("queued Learn job crashed: %s", job.manifest_id)
        finally:
            _LEARN_QUEUE.task_done()


def _execute_queued_job(job: _LearnJob) -> None:
    """Pop-and-run. Transitions the manifest queued → running, then hands
    off to the existing _run_with_dir pipeline."""
    # The user may have deleted the run while it was in queue; check that
    # the manifest is still there and still queued before doing anything.
    manifest_path = job.rdir / runs_mod.MANIFEST_NAME
    if not manifest_path.exists():
        log.info("skipping job %s — run dir deleted", job.manifest_id)
        return
    manifest = runs_mod.read_manifest(job.rdir)
    if manifest.status not in ("queued", "running"):
        log.info(
            "skipping job %s — status is %s, not queued",
            job.manifest_id,
            manifest.status,
        )
        return

    manifest.status = "running"
    runs_mod.write_manifest(job.rdir, manifest)
    runs_mod.write_progress(
        job.rdir,
        runs_mod.RunProgress(
            stage="loading_models",
            message="Worker started — preparing pipeline",
            total_frames=job.total_frames,
            started_at=job.started_at,
            updated_at=_now_iso(),
        ),
    )
    log.info("worker thread started for run %s", manifest.id)

    _run_with_dir(
        rdir=job.rdir,
        manifest=manifest,
        task=job.task,
        prompt=job.prompt,
        prompts=job.prompts,
        video_path=job.video_path,
        total_frames=job.total_frames,
        started_at=job.started_at,
        detect_impl=job.detect_impl,
        segment_impl=job.segment_impl,
        reid_impl=job.reid_impl,
        track_impl=job.track_impl,
        max_frames=job.max_frames,
    )


# Default impl picks per task. These are what shows up if the wizard
# doesn't override — the "opinionated default" that lets Learn stay
# three-fields-and-a-button. Each becomes a knob in Optimize later.
DEFAULTS = {
    "detection": {
        "detect": "groundingdino",
        "track": "bytetrack",
    },
    "segmentation": {
        "detect": "groundingdino",
        "segment": "sam2-tiny",
        "reid": "dinov3-vits16",
        "track": "bytetrack",
    },
}


# How often to flush progress.json. Every frame would be wasteful; every
# 5 frames keeps the UI feeling live without thrashing the disk.
PROGRESS_FLUSH_EVERY = 5


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _write_teacher_coco(rdir: Path, *, prompt: str, video_path: str) -> None:
    """Build labels/coco.json from the just-written per_frame.jsonl.

    Phase 3: this writes the *unfiltered* COCO — every detection at the
    score floor is persisted, with `det_idx` carried on each annotation.
    The Student trainer reads this and applies per-frame review state +
    threshold filtering itself in `pipeline.distill.prepare_yolo_dataset`.

    Frame dimensions come from cv2 — one capture is cheap and gives us the
    canonical (height, width) the boxes were produced at.
    """
    import json

    per_frame_path = rdir / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    if not per_frame_path.exists():
        log.warning("no per_frame.jsonl at %s; skipping coco export", per_frame_path)
        return

    cap = cv2.VideoCapture(video_path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    images: list[dict] = []
    annotations: list[dict] = []
    class_names: list[str] = []
    class_to_id: dict[str, int] = {}
    ann_id = 1

    with per_frame_path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx = int(rec.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            dets = rec.get("detections") or []

            # Always record the image, even if no detections — the Student
            # needs negative frames too (frames where the target is
            # genuinely absent contribute precision signal).
            images.append(
                {
                    "id": frame_idx,
                    "file_name": f"frame_{frame_idx:06d}.jpg",
                    "width": w,
                    "height": h,
                }
            )
            for det_idx, d in enumerate(dets):
                cname = d.get("class_name") or "object"
                if cname not in class_to_id:
                    class_to_id[cname] = len(class_names) + 1
                    class_names.append(cname)
                cid = class_to_id[cname]
                x1, y1, x2, y2 = d["bbox_xyxy"]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": frame_idx,
                        # Phase 3: stable position-within-frame so distill
                        # can match against rejected_dets in frame_states.json.
                        "det_idx": det_idx,
                        "category_id": cid,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "score": float(d.get("score", 1.0)),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    runs_mod.write_coco(
        rdir,
        prompt=prompt,
        image_records=images,
        annotation_records=annotations,
        category_names=class_names,
    )
    log.info(
        "wrote coco.json: %d images, %d annotations, %d classes",
        len(images),
        len(annotations),
        len(class_names),
    )


def _count_frames(video_path: str, max_frames: Optional[int]) -> int:
    """Best-effort frame count for the progress bar.

    cv2's CAP_PROP_FRAME_COUNT can be off (some codecs report 0) — we treat
    0 as "unknown" so the UI shows an indeterminate spinner instead of a
    bogus percentage. Capped by max_frames when set.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if max_frames is not None:
        n = min(n, max_frames) if n > 0 else max_frames
    return max(0, n)


def pick_models(
    *,
    task: str,
    detect_impl: Optional[str] = None,
    segment_impl: Optional[str] = None,
    reid_impl: Optional[str] = None,
    track_impl: Optional[str] = None,
) -> dict[str, str]:
    """Resolve impl choices for each stage of the Teacher pipeline.

    Pulled out of `build_graph` so the server can compute the model picks
    *synchronously* during POST /learn — that way the initial manifest
    written to disk (and returned to the client) already carries the
    actual impls, instead of starting empty and being patched in by the
    worker thread a few microseconds later.
    """
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"unknown task: {task!r}")
    defaults = DEFAULTS[task]
    models: dict[str, str] = {"input": "opencv"}
    models["detect"] = detect_impl or defaults["detect"]
    if task == "segmentation":
        models["segment"] = segment_impl or defaults["segment"]
        models["reid"] = reid_impl or defaults["reid"]
    models["track"] = track_impl or defaults["track"]
    models["output"] = "overlay-mp4"
    return models


def build_graph(
    *,
    task: str,
    prompt: str,
    video_path: str,
    overlay_path: Path,
    prompts: Optional[list[str]] = None,
    detect_impl: Optional[str] = None,
    segment_impl: Optional[str] = None,
    reid_impl: Optional[str] = None,
    track_impl: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> tuple[GraphSpec, dict[str, str]]:
    """Translate a LearnRequest's intent into a linear GraphSpec.

    The detector receives both `params.prompts` (the canonical chip list,
    one entry per user-intended class) and `params.prompt` (a joined
    display string) so adapters can use whichever is more convenient.
    Returns (graph, models_used).
    """
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"unknown task: {task!r}")

    defaults = DEFAULTS[task]
    models: dict[str, str] = {}
    nodes: list[NodeSpec] = []

    input_params: dict = {"path": video_path}
    if max_frames is not None:
        input_params["max_frames"] = max_frames
    nodes.append(NodeSpec(id="n_input", kind=BlockKind.INPUT, impl="opencv", params=input_params))
    models["input"] = "opencv"

    det_impl = detect_impl or defaults["detect"]
    # Carry both: `prompts` is what the GroundingDINO adapter actually uses
    # to label detections; `prompt` is kept for any adapter that only knows
    # the legacy single-string form.
    detect_params: dict = {"prompt": prompt}
    if prompts:
        detect_params["prompts"] = list(prompts)
    nodes.append(
        NodeSpec(
            id="n_detect",
            kind=BlockKind.DETECT,
            impl=det_impl,
            params=detect_params,
        )
    )
    models["detect"] = det_impl

    if task == "segmentation":
        seg_impl = segment_impl or defaults["segment"]
        nodes.append(
            NodeSpec(id="n_segment", kind=BlockKind.SEGMENT, impl=seg_impl, params={})
        )
        models["segment"] = seg_impl

        rid_impl = reid_impl or defaults["reid"]
        nodes.append(NodeSpec(id="n_reid", kind=BlockKind.REID, impl=rid_impl, params={}))
        models["reid"] = rid_impl

    trk_impl = track_impl or defaults["track"]
    nodes.append(NodeSpec(id="n_track", kind=BlockKind.TRACK, impl=trk_impl, params={}))
    models["track"] = trk_impl

    nodes.append(
        NodeSpec(
            id="n_output",
            kind=BlockKind.OUTPUT,
            impl="overlay-mp4",
            params={"path": str(overlay_path)},
        )
    )
    models["output"] = "overlay-mp4"

    edges: list[tuple[str, str]] = []
    for a, b in zip(nodes, nodes[1:]):
        edges.append((a.id, b.id))

    return GraphSpec(nodes=nodes, edges=edges), models


def _resolve_prompts(
    prompt: Optional[str], prompts: Optional[list[str]]
) -> tuple[list[str], str]:
    """Normalize the (prompt, prompts) inputs into a (chips, display) pair.

    chips   — list[str] used by the GroundingDINO adapter for class labels.
    display — joined string for the manifest, COCO export, run dir name,
              and any UI element that wants a human-readable summary.
    """
    chips: list[str] = []
    if prompts:
        for p in prompts:
            if p is None:
                continue
            s = str(p).strip()
            if s:
                chips.append(s)
    if not chips and prompt:
        # Legacy single-string entry. Treat as ONE chip — we deliberately
        # do NOT split on '.' here because that would silently turn the
        # user's "soccer ball" into an unrelated word soup. If they want
        # multi-class, they pass `prompts`.
        s = prompt.strip()
        if s:
            chips.append(s)
    if not chips:
        raise ValueError(
            "LearnRequest needs a non-empty prompt — pass `prompts: [...]` "
            "(preferred) or `prompt: 'soccer ball'`."
        )
    display = ", ".join(chips)
    return chips, display


def run_learn(
    *,
    project_id: str,
    task: str,
    video_path: str,
    prompt: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    detect_impl: Optional[str] = None,
    segment_impl: Optional[str] = None,
    reid_impl: Optional[str] = None,
    track_impl: Optional[str] = None,
    max_frames: Optional[int] = None,
    runs_root: Path = runs_mod.RUNS_DIR,
) -> runs_mod.RunManifest:
    """Synchronous Learn run. Used by CLI/tests.

    Server callers should prefer `run_learn_in_background` so the HTTP
    request returns immediately while the pipeline grinds in a worker
    thread.
    """
    chips, display = _resolve_prompts(prompt, prompts)
    rdir, manifest = runs_mod.create_run(
        project_id=project_id,
        task=task,
        prompt=display,
        video_path=video_path,
        models={},
        runs_root=runs_root,
    )
    started_at = _now_iso()
    total_frames = _count_frames(video_path, max_frames)
    return _run_with_dir(
        rdir=rdir,
        manifest=manifest,
        task=task,
        prompt=display,
        prompts=chips,
        video_path=video_path,
        total_frames=total_frames,
        started_at=started_at,
        detect_impl=detect_impl,
        segment_impl=segment_impl,
        reid_impl=reid_impl,
        track_impl=track_impl,
        max_frames=max_frames,
    )


def run_learn_in_background(
    *,
    project_id: str,
    task: str,
    video_path: str,
    prompt: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    detect_impl: Optional[str] = None,
    segment_impl: Optional[str] = None,
    reid_impl: Optional[str] = None,
    track_impl: Optional[str] = None,
    max_frames: Optional[int] = None,
    runs_root: Path = runs_mod.RUNS_DIR,
) -> runs_mod.RunManifest:
    """Allocate the run dir + manifest, then process in a daemon thread.

    Returns the manifest immediately so the HTTP response can carry the
    run_id back. The worker thread runs the pipeline, writes progress.json
    periodically, and flips status to "completed"/"failed" when done.
    """
    # Resolve the impl picks synchronously so the first manifest written to
    # disk (and the response we hand back to the client) already shows the
    # actual models — the worker thread only updates dynamic fields after
    # this point (status, ended_at, error).
    chips, display = _resolve_prompts(prompt, prompts)
    try:
        models = pick_models(
            task=task,
            detect_impl=detect_impl,
            segment_impl=segment_impl,
            reid_impl=reid_impl,
            track_impl=track_impl,
        )
    except ValueError:
        raise

    rdir, manifest = runs_mod.create_run(
        project_id=project_id,
        task=task,
        prompt=display,
        video_path=video_path,
        models=models,
        runs_root=runs_root,
    )

    # Probe the source video ONCE on the main thread so the worker doesn't
    # have to repeat a cv2.VideoCapture open (which has thread-safety
    # quirks on some platforms and can stall a worker for many seconds).
    started_at = _now_iso()
    try:
        total_frames = _count_frames(video_path, max_frames)
    except Exception as e:
        log.warning("frame count probe failed for %s: %s", video_path, e)
        total_frames = max_frames or 0

    # Mark the manifest as 'queued' so the UI can render an honest "waiting
    # in line" state. The worker flips it to 'running' the moment it pops
    # this job off the queue.
    manifest.status = "queued"
    runs_mod.write_manifest(rdir, manifest)

    ahead = _LEARN_QUEUE.qsize()  # not counting the active run (already popped)
    queued_msg = (
        "Queued — runs next"
        if ahead == 0
        else f"Queued — {ahead} job{'s' if ahead != 1 else ''} ahead"
    )
    runs_mod.write_progress(
        rdir,
        runs_mod.RunProgress(
            stage="queued",
            message=queued_msg,
            total_frames=total_frames,
            started_at=started_at,
            updated_at=_now_iso(),
        ),
    )

    job = _LearnJob(
        rdir=rdir,
        project_id=project_id,
        manifest_id=manifest.id,
        task=task,
        prompt=display,
        prompts=chips,
        video_path=video_path,
        total_frames=total_frames,
        started_at=started_at,
        detect_impl=detect_impl,
        segment_impl=segment_impl,
        reid_impl=reid_impl,
        track_impl=track_impl,
        max_frames=max_frames,
    )
    ensure_learn_worker_started()
    _LEARN_QUEUE.put(job)
    log.info(
        "enqueued Learn job %s (%d ahead of it in queue)",
        manifest.id,
        ahead,
    )
    return manifest


def _run_with_dir(
    *,
    rdir: Path,
    manifest: runs_mod.RunManifest,
    task: str,
    prompt: str,
    prompts: list[str],
    video_path: str,
    total_frames: int,
    started_at: str,
    detect_impl: Optional[str],
    segment_impl: Optional[str],
    reid_impl: Optional[str],
    track_impl: Optional[str],
    max_frames: Optional[int],
) -> runs_mod.RunManifest:
    """Shared body for sync + background entry points.

    `total_frames` and `started_at` are passed in (rather than recomputed)
    so a single cv2.VideoCapture open suffices per run — duplicate opens
    were causing intermittent multi-second stalls on the worker thread.
    """
    overlay_path = rdir / runs_mod.OVERLAY_NAME
    total = total_frames

    def _progress(stage: str, message: str, current: int, hits: int) -> None:
        runs_mod.write_progress(
            rdir,
            runs_mod.RunProgress(
                stage=stage,
                message=message,
                current_frame=current,
                total_frames=total,
                frames_with_detections=hits,
                started_at=started_at,
                updated_at=_now_iso(),
            ),
        )

    # Customize the loading-stage message based on actual cache state. This
    # is wrapped in a try-block because cache_status touches huggingface_hub
    # — if that import or the disk probe ever stalled, we'd be stuck before
    # progress could ever flow. Falling back to a generic message keeps the
    # UI moving.
    det_impl_for_msg = detect_impl or DEFAULTS[task]["detect"]
    try:
        cs = model_cache.cache_status(det_impl_for_msg)
        if not cs.cached and cs.estimated_bytes > 0:
            mb = cs.estimated_bytes // 1_000_000
            load_msg = f"Downloading {det_impl_for_msg} weights (~{mb} MB)"
        else:
            load_msg = f"Loading {det_impl_for_msg} from cache"
    except Exception as e:
        log.warning("cache probe failed: %s", e)
        load_msg = f"Loading {det_impl_for_msg}"
    _progress("loading_models", load_msg, 0, 0)
    log.info("run %s: load_msg=%r", manifest.id, load_msg)

    try:
        graph, models = build_graph(
            task=task,
            prompt=prompt,
            prompts=prompts,
            video_path=video_path,
            overlay_path=overlay_path,
            detect_impl=detect_impl,
            segment_impl=segment_impl,
            reid_impl=reid_impl,
            track_impl=track_impl,
            max_frames=max_frames,
        )
    except ValueError as e:
        runs_mod.mark_failed(rdir, str(e))
        raise

    manifest.models = models
    runs_mod.write_manifest(rdir, manifest)
    log.info("run %s: graph built; entering frame loop", manifest.id)

    per_frame_ms: list[float] = []
    n_detections_total = 0
    frames_with_detections = 0

    try:
        with runs_mod.PerFrameWriter(rdir) as pfw:
            t_start = time.perf_counter()
            t_prev = t_start
            iterator = run_graph(graph)
            for batch in iterator:
                t_now = time.perf_counter()
                ms = (t_now - t_prev) * 1000.0
                t_prev = t_now
                per_frame_ms.append(ms)

                dets = [runs_mod.detection_to_dict(d) for d in batch.detections]
                if dets:
                    frames_with_detections += 1
                    n_detections_total += len(dets)
                pfw.write(batch.frame_index, dets, masks=None)

                if (batch.frame_index + 1) % PROGRESS_FLUSH_EVERY == 0:
                    _progress(
                        "running",
                        f"Frame {batch.frame_index + 1} — {ms:.0f} ms last frame",
                        batch.frame_index + 1,
                        frames_with_detections,
                    )
    except Exception as e:
        log.exception("learn run failed: %s", e)
        runs_mod.mark_failed(rdir, str(e))
        return runs_mod.read_manifest(rdir)

    _progress("finalizing", "Writing COCO export and stats", len(per_frame_ms), frames_with_detections)

    # COCO export — rejection-aware. Reads back the per_frame.jsonl we just
    # wrote, applies any current rejections, and emits labels/coco.json. The
    # Student trainer (next turn) consumes this as ground-truth-equivalent.
    _write_teacher_coco(rdir, prompt=prompt, video_path=video_path)

    n = max(1, len(per_frame_ms))
    sorted_ms = sorted(per_frame_ms)
    stats = runs_mod.RunStats(
        frames_processed=len(per_frame_ms),
        frames_with_detections=frames_with_detections,
        total_ms=sum(per_frame_ms),
        avg_ms_per_frame=sum(per_frame_ms) / n,
        p50_ms_per_frame=sorted_ms[n // 2] if sorted_ms else 0.0,
        p95_ms_per_frame=sorted_ms[min(n - 1, int(n * 0.95))] if sorted_ms else 0.0,
        n_detections_total=n_detections_total,
    )
    return runs_mod.mark_completed(rdir, stats)
