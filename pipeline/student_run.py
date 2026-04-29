"""Student-run orchestrator (Phase 5).

A "student run" is a trained Student running inference against an arbitrary
input — either a video file under `data/` or any Teacher-labeled dataset
in the same project. The result lives at:

    runs/projects/<pid>/student_runs/<sid>/<rid>/
        manifest.json         — kind/ref/status/error
        stats.json            — frame count, latency, mAP if applicable
        progress.json         — live updates while running
        predictions/
            per_frame.jsonl   — one line per frame, same shape as Teacher
            coco.json         — student predictions, COCO format
        overlay.mp4           — boxes drawn on source frames

Background execution mirrors `pipeline.learn`'s queue model: a single
persistent daemon thread pulls `_StudentRunJob`s in order. We deliberately
keep this queue separate from Learn (Teacher) and from Optimize (training)
so a long Optimize doesn't block a quick student-run preview, and a
student-run doesn't fight a Teacher for GroundingDINO memory.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from pipeline import runs as runs_mod
from pipeline.students.registry import make_trainer

log = logging.getLogger(__name__)


# ---- Sequential queue ------------------------------------------------------


@dataclass
class _StudentRunJob:
    """Frozen payload the worker thread needs to execute one run."""

    rdir: Path
    project_id: str
    student_id: str
    run_id: str
    input_kind: str
    input_ref: str
    video_path: str          # resolved source video (regardless of input_kind)
    weights_path: str
    architecture: str
    started_at: str
    total_frames: int
    teacher_id_for_eval: Optional[str]  # set when input_kind == "teacher_dataset"


_STUDENT_RUN_QUEUE: "queue.Queue[Optional[_StudentRunJob]]" = queue.Queue()
_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: Optional[threading.Thread] = None


def ensure_student_run_worker_started() -> None:
    """Idempotent — spawn the singleton student-run worker if not alive.

    Mirrors `pipeline.learn.ensure_learn_worker_started`. Called by the
    server's lifespan startup hook and as a safety net inside
    `run_student_run_in_background`.
    """
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _WORKER_THREAD = threading.Thread(
            target=_worker_loop,
            name="moderncv-studentrun-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()
        log.info("student-run worker thread started")


def stop_student_run_worker(timeout: float = 2.0) -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        th = _WORKER_THREAD
        if th is None or not th.is_alive():
            return
    try:
        _STUDENT_RUN_QUEUE.put_nowait(None)
    except Exception:
        pass
    th.join(timeout=timeout)


def _worker_loop() -> None:
    while True:
        job = _STUDENT_RUN_QUEUE.get()
        if job is None:
            log.info("student-run worker received shutdown sentinel")
            _STUDENT_RUN_QUEUE.task_done()
            break
        try:
            _execute_job(job)
        except Exception:
            log.exception("queued student-run job crashed: %s", job.run_id)
        finally:
            _STUDENT_RUN_QUEUE.task_done()


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


# ---- How often to flush progress.json. Same value Learn uses — keeps the
# UI feeling live without thrashing the disk for short clips.
PROGRESS_FLUSH_EVERY = 5


def _execute_job(job: _StudentRunJob) -> None:
    """Pop-and-run. Flips manifest queued → running, drives inference,
    optionally computes mAP, then flips to completed/failed."""
    manifest_path = job.rdir / runs_mod.MANIFEST_NAME
    if not manifest_path.exists():
        log.info("skipping student-run %s — dir deleted", job.run_id)
        return
    try:
        manifest = runs_mod.read_student_run_manifest(job.rdir)
    except Exception as e:
        log.warning("could not read student-run manifest at %s: %s", job.rdir, e)
        return
    if manifest.status not in ("queued", "running"):
        log.info(
            "skipping student-run %s — status is %s", job.run_id, manifest.status
        )
        return

    manifest.status = "running"
    runs_mod.write_student_run_manifest(job.rdir, manifest)
    _progress(job, "loading_model", "Worker started — loading weights", 0, 0)

    try:
        stats = _run_inference(job)
    except Exception as e:
        log.exception("student-run %s failed", job.run_id)
        runs_mod.mark_student_run_failed(job.rdir, str(e))
        return

    if job.input_kind == "teacher_dataset" and job.teacher_id_for_eval:
        try:
            map50, map5095 = _compute_map_against_teacher(
                project_id=job.project_id,
                student_id=job.student_id,
                run_dir=job.rdir,
                teacher_id=job.teacher_id_for_eval,
                weights_path=Path(job.weights_path),
                architecture=job.architecture,
            )
            stats.map50 = map50
            stats.map50_95 = map5095
        except Exception as e:
            # mAP is a value-add — failing here shouldn't tank the whole
            # run. Log and leave the fields at None.
            log.warning(
                "mAP computation failed for student-run %s: %s", job.run_id, e
            )

    runs_mod.mark_student_run_completed(job.rdir, stats)
    _progress(job, "completed", "Done", stats.n_frames, stats.n_detections)


def _progress(
    job: _StudentRunJob, stage: str, message: str, current: int, hits: int
) -> None:
    runs_mod.write_progress(
        job.rdir,
        runs_mod.RunProgress(
            stage=stage,
            message=message,
            current_frame=current,
            total_frames=job.total_frames,
            frames_with_detections=hits,
            started_at=job.started_at,
            updated_at=_now_iso(),
        ),
    )


def _run_inference(job: _StudentRunJob) -> runs_mod.StudentRunStats:
    """Drive the trainer's `predict()` over every frame in the video.

    Writes `predictions/per_frame.jsonl` and `predictions/coco.json` plus
    an `overlay.mp4`. Returns a populated `StudentRunStats` (mAP fields
    left at `None` — the caller fills them in for teacher_dataset inputs).
    """
    trainer = make_trainer(job.architecture)
    overlay_path = job.rdir / runs_mod.OVERLAY_NAME

    cap = cv2.VideoCapture(job.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {job.video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Frame producer — yields BGR uint8 numpy arrays, the shape
        # `YoloTrainer.predict` expects (Ultralytics will accept them
        # directly without preprocessing).
        def _frame_iter():
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame

        prediction_iter = trainer.predict(
            weights=Path(job.weights_path),
            frames=_frame_iter(),
        )

        per_frame_ms: list[float] = []
        n_detections = 0
        frames_with_detections = 0
        writer: Optional[cv2.VideoWriter] = None
        all_records: list[dict] = []  # for coco export
        frame_h: int = 0
        frame_w: int = 0

        try:
            with runs_mod.PerFrameWriter(
                job.rdir, subdir=runs_mod.PREDICTIONS_DIR
            ) as pfw:
                # Re-open capture for overlay — the prediction path
                # already consumed `cap`. Two separate captures is
                # cheap and avoids interleaving the read positions.
                cap_for_overlay = cv2.VideoCapture(job.video_path)
                try:
                    if not cap_for_overlay.isOpened():
                        raise RuntimeError(
                            f"cannot reopen video for overlay {job.video_path}"
                        )
                    frame_idx = 0
                    t_prev = time.perf_counter()
                    for dets in prediction_iter:
                        ok, raw_frame = cap_for_overlay.read()
                        if not ok or raw_frame is None:
                            # Predictions ran past the end of the video —
                            # shouldn't happen because we share the same
                            # iterator source, but be defensive.
                            break
                        if frame_h == 0:
                            frame_h, frame_w = raw_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(
                                str(overlay_path),
                                fourcc,
                                float(fps),
                                (frame_w, frame_h),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(
                                    f"cannot open VideoWriter at {overlay_path}"
                                )

                        t_now = time.perf_counter()
                        per_frame_ms.append((t_now - t_prev) * 1000.0)
                        t_prev = t_now

                        if dets:
                            frames_with_detections += 1
                            n_detections += len(dets)

                        pfw.write(frame_idx, dets, masks=None)
                        all_records.append({"frame_idx": frame_idx, "detections": dets})

                        # Render overlay — green boxes, score label.
                        # Mirrors `pipeline.models.video._draw_detections`'s
                        # style so the user sees the same rendering they
                        # see for Teacher overlays.
                        overlay_frame = raw_frame.copy()
                        for d in dets:
                            x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
                            cv2.rectangle(
                                overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )
                            label = f"{d['class_name']} {d['score']:.2f}"
                            cv2.putText(
                                overlay_frame,
                                label,
                                (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        if writer is not None:
                            writer.write(overlay_frame)

                        if (frame_idx + 1) % PROGRESS_FLUSH_EVERY == 0:
                            _progress(
                                job,
                                "running",
                                f"Frame {frame_idx + 1} — {per_frame_ms[-1]:.0f} ms last",
                                frame_idx + 1,
                                frames_with_detections,
                            )
                        frame_idx += 1
                finally:
                    cap_for_overlay.release()
                    if writer is not None:
                        writer.release()
        finally:
            pass
    finally:
        cap.release()

    # Write COCO predictions for downstream consumption.
    _write_predictions_coco(
        job.rdir,
        records=all_records,
        video_w=frame_w,
        video_h=frame_h,
    )

    n = max(1, len(per_frame_ms))
    sorted_ms = sorted(per_frame_ms)
    return runs_mod.StudentRunStats(
        n_frames=len(per_frame_ms),
        n_detections=n_detections,
        avg_inference_ms=sum(per_frame_ms) / n,
        p50_inference_ms=sorted_ms[n // 2] if sorted_ms else 0.0,
        p95_inference_ms=(
            sorted_ms[min(n - 1, int(n * 0.95))] if sorted_ms else 0.0
        ),
    )


def _write_predictions_coco(
    rdir: Path, *, records: list[dict], video_w: int, video_h: int
) -> None:
    """Emit `predictions/coco.json` from the in-memory record list.

    Using the in-memory list (rather than re-reading per_frame.jsonl from
    disk) keeps this cheap and avoids one more file open. Class vocabulary
    is built incrementally from what the Student actually predicted —
    there's no project-level vocabulary to inherit from at run time.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    class_names: list[str] = []
    class_to_id: dict[str, int] = {}
    ann_id = 1
    for rec in records:
        frame_idx = int(rec["frame_idx"])
        images.append(
            {
                "id": frame_idx,
                "file_name": f"frame_{frame_idx:06d}.jpg",
                "width": video_w,
                "height": video_h,
            }
        )
        for det_idx, d in enumerate(rec["detections"]):
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
                    "det_idx": det_idx,
                    "category_id": cid,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "score": float(d.get("score", 1.0)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": "ModernCV student-run predictions"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": n, "supercategory": "thing"}
            for i, n in enumerate(class_names)
        ],
    }
    (rdir / runs_mod.PREDICTIONS_DIR / runs_mod.COCO_NAME).write_text(json.dumps(coco))


def _compute_map_against_teacher(
    *,
    project_id: str,
    student_id: str,
    run_dir: Path,
    teacher_id: str,
    weights_path: Path,
    architecture: str,
) -> tuple[float, float]:
    """Build a one-teacher YOLO eval dataset and run trainer.eval().

    Reuses `pipeline.distill.prepare_eval_dataset` — the same helper the
    Student trainer uses for per-eval-teacher mAP at training time. The
    eval dataset lands under the student-run dir so artefacts stay
    self-contained. Class vocabulary is derived from the Student's own
    training-time vocab (read from the trainer's `best.pt` if available;
    otherwise from the teacher's COCO categories).
    """
    from pipeline import distill

    teacher_rdir = runs_mod.run_dir(project_id, teacher_id)
    if not (teacher_rdir / runs_mod.MANIFEST_NAME).exists():
        raise FileNotFoundError(f"no such teacher: {teacher_id}")
    coco_path = teacher_rdir / runs_mod.LABELS_DIR / runs_mod.COCO_NAME
    if not coco_path.exists():
        raise FileNotFoundError(f"teacher has no coco.json: {teacher_id}")
    coco = json.loads(coco_path.read_text())
    class_names = sorted({c.get("name", "object") for c in coco.get("categories", [])})
    if not class_names:
        # Fallback: use a single dummy class so YOLO still parses the
        # data.yaml. mAP will be 0 but at least the run completes.
        class_names = ["object"]

    data_yaml, _, _ = distill.prepare_eval_dataset(
        project_id=project_id,
        student_dir=run_dir,
        eval_teacher_id=teacher_id,
        class_names=class_names,
    )
    trainer = make_trainer(architecture)
    map50, map5095 = trainer.eval(weights=weights_path, data_yaml=data_yaml)
    return float(map50), float(map5095)


# ---- Public entry points --------------------------------------------------


def _resolve_input(
    *, project_id: str, input_kind: str, input_ref: str
) -> tuple[str, Optional[str]]:
    """Validate the input and return (video_path, teacher_id_for_eval).

    For `video`, `input_ref` is taken as a literal path — the caller is
    responsible for keeping it inside `data/`. For `teacher_dataset`, we
    look up the teacher's manifest and use its `video_path`.
    """
    if input_kind == "video":
        if not input_ref:
            raise ValueError("input_ref is required when input_kind='video'")
        if not Path(input_ref).exists():
            raise FileNotFoundError(f"video not found: {input_ref}")
        return input_ref, None
    if input_kind == "teacher_dataset":
        tdir = runs_mod.run_dir(project_id, input_ref)
        if not (tdir / runs_mod.MANIFEST_NAME).exists():
            raise FileNotFoundError(f"no such teacher: {input_ref}")
        m = runs_mod.read_manifest(tdir)
        if m.status != "completed":
            raise ValueError(
                f"teacher {input_ref!r} status is {m.status!r}; "
                "wait for the Learn run to finish before running the Student against it"
            )
        if not Path(m.video_path).exists():
            raise FileNotFoundError(
                f"teacher's source video missing: {m.video_path}"
            )
        return m.video_path, input_ref
    raise ValueError(f"unknown input_kind: {input_kind!r}")


def _count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(0, n)


def run_student_run_in_background(
    *,
    project_id: str,
    student_id: str,
    input_kind: str,
    input_ref: str,
    runs_root: Path = runs_mod.RUNS_DIR,
) -> runs_mod.StudentRunManifest:
    """Allocate the student-run dir, then enqueue the actual work.

    Returns the manifest immediately (with status='queued') so the HTTP
    response can carry the run id back to the client. The worker thread
    flips status → 'running' when it pops the job.
    """
    sdir = runs_mod.student_dir(project_id, student_id, runs_root)
    if not (sdir / runs_mod.MANIFEST_NAME).exists():
        raise FileNotFoundError(f"no such student: {student_id}")
    student_manifest = runs_mod.read_student_manifest(sdir)
    if student_manifest.status != "completed":
        raise ValueError(
            f"student {student_id!r} status is {student_manifest.status!r}; "
            "train it to completion before running"
        )
    weights = sdir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"student weights missing: {weights}")

    video_path, teacher_for_eval = _resolve_input(
        project_id=project_id, input_kind=input_kind, input_ref=input_ref
    )
    total_frames = _count_frames(video_path)

    rdir, manifest = runs_mod.create_student_run(
        project_id=project_id,
        student_id=student_id,
        input_kind=input_kind,
        input_ref=input_ref,
        runs_root=runs_root,
    )

    manifest.status = "queued"
    runs_mod.write_student_run_manifest(rdir, manifest)
    started_at = _now_iso()
    runs_mod.write_progress(
        rdir,
        runs_mod.RunProgress(
            stage="queued",
            message="Queued — runs next",
            total_frames=total_frames,
            started_at=started_at,
            updated_at=_now_iso(),
        ),
    )

    job = _StudentRunJob(
        rdir=rdir,
        project_id=project_id,
        student_id=student_id,
        run_id=manifest.id,
        input_kind=input_kind,
        input_ref=input_ref,
        video_path=video_path,
        weights_path=str(weights),
        architecture=student_manifest.architecture,
        started_at=started_at,
        total_frames=total_frames,
        teacher_id_for_eval=teacher_for_eval,
    )
    ensure_student_run_worker_started()
    _STUDENT_RUN_QUEUE.put(job)
    log.info("enqueued student-run %s for student %s", manifest.id, student_id)
    return manifest
