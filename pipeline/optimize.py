"""Optimize-mode runner.

A Student is a small fast model trained on one or more Teachers' curated
COCO labels. This module owns the orchestration — directory + manifest
allocation, the background worker that drives the trainer, progress writes,
status transitions. The actual training mechanics (Ultralytics YOLO,
COCO→YOLO conversion, per-eval-teacher mAP, inference timing) live in
`pipeline.distill` so they can be tested standalone.

Why split this from `pipeline.distill`
--------------------------------------
This file is short and threading-aware: it has to run in a daemon thread,
write progress files atomically, and never crash the FastAPI worker on
training errors. The trainer is heavy and CPU/GPU-bound — keeping them
separate means the trainer can be unit-tested without spinning up the
whole web app, and changes to the orchestration (e.g. switching to a
queue) don't churn the training code.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline import distill, runs as runs_mod
# Importing the students package triggers each trainer module's
# `@register(...)` side effect — must happen before `make_trainer` is
# called below.
from pipeline.students import make_trainer
from pipeline.students.yolo import _pick_device

log = logging.getLogger(__name__)


# Default image size passed to every trainer's `train()`. Hoisted to a
# module constant so the same value gets stamped on `StudentStats.imgsz`
# (Phase 2.2) without two literal-640s drifting. When per-architecture
# overrides land, lift this to a `make_trainer()`-returned default.
_DEFAULT_IMGSZ = 640


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _seed_progress(rdir: Path, *, message: str, total_epochs: int = 50) -> None:
    """Reuse the Teacher progress shape so the GUI can render with one
    component. We treat 'epochs' as the unit instead of 'frames'."""
    runs_mod.write_progress(
        rdir,
        runs_mod.RunProgress(
            stage="loading_models",
            message=message,
            current_frame=0,
            total_frames=total_epochs,
            frames_with_detections=0,
            started_at=_now_iso(),
            updated_at=_now_iso(),
        ),
    )


def _read_teacher_coco_summary(
    project_id: str, teacher_id: str
) -> tuple[int, int, list[str]]:
    """Quick stats from the teacher's coco.json: (n_images, n_annotations,
    class_names). Used to populate StudentStats and decide whether the
    teacher is even ready to distill from."""
    path = (
        runs_mod.run_dir(project_id, teacher_id)
        / runs_mod.LABELS_DIR
        / runs_mod.COCO_NAME
    )
    if not path.exists():
        raise FileNotFoundError(
            f"teacher {teacher_id!r} has no coco.json yet — finish a Learn run first"
        )
    coco = json.loads(path.read_text())
    return (
        len(coco.get("images", [])),
        len(coco.get("annotations", [])),
        [c["name"] for c in coco.get("categories", [])],
    )


def _summarize_teacher_set(
    project_id: str,
    teacher_ids: list[str],
) -> tuple[int, int, list[str]]:
    """Aggregate (images, annotations, union-of-class-names) across many
    teachers.

    Used to log what the Student is actually about to train on (or evaluate
    against). Not a substitute for the real trainer's data preparation —
    that'll need to dedupe class IDs across teachers — but it's the right
    signal for the progress message and for surfacing "you picked teachers
    with disjoint vocabularies" warnings later.
    """
    total_imgs = 0
    total_anns = 0
    classes: list[str] = []
    seen: set[str] = set()
    for tid in teacher_ids:
        try:
            imgs, anns, cnames = _read_teacher_coco_summary(project_id, tid)
        except FileNotFoundError as e:
            # A teacher in the user's list might be in a weird state (e.g.
            # they marked it completed but the COCO export failed). Skip
            # with a warning so the Student can still attempt to run on
            # the remaining teachers — the trainer will hard-fail later
            # if there's nothing usable.
            log.warning("skipping teacher %s for summary: %s", tid, e)
            continue
        total_imgs += imgs
        total_anns += anns
        for c in cnames:
            if c not in seen:
                seen.add(c)
                classes.append(c)
    return total_imgs, total_anns, classes


def run_optimize_in_background(
    *,
    project_id: str,
    train_teacher_ids: list[str],
    eval_teacher_ids: list[str],
    task: str,
    detect_impl: Optional[str] = None,
    segment_impl: Optional[str] = None,
    track_impl: Optional[str] = None,
    epochs: int = 50,
    t_high: float = 0.35,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    architecture: str = "yolov8n",
    runs_root: Path = runs_mod.RUNS_DIR,
) -> runs_mod.StudentManifest:
    """Allocate a Student dir, return its manifest, run training on a
    daemon thread.

    The Student manifest carries two lists:
      • `train_teacher_ids` — Teachers whose curated COCOs get merged into
        the training set.
      • `eval_teacher_ids`  — Teachers held out for transferability scoring;
        the Student is evaluated against them but never sees them at train
        time. Empty list = "no eval, just train" (the trainer skips mAP).

    The background worker:
      1. Reads each train teacher's coco.json + sums up totals.
      2. Reads each eval teacher's coco.json so the user sees the eval set
         size in the progress message.
      3. (Stub) sleeps briefly, marks `failed` with "trainer not yet wired"
         so the UI flow is end-to-end testable today.

    When the real trainer lands, only step 3 changes — steps 1+2 remain
    the data-preparation prelude.
    """
    if not train_teacher_ids:
        raise ValueError("train_teacher_ids must contain at least one teacher")

    # Validate train teachers up front. Eval teachers are validated lazily
    # in the worker — a missing eval teacher is recoverable (we just skip
    # eval), but a missing train teacher means the run is meaningless.
    for tid in train_teacher_ids:
        teacher = runs_mod.read_manifest(runs_mod.run_dir(project_id, tid))
        if teacher.status != "completed":
            raise ValueError(
                f"teacher {tid!r} status is {teacher.status!r}; "
                "wait for the Learn run to finish before optimizing"
            )

    # The display prompt comes from the first train teacher. Multi-teacher
    # students don't have a single "prompt" — listing all of them would
    # blow up the sidebar — so we use the first as the canonical label and
    # the train_teacher_ids list carries the rest.
    first_train = runs_mod.read_manifest(
        runs_mod.run_dir(project_id, train_teacher_ids[0])
    )
    display_prompt = first_train.prompt
    if len(train_teacher_ids) > 1:
        display_prompt = f"{display_prompt} (+{len(train_teacher_ids) - 1} more)"

    # Default impls per task — the *Student* toolchain.
    det = detect_impl or "yolov8n"
    models = {"detect": det}
    if task == "segmentation":
        models["segment"] = segment_impl or "fastsam"
    models["track"] = track_impl or "bytetrack"

    rdir, manifest = runs_mod.create_student(
        project_id=project_id,
        train_teacher_ids=train_teacher_ids,
        eval_teacher_ids=eval_teacher_ids,
        task=task,
        prompt=display_prompt,
        models=models,
        t_high=t_high,
        t_low=t_low,
        treat_empty_as_negative=treat_empty_as_negative,
        architecture=architecture,
        runs_root=runs_root,
    )

    _seed_progress(
        rdir,
        message=(
            f"Preparing dataset from {len(train_teacher_ids)} train teacher(s)"
            + (f" + {len(eval_teacher_ids)} eval" if eval_teacher_ids else "")
        ),
        total_epochs=epochs,
    )

    def _worker() -> None:
        try:
            _run_distillation(
                project_id=project_id,
                rdir=rdir,
                train_teacher_ids=train_teacher_ids,
                eval_teacher_ids=eval_teacher_ids,
                task=task,
                epochs=epochs,
                t_high=t_high,
                t_low=t_low,
                treat_empty_as_negative=treat_empty_as_negative,
                architecture=architecture,
            )
        except Exception as e:
            log.exception("optimize worker crashed: %s", e)
            try:
                runs_mod.mark_student_failed(rdir, str(e))
            except Exception:
                pass

    th = threading.Thread(target=_worker, name=f"optimize-{manifest.id}", daemon=True)
    th.start()
    return manifest


# ---- The actual distillation flow -----------------------------------------


def _progress_writer(rdir: Path, total_epochs: int, started_at: str):
    """Build a `progress(stage, message, current, total)` closure that
    overwrites progress.json. Pulled out so the long worker body reads as
    a list of phases without inline `runs_mod.write_progress(...)` blocks."""

    def write(stage: str, message: str, current: int, total: int) -> None:
        runs_mod.write_progress(
            rdir,
            runs_mod.RunProgress(
                stage=stage,
                message=message,
                current_frame=current,
                # Reuse `total_frames` as the unit progress bar denominator —
                # during prep that's frame count, during training it's epochs.
                total_frames=total or total_epochs,
                frames_with_detections=0,
                started_at=started_at,
                updated_at=_now_iso(),
            ),
        )

    return write


def _run_distillation(
    *,
    project_id: str,
    rdir: Path,
    train_teacher_ids: list[str],
    eval_teacher_ids: list[str],
    task: str,
    epochs: int,
    t_high: float = 0.35,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    architecture: str = "yolov8n",
) -> None:
    """Drive the four phases (prep → train → eval → time) and persist stats.

    Each phase writes a progress.json before starting so the GUI's polling
    sidebar shows the live state. Errors anywhere fail the run with a
    descriptive message — the most common ones are missing teacher coco.json
    files (Teacher run never completed) and Ultralytics weight-download
    failures (no network on first run).
    """
    # Detection-only this turn. Segmentation will need a parallel path with
    # yolov8n-seg + polygon parsing; deferred so we ship the common case first.
    if task != "detection":
        raise RuntimeError(
            f"task={task!r} not supported by the distill trainer yet — "
            "only 'detection' is wired this turn"
        )

    started_at = _now_iso()
    progress = _progress_writer(rdir, total_epochs=epochs, started_at=started_at)

    # The trainer dispatch (Phase 1.3): pick the architecture's trainer
    # implementation up front so an unknown name fails fast — before we
    # extract any frames or write any progress files. The students package
    # populated the registry at import time (top of this module).
    trainer = make_trainer(architecture)

    # --- Phase 1: prepare the merged YOLO dataset ---------------------------
    progress("loading_models",
             f"Preparing dataset from {len(train_teacher_ids)} train teacher(s)",
             0, 0)

    def _prep_progress(msg: str, cur: int, tot: int) -> None:
        progress("loading_models", msg, cur, tot)

    summary = distill.prepare_yolo_dataset(
        project_id=project_id,
        student_dir=rdir,
        train_teacher_ids=train_teacher_ids,
        t_high=t_high,
        t_low=t_low,
        treat_empty_as_negative=treat_empty_as_negative,
        progress=_prep_progress,
    )
    log.info(
        "Distill prep: %d train images, %d train annotations, %d val images, "
        "classes=%s",
        summary.n_train_images,
        summary.n_train_annotations,
        summary.n_val_images,
        summary.class_names,
    )
    if summary.n_train_images == 0:
        raise RuntimeError(
            "No usable training frames after extraction — check that the "
            "Teacher's source video still exists at the path in its manifest"
        )

    # --- Phase 2: train ----------------------------------------------------
    progress("running", f"Training {architecture} for {epochs} epochs", 0, epochs)

    def _epoch_progress(epoch: int, total: int) -> None:
        progress("running", f"Training epoch {epoch}/{total}", epoch, total)

    train_result = trainer.train(
        data_yaml=summary.data_yaml,
        student_dir=rdir,
        epochs=epochs,
        # Dispatcher uses the architecture's framework default for imgsz —
        # YOLO's 640 today, will diverge once RT-DETR/DINOv3 land. We pass
        # 640 explicitly so the YOLO trainer's behaviour is byte-identical
        # to the pre-Phase-1 code path (`train_yolo` defaulted imgsz=640).
        imgsz=_DEFAULT_IMGSZ,
        progress=_epoch_progress,
    )
    weights = train_result.weights_path
    train_seconds = train_result.train_seconds

    # --- Phase 3: per-eval-teacher mAP -------------------------------------
    per_eval: list[dict] = []
    if eval_teacher_ids:
        for i, etid in enumerate(eval_teacher_ids, start=1):
            progress(
                "finalizing",
                f"Evaluating on eval teacher {i}/{len(eval_teacher_ids)} ({etid})",
                i,
                len(eval_teacher_ids),
            )
            try:
                eval_yaml, n_imgs, n_anns = distill.prepare_eval_dataset(
                    project_id=project_id,
                    student_dir=rdir,
                    eval_teacher_id=etid,
                    class_names=summary.class_names,
                )
                map50, map5095 = trainer.eval(weights=weights, data_yaml=eval_yaml)
            except Exception as e:
                # One bad eval teacher (e.g. missing source video) shouldn't
                # tank the whole run — record a zero row and carry on. The
                # error will appear in logs; the per-teacher table will show
                # the gap.
                log.exception("eval teacher %s failed: %s", etid, e)
                per_eval.append({
                    "teacher_id": etid,
                    "n_images": 0,
                    "n_annotations": 0,
                    "map50": 0.0,
                    "map50_95": 0.0,
                    "error": str(e),
                })
                continue
            per_eval.append({
                "teacher_id": etid,
                "n_images": n_imgs,
                "n_annotations": n_anns,
                "map50": map50,
                "map50_95": map5095,
            })
            log.info(
                "Eval teacher %s: %d images, mAP@0.5=%.3f, mAP@0.5:0.95=%.3f",
                etid, n_imgs, map50, map5095,
            )

    # --- Phase 4: inference timing -----------------------------------------
    progress("finalizing", "Timing inference on a small batch", 0, 0)
    # Prefer the eval set for timing since it's the closest proxy for "real"
    # data the Student will see; fall back to the train val split otherwise.
    timing_dir: Optional[Path] = None
    if eval_teacher_ids:
        first_eval = rdir / "dataset" / "eval" / eval_teacher_ids[0] / "images" / "val"
        if first_eval.exists():
            timing_dir = first_eval
    if timing_dir is None:
        timing_dir = rdir / "dataset" / "images" / "val"
    avg_ms, p50_ms, p95_ms = trainer.time_inference(
        weights=weights, sample_image_dir=timing_dir,
    )

    # --- Persist stats + flip status ---------------------------------------
    map50_mean = (
        sum(r["map50"] for r in per_eval) / len(per_eval) if per_eval else 0.0
    )
    map5095_mean = (
        sum(r["map50_95"] for r in per_eval) / len(per_eval) if per_eval else 0.0
    )

    stats = runs_mod.StudentStats(
        train_images=summary.n_train_images,
        train_annotations=summary.n_train_annotations,
        train_seconds=train_seconds,
        epochs=epochs,
        map50=map50_mean,
        map50_95=map5095_mean,
        avg_inference_ms=avg_ms,
        p50_inference_ms=p50_ms,
        p95_inference_ms=p95_ms,
        model_size_mb=distill.model_size_mb(weights),
        per_eval_teacher=per_eval,
        # Phase 0.5/0.6 — frame-bucket breakdown copied from the prep
        # summary plus the thresholds we used. Stamping the thresholds
        # here means the detail card can render the bucket counts in
        # context without re-reading the manifest.
        n_positive_frames=summary.n_positive_frames,
        n_uncertain_dropped=summary.n_uncertain_dropped,
        n_true_negative_frames=summary.n_true_negative_frames,
        per_teacher_buckets=summary.per_teacher_buckets,
        t_high=t_high,
        t_low=t_low,
        treat_empty_as_negative=treat_empty_as_negative,
        # Phase 2.2 comparability fields. `imgsz` mirrors what we passed
        # into `trainer.train()`. `device` is what the trainer actually
        # used (re-evaluating `_pick_device()` here matches the trainer's
        # internal pick — same function, same machine). The warmup-discard
        # flag is a fixed fact of `YoloTrainer.time_inference`'s body
        # (it always drops call #1); recording it here means future
        # trainers that don't can flip it to False without touching the
        # caller. Phase 3 surfaces these on the compare view.
        imgsz=_DEFAULT_IMGSZ,
        device=_pick_device(),
        inference_warmup_discarded=True,
    )
    runs_mod.mark_student_completed(rdir, stats)
    log.info(
        "Student done: %d train imgs, %d epochs, mean mAP50=%.3f across %d eval teachers",
        summary.n_train_images, epochs, map50_mean, len(per_eval),
    )
