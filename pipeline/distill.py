"""Knowledge distillation: turn Teacher COCO labels into a trained Student.

This module owns the actual trainer — frame extraction, COCO→YOLO
conversion, Ultralytics training, per-eval-teacher mAP, and inference
timing. `pipeline.optimize` is the queue/orchestration layer that calls
in here from a worker thread; this file is pure mechanics, no I/O on
manifest/progress files.

Why this lives in its own module
--------------------------------
The trainer is the heaviest piece of the project — heavy in deps
(torch + ultralytics + cv2 frame extraction), heavy in side-effects (it
materialises ~hundreds of JPGs on disk and a YOLO checkpoint), and heavy
in failure modes (CUDA/MPS device picking, Ultralytics version drift,
COCO category-id remapping). Keeping it isolated from the queue/orchestration
in `optimize.py` makes it testable in a notebook without spinning up the
FastAPI app, and lets the worker stay short and readable.

On-disk layout under runs/student_<id>/
---------------------------------------
    dataset/
        data.yaml                  # YOLO dataset descriptor
        images/
            train/teacher_<id>/<frame>.jpg
            val/teacher_<id>/<frame>.jpg
        labels/
            train/teacher_<id>/<frame>.txt
            val/teacher_<id>/<frame>.txt
        eval/<eval_teacher_id>/    # one self-contained YOLO dataset per eval teacher
            data.yaml
            images/val/...
            labels/val/...
    ultralytics/                   # Ultralytics' run output (weights/, results.csv)
    best.pt                        # trained weights (copied out of ultralytics/)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from pipeline import runs as runs_mod

log = logging.getLogger(__name__)


# Default 80/20 train/val split inside the merged train set. The split is
# *random* per-frame; for a few hundred frames per teacher that's fine,
# and it keeps the eval teachers strictly held out from any training
# signal (early stopping, learning-rate scheduling, anything).
DEFAULT_TRAIN_VAL_SPLIT = 0.8

# Minimum number of positive boxes in the merged train set before we even
# attempt training. Below this, YOLO will burn epochs producing nothing
# useful and silently skip best.pt because there's no fitness signal —
# better to fail loud at prep time and point the user back at the Teacher
# run, where the real problem lives (prompt didn't catch the object,
# threshold too high, etc.).
MIN_TRAIN_ANNOTATIONS = 5

# Frames sampled for inference timing. 30 covers ~1 second at 30fps, which
# is enough to get a stable median; more would burn time without changing
# the number much.
INFERENCE_TIMING_SAMPLES = 30


# ---- Data preparation ------------------------------------------------------


@dataclass
class DatasetSummary:
    """Returned from prepare_yolo_dataset so the worker can populate stats."""

    data_yaml: Path
    n_train_images: int
    n_train_annotations: int
    n_val_images: int
    class_names: list[str]
    # Confidence-band bucket counts (Phase 0.3). Defaulted so test code or
    # callers that don't care about bucketing still construct cleanly.
    n_positive_frames: int = 0
    n_uncertain_dropped: int = 0
    n_true_negative_frames: int = 0
    # Per-train-teacher bucket breakdown:
    # [{"teacher_id": str, "positive": int, "uncertain": int, "true_negative": int}, ...]
    per_teacher_buckets: list[dict] = field(default_factory=list)


@dataclass
class FrameBuckets:
    """Per-frame confidence-band classification for one teacher's COCO.

    Frames are bucketed by the teacher's *highest* per-frame detection
    score:

      • `positive`      — at least one annotation with score ≥ t_high.
                          Frame is kept; low-score boxes on it get
                          dropped at YOLO-label time (`min_score=t_high`).
      • `uncertain`     — every annotation is in `[t_low, t_high)` (and
                          there's at least one). Teacher had something
                          to say but wasn't confident enough to commit.
                          These frames are dropped entirely from training
                          unless `treat_empty_as_negative=True`.
      • `true_negative` — no annotations at all, OR every annotation is
                          strictly below `t_low`. Strongest "really
                          empty" signal — kept as a YOLO empty .txt.

    Frame ids are the COCO image ids (== source video frame index, by
    convention in this project).
    """

    positive: list[int] = field(default_factory=list)
    uncertain: list[int] = field(default_factory=list)
    true_negative: list[int] = field(default_factory=list)


def classify_frames(coco: dict, *, t_high: float, t_low: float) -> FrameBuckets:
    """Pure helper: bucket every frame in a COCO dict by teacher confidence.

    Reused by the trainer (Phase 0.3) and the GUI preview endpoint (Phase
    0.4). No I/O, no numpy — small enough to run on every keystroke in
    the GUI.

    Bucket rules (match the dataclass docstring exactly):
      • `positive`: any ann score ≥ t_high.
      • `uncertain`: at least one ann in [t_low, t_high) AND no ann
        ≥ t_high.
      • `true_negative`: no annotations OR all anns < t_low.

    Annotations missing a `score` key are treated as score=1.0 (same
    convention as `_coco_to_yolo_lines`).
    """
    if t_low > t_high:
        raise ValueError(
            f"t_low ({t_low}) must be <= t_high ({t_high}) — "
            "the uncertain band [t_low, t_high) would otherwise be empty/inverted."
        )

    # Index annotations by image id so each frame's verdict is one pass.
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    buckets = FrameBuckets()
    for img in coco.get("images", []):
        frame_id = int(img["id"])
        anns = anns_by_image.get(frame_id, [])
        if not anns:
            buckets.true_negative.append(frame_id)
            continue
        scores = [float(a.get("score", 1.0)) for a in anns]
        max_score = max(scores)
        if max_score >= t_high:
            buckets.positive.append(frame_id)
        elif max_score >= t_low:
            # At least one ann sits in [t_low, t_high) and none cross t_high.
            buckets.uncertain.append(frame_id)
        else:
            # Every ann is strictly below t_low — teacher tried, came up
            # with nothing convincing. Strongest "really empty" signal.
            buckets.true_negative.append(frame_id)
    return buckets


def _read_teacher_coco(teacher_id: str) -> tuple[dict, str]:
    """Load a teacher's coco.json and return (coco_dict, video_path).

    `video_path` is read from the teacher's manifest — that's where we
    extract frames from on demand, since we deliberately don't pre-extract
    JPGs at Learn time (would add hundreds of MB per Teacher run).
    """
    rdir = runs_mod.run_dir(teacher_id)
    if not (rdir / runs_mod.MANIFEST_NAME).exists():
        raise FileNotFoundError(f"teacher dir missing: {rdir}")
    coco_path = rdir / runs_mod.LABELS_DIR / runs_mod.COCO_NAME
    if not coco_path.exists():
        raise FileNotFoundError(
            f"teacher {teacher_id!r} has no coco.json at {coco_path}"
        )
    coco = json.loads(coco_path.read_text())
    manifest = runs_mod.read_manifest(rdir)
    return coco, manifest.video_path


def _extract_frames(
    video_path: str,
    frame_indices: list[int],
    out_paths: dict[int, Path],
    progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Pull a sparse set of frames out of a video and write them as JPGs.

    cv2's `set(CAP_PROP_POS_FRAMES, idx)` is correct but not strictly
    O(1) — for many codecs each seek may decode from the previous keyframe.
    For our use case (a few hundred frames from one short clip) it's fast
    enough; if students start training on long clips this is the first
    place to optimise (sort indices + sequential read with skip).
    """
    if not frame_indices:
        return
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video for frame extraction: {video_path}")
        # Sort indices so seeks march forward — much cheaper for inter-frame
        # codecs than random-access ones.
        sorted_idx = sorted(frame_indices)
        total = len(sorted_idx)
        for i, idx in enumerate(sorted_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("frame %d unreadable in %s — skipping", idx, video_path)
                continue
            out_path = out_paths[idx]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if progress and (i + 1) % 25 == 0:
                progress(i + 1, total)
        if progress:
            progress(total, total)
    finally:
        cap.release()


def _coco_to_yolo_lines(
    annotations: list[dict],
    img_w: int,
    img_h: int,
    coco_to_global: dict[int, int],
    min_score: float = 0.0,
) -> list[str]:
    """Convert COCO annotations (x,y,w,h in pixels) to YOLO's normalized
    `class cx cy w h` (all in [0,1]).

    `coco_to_global` maps THIS teacher's COCO category_id → the merged
    global class id used across all train teachers. Annotations whose
    category isn't in the map (because their class isn't in the merged
    vocabulary) are dropped silently — that only happens when called for
    an eval teacher whose vocab is a superset of train's, and the trainer
    already logged a warning in that case.

    `min_score` filters annotations by `ann["score"]`. Default 0.0
    preserves prior behaviour (no filtering). Annotations missing a
    `score` key are treated as score=1.0 so future user-added labels (no
    teacher-confidence attached) pass through unchanged.
    """
    if img_w <= 0 or img_h <= 0:
        return []
    lines: list[str] = []
    for ann in annotations:
        cat = ann.get("category_id")
        if cat not in coco_to_global:
            continue
        if float(ann.get("score", 1.0)) < min_score:
            continue
        x, y, w, h = ann["bbox"]
        # COCO bbox can occasionally be slightly out of image due to int
        # rounding upstream; clip to [0, dim] before normalising.
        x = max(0.0, min(float(x), img_w))
        y = max(0.0, min(float(y), img_h))
        w = max(0.0, min(float(w), img_w - x))
        h = max(0.0, min(float(h), img_h - y))
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        lines.append(f"{coco_to_global[cat]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _merge_class_vocabs(teacher_ids: list[str]) -> tuple[list[str], dict[str, dict[int, int]]]:
    """Build a global class vocabulary across teachers.

    Returns:
      class_names: list[str] — global class id = index into this list.
      per_teacher: {teacher_id: {teacher_coco_cat_id: global_class_id}}

    A class name seen by multiple teachers gets the same global id —
    that's the whole point of allowing multi-teacher distillation.
    """
    class_to_global: dict[str, int] = {}
    class_names: list[str] = []
    per_teacher: dict[str, dict[int, int]] = {}
    for tid in teacher_ids:
        coco, _ = _read_teacher_coco(tid)
        local_map: dict[int, int] = {}
        for cat in coco.get("categories", []):
            cname = cat["name"]
            cid = int(cat["id"])
            if cname not in class_to_global:
                class_to_global[cname] = len(class_names)
                class_names.append(cname)
            local_map[cid] = class_to_global[cname]
        per_teacher[tid] = local_map
    return class_names, per_teacher


def prepare_yolo_dataset(
    *,
    student_dir: Path,
    train_teacher_ids: list[str],
    split_ratio: float = DEFAULT_TRAIN_VAL_SPLIT,
    seed: int = 42,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> DatasetSummary:
    """Materialise a YOLO-format dataset under student_dir/dataset/.

    Each train teacher contributes its frames + curated annotations; we
    union the class vocabularies and split each teacher's frames 80/20
    train/val. `progress` callback receives (stage_message, current,
    total) so the worker can write a live progress.json.
    """
    rng = random.Random(seed)
    class_names, per_teacher_map = _merge_class_vocabs(train_teacher_ids)
    if not class_names:
        raise ValueError(
            "merged train teachers have no classes — Teacher COCOs are empty?"
        )

    dataset_dir = student_dir / "dataset"
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    n_train_imgs = 0
    n_val_imgs = 0
    n_train_anns = 0

    for tid in train_teacher_ids:
        coco, video_path = _read_teacher_coco(tid)
        cat_map = per_teacher_map[tid]
        # Index COCO annotations by image_id so we don't re-scan per image.
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        images = coco.get("images", [])
        # Shuffle once per teacher so the val split isn't all the
        # last-frames-of-the-clip (which would skew toward late-game
        # state on soccer footage).
        order = list(range(len(images)))
        rng.shuffle(order)
        cut = int(len(order) * split_ratio)
        train_idx = set(order[:cut])

        # Plan frame extraction for both splits in one cv2 pass.
        out_paths: dict[int, Path] = {}
        per_image_split: dict[int, str] = {}
        for i, img in enumerate(images):
            frame_idx = int(img["id"])
            split = "train" if i in train_idx else "val"
            per_image_split[frame_idx] = split
            sub = (images_train if split == "train" else images_val) / tid
            out_paths[frame_idx] = sub / f"frame_{frame_idx:06d}.jpg"

        if progress:
            progress(f"Extracting {len(images)} frames from {tid}", 0, len(images))

        def _frame_progress(cur: int, tot: int, _tid: str = tid) -> None:
            if progress:
                progress(f"Extracting frames from {_tid}", cur, tot)

        _extract_frames(
            video_path=video_path,
            frame_indices=list(out_paths.keys()),
            out_paths=out_paths,
            progress=_frame_progress,
        )

        # Now write the YOLO label files alongside each extracted frame.
        for img in images:
            frame_idx = int(img["id"])
            split = per_image_split[frame_idx]
            label_root = labels_train if split == "train" else labels_val
            sub = label_root / tid
            sub.mkdir(parents=True, exist_ok=True)
            label_path = sub / f"frame_{frame_idx:06d}.txt"

            jpg_path = out_paths[frame_idx]
            if not jpg_path.exists():
                # Frame was unreadable — skip both image + label.
                continue

            anns = anns_by_image.get(frame_idx, [])
            lines = _coco_to_yolo_lines(
                anns,
                img_w=int(img.get("width") or 0),
                img_h=int(img.get("height") or 0),
                coco_to_global=cat_map,
            )
            # Empty file is the YOLO convention for "this is a negative
            # frame, no objects" — it still contributes a precision signal
            # during training because false-positives on it count.
            label_path.write_text("\n".join(lines))

            if split == "train":
                n_train_imgs += 1
                n_train_anns += len(lines)
            else:
                n_val_imgs += 1

    # Data-sufficiency check. If the Teacher's COCO is essentially empty
    # (e.g. GroundingDINO whiffed at the prompt, threshold too high, the
    # object class is too rare in the clip), there's no point firing up
    # the trainer — Ultralytics will run all the requested epochs against
    # mostly-negative frames, fail to learn anything, and silently skip
    # best.pt because val mAP never improves. The error you'd see is the
    # opaque "Ultralytics produced no weights" RuntimeError below; better
    # to surface the real reason here.
    if n_train_anns < MIN_TRAIN_ANNOTATIONS:
        raise RuntimeError(
            f"Training set has only {n_train_anns} positive box(es) across "
            f"{n_train_imgs} train frame(s) — need at least "
            f"{MIN_TRAIN_ANNOTATIONS} to attempt training. "
            "The Teacher run(s) likely didn't detect the target object — "
            "check the Teacher's per-frame labels (Inspector) and consider "
            "lowering box_threshold/text_threshold or rewording the prompt."
        )

    # Build the YOLO data.yaml. Paths are absolute so YOLO doesn't try to
    # resolve them relative to its own runs/ output dir.
    data_yaml = dataset_dir / "data.yaml"
    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, cname in enumerate(class_names):
        # Single quotes around names so a class name with special chars
        # (spaces, colons) doesn't break the YAML parser.
        safe = cname.replace("'", "''")
        yaml_lines.append(f"  {i}: '{safe}'")
    data_yaml.write_text("\n".join(yaml_lines) + "\n")

    return DatasetSummary(
        data_yaml=data_yaml,
        n_train_images=n_train_imgs,
        n_train_annotations=n_train_anns,
        n_val_images=n_val_imgs,
        class_names=class_names,
    )


def prepare_eval_dataset(
    *,
    student_dir: Path,
    eval_teacher_id: str,
    class_names: list[str],
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> tuple[Path, int, int]:
    """Build a one-teacher YOLO dataset under student_dir/dataset/eval/<id>/.

    Used for per-eval-teacher mAP computation. The class_names list comes
    from the trained Student's vocabulary — eval annotations whose class
    isn't in that list are dropped (which lowers mAP, as it should: the
    Student literally cannot predict an unknown class).

    Returns (data_yaml_path, n_images, n_annotations).
    """
    coco, video_path = _read_teacher_coco(eval_teacher_id)
    # Map this eval teacher's COCO categories → the trained vocab's ids.
    # If a category name doesn't exist in the Student's vocab, we skip it
    # (annotations with that cat get dropped, which is the right semantics
    # for evaluating "can the trained Student handle this class").
    name_to_global = {n: i for i, n in enumerate(class_names)}
    cat_map: dict[int, int] = {}
    for cat in coco.get("categories", []):
        cname = cat["name"]
        if cname in name_to_global:
            cat_map[int(cat["id"])] = name_to_global[cname]

    eval_dir = student_dir / "dataset" / "eval" / eval_teacher_id
    images_val = eval_dir / "images" / "val"
    labels_val = eval_dir / "labels" / "val"
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    images = coco.get("images", [])
    out_paths = {
        int(img["id"]): images_val / f"frame_{int(img['id']):06d}.jpg"
        for img in images
    }

    if progress:
        progress(f"Eval: extracting {len(images)} frames from {eval_teacher_id}", 0, len(images))

    def _frame_progress(cur: int, tot: int) -> None:
        if progress:
            progress(f"Eval: extracting frames from {eval_teacher_id}", cur, tot)

    _extract_frames(
        video_path=video_path,
        frame_indices=list(out_paths.keys()),
        out_paths=out_paths,
        progress=_frame_progress,
    )

    n_imgs = 0
    n_anns = 0
    for img in images:
        frame_idx = int(img["id"])
        jpg = out_paths[frame_idx]
        if not jpg.exists():
            continue
        label_path = labels_val / f"frame_{frame_idx:06d}.txt"
        lines = _coco_to_yolo_lines(
            anns_by_image.get(frame_idx, []),
            img_w=int(img.get("width") or 0),
            img_h=int(img.get("height") or 0),
            coco_to_global=cat_map,
        )
        label_path.write_text("\n".join(lines))
        n_imgs += 1
        n_anns += len(lines)

    # YOLO requires both `train:` and `val:` even for eval-only — point
    # train at the same val dir so it parses cleanly. We never call
    # train() on this yaml.
    data_yaml = eval_dir / "data.yaml"
    yaml_lines = [
        f"path: {eval_dir.resolve()}",
        "train: images/val",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, cname in enumerate(class_names):
        safe = cname.replace("'", "''")
        yaml_lines.append(f"  {i}: '{safe}'")
    data_yaml.write_text("\n".join(yaml_lines) + "\n")

    return data_yaml, n_imgs, n_anns


# ---- Training --------------------------------------------------------------


def _pick_device() -> str:
    """Best available torch device for Ultralytics on this box.

    On Apple Silicon (the project's primary target) MPS is dramatically
    faster than CPU — Ultralytics auto-detects it but we set it explicitly
    so the progress logs show the right thing.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dep
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo(
    *,
    data_yaml: Path,
    student_dir: Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[Path, float]:
    """Train YOLOv8 on the prepared dataset; return (best_weights_path, seconds).

    `progress(epoch, total_epochs)` is invoked after each epoch via an
    Ultralytics callback so the GUI's progress bar tracks training rather
    than freezing for the entire train_seconds duration.
    """
    from ultralytics import YOLO  # imported lazily — heavy module

    device = _pick_device()
    log.info("Distill: training %s on %s, %d epochs, imgsz=%d",
             base_model, device, epochs, imgsz)

    model = YOLO(base_model)

    if progress:
        # Ultralytics calls each callback with the trainer object; we
        # peek at trainer.epoch (0-indexed) to report 1-based progress.
        def _on_epoch_end(trainer):
            try:
                progress(int(trainer.epoch) + 1, int(epochs))
            except Exception as e:  # pragma: no cover — never let progress kill training
                log.warning("progress callback failed: %s", e)

        model.add_callback("on_train_epoch_end", _on_epoch_end)

    # Ultralytics' `project` arg is interpreted relative to the current CWD
    # for filesystem ops, but newer versions (8.3+) can also resolve it
    # against SETTINGS["runs_dir"] in some code paths. Pass an absolute
    # path so output ends up exactly where we look for it on success and
    # error reporting stays accurate.
    project_dir = (student_dir / "ultralytics").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Tee Ultralytics' chatter into a per-run log file so post-mortems
    # don't have to scroll the uvicorn console. We attach a FileHandler
    # to the "ultralytics" logger (catches LOGGER.warning/info), and also
    # redirect stdout/stderr (catches print() and tqdm). The two streams
    # may interleave, but for debugging "training silently produced no
    # weights" that's exactly what we want.
    log_path = student_dir / "train.log"
    ult_logger = logging.getLogger("ultralytics")
    fh = logging.FileHandler(str(log_path))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ult_logger.addHandler(fh)

    t0 = time.perf_counter()
    try:
        with open(log_path, "a") as logf, redirect_stdout(logf), redirect_stderr(logf):
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                project=str(project_dir),
                name="train",
                exist_ok=True,
                verbose=True,  # log everything; we capture it to train.log
                plots=False,  # save the disk space; we don't render Ultralytics plots in the UI
            )
    finally:
        ult_logger.removeHandler(fh)
        fh.close()
    elapsed = time.perf_counter() - t0

    # Locate best.pt — Ultralytics writes it into project/name/weights/best.pt.
    best = project_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        # Fallback: last.pt is usually present even if best.pt save was suppressed.
        last = project_dir / "train" / "weights" / "last.pt"
        if not last.exists():
            raise RuntimeError(
                f"Ultralytics produced no weights under {best.parent} — "
                f"training may have failed silently. See {log_path} for the "
                "captured stdout/stderr from the trainer."
            )
        best = last

    # Copy to a stable, predictable location at the student dir root so
    # downstream callers don't have to know about the ultralytics subtree.
    out = student_dir / "best.pt"
    shutil.copy2(best, out)
    log.info("Distill: training done in %.1fs; weights -> %s", elapsed, out)
    return out, elapsed


# ---- Evaluation ------------------------------------------------------------


def eval_yolo(
    *,
    weights: Path,
    data_yaml: Path,
) -> tuple[float, float]:
    """Compute (map50, map50_95) for a YOLO checkpoint on a YOLO data.yaml.

    Used per eval teacher to populate the per-eval-teacher transferability
    table. Ultralytics' `model.val()` returns a `DetMetrics` object with
    `.box.map50` and `.box.map`.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(
        data=str(data_yaml),
        device=_pick_device(),
        verbose=False,
        plots=False,
        save_json=False,
    )
    map50 = float(getattr(res.box, "map50", 0.0) or 0.0)
    map5095 = float(getattr(res.box, "map", 0.0) or 0.0)  # the unsuffixed `.map` IS map@0.5:0.95
    return map50, map5095


# ---- Inference timing ------------------------------------------------------


def time_inference(
    *,
    weights: Path,
    sample_image_dir: Path,
    n_samples: int = INFERENCE_TIMING_SAMPLES,
) -> tuple[float, float, float]:
    """Run inference on up to `n_samples` JPGs and return (avg_ms, p50_ms, p95_ms).

    We exclude the first call from the average — the first inference pays
    for kernel compilation, JIT warm-up, and weight-to-device transfer.
    Subsequent calls are what the user will actually feel at runtime.
    """
    from ultralytics import YOLO

    jpgs = sorted(sample_image_dir.rglob("*.jpg"))
    if not jpgs:
        log.warning("no images for timing under %s — skipping", sample_image_dir)
        return 0.0, 0.0, 0.0
    sample = jpgs[:n_samples]

    model = YOLO(str(weights))
    device = _pick_device()

    # Warmup pass — discarded.
    model.predict(str(sample[0]), device=device, verbose=False)

    timings: list[float] = []
    for p in sample:
        t0 = time.perf_counter()
        model.predict(str(p), device=device, verbose=False)
        timings.append((time.perf_counter() - t0) * 1000.0)

    if not timings:
        return 0.0, 0.0, 0.0
    timings.sort()
    avg = sum(timings) / len(timings)
    p50 = timings[len(timings) // 2]
    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
    return avg, p50, p95


# ---- Convenience -----------------------------------------------------------


def model_size_mb(weights: Path) -> float:
    """Disk size of the trained checkpoint, in MB. Used for StudentStats."""
    if not weights.exists():
        return 0.0
    return weights.stat().st_size / 1_000_000.0
