"""Knowledge-distillation data preparation (Phase 1.2 shim).

What lives here now
-------------------
The architecture-agnostic *data* side of distillation:

    • `prepare_yolo_dataset`  — extract + bucket + label train+val frames.
    • `prepare_eval_dataset`  — same shape, one held-out eval teacher.
    • `classify_frames` / `FrameBuckets` — pure confidence-band bucketing.
    • `_coco_to_yolo_lines` — COCO bbox → YOLO line conversion.
    • `_read_teacher_coco`, `_extract_frames` — internal I/O helpers
      (monkey-patched by the regression test, hence still public-ish).
    • `DatasetSummary`, `MIN_TRAIN_ANNOTATIONS`, `INFERENCE_TIMING_SAMPLES`,
      `model_size_mb`, `_pick_device` — utilities consumed by the
      orchestration layer.

What moved out
--------------
The actual trainer — `train_yolo`, `eval_yolo`, `time_inference` — moved
to `pipeline.students.yolo` as methods on the `YoloTrainer` class.
Phase 1's dispatcher (`pipeline.students.make_trainer(name)`) is the
new entry point and `pipeline.optimize` calls it directly. The bodies
were ported byte-for-byte so a yolov8n run through the dispatcher
produces numerically identical results to the pre-Phase-1 path
(spec acceptance gate Phase 1.5 #2).

Why this file still exists
--------------------------
The data-prep helpers above are framework-agnostic — every architecture
the project will support (YOLO, RT-DETR, DINOv3) consumes a YOLO-format
dataset. Keeping them out of the trainer registry means new trainer
files don't have to re-import frame-extraction or COCO conversion.

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
    # [{"teacher_id": str, "positive": int, "uncertain": int, "true_negative": int,
    #   "n_frames_curated": int, "n_frames_confirmed_empty": int,
    #   "n_frames_marked_missed": int, "n_frames_unreviewed_used": int}, ...]
    per_teacher_buckets: list[dict] = field(default_factory=list)
    # Phase 3 review-source counters — see schemas.StudentStatsModel for
    # the exact semantics. Sum across teachers; per-teacher buckets carry
    # the same split.
    n_frames_curated: int = 0
    n_frames_confirmed_empty: int = 0
    n_frames_marked_missed: int = 0
    n_frames_unreviewed_used: int = 0


@dataclass
class FrameBuckets:
    """Per-frame confidence-band classification for one teacher's COCO.

    Frames are bucketed by the teacher's *highest* per-frame detection
    score:

      • `positive`      — at least one annotation with score ≥
                          `export_threshold`. Frame is kept; low-score
                          boxes on it get dropped at YOLO-label time
                          (`min_score=export_threshold`).
      • `uncertain`     — every annotation is in
                          `[t_low, export_threshold)` (and there's at
                          least one). Teacher had something to say but
                          wasn't confident enough to commit. These frames
                          are dropped entirely from training unless
                          `treat_empty_as_negative=True`.
      • `true_negative` — no annotations at all, OR every annotation is
                          strictly below `t_low`. Strongest "really
                          empty" signal — kept as a YOLO empty .txt.

    Frame ids are the COCO image ids (== source video frame index, by
    convention in this project).
    """

    positive: list[int] = field(default_factory=list)
    uncertain: list[int] = field(default_factory=list)
    true_negative: list[int] = field(default_factory=list)


def classify_frames(
    coco: dict, *, export_threshold: float, t_low: float
) -> FrameBuckets:
    """Pure helper: bucket every frame in a COCO dict by teacher confidence.

    Reused by the trainer (Phase 0.3) and the GUI preview endpoint (Phase
    0.4). No I/O, no numpy — small enough to run on every keystroke in
    the GUI.

    Bucket rules (match the dataclass docstring exactly):
      • `positive`: any ann score ≥ `export_threshold`.
      • `uncertain`: at least one ann in [t_low, export_threshold) AND no
        ann ≥ `export_threshold`.
      • `true_negative`: no annotations OR all anns < t_low.

    Annotations missing a `score` key are treated as score=1.0 (same
    convention as `_coco_to_yolo_lines`).
    """
    if t_low > export_threshold:
        raise ValueError(
            f"t_low ({t_low}) must be <= export_threshold ({export_threshold}) — "
            "the uncertain band [t_low, export_threshold) would otherwise "
            "be empty/inverted."
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
        if max_score >= export_threshold:
            buckets.positive.append(frame_id)
        elif max_score >= t_low:
            # At least one ann sits in [t_low, export_threshold) and none
            # cross export_threshold.
            buckets.uncertain.append(frame_id)
        else:
            # Every ann is strictly below t_low — teacher tried, came up
            # with nothing convincing. Strongest "really empty" signal.
            buckets.true_negative.append(frame_id)
    return buckets


def _apply_frame_state_overrides(
    coco: dict, frame_states: dict[int, dict]
) -> dict[str, int]:
    """Filter a teacher's COCO dict in place per Phase 3 review state.

    Three rules, applied in order:

      • `marked_missed` — image and all its annotations are removed entirely.
        These frames represent "the model was wrong here" — we can't trust
        the labels, so the trainer never sees them.
      • `confirmed_empty` — image kept, all its annotations dropped. Forces
        the frame into the `true_negative` bucket regardless of
        `treat_empty_as_negative`. The Student trains on it as a true
        background.
      • `curated` — image kept, annotations whose `det_idx` is in the
        user's `rejected_dets` are dropped; surviving annotations flow
        into the threshold bucketing as usual.

    Frames without a state entry are unchanged. Returns a counters dict
    with the four review-source totals.

    Caller passes a fresh COCO (loaded per-call from JSON), so in-place
    mutation is safe.
    """
    counters = {
        "n_frames_curated": 0,
        "n_frames_confirmed_empty": 0,
        "n_frames_marked_missed": 0,
    }
    if not frame_states:
        return counters

    images_kept: list[dict] = []
    dropped_image_ids: set[int] = set()
    for img in coco.get("images", []):
        fi = int(img["id"])
        entry = frame_states.get(fi)
        if entry is None:
            images_kept.append(img)
            continue
        state = entry.get("state")
        if state == "marked_missed":
            counters["n_frames_marked_missed"] += 1
            dropped_image_ids.add(fi)
            continue
        images_kept.append(img)
        if state == "confirmed_empty":
            counters["n_frames_confirmed_empty"] += 1
        elif state == "curated":
            counters["n_frames_curated"] += 1
    coco["images"] = images_kept

    new_anns: list[dict] = []
    for ann in coco.get("annotations", []):
        fi = int(ann.get("image_id", -1))
        if fi in dropped_image_ids:
            continue
        entry = frame_states.get(fi)
        if entry is None:
            new_anns.append(ann)
            continue
        state = entry.get("state")
        if state == "confirmed_empty":
            continue  # forced negative — drop every annotation
        if state == "curated":
            rejected = set(entry.get("rejected_dets") or [])
            det_idx = ann.get("det_idx")
            # det_idx was added by `_write_teacher_coco` (Phase 3). Pre-Phase-3
            # COCOs without det_idx can't be safely filtered here — but the
            # `runs/` wipe at Phase 1 means we never see them in practice.
            if det_idx is not None and int(det_idx) in rejected:
                continue
        new_anns.append(ann)
    coco["annotations"] = new_anns
    return counters


def _read_teacher_coco(project_id: str, teacher_id: str) -> tuple[dict, str]:
    """Load a teacher's coco.json and return (coco_dict, video_path).

    `video_path` is read from the teacher's manifest — that's where we
    extract frames from on demand, since we deliberately don't pre-extract
    JPGs at Learn time (would add hundreds of MB per Teacher run).
    """
    rdir = runs_mod.run_dir(project_id, teacher_id)
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


def _merge_class_vocabs(
    project_id: str, teacher_ids: list[str]
) -> tuple[list[str], dict[str, dict[int, int]]]:
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
        coco, _ = _read_teacher_coco(project_id, tid)
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
    project_id: str,
    student_dir: Path,
    train_teacher_ids: list[str],
    export_threshold: float = 0.30,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    split_ratio: float = DEFAULT_TRAIN_VAL_SPLIT,
    seed: int = 42,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> DatasetSummary:
    """Materialise a YOLO-format dataset under student_dir/dataset/.

    Each train teacher contributes its frames + curated annotations; we
    union the class vocabularies and split each teacher's frames 80/20
    train/val.

    Frame bucketing (Phase 0.3)
    ---------------------------
    Per teacher, every frame is classified by its highest detection
    score:

      • `positive`      (max score ≥ export_threshold) — extracted,
        labelled with `_coco_to_yolo_lines(min_score=export_threshold)`
        so low-confidence boxes on otherwise-good frames don't leak into
        the training set.
      • `uncertain`     (max in [t_low, export_threshold)) — *dropped
        entirely*. No frame extraction, no label file. The teacher saw
        something but wasn't sure, and using these as either positive
        *or* negative training examples both bias the student.
      • `true_negative` (no anns OR max < t_low) — extracted with an
        empty `.txt`. This is the "really empty" signal — the teacher
        either didn't fire at all or fired only on noise.

    `treat_empty_as_negative=True` is the opt-in escape hatch for users
    who trust their teacher: uncertain frames are reclassified to
    `true_negative` *before* extraction, so the trainer sees the same
    set of frames it did before the bucketing change. Combined with the
    `min_score=export_threshold` filter on positive labels, this
    reproduces the old training set exactly *as long as the teacher's
    detections all sit at or above `export_threshold`*.

    `progress` callback receives (stage_message, current, total) so the
    worker can write a live progress.json.
    """
    if t_low > export_threshold:
        raise ValueError(
            f"t_low ({t_low}) must be <= export_threshold ({export_threshold}); "
            "the uncertain band [t_low, export_threshold) would otherwise "
            "be empty/inverted."
        )

    rng = random.Random(seed)
    class_names, per_teacher_map = _merge_class_vocabs(project_id, train_teacher_ids)
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

    # Aggregate bucket counts surfaced on DatasetSummary → StudentStats.
    n_positive_total = 0
    n_uncertain_dropped_total = 0
    n_true_negative_total = 0
    per_teacher_buckets: list[dict] = []

    # Phase 3 review-source totals.
    n_curated_total = 0
    n_confirmed_empty_total = 0
    n_marked_missed_total = 0
    n_unreviewed_used_total = 0

    for tid in train_teacher_ids:
        coco, video_path = _read_teacher_coco(project_id, tid)
        cat_map = per_teacher_map[tid]

        # Phase 3: load this teacher's per-frame review state and apply the
        # three overrides (curated / confirmed_empty / marked_missed) BEFORE
        # the threshold bucketing. The bucketing then sees a COCO that
        # already reflects the user's verdict — no special-casing inside
        # `classify_frames` needed.
        rdir = runs_mod.run_dir(project_id, tid)
        frame_states = runs_mod.read_frame_states(rdir)
        override_counts = _apply_frame_state_overrides(coco, frame_states)
        n_curated_total += override_counts["n_frames_curated"]
        n_confirmed_empty_total += override_counts["n_frames_confirmed_empty"]
        n_marked_missed_total += override_counts["n_frames_marked_missed"]

        # Index COCO annotations by image_id so we don't re-scan per image.
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        # Bucket the teacher's frames. The breakdown is reported even
        # when treat_empty_as_negative=True so the user sees what would
        # have been dropped by the strict policy.
        buckets = classify_frames(
            coco, export_threshold=export_threshold, t_low=t_low
        )
        n_positive = len(buckets.positive)
        n_uncertain = len(buckets.uncertain)
        n_true_negative = len(buckets.true_negative)

        # Count unreviewed frames that survived bucketing into a kept
        # bucket (positive or true_negative). These are frames the user
        # never looked at; the threshold bucketing decided their fate.
        # `uncertain` frames that get dropped don't count as "used".
        kept_for_training = set(buckets.positive) | set(buckets.true_negative)
        if treat_empty_as_negative:
            kept_for_training |= set(buckets.uncertain)
        teacher_unreviewed_used = sum(
            1 for fi in kept_for_training if fi not in frame_states
        )
        n_unreviewed_used_total += teacher_unreviewed_used

        per_teacher_buckets.append({
            "teacher_id": tid,
            "positive": n_positive,
            "uncertain": n_uncertain,
            "true_negative": n_true_negative,
            "n_frames_curated": override_counts["n_frames_curated"],
            "n_frames_confirmed_empty": override_counts["n_frames_confirmed_empty"],
            "n_frames_marked_missed": override_counts["n_frames_marked_missed"],
            "n_frames_unreviewed_used": teacher_unreviewed_used,
        })

        # Effective bucketing for this run: in escape-hatch mode the
        # uncertain frames join true_negative before extraction. We
        # don't mutate `buckets` itself so the GUI gets the strict
        # numbers in the per-teacher breakdown.
        if treat_empty_as_negative:
            extract_positive = list(buckets.positive)
            extract_true_neg = list(buckets.true_negative) + list(buckets.uncertain)
            n_uncertain_dropped_total += 0  # nothing dropped this mode
        else:
            extract_positive = list(buckets.positive)
            extract_true_neg = list(buckets.true_negative)
            n_uncertain_dropped_total += n_uncertain

        n_positive_total += n_positive
        n_true_negative_total += n_true_negative

        # Frames the trainer actually sees (set, for fast lookup later).
        kept_frames = set(extract_positive) | set(extract_true_neg)

        # Build the index map needed for split assignment. Iterate
        # `images` in original COCO order so the val split is reproducible
        # for a given seed regardless of which buckets got dropped.
        images = coco.get("images", [])
        kept_images = [img for img in images if int(img["id"]) in kept_frames]
        if not kept_images:
            # Teacher contributed nothing — skip extraction; aggregate
            # counts already reflect this.
            continue

        # Shuffle once per teacher so the val split isn't all the
        # last-frames-of-the-clip (which would skew toward late-game
        # state on soccer footage).
        order = list(range(len(kept_images)))
        rng.shuffle(order)
        cut = int(len(order) * split_ratio)
        train_idx = set(order[:cut])

        # Plan frame extraction for both splits in one cv2 pass.
        out_paths: dict[int, Path] = {}
        per_image_split: dict[int, str] = {}
        for i, img in enumerate(kept_images):
            frame_idx = int(img["id"])
            split = "train" if i in train_idx else "val"
            per_image_split[frame_idx] = split
            sub = (images_train if split == "train" else images_val) / tid
            out_paths[frame_idx] = sub / f"frame_{frame_idx:06d}.jpg"

        if progress:
            progress(f"Extracting {len(kept_images)} frames from {tid}", 0, len(kept_images))

        def _frame_progress(cur: int, tot: int, _tid: str = tid) -> None:
            if progress:
                progress(f"Extracting frames from {_tid}", cur, tot)

        _extract_frames(
            video_path=video_path,
            frame_indices=list(out_paths.keys()),
            out_paths=out_paths,
            progress=_frame_progress,
        )

        positive_set = set(extract_positive)

        # Now write the YOLO label files alongside each extracted frame.
        for img in kept_images:
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

            if frame_idx in positive_set:
                # Positive frame — write filtered labels. Any teacher
                # detection below `export_threshold` is a low-confidence
                # box on an otherwise good frame; dropping it keeps the
                # label set honest at the cost of a few real positives
                # slipping through as background. The classify_frames
                # bucketing already guaranteed there's at least one ann
                # ≥ export_threshold so the resulting line list is
                # non-empty in the normal case.
                anns = anns_by_image.get(frame_idx, [])
                lines = _coco_to_yolo_lines(
                    anns,
                    img_w=int(img.get("width") or 0),
                    img_h=int(img.get("height") or 0),
                    coco_to_global=cat_map,
                    min_score=export_threshold,
                )
            else:
                # True negative (or escape-hatch reclassified uncertain) —
                # genuine background. YOLO convention: empty .txt means
                # "this frame contains no objects". Still useful at train
                # time because false-positive predictions count against
                # precision.
                lines = []

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
            "lowering export_threshold or rewording the prompt."
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
        n_positive_frames=n_positive_total,
        n_uncertain_dropped=n_uncertain_dropped_total,
        n_true_negative_frames=n_true_negative_total,
        per_teacher_buckets=per_teacher_buckets,
        n_frames_curated=n_curated_total,
        n_frames_confirmed_empty=n_confirmed_empty_total,
        n_frames_marked_missed=n_marked_missed_total,
        n_frames_unreviewed_used=n_unreviewed_used_total,
    )


def prepare_eval_dataset(
    *,
    project_id: str,
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
    coco, video_path = _read_teacher_coco(project_id, eval_teacher_id)
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


# ---- Trainer-side utilities ------------------------------------------------
#
# `_pick_device` historically lived in this file — its only callers today
# are the YOLO trainer (now in `pipeline/students/yolo.py`) and the eval
# helpers above. We re-export the canonical implementation from there so
# the device pick logic has exactly one home; existing callers that did
# `from pipeline.distill import _pick_device` keep working unchanged.

from pipeline.students.yolo import _pick_device  # noqa: E402, F401 — re-export


# ---- Convenience -----------------------------------------------------------


def model_size_mb(weights: Path) -> float:
    """Disk size of the trained checkpoint, in MB. Used for StudentStats."""
    if not weights.exists():
        return 0.0
    return weights.stat().st_size / 1_000_000.0
