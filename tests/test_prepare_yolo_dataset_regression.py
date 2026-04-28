"""Phase 0.7 acceptance #2 — `treat_empty_as_negative=True` regression check.

This is the most important test in this PR. It guarantees the
escape-hatch mode reproduces the old behaviour exactly on a synthetic
two-frame two-class COCO, so existing users who flip the flag get the
same training set they had before bucketing landed.

The construction stays inside the regime where exact reproduction is
defined: every teacher detection has score ≥ t_high (so no anns fall
in the uncertain band, which is the typical case given the project's
GroundingDINO `box_threshold=0.25-0.30` defaults). With that:

  • old code: extract every frame in coco.images; write
    _coco_to_yolo_lines(no filter) for each.
  • new code with treat_empty_as_negative=True: bucket runs (uncertain
    is empty → no reclassification effect), positive frames write
    _coco_to_yolo_lines(min_score=t_high) which keeps every box because
    every box is ≥ t_high; true_negative frames write empty .txt
    (matches old behaviour for empty-ann frames).

We compare image counts and total annotation counts on the train set
(plus val image count). The split shuffle is seeded the same way in
both runs, so even file-by-file the outputs are identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline import distill


# ---- Synthetic COCO fixture ------------------------------------------------


def _make_synthetic_coco() -> dict:
    """Small two-class COCO sized to clear MIN_TRAIN_ANNOTATIONS=5.

    Frames 0-4: each has two anns, both at score ≥ t_high=0.35
                (cat 0 at 0.9, cat 1 at 0.5)              — positive.
    Frame 5:    no anns                                   — true negative.

    Six frames, ten total anns. With the trainer's 80/20 split (seed=42)
    that's plenty of train anns regardless of which frames land in val.
    All scores live above the strict bucket boundary so the uncertain
    band is empty — both `treat_empty_as_negative=True` and the strict
    default produce the same image set, which is exactly what makes the
    regression compare meaningful.
    """
    images = [
        {"id": i, "width": 200, "height": 100, "file_name": f"frame_{i:06d}.jpg"}
        for i in range(6)
    ]
    annotations: list[dict] = []
    for i in range(5):  # frames 0..4 are positive
        annotations.append({
            "image_id": i, "category_id": 0,
            "bbox": [10, 20, 50, 30], "score": 0.9,
        })
        annotations.append({
            "image_id": i, "category_id": 1,
            "bbox": [80, 40, 60, 30], "score": 0.5,
        })
    # frame 5 has no annotations — true negative
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "ball"},
            {"id": 1, "name": "player"},
        ],
    }


# ---- Test plumbing --------------------------------------------------------


@dataclass
class _Manifest:
    """Minimal stand-in for runs.RunManifest. Only `.video_path` is read."""

    video_path: str


def _patch_io(monkeypatch: pytest.MonkeyPatch, coco: dict) -> None:
    """Replace the two disk-touching helpers in pipeline.distill so this
    test runs without real teacher dirs or videos.

      • `_read_teacher_coco` returns the synthetic dict + a fake video
        path. Both `prepare_yolo_dataset` and `_merge_class_vocabs` call
        through here, so one patch covers both.
      • `_extract_frames` writes a 1-byte placeholder JPG at each output
        path so the `if not jpg_path.exists()` guard in the trainer
        falls through to the label-writing branch. We don't need real
        pixels — only the label text matters for this comparison.
    """

    def fake_read_teacher_coco(project_id: str, tid: str) -> tuple[dict, str]:
        return coco, f"/fake/{tid}.mp4"

    def fake_extract_frames(*, video_path, frame_indices, out_paths, progress=None):
        for idx, path in out_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00")  # placeholder; trainer only checks .exists()
        if progress:
            progress(len(frame_indices), len(frame_indices))

    monkeypatch.setattr(distill, "_read_teacher_coco", fake_read_teacher_coco)
    monkeypatch.setattr(distill, "_extract_frames", fake_extract_frames)


# ---- Tests ----------------------------------------------------------------


def test_treat_empty_as_negative_reproduces_old_image_and_annotation_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 0.7 acceptance #2 — the load-bearing check."""
    coco = _make_synthetic_coco()
    _patch_io(monkeypatch, coco)

    summary = distill.prepare_yolo_dataset(
        project_id="proj_test",
        student_dir=tmp_path,
        train_teacher_ids=["teacher_a"],
        treat_empty_as_negative=True,
        # Defaults: t_high=0.35, t_low=0.15. Synthesized COCO has no
        # anns in the uncertain band, so even strict mode would produce
        # the same numbers.
    )

    # Old behaviour, computed from the synthetic input directly:
    #   - every frame in coco.images was extracted
    #   - every annotation made it into a YOLO label (no min_score filter)
    n_total_images = len(coco["images"])
    n_total_anns = len(coco["annotations"])

    # Regression check #1: same image count.
    assert summary.n_train_images + summary.n_val_images == n_total_images
    # Regression check #2: same total annotation count across train+val.
    assert summary.n_train_annotations + _count_val_anns(tmp_path) == n_total_anns

    # And the bucket breakdown surfaces the right counts even in
    # escape-hatch mode (the GUI shows these to the user).
    assert summary.n_positive_frames == 5
    assert summary.n_uncertain_dropped == 0  # explicitly zeroed in escape-hatch mode
    assert summary.n_true_negative_frames == 1
    assert summary.per_teacher_buckets == [
        {"teacher_id": "teacher_a", "positive": 5, "uncertain": 0, "true_negative": 1}
    ]


def test_strict_mode_drops_uncertain_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror of the regression test — confirm strict mode actually
    diverges from the old training set (otherwise the regression check
    above would pass trivially even if the bucketing was a no-op)."""
    # Same fixture but add an extra frame whose only ann sits in [t_low, t_high).
    coco = _make_synthetic_coco()
    uncertain_frame_id = max(int(img["id"]) for img in coco["images"]) + 1
    coco["images"].append(
        {"id": uncertain_frame_id, "width": 200, "height": 100,
         "file_name": f"frame_{uncertain_frame_id:06d}.jpg"}
    )
    coco["annotations"].append(
        {"image_id": uncertain_frame_id, "category_id": 0,
         "bbox": [0, 0, 10, 10], "score": 0.20}
    )
    _patch_io(monkeypatch, coco)

    n_total_frames = len(coco["images"])  # 7 = 5 pos + 1 neg + 1 uncertain

    strict = distill.prepare_yolo_dataset(
        project_id="proj_test",
        student_dir=tmp_path / "strict",
        train_teacher_ids=["teacher_a"],
        treat_empty_as_negative=False,
    )
    # The added frame was uncertain — strict mode drops it entirely.
    assert strict.n_uncertain_dropped == 1
    # Total extracted frames == positive + true_negative == n_total - 1.
    assert strict.n_train_images + strict.n_val_images == n_total_frames - 1

    escape = distill.prepare_yolo_dataset(
        project_id="proj_test",
        student_dir=tmp_path / "escape",
        train_teacher_ids=["teacher_a"],
        treat_empty_as_negative=True,
    )
    # Escape-hatch keeps the uncertain frame (reclassified as negative).
    assert escape.n_uncertain_dropped == 0
    assert escape.n_train_images + escape.n_val_images == n_total_frames


def test_t_low_above_t_high_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coco = _make_synthetic_coco()
    _patch_io(monkeypatch, coco)
    with pytest.raises(ValueError, match="t_low.*t_high"):
        distill.prepare_yolo_dataset(
            project_id="proj_test",
            student_dir=tmp_path,
            train_teacher_ids=["teacher_a"],
            t_high=0.10,
            t_low=0.50,
        )


# ---- Helpers --------------------------------------------------------------


def _count_val_anns(student_dir: Path) -> int:
    """Sum line counts across val labels — needed because the random
    seed may put the positive frame in val."""
    val_dir = student_dir / "dataset" / "labels" / "val"
    if not val_dir.exists():
        return 0
    total = 0
    for txt in val_dir.rglob("*.txt"):
        content = txt.read_text().strip()
        if not content:
            continue
        total += len(content.splitlines())
    return total
