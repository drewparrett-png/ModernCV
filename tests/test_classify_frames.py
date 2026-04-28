"""Unit tests for `pipeline.distill.classify_frames`.

The bucketing helper is the heart of Phase 0 — every other piece of the
new pipeline (the GUI preview endpoint, the trainer, the regression
test) reads off these buckets. The bugs that hurt are the fence-post
ones — frames whose max score is *exactly* `t_high` or `t_low` should
land in a defined bucket, not get silently dropped.
"""

from __future__ import annotations

import pytest

from pipeline.distill import FrameBuckets, classify_frames


def _coco(images: list[dict], annotations: list[dict]) -> dict:
    """Helper: build a minimal COCO dict — only the keys classify_frames reads."""
    return {"images": images, "annotations": annotations}


def test_empty_coco_returns_empty_buckets() -> None:
    out = classify_frames(_coco([], []), t_high=0.35, t_low=0.15)
    assert out == FrameBuckets(positive=[], uncertain=[], true_negative=[])


def test_frame_with_no_annotations_is_true_negative() -> None:
    out = classify_frames(
        _coco(
            images=[{"id": 0, "width": 100, "height": 100}],
            annotations=[],
        ),
        t_high=0.35,
        t_low=0.15,
    )
    assert out.true_negative == [0]
    assert out.positive == []
    assert out.uncertain == []


def test_bucketing_by_max_score_only() -> None:
    # Frame 0: max=0.9 → positive (even though it has a 0.05 ann too)
    # Frame 1: max=0.20 → uncertain (in [0.15, 0.35))
    # Frame 2: max=0.05 → true_negative (all anns below t_low=0.15)
    # Frame 3: no anns → true_negative
    images = [{"id": i, "width": 100, "height": 100} for i in range(4)]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.05},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.20},
        {"image_id": 2, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.05},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.positive == [0]
    assert out.uncertain == [1]
    assert sorted(out.true_negative) == [2, 3]


def test_score_exactly_at_t_high_is_positive() -> None:
    # `>= t_high` is the positive boundary — exactly t_high goes positive,
    # not uncertain.
    images = [{"id": 0, "width": 100, "height": 100}]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.35},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.positive == [0]
    assert out.uncertain == []
    assert out.true_negative == []


def test_score_exactly_at_t_low_is_uncertain() -> None:
    # `>= t_low` enters the uncertain band; the positive boundary is
    # strict so a single-ann frame at score=t_low is uncertain, not
    # true_negative.
    images = [{"id": 0, "width": 100, "height": 100}]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.15},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.uncertain == [0]
    assert out.positive == []
    assert out.true_negative == []


def test_score_just_below_t_low_is_true_negative() -> None:
    images = [{"id": 0, "width": 100, "height": 100}]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.149},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.true_negative == [0]


def test_missing_score_treated_as_one() -> None:
    # Forward-compatibility: future user-added labels will not carry a
    # teacher confidence score. `classify_frames` should treat them as
    # 1.0 so the frame goes positive (matches `_coco_to_yolo_lines`'s
    # `ann.get("score", 1.0)` convention).
    images = [{"id": 0, "width": 100, "height": 100}]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10]},  # no score key
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.positive == [0]


def test_t_low_above_t_high_is_rejected() -> None:
    with pytest.raises(ValueError, match="t_low.*t_high"):
        classify_frames(_coco([], []), t_high=0.15, t_low=0.35)


def test_t_low_equal_t_high_is_allowed() -> None:
    # Edge case: t_low == t_high collapses the uncertain band — every
    # frame is either positive or true_negative. We allow this rather
    # than raising; it's a sensible "no uncertainty band" mode.
    images = [
        {"id": 0, "width": 100, "height": 100},
        {"id": 1, "width": 100, "height": 100},
    ]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.30},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.20},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.30, t_low=0.30)
    assert out.positive == [0]
    assert out.uncertain == []
    assert out.true_negative == [1]


def test_positive_dominates_when_mixed() -> None:
    # A single high-score box on a frame is enough to make it positive,
    # even if every other ann is in the uncertain band. The trainer
    # will apply min_score=t_high to drop the low-score boxes from the
    # YOLO label.
    images = [{"id": 0, "width": 100, "height": 100}]
    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.20},
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.18},
    ]
    out = classify_frames(_coco(images, annotations), t_high=0.35, t_low=0.15)
    assert out.positive == [0]
