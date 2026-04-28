"""Unit tests for `pipeline.distill._coco_to_yolo_lines`'s `min_score` filter (Phase 0.1).

`min_score` is the only behaviour change to the COCO→YOLO converter in
this PR. Default 0.0 must preserve prior behaviour exactly so the
function stays a drop-in for the eval-dataset path that doesn't care
about teacher confidence.
"""

from __future__ import annotations

from pipeline.distill import _coco_to_yolo_lines


def _ann(score: float, category_id: int = 0, bbox: tuple[float, float, float, float] = (0, 0, 10, 10)) -> dict:
    return {"image_id": 0, "category_id": category_id, "bbox": list(bbox), "score": score}


def test_default_min_score_keeps_every_annotation() -> None:
    # Default 0.0 is the no-op: a score=0.01 ann still passes through.
    anns = [_ann(0.01), _ann(0.5), _ann(0.99)]
    lines = _coco_to_yolo_lines(anns, img_w=100, img_h=100, coco_to_global={0: 0})
    assert len(lines) == 3


def test_min_score_drops_below_threshold() -> None:
    anns = [_ann(0.05), _ann(0.20), _ann(0.50), _ann(0.99)]
    lines = _coco_to_yolo_lines(
        anns, img_w=100, img_h=100, coco_to_global={0: 0}, min_score=0.35
    )
    # Only the 0.50 and 0.99 anns survive.
    assert len(lines) == 2


def test_min_score_at_exact_threshold_keeps_annotation() -> None:
    # Filter is `score < min_score` → score == min_score passes through.
    # Matches the `>= t_high` convention in classify_frames so a frame
    # bucketed positive can't end up with zero label lines.
    anns = [_ann(0.35)]
    lines = _coco_to_yolo_lines(
        anns, img_w=100, img_h=100, coco_to_global={0: 0}, min_score=0.35
    )
    assert len(lines) == 1


def test_missing_score_treated_as_one() -> None:
    # Future user-added labels (no teacher confidence) must not be
    # filtered out by min_score. They get the implicit 1.0.
    anns = [{"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10]}]  # no score
    lines = _coco_to_yolo_lines(
        anns, img_w=100, img_h=100, coco_to_global={0: 0}, min_score=0.99
    )
    assert len(lines) == 1


def test_unmapped_category_still_dropped_silently() -> None:
    # Pre-existing behaviour — unmapped categories drop. Filter doesn't
    # interact with this.
    anns = [_ann(0.99, category_id=42)]
    lines = _coco_to_yolo_lines(
        anns, img_w=100, img_h=100, coco_to_global={0: 0}, min_score=0.0
    )
    assert lines == []


def test_zero_size_image_returns_empty() -> None:
    # Pre-existing guard. Re-asserted to make sure min_score didn't
    # regress it.
    anns = [_ann(0.99)]
    lines = _coco_to_yolo_lines(
        anns, img_w=0, img_h=100, coco_to_global={0: 0}, min_score=0.5
    )
    assert lines == []


def test_round_trip_normalization_preserved() -> None:
    # Sanity: the bbox math hasn't drifted. A 50x50 box at (10,20) on a
    # 100x100 image normalises to cx=0.35, cy=0.45, w=0.5, h=0.5.
    anns = [_ann(0.9, bbox=(10, 20, 50, 50))]
    lines = _coco_to_yolo_lines(
        anns, img_w=100, img_h=100, coco_to_global={0: 7}, min_score=0.0
    )
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "7"
    assert float(parts[1]) == 0.35
    assert float(parts[2]) == 0.45
    assert float(parts[3]) == 0.50
    assert float(parts[4]) == 0.50
