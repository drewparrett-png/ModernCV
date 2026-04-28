"""Phase 3 — distill's per-frame-state override pass.

`_apply_frame_state_overrides` mutates a teacher's COCO dict in place
according to the user's verdict per frame:
  • marked_missed → image and annotations dropped entirely.
  • confirmed_empty → image kept, annotations cleared.
  • curated → image kept, annotations whose det_idx ∈ rejected_dets dropped.

The test is pure (no disk, no project) — `_apply_frame_state_overrides`
takes the COCO + frame_states dict directly, which keeps this fast and
makes the contract obvious.
"""

from __future__ import annotations

from pipeline import distill


def _coco() -> dict:
    """Three frames, two annotations each, det_idx stamped per annotation."""
    return {
        "images": [
            {"id": 0, "width": 100, "height": 100, "file_name": "0.jpg"},
            {"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "2.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "det_idx": 0, "category_id": 0,
             "bbox": [0, 0, 10, 10], "score": 0.9},
            {"id": 2, "image_id": 0, "det_idx": 1, "category_id": 0,
             "bbox": [20, 20, 10, 10], "score": 0.8},
            {"id": 3, "image_id": 1, "det_idx": 0, "category_id": 0,
             "bbox": [0, 0, 10, 10], "score": 0.9},
            {"id": 4, "image_id": 1, "det_idx": 1, "category_id": 0,
             "bbox": [20, 20, 10, 10], "score": 0.8},
            {"id": 5, "image_id": 2, "det_idx": 0, "category_id": 0,
             "bbox": [0, 0, 10, 10], "score": 0.9},
            {"id": 6, "image_id": 2, "det_idx": 1, "category_id": 0,
             "bbox": [20, 20, 10, 10], "score": 0.8},
        ],
        "categories": [{"id": 0, "name": "x"}],
    }


def test_no_states_is_noop() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(coco, {})
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 6
    assert counts == {
        "n_frames_curated": 0,
        "n_frames_confirmed_empty": 0,
        "n_frames_marked_missed": 0,
    }


def test_marked_missed_drops_image_and_anns() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(
        coco, {1: {"state": "marked_missed", "rejected_dets": []}}
    )
    image_ids = [img["id"] for img in coco["images"]]
    assert image_ids == [0, 2]
    # Annotations on frame 1 are all gone.
    assert all(ann["image_id"] != 1 for ann in coco["annotations"])
    assert len(coco["annotations"]) == 4
    assert counts["n_frames_marked_missed"] == 1


def test_confirmed_empty_keeps_image_drops_anns() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(
        coco, {1: {"state": "confirmed_empty", "rejected_dets": []}}
    )
    # Image stays.
    assert [img["id"] for img in coco["images"]] == [0, 1, 2]
    # Annotations on frame 1 dropped.
    assert all(ann["image_id"] != 1 for ann in coco["annotations"])
    assert len(coco["annotations"]) == 4
    assert counts["n_frames_confirmed_empty"] == 1


def test_curated_drops_only_rejected_dets() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(
        coco, {1: {"state": "curated", "rejected_dets": [1]}}
    )
    # Image stays.
    assert [img["id"] for img in coco["images"]] == [0, 1, 2]
    # Frame 1 keeps only det_idx=0.
    frame1_anns = [a for a in coco["annotations"] if a["image_id"] == 1]
    assert len(frame1_anns) == 1
    assert frame1_anns[0]["det_idx"] == 0
    # Other frames untouched.
    assert sum(1 for a in coco["annotations"] if a["image_id"] == 0) == 2
    assert sum(1 for a in coco["annotations"] if a["image_id"] == 2) == 2
    assert counts["n_frames_curated"] == 1


def test_curated_with_no_rejections_keeps_everything() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(
        coco, {1: {"state": "curated", "rejected_dets": []}}
    )
    assert len(coco["annotations"]) == 6
    assert counts["n_frames_curated"] == 1


def test_mixed_states_compose() -> None:
    coco = _coco()
    counts = distill._apply_frame_state_overrides(
        coco,
        {
            0: {"state": "curated", "rejected_dets": [0]},
            1: {"state": "confirmed_empty", "rejected_dets": []},
            2: {"state": "marked_missed", "rejected_dets": []},
        },
    )
    # Frame 2 dropped entirely; frames 0 and 1 remain.
    assert [img["id"] for img in coco["images"]] == [0, 1]
    # Frame 0 keeps only det_idx=1; frame 1 has no anns.
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["image_id"] == 0
    assert ann["det_idx"] == 1
    assert counts == {
        "n_frames_curated": 1,
        "n_frames_confirmed_empty": 1,
        "n_frames_marked_missed": 1,
    }
