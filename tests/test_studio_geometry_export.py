"""Prompt rasterisation, paint-to-labels, and YOLO dataset export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.studio import prompting
from pipeline.studio.export import export_yolo_dataset, label_line
from pipeline.studio.geometry import sample_stroke_points, stroke_mask
from pipeline.studio.store import StudioStore


@pytest.fixture()
def store(tmp_path: Path) -> StudioStore:
    st = StudioStore(tmp_path / "studio")
    st.set_classes([{"name": "carton"}, {"name": "pallet"}])
    return st


def _img(st: StudioStore, w: int = 200, h: int = 100) -> dict:
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    return st.add_image_bytes(buf.tobytes(), "x.png")


def test_stroke_mask_has_brush_width() -> None:
    m = stroke_mask([[10, 50], [90, 50]], radius=5, shape=(100, 100))
    col = m[:, 50]
    assert 9 <= col.sum() <= 12
    assert m[50, 5] and not m[50, 97]  # round caps extend past the end points


def test_sample_stroke_points_spreads_along_length() -> None:
    pts = sample_stroke_points([[0, 0], [100, 0], [100, 100]], max_points=5)
    assert len(pts) == 5
    assert pts[0] == [0.0, 0.0] and pts[-1] == [100.0, 100.0]
    assert sample_stroke_points([[3, 4]]) == [[3.0, 4.0]]


def test_paint_splits_blobs_and_respects_erase(store: StudioStore) -> None:
    rec = _img(store)
    prompts = [
        {"type": "stroke", "points": [[20, 50], [40, 50]], "radius": 8, "class_id": 0, "polarity": 1},
        {"type": "stroke", "points": [[150, 50], [170, 50]], "radius": 8, "class_id": 0, "polarity": 1},
        {"type": "stroke", "points": [[160, 30], [160, 70]], "radius": 3, "class_id": 0, "polarity": -1},
        {"type": "box", "bbox": [60, 10, 90, 40], "class_id": 1, "polarity": 1},
    ]
    anns = prompting.paint_to_annotations(store, rec["id"], prompts)
    masks = [a for a in anns if a["polygon"]]
    boxes = [a for a in anns if not a["polygon"]]
    assert len(boxes) == 1 and boxes[0]["class_id"] == 1
    # Left blob stays whole; the erase stroke cuts the right blob in two.
    assert len(masks) == 3


def test_label_lines_per_task() -> None:
    ann = {"bbox": [10, 20, 30, 60], "polygon": None}
    assert label_line("detect", 2, ann, 100, 100)[0] == "2 0.200000 0.400000 0.200000 0.400000"
    line, converted = label_line("segment", 0, ann, 100, 100)
    assert converted and len(line.split()) == 9
    rot = {"bbox": [0, 0, 0, 0], "polygon": [[50, 10], [90, 50], [50, 90], [10, 50]]}
    line, converted = label_line("obb", 0, rot, 100, 100)
    vals = [float(v) for v in line.split()[1:]]
    assert not converted and len(vals) == 8
    assert min(vals) == pytest.approx(0.1, abs=1e-3) and max(vals) == pytest.approx(0.9, abs=1e-3)


def test_export_keeps_class_order_and_skips_unlabelled(store: StudioStore, tmp_path: Path) -> None:
    a, b, c = _img(store), _img(store), _img(store)
    store.set_annotations(a["id"], [{"class_id": 1, "bbox": [1, 1, 50, 50]}])
    store.set_annotations(b["id"], [{"class_id": 0, "polygon": [[1, 1], [40, 1], [40, 30]]}])
    store.update_image(c["id"], negative=True)
    store.add_image_bytes(cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes(), "unlabelled.png")
    s = export_yolo_dataset(store, tmp_path / "ds", "segment", class_ids=[1, 0])
    assert s.names == ["pallet", "carton"]
    assert sum(s.counts.values()) == 3  # a, b and the negative; not the unlabelled one
    assert s.counts["train"] >= 1 and s.counts["val"] >= 1
    labels = {p.stem: p.read_text() for p in (tmp_path / "ds" / "labels").rglob("*.txt")}
    assert labels[Path(a["file"]).stem].startswith("0 ")  # pallet → 0 in this order
    assert labels[Path(c["file"]).stem] == ""
    yaml = (tmp_path / "ds" / "data.yaml").read_text()
    assert '0: "pallet"' in yaml and '1: "carton"' in yaml


def test_export_single_image_warns_about_val(store: StudioStore, tmp_path: Path) -> None:
    a = _img(store)
    store.set_annotations(a["id"], [{"class_id": 0, "bbox": [1, 1, 50, 50]}])
    s = export_yolo_dataset(store, tmp_path / "ds", "detect")
    assert s.counts["train"] == 1
    assert any("validating on the training set" in w for w in s.warnings)
    assert "val: images/train" in (tmp_path / "ds" / "data.yaml").read_text()
