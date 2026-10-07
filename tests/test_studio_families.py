"""YOLO26 / YOLO11 family support across naming, training requests and specs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.studio import engines, training
from pipeline.studio.store import Invalid, StudioStore


def test_yolo_names_per_family() -> None:
    assert engines.yolo_name("26", "detect", "n") == "yolo26n.pt"
    assert engines.yolo_name("11", "segment", "s") == "yolo11s-seg.pt"
    assert engines.yolo_name("11", "obb", "x") == "yolo11x-obb.pt"
    assert engines.yolo26_name("pose", "m") == "yolo26m-pose.pt"
    with pytest.raises(ValueError):
        engines.yolo_name("8", "detect", "n")


def test_spec_family_accepts_legacy_kind() -> None:
    assert engines.spec_family({"kind": "yolo26", "task": "detect"}) == "26"
    assert engines.spec_family({"kind": "yolo", "family": "11"}) == "11"
    assert engines.spec_family({"kind": "yolo"}) == "26"
    with pytest.raises(ValueError):
        engines.spec_family({"kind": "yolo", "family": "5"})


def test_catalog_lists_both_families() -> None:
    cat = engines.catalog()
    assert set(cat["yolo"]) == {"26", "11"}
    assert cat["yolo"]["11"]["segment"][0]["weights"] == "yolo11n-seg.pt"


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StudioStore:
    # Queue jobs without starting a real trainer.
    monkeypatch.setattr(training, "ensure_worker_started", lambda: None)
    monkeypatch.setattr(training._QUEUE, "put", lambda item: None)
    st = StudioStore(tmp_path / "studio")
    st.set_classes([{"name": "carton"}])
    for _ in range(2):
        rec = st.add_image_bytes(cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes(), "a.png")
        st.set_annotations(rec["id"], [{"class_id": 0, "bbox": [4, 4, 40, 40]}])
    return st


def test_training_records_family(store: StudioStore) -> None:
    m = training.start_training(store, {"task": "segment", "size": "n", "family": "11", "config": {"epochs": 1}})
    assert m["family"] == "11"
    assert m["base_weights"] == "yolo11n-seg.pt"
    assert m["name"].startswith("yolo11n-segment")
    assert "_y11n-" in m["id"]
    with pytest.raises(Invalid):
        training.start_training(store, {"task": "segment", "size": "n", "family": "7"})


def test_finetune_inherits_parent_family_and_size(store: StudioStore) -> None:
    parent = training.start_training(store, {"task": "detect", "size": "s", "family": "11", "config": {"epochs": 1}})
    pdir = store.models_dir / parent["id"]
    (pdir / "best.pt").write_bytes(b"weights")
    child = training.start_training(store, {"task": "detect", "size": "n", "family": "26", "base": f"model:{parent['id']}"})
    assert (child["family"], child["size"]) == ("11", "s")


@pytest.mark.skipif(not engines.is_cached("yolo11n.pt"), reason="yolo11n.pt not cached")
def test_yolo11_predict_has_no_end2end_head(tmp_path: Path) -> None:
    from pipeline.studio.infer import run_predict

    st = StudioStore(tmp_path / "studio")
    img = np.zeros((96, 128, 3), np.uint8)
    out = run_predict(st, {"kind": "yolo", "family": "11", "task": "detect", "size": "n"}, img, {"end2end": False})
    assert out["model_label"] == "yolo11n" and out["end2end"] is None
