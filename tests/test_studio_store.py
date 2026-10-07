"""Studio dataset store: classes, images, annotations, suggestions, stats."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.studio import prompting
from pipeline.studio.store import Invalid, NotFound, StudioStore, auto_split


@pytest.fixture()
def store(tmp_path: Path) -> StudioStore:
    return StudioStore(tmp_path / "studio")


def _png(w: int = 64, h: int = 48) -> bytes:
    img = np.full((h, w, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_classes_ids_are_stable_and_names_unique(store: StudioStore) -> None:
    out = store.set_classes([{"name": "carton"}, {"name": "pallet"}])
    assert [c["id"] for c in out] == [0, 1]
    out = store.set_classes([{"id": 1, "name": "pallet deck"}, {"name": "tote"}])
    assert {c["id"]: c["name"] for c in out} == {1: "pallet deck", 2: "tote"}
    with pytest.raises(Invalid):
        store.set_classes([{"name": "a"}, {"name": "A"}])


def test_removing_a_class_drops_its_annotations(store: StudioStore) -> None:
    store.set_classes([{"name": "carton"}, {"name": "pallet"}])
    rec = store.add_image_bytes(_png(), "a.png")
    store.set_annotations(rec["id"], [
        {"class_id": 0, "bbox": [1, 1, 10, 10]},
        {"class_id": 1, "bbox": [5, 5, 20, 20]},
    ])
    store.set_classes([{"id": 0, "name": "carton"}])
    anns = store.annotations(rec["id"])
    assert [a["class_id"] for a in anns] == [0]
    assert store.get_image(rec["id"])["n_annotations"] == 1


def test_add_image_rejects_garbage(store: StudioStore) -> None:
    with pytest.raises(Invalid):
        store.add_image_bytes(b"not an image", "x.jpg")


def test_non_web_formats_are_reencoded(store: StudioStore) -> None:
    img = np.zeros((10, 12, 3), np.uint8)
    ok, buf = cv2.imencode(".bmp", img)
    rec = store.add_image_bytes(buf.tobytes(), "scan.bmp")
    assert rec["file"].endswith(".png")
    assert (rec["width"], rec["height"]) == (12, 10)


def test_polygon_is_authoritative_and_clamped(store: StudioStore) -> None:
    store.set_classes([{"name": "carton"}])
    rec = store.add_image_bytes(_png(64, 48), "a.png")
    [a] = store.set_annotations(rec["id"], [
        {"class_id": 0, "bbox": [0, 0, 1, 1], "polygon": [[10, 10], [80, 10], [80, 30], [10, 30]]},
    ])
    assert a["bbox"] == [10.0, 10.0, 64.0, 30.0]
    assert a["polygon"][1] == [64.0, 10.0]
    with pytest.raises(Invalid):
        store.set_annotations(rec["id"], [{"class_id": 9, "bbox": [1, 1, 5, 5]}])


def test_delete_image_removes_sidecars(store: StudioStore) -> None:
    store.set_classes([{"name": "carton"}])
    rec = store.add_image_bytes(_png(), "a.png")
    store.set_annotations(rec["id"], [{"class_id": 0, "bbox": [1, 1, 9, 9]}])
    store.thumbnail(rec["id"])
    store.delete_image(rec["id"])
    with pytest.raises(NotFound):
        store.get_image(rec["id"])
    assert not list((store.root / "annotations").glob("*.json"))
    assert not list((store.root / "cache" / "thumbs").glob("*"))


def test_depth_roundtrip_and_resize(store: StudioStore) -> None:
    rec = store.add_image_bytes(_png(64, 48), "a.png")
    depth = np.full((24, 32), 1500, np.uint16)  # half resolution, 1.5 m
    store.set_depth(rec["id"], depth, {"fx": 50, "fy": 50, "cx": 32, "cy": 24})
    d = store.read_depth_m(rec["id"])
    assert d.shape == (48, 64)
    assert np.allclose(d, 1.5)
    assert store.get_image(rec["id"])["has_depth"]


def test_depth_rejects_colour_and_8bit(store: StudioStore) -> None:
    rec = store.add_image_bytes(_png(), "a.png")
    with pytest.raises(Invalid):
        store.set_depth_bytes(rec["id"], _png(), "depth.png")  # 3-channel
    ok, buf = cv2.imencode(".png", np.zeros((48, 64), np.uint8))
    with pytest.raises(Invalid):
        store.set_depth_bytes(rec["id"], buf.tobytes(), "depth.png")


def test_accept_suggestions_creates_missing_class(store: StudioStore) -> None:
    store.set_classes([{"name": "carton"}])
    rec = store.add_image_bytes(_png(), "a.png")
    store.set_suggestions(rec["id"], [
        {"class_id": -1, "class_name": "Pallet", "bbox": [1, 1, 30, 20], "score": 0.9},
        {"class_id": 0, "class_name": "carton", "bbox": [5, 5, 15, 15], "score": 0.2},
        {"class_id": 0, "class_name": "carton", "bbox": [0, 0, 0.2, 0.2], "score": 0.9},  # degenerate → dropped
    ], "test")
    assert store.get_image(rec["id"])["n_suggestions"] == 2
    out = prompting.accept_suggestions(store, rec["id"], min_score=0.5)
    assert out["accepted"] == 1
    names = {c["name"] for c in store.classes()}
    assert "Pallet" in names
    assert len(out["suggestions"]["items"]) == 1  # the low-score one stays for review


def test_auto_split_is_deterministic_and_stats_count_labelled_only(store: StudioStore) -> None:
    assert auto_split("img_abc") == auto_split("img_abc")
    store.set_classes([{"name": "carton"}])
    a = store.add_image_bytes(_png(), "a.png")
    store.add_image_bytes(_png(), "b.png")
    store.set_annotations(a["id"], [{"class_id": 0, "bbox": [1, 1, 9, 9]}])
    st = store.stats()
    assert st["n_images"] == 2 and st["n_labeled"] == 1 and st["n_unlabeled"] == 1
    assert st["n_box_only"] == 1
    assert sum(st["splits"].values()) == 1


def test_recover_jobs_requeues_and_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from pipeline.studio import training

    queued = []
    monkeypatch.setattr(training, "ensure_worker_started", lambda: None)
    monkeypatch.setattr(training._QUEUE, "put", lambda item: queued.append(item))
    base = tmp_path / "projects" / "p1" / "studio" / "models"
    for mid, status, pid in [("m_run", "running", 999999), ("m_q", "queued", None), ("m_done", "completed", None)]:
        (base / mid).mkdir(parents=True)
        (base / mid / "model.json").write_text(json.dumps({"id": mid, "status": status, "pid": pid}))
    counts = training.recover_jobs(tmp_path)
    assert counts == {"adopted": 0, "failed": 1, "requeued": 1}
    assert json.loads((base / "m_run" / "model.json").read_text())["status"] == "failed"
    assert [q.name for q in queued] == ["m_q"]
