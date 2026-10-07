"""Regressions for the Studio code-review findings (2026-10-07)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline.studio import track, training
from pipeline.studio.pallet import Instance, PalletParams, analyze, colorize, fit_plane_ransac
from pipeline.studio.store import Invalid, StudioStore, check_id
from pipeline.studio.synth import render_pallet_scene
from server.main import app


def test_colorize_reversed_range_keeps_spread() -> None:
    """Near = warm uses lo > hi; it used to collapse to two colours."""
    depth = np.linspace(1.0, 3.0, 200 * 50, dtype=np.float32).reshape(50, 200)
    img = colorize(depth, 3.0, 1.0)
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 50
    assert not np.array_equal(img[0, 0], img[0, -1])


def test_ransac_inliers_cover_all_points_when_subsampled() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, size=(70_000, 2))
    pts = np.column_stack([xy, np.full(len(xy), 2.0)])
    n, d, inl = fit_plane_ransac(pts, thresh=0.01, max_points=60_000)
    assert inl.shape == (70_000,) and inl.all()


def test_boxes_mode_with_large_masks() -> None:
    """A carton mask with >60k depth pixels crashed 'Lowest carton' mode."""
    h, w = 600, 800
    depth = np.full((h, w), 2.5, np.float32)  # floor, camera looking straight down
    big = np.zeros((h, w), bool)
    big[100:420, 100:420] = True  # 102k px carton top, 40 cm tall
    small = np.zeros((h, w), bool)
    small[150:300, 500:700] = True  # 30 cm tall
    depth[big] = 2.1
    depth[small] = 2.2
    k = {"fx": 600.0, "fy": 600.0, "cx": w / 2, "cy": h / 2}
    insts = [Instance(id="big", class_id=0, mask=big), Instance(id="small", class_id=0, mask=small)]
    res = analyze(depth, k, insts, PalletParams(plane_mode="boxes"))
    hs = {b["id"]: b["height_m"] for b in res["boxes"]}
    assert res["plane"]["mode"] == "boxes"
    assert hs["small"] == pytest.approx(0.0, abs=0.005)  # lowest top is the zero
    assert hs["big"] == pytest.approx(0.1, abs=0.005)


def test_rotated_pallet_rows_follow_the_grid() -> None:
    """Pick order walks rows of the pallet grid even when it's rotated."""
    sc = render_pallet_scene(seed=12)  # oblique view with a rotated pallet
    insts = [Instance(id=str(i), class_id=0, mask=g["mask"]) for i, g in enumerate(sc.instances)]
    res = analyze(sc.depth_mm.astype(np.float32) / 1000, sc.intrinsics, insts, PalletParams())
    orders = sorted(b["pick_order"] for b in res["boxes"] if b.get("pick_order"))
    assert orders == list(range(1, len(orders) + 1))


def test_check_id_rejects_path_escapes() -> None:
    assert check_id("m_20261007-170409_n-segment") == "m_20261007-170409_n-segment"
    assert check_id("best.onnx") == "best.onnx"
    for bad in ("../other", "/Users/x/evil.pt", "a/b", "..", ".", "", ".hidden"):
        with pytest.raises(Invalid):
            check_id(bad)


def test_export_zip_name_keeps_dotted_directories(tmp_path: Path) -> None:
    st = StudioStore(tmp_path / "studio")
    mdir = st.models_dir / "m_x"
    (mdir / "exports" / "best.mlpackage" / "Data").mkdir(parents=True)
    (mdir / "exports" / "best.mlpackage" / "Data" / "weights.bin").write_bytes(b"x" * 10)
    (mdir / "model.json").write_text(json.dumps({"id": "m_x", "status": "completed"}))
    p = training.export_download_path(st, "m_x", "best.mlpackage")
    assert p.exists() and p.name == "best.mlpackage.zip"
    assert any(n.endswith("weights.bin") for n in zipfile.ZipFile(p).namelist())


def test_npy_depth_is_metres_regardless_of_scale(tmp_path: Path) -> None:
    st = StudioStore(tmp_path / "studio")
    rec = st.add_image_bytes(cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))[1].tobytes(), "a.png")
    buf = io.BytesIO()
    np.save(buf, np.full((20, 30), 2.5, np.float32))
    st.set_depth_bytes(rec["id"], buf.getvalue(), "depth.npy", scale_to_mm=1.0)
    assert np.allclose(st.read_depth_m(rec["id"]), 2.5)


def test_delete_image_removes_depth_cache(tmp_path: Path) -> None:
    st = StudioStore(tmp_path / "studio")
    rec = st.add_image_bytes(cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))[1].tobytes(), "a.png")
    cache = st.cache_dir / "depth" / f"{rec['id']}_da2-metric-indoor-b.npy"
    cache.parent.mkdir(parents=True)
    np.save(cache, np.zeros((2, 2)))
    st.delete_image(rec["id"])
    assert not cache.exists()


def test_track_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list = []
    monkeypatch.setattr(track, "ensure_worker_started", lambda: None)
    monkeypatch.setattr(track._QUEUE, "put", lambda item: queued.append(item))
    base = tmp_path / "projects" / "p" / "studio" / "tracks"
    for tid, status in [("trk_run", "running"), ("trk_q", "queued"), ("trk_c", "cancelled"), ("trk_ok", "completed")]:
        (base / tid).mkdir(parents=True)
        (base / tid / "track.json").write_text(json.dumps({"id": tid, "status": status}))
    assert track.recover_jobs(tmp_path) == {"failed": 1, "requeued": 1}
    assert json.loads((base / "trk_run" / "track.json").read_text())["status"] == "failed"
    assert [t[1].name for t in queued] == ["trk_q"]
    assert not (base / "trk_c").exists() and (base / "trk_ok").exists()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


@pytest.fixture()
def pid(client: TestClient) -> str:
    return client.post("/projects", json={"name": "x", "task": "segmentation", "prompts": ["carton"]}).json()["id"]


def test_unknown_model_is_404_not_500(client: TestClient, pid: str) -> None:
    for path, body in [("val", {"split": "val"}), ("export", {"format": "onnx"}), ("benchmark", {"n_images": 2})]:
        res = client.post(f"/projects/{pid}/studio/models/m_gone/{path}", json=body)
        assert res.status_code == 404, (path, res.status_code, res.text)


def test_predict_rejects_artifact_paths(client: TestClient, pid: str, tmp_path: Path) -> None:
    mdir = tmp_path / "runs" / "projects" / pid / "studio" / "models" / "m_x"
    mdir.mkdir(parents=True)
    (mdir / "model.json").write_text(json.dumps({"id": "m_x", "status": "completed"}))
    iid = client.post(
        f"/projects/{pid}/studio/images",
        files=[("files", ("a.jpg", cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))[1].tobytes(), "image/jpeg"))],
    ).json()["images"][0]["id"]
    for spec in (
        {"kind": "trained", "model_id": "../../etc", "artifact": "best.pt"},
        {"kind": "trained", "model_id": "m_x", "artifact": "/tmp/evil.pt"},
    ):
        res = client.post(f"/projects/{pid}/studio/predict", json={"model": spec, "image_id": iid, "params": {}})
        assert res.status_code == 422, res.text


def test_dataset_zip_yaml_has_no_path_key(client: TestClient, pid: str) -> None:
    iid = client.post(
        f"/projects/{pid}/studio/images",
        files=[("files", ("a.jpg", cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes(), "image/jpeg"))],
    ).json()["images"][0]["id"]
    cid = client.put(f"/projects/{pid}/studio/classes", json={"classes": [{"name": "carton"}]}).json()["classes"][0]["id"]
    client.put(f"/projects/{pid}/studio/images/{iid}/annotations", json={"annotations": [{"class_id": cid, "bbox": [2, 2, 20, 20]}]})
    res = client.get(f"/projects/{pid}/studio/dataset.zip", params={"task": "detect"})
    yaml = zipfile.ZipFile(io.BytesIO(res.content)).read("data.yaml").decode()
    assert "path:" not in yaml and "train: images/train" in yaml
