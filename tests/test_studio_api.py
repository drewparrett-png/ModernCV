"""Studio HTTP surface (no model weights needed for these paths)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


@pytest.fixture()
def pid(client: TestClient) -> str:
    res = client.post("/projects", json={"name": "Depal", "task": "segmentation", "prompts": ["carton"]})
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _jpeg(w: int = 80, h: int = 60) -> bytes:
    return cv2.imencode(".jpg", np.full((h, w, 3), 90, np.uint8))[1].tobytes()


def test_unknown_project_is_404(client: TestClient) -> None:
    assert client.get("/projects/nope/studio").status_code == 404


def test_upload_label_and_download_dataset(client: TestClient, pid: str) -> None:
    res = client.post(
        f"/projects/{pid}/studio/images",
        files=[("files", ("a.jpg", _jpeg(), "image/jpeg")), ("files", ("bad.jpg", b"xx", "image/jpeg"))],
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["images"]) == 1 and len(body["errors"]) == 1
    iid = body["images"][0]["id"]

    res = client.put(f"/projects/{pid}/studio/classes", json={"classes": [{"name": "carton"}]})
    cid = res.json()["classes"][0]["id"]

    res = client.post(f"/projects/{pid}/studio/prompt/paint", json={
        "image_id": iid,
        "prompts": [{"type": "stroke", "points": [[20, 30], [50, 30]], "radius": 6, "class_id": cid}],
    })
    assert res.status_code == 200, res.text
    assert res.json()["added"] == 1

    state = client.get(f"/projects/{pid}/studio").json()
    assert state["stats"]["n_labeled"] == 1

    res = client.get(f"/projects/{pid}/studio/dataset.zip", params={"task": "segment"})
    assert res.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    names = zf.namelist()
    assert "data.yaml" in names
    assert "path:" not in zf.read("data.yaml").decode()
    assert any(n.startswith("labels/") and n.endswith(".txt") for n in names)


def test_thumb_and_file(client: TestClient, pid: str) -> None:
    iid = client.post(f"/projects/{pid}/studio/images", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["images"][0]["id"]
    assert client.get(f"/projects/{pid}/studio/images/{iid}/file").status_code == 200
    thumb = client.get(f"/projects/{pid}/studio/images/{iid}/thumb", params={"size": 40})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/projects/{pid}/studio/images/missing/file").status_code == 404


def test_depth_upload_and_preview(client: TestClient, pid: str) -> None:
    iid = client.post(f"/projects/{pid}/studio/images", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["images"][0]["id"]
    depth = (np.linspace(1000, 2000, 80 * 60).reshape(60, 80)).astype(np.uint16)
    png = cv2.imencode(".png", depth)[1].tobytes()
    res = client.put(
        f"/projects/{pid}/studio/images/{iid}/depth",
        files={"file": ("d.png", png, "image/png")},
        data={"fx": "70", "fy": "70", "cx": "40", "cy": "30"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["has_depth"] and res.json()["intrinsics"]["fx"] == 70
    assert client.get(f"/projects/{pid}/studio/images/{iid}/depth.png").status_code == 200


def test_synthetic_scene_and_pallet_analysis(client: TestClient, pid: str) -> None:
    res = client.post(f"/projects/{pid}/studio/images/synthetic", json={"count": 1, "seed": 3, "width": 480, "height": 360})
    assert res.status_code == 200, res.text
    img = res.json()["images"][0]
    assert img["has_depth"] and img["n_annotations"] > 0 and img["intrinsics"]
    res = client.post(f"/projects/{pid}/studio/pallet/{img['id']}", json={"depth": {"source": "sensor"}})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["depth_source"] == "sensor"
    assert out["summary"]["n_layers"] >= 1
    assert out["ground_truth"]["mean_abs_err_m"] < 0.01
    assert out["height_vis"].startswith("data:image/jpeg;base64,")


def test_training_validation_errors(client: TestClient, pid: str) -> None:
    res = client.post(f"/projects/{pid}/studio/models", json={"task": "pose", "size": "n"})
    assert res.status_code == 422
    res = client.post(f"/projects/{pid}/studio/models", json={"task": "detect", "size": "n"})
    assert res.status_code == 422  # no classes / labels yet
    res = client.post(f"/projects/{pid}/studio/models", json={"task": "detect", "size": "n", "config": {"augment": {"bogus": 1}}})
    assert res.status_code == 422


def test_track_rejects_paths_outside_data(client: TestClient, pid: str) -> None:
    res = client.post(f"/projects/{pid}/studio/tracks", json={
        "video_path": "/etc/passwd", "model": {"kind": "yolo26", "task": "detect", "size": "n"},
    })
    assert res.status_code == 422
