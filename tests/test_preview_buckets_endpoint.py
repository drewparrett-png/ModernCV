"""Tests for `POST /projects/{pid}/students/preview-buckets` (Phase 0.4,
project-scoped under the Phase 1 rearchitecture).

Same shape as before: the endpoint is the live preview that powers the New
Student form. We lay out a project + a couple of fake teacher dirs in a
tmp tree and hit the endpoint via TestClient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app


# ---- Fixtures -------------------------------------------------------------


def _write_teacher(
    project_id: str,
    teacher_id: str,
    *,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> None:
    """Lay out a fake completed Teacher dir inside a project's teachers/ dir."""
    tdir = runs_mod.run_dir(project_id, teacher_id)
    (tdir / runs_mod.LABELS_DIR).mkdir(parents=True, exist_ok=True)
    manifest = runs_mod.RunManifest(
        id=teacher_id,
        task="detection",
        prompt="test",
        video_path="/fake/video.mp4",
        started_at="2026-04-28T12:00:00Z",
        ended_at="2026-04-28T12:01:00Z",
        status="completed",
        models={"detect": "groundingdino"},
    )
    (tdir / runs_mod.MANIFEST_NAME).write_text(manifest.to_json())
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [{"id": 0, "name": "ball"}, {"id": 1, "name": "player"}],
    }
    (tdir / runs_mod.LABELS_DIR / runs_mod.COCO_NAME).write_text(json.dumps(coco))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


@pytest.fixture()
def project(client: TestClient) -> str:
    """Create a project and return its id. Depends on `client` to ensure
    the chdir + runs/ setup has happened first."""
    project = runs_mod.create_project(
        name="Test project",
        task="detection",
        prompts=["ball", "player"],
    )
    return project.id


def _coco_classifiable(
    *, n_positive: int, n_uncertain: int, n_true_negative: int
) -> tuple[list[dict], list[dict]]:
    images: list[dict] = []
    anns: list[dict] = []
    nxt_frame = 0
    nxt_ann = 0
    for _ in range(n_positive):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        anns.append({"id": nxt_ann, "image_id": nxt_frame, "category_id": 0,
                     "bbox": [0, 0, 10, 10], "score": 0.9})
        nxt_frame += 1
        nxt_ann += 1
    for _ in range(n_uncertain):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        anns.append({"id": nxt_ann, "image_id": nxt_frame, "category_id": 1,
                     "bbox": [0, 0, 10, 10], "score": 0.20})
        nxt_frame += 1
        nxt_ann += 1
    for _ in range(n_true_negative):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        nxt_frame += 1
    return images, anns


# ---- Tests ----------------------------------------------------------------


def test_single_teacher_strict_mode(client: TestClient, project: str) -> None:
    images, anns = _coco_classifiable(n_positive=3, n_uncertain=2, n_true_negative=1)
    _write_teacher(project, "teacher_a", images=images, annotations=anns)

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"] == {
        "positive": 3,
        "uncertain": 2,
        "true_negative": 1,
        "n_classes": 2,
        "class_names": ["ball", "player"],
    }
    assert body["per_teacher"] == [
        {"teacher_id": "teacher_a", "positive": 3, "uncertain": 2, "true_negative": 1}
    ]


def test_aggregation_across_teachers(client: TestClient, project: str) -> None:
    a_imgs, a_anns = _coco_classifiable(n_positive=2, n_uncertain=1, n_true_negative=0)
    b_imgs, b_anns = _coco_classifiable(n_positive=1, n_uncertain=2, n_true_negative=3)
    _write_teacher(
        project, "teacher_a",
        images=a_imgs, annotations=a_anns,
        categories=[{"id": 0, "name": "ball"}, {"id": 1, "name": "player"}],
    )
    _write_teacher(
        project, "teacher_b",
        images=b_imgs, annotations=b_anns,
        categories=[{"id": 0, "name": "ball"}, {"id": 1, "name": "ref"}],
    )

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a", "teacher_b"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["positive"] == 3
    assert body["aggregate"]["uncertain"] == 3
    assert body["aggregate"]["true_negative"] == 3
    assert body["aggregate"]["class_names"] == ["ball", "player", "ref"]
    assert body["aggregate"]["n_classes"] == 3
    assert [t["teacher_id"] for t in body["per_teacher"]] == [
        "teacher_a", "teacher_b",
    ]


def test_treat_empty_as_negative_reclassifies_in_response(
    client: TestClient, project: str,
) -> None:
    images, anns = _coco_classifiable(n_positive=2, n_uncertain=4, n_true_negative=1)
    _write_teacher(project, "teacher_a", images=images, annotations=anns)

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"], "treat_empty_as_negative": True},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["positive"] == 2
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 5
    assert body["per_teacher"][0] == {
        "teacher_id": "teacher_a",
        "positive": 2,
        "uncertain": 0,
        "true_negative": 5,
    }


def test_custom_thresholds_change_buckets(client: TestClient, project: str) -> None:
    images, anns = _coco_classifiable(n_positive=1, n_uncertain=3, n_true_negative=0)
    _write_teacher(project, "teacher_a", images=images, annotations=anns)

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"], "export_threshold": 0.10, "t_low": 0.05},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["positive"] == 4
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 0


def test_t_low_above_export_threshold_rejected(client: TestClient, project: str) -> None:
    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": [], "export_threshold": 0.10, "t_low": 0.50},
    )
    assert res.status_code == 422
    detail = json.dumps(res.json())
    assert "t_low" in detail and "export_threshold" in detail


def test_missing_teacher_returns_404(client: TestClient, project: str) -> None:
    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["does-not-exist"]},
    )
    assert res.status_code == 404
    assert "does-not-exist" in res.text


def test_missing_project_returns_404(client: TestClient) -> None:
    res = client.post(
        "/projects/proj_nope/students/preview-buckets",
        json={"teacher_ids": []},
    )
    assert res.status_code == 404


def test_teacher_without_coco_returns_400(
    client: TestClient, project: str,
) -> None:
    tdir = runs_mod.run_dir(project, "teacher_no_coco")
    (tdir / runs_mod.LABELS_DIR).mkdir(parents=True)
    manifest = runs_mod.RunManifest(
        id="teacher_no_coco",
        task="detection",
        prompt="test",
        video_path="/fake/video.mp4",
        started_at="2026-04-28T12:00:00Z",
        status="running",
    )
    (tdir / runs_mod.MANIFEST_NAME).write_text(manifest.to_json())

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_no_coco"]},
    )
    assert res.status_code == 400
    assert "coco.json" in res.text


def test_empty_teacher_list_returns_zero_aggregate(
    client: TestClient, project: str
) -> None:
    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": []},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["aggregate"]["positive"] == 0
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 0
    assert body["aggregate"]["n_classes"] == 0
    assert body["aggregate"]["class_names"] == []
    assert body["per_teacher"] == []


# ---- Frame-state integration tests ----------------------------------------


def _write_frame_states(project_id: str, teacher_id: str, states: dict) -> None:
    tdir = runs_mod.run_dir(project_id, teacher_id)
    runs_mod.write_frame_states(tdir, states)


def test_marked_missed_frame_not_counted(client: TestClient, project: str) -> None:
    """marked_missed frames are excluded entirely — positive count drops."""
    images, anns = _coco_classifiable(n_positive=2, n_uncertain=0, n_true_negative=0)
    _write_teacher(project, "teacher_a", images=images, annotations=anns)
    # frame 0 is positive (score 0.9); mark it missed — should disappear
    _write_frame_states(project, "teacher_a", {0: {"state": "marked_missed"}})

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"]},
    )
    assert res.status_code == 200
    agg = res.json()["aggregate"]
    assert agg["positive"] == 1
    assert agg["uncertain"] == 0
    assert agg["true_negative"] == 0


def test_confirmed_empty_frame_becomes_true_negative(
    client: TestClient, project: str
) -> None:
    """confirmed_empty clears annotations — previously-positive frame becomes true_negative."""
    images, anns = _coco_classifiable(n_positive=1, n_uncertain=0, n_true_negative=1)
    _write_teacher(project, "teacher_a", images=images, annotations=anns)
    # frame 0 is positive; confirm it is empty → should become true_negative
    _write_frame_states(project, "teacher_a", {0: {"state": "confirmed_empty"}})

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"]},
    )
    assert res.status_code == 200
    agg = res.json()["aggregate"]
    assert agg["positive"] == 0
    assert agg["uncertain"] == 0
    assert agg["true_negative"] == 2


def test_curated_rejected_detection_shifts_bucket(
    client: TestClient, project: str
) -> None:
    """curated + rejected detection: frame's only annotation removed → true_negative."""
    images = [{"id": 0, "width": 100, "height": 100}]
    anns = [
        {
            "id": 0, "image_id": 0, "category_id": 0,
            "bbox": [0, 0, 10, 10], "score": 0.9, "det_idx": 0,
        }
    ]
    _write_teacher(project, "teacher_a", images=images, annotations=anns)
    # reject the one detection — frame now has no annotations → true_negative
    _write_frame_states(
        project, "teacher_a", {0: {"state": "curated", "rejected_dets": [0]}}
    )

    res = client.post(
        f"/projects/{project}/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"]},
    )
    assert res.status_code == 200
    agg = res.json()["aggregate"]
    assert agg["positive"] == 0
    assert agg["uncertain"] == 0
    assert agg["true_negative"] == 1
