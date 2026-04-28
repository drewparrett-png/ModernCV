"""Phase 0: dataset review state — project-scoped (Phase 1 rearchitecture).

Two surfaces under test:

1. The disk-side derivation rule (`derive_review_status`) — `approved_at`
   wins; otherwise non-empty rejections file → "reviewed"; otherwise →
   "unreviewed". Plus the cheap `has_any_rejections` helper.
2. The /projects/{pid}/runs/{id}/approve · /unapprove endpoints — 404 / 400
   / no-op semantics, and round-trip persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def project(runs_root: Path) -> runs_mod.Project:
    return runs_mod.create_project(
        name="Soccer ball test",
        task="detection",
        prompts=["soccer ball"],
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_completed_run(
    project_id: str, prompt: str = "test prompt"
) -> Path:
    rdir, _ = runs_mod.create_run(
        project_id=project_id,
        task="detection",
        prompt=prompt,
        video_path="data/x.mp4",
        models={"detect": "groundingdino"},
    )
    runs_mod.mark_completed(rdir)
    return rdir


# ---- has_any_rejections ----------------------------------------------------


def test_has_any_rejections_missing_file(tmp_path: Path) -> None:
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_empty_dict(tmp_path: Path) -> None:
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text("{}")
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_only_empty_lists(tmp_path: Path) -> None:
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text(json.dumps({"5": []}))
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_populated(tmp_path: Path) -> None:
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text(
        json.dumps({"5": [0, 2], "12": [1]})
    )
    assert runs_mod.has_any_rejections(tmp_path) is True


def test_has_any_rejections_malformed_json_is_false(tmp_path: Path) -> None:
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text("{not valid json")
    assert runs_mod.has_any_rejections(tmp_path) is False


# ---- derive_review_status --------------------------------------------------


def test_review_status_unreviewed_default(project: runs_mod.Project) -> None:
    rdir = _make_completed_run(project.id)
    assert runs_mod.derive_review_status(rdir) == "unreviewed"


def test_review_status_reviewed_when_rejections_present(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)
    assert runs_mod.derive_review_status(rdir) == "reviewed"


def test_review_status_approved_overrides_rejections(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)
    runs_mod.approve_run(rdir)
    assert runs_mod.derive_review_status(rdir) == "approved"


def test_review_status_unapprove_falls_back_to_reviewed(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)
    runs_mod.approve_run(rdir)
    runs_mod.unapprove_run(rdir)
    assert runs_mod.derive_review_status(rdir) == "reviewed"


def test_review_status_unapprove_with_no_rejections_is_unreviewed(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    runs_mod.approve_run(rdir)
    runs_mod.unapprove_run(rdir)
    assert runs_mod.derive_review_status(rdir) == "unreviewed"


def test_legacy_manifest_without_approved_at_field_loads(
    runs_root: Path, project: runs_mod.Project,
) -> None:
    """Manifests written before the `approved_at` field landed must still
    load — the dataclass tolerates unknown keys and defaults the missing
    field to None."""
    rdir = runs_mod.run_dir(project.id, "teacher_legacy")
    rdir.mkdir()
    legacy_manifest = {
        "id": "teacher_legacy",
        "task": "detection",
        "prompt": "old prompt",
        "video_path": "data/old.mp4",
        "started_at": "2025-01-01T00:00:00Z",
        "ended_at": "2025-01-01T00:01:00Z",
        "status": "completed",
        "models": {"detect": "groundingdino"},
        "error": None,
    }
    (rdir / runs_mod.MANIFEST_NAME).write_text(json.dumps(legacy_manifest))

    m = runs_mod.read_manifest(rdir)
    assert m.approved_at is None
    assert runs_mod.derive_review_status(rdir) == "unreviewed"


# ---- Persistence ----------------------------------------------------------


def test_approve_persists_across_reread(project: runs_mod.Project) -> None:
    rdir = _make_completed_run(project.id)
    runs_mod.approve_run(rdir)
    reread = runs_mod.read_manifest(rdir)
    assert reread.approved_at is not None
    raw = json.loads((rdir / runs_mod.MANIFEST_NAME).read_text())
    assert raw["approved_at"] == reread.approved_at


def test_approve_is_idempotent_does_not_drift_timestamp(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    first = runs_mod.approve_run(rdir)
    second = runs_mod.approve_run(rdir)
    assert first.approved_at is not None
    assert second.approved_at == first.approved_at


# ---- Endpoints -------------------------------------------------------------


def test_approve_endpoint_happy_path(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    res = client.post(f"/projects/{project.id}/runs/{rdir.name}/approve")
    assert res.status_code == 200
    body = res.json()
    assert body["manifest"]["approved_at"] is not None
    assert body["manifest"]["review_status"] == "approved"
    assert body["manifest"]["project_id"] == project.id


def test_approve_endpoint_404_for_missing_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.post(f"/projects/{project.id}/runs/teacher_does_not_exist/approve")
    assert res.status_code == 404


def test_approve_endpoint_404_for_missing_project(client: TestClient) -> None:
    res = client.post("/projects/proj_no_such/runs/teacher_x/approve")
    assert res.status_code == 404


def test_approve_endpoint_400_for_running_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="still running",
        video_path="data/x.mp4",
        models={},
    )
    res = client.post(f"/projects/{project.id}/runs/{rdir.name}/approve")
    assert res.status_code == 400
    assert "completed" in res.json()["detail"]


def test_approve_endpoint_400_for_failed_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="will fail",
        video_path="data/x.mp4",
        models={},
    )
    runs_mod.mark_failed(rdir, error="boom")
    res = client.post(f"/projects/{project.id}/runs/{rdir.name}/approve")
    assert res.status_code == 400


def test_approve_endpoint_idempotent_on_already_approved(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    first = client.post(f"/projects/{project.id}/runs/{rdir.name}/approve").json()
    second = client.post(f"/projects/{project.id}/runs/{rdir.name}/approve").json()
    assert first["manifest"]["approved_at"] == second["manifest"]["approved_at"]


def test_unapprove_endpoint_clears_field(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    client.post(f"/projects/{project.id}/runs/{rdir.name}/approve")
    res = client.post(f"/projects/{project.id}/runs/{rdir.name}/unapprove")
    assert res.status_code == 200
    body = res.json()
    assert body["manifest"]["approved_at"] is None
    assert body["manifest"]["review_status"] == "unreviewed"


def test_unapprove_endpoint_404_for_missing_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.post(
        f"/projects/{project.id}/runs/teacher_does_not_exist/unapprove"
    )
    assert res.status_code == 404


def test_unapprove_endpoint_no_op_on_unapproved_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    res = client.post(f"/projects/{project.id}/runs/{rdir.name}/unapprove")
    assert res.status_code == 200
    assert res.json()["manifest"]["approved_at"] is None


def test_runs_list_serves_review_status_for_each(
    project: runs_mod.Project, client: TestClient
) -> None:
    a = _make_completed_run(project.id, prompt="alpha")
    b = _make_completed_run(project.id, prompt="beta")
    runs_mod.approve_run(b)
    runs_mod.toggle_rejection(a, frame_idx=0, det_idx=0)

    res = client.get(f"/projects/{project.id}/runs")
    assert res.status_code == 200
    statuses = {r["id"]: r["review_status"] for r in res.json()["runs"]}
    assert statuses[a.name] == "reviewed"
    assert statuses[b.name] == "approved"
