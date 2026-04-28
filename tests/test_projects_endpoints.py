"""Phase 1: Project lifecycle endpoints + summary counters.

Covers the new `/projects` surface: create, list-with-summary, get,
patch-name-only, delete. Plus the Phase 1 invariants:

  • task + prompts are LOCKED at creation (PATCH refuses both).
  • Summary counters reflect what's on disk:
      n_running, n_teacher_datasets, n_human_reviewed_datasets, n_students.
  • Teacher / Student endpoints under a missing project return 404 cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


# ---- Create / list / get --------------------------------------------------


def test_create_project_happy_path(client: TestClient) -> None:
    res = client.post(
        "/projects",
        json={
            "name": "Soccer ball",
            "task": "detection",
            "prompts": ["soccer ball"],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "Soccer ball"
    assert body["task"] == "detection"
    assert body["prompts"] == ["soccer ball"]
    assert body["id"].startswith("proj_")
    assert body["created_at"]


def test_create_project_dedupes_and_strips_prompts(client: TestClient) -> None:
    res = client.post(
        "/projects",
        json={
            "name": "x",
            "task": "detection",
            "prompts": ["  ball ", "ball", "player "],
        },
    )
    assert res.status_code == 200
    assert res.json()["prompts"] == ["ball", "player"]


def test_create_project_rejects_empty_prompts_list(client: TestClient) -> None:
    res = client.post(
        "/projects",
        json={"name": "x", "task": "detection", "prompts": []},
    )
    assert res.status_code == 422


def test_create_project_rejects_blank_prompt_entry(client: TestClient) -> None:
    res = client.post(
        "/projects",
        json={"name": "x", "task": "detection", "prompts": ["   "]},
    )
    assert res.status_code == 422


def test_create_project_rejects_unknown_task(client: TestClient) -> None:
    res = client.post(
        "/projects",
        json={"name": "x", "task": "classification", "prompts": ["a"]},
    )
    assert res.status_code == 422


def test_list_projects_empty(client: TestClient) -> None:
    res = client.get("/projects")
    assert res.status_code == 200
    assert res.json()["projects"] == []


def test_list_projects_returns_summary_with_zero_counts(client: TestClient) -> None:
    p = client.post(
        "/projects",
        json={"name": "p1", "task": "detection", "prompts": ["a"]},
    ).json()

    res = client.get("/projects")
    assert res.status_code == 200
    rows = res.json()["projects"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == p["id"]
    assert row["n_running"] == 0
    assert row["n_teacher_datasets"] == 0
    assert row["n_human_reviewed_datasets"] == 0
    assert row["n_students"] == 0


def test_get_project_404(client: TestClient) -> None:
    res = client.get("/projects/proj_no_such")
    assert res.status_code == 404


def test_get_project_returns_full_record(client: TestClient) -> None:
    p = client.post(
        "/projects",
        json={"name": "p1", "task": "segmentation", "prompts": ["dog"]},
    ).json()
    res = client.get(f"/projects/{p['id']}")
    assert res.status_code == 200
    assert res.json() == p


# ---- Rename ---------------------------------------------------------------


def test_patch_renames_only(client: TestClient) -> None:
    p = client.post(
        "/projects",
        json={"name": "old", "task": "detection", "prompts": ["a"]},
    ).json()
    res = client.patch(f"/projects/{p['id']}", json={"name": "new"})
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "new"
    # Locked fields unchanged.
    assert body["task"] == "detection"
    assert body["prompts"] == ["a"]


def test_patch_rejects_task_or_prompt_changes(client: TestClient) -> None:
    """The schema only declares `name` so any extra body keys fail
    pydantic validation — task + prompts cannot be slipped through."""
    p = client.post(
        "/projects",
        json={"name": "p", "task": "detection", "prompts": ["a"]},
    ).json()
    res = client.patch(
        f"/projects/{p['id']}",
        json={"name": "p", "task": "segmentation"},
    )
    assert res.status_code == 422


def test_patch_404(client: TestClient) -> None:
    res = client.patch("/projects/proj_no_such", json={"name": "x"})
    assert res.status_code == 404


# ---- Delete ---------------------------------------------------------------


def test_delete_project_removes_dir(client: TestClient) -> None:
    p = client.post(
        "/projects",
        json={"name": "p", "task": "detection", "prompts": ["a"]},
    ).json()
    pdir = runs_mod.project_dir(p["id"])
    assert pdir.exists()
    res = client.delete(f"/projects/{p['id']}")
    assert res.status_code == 200
    assert not pdir.exists()


def test_delete_project_404(client: TestClient) -> None:
    res = client.delete("/projects/proj_no_such")
    assert res.status_code == 404


# ---- Summary counters reflect runs on disk --------------------------------


def test_summary_counts_teachers_and_students(
    client: TestClient,
) -> None:
    p = client.post(
        "/projects",
        json={"name": "p", "task": "detection", "prompts": ["a"]},
    ).json()
    pid = p["id"]

    # Two completed teachers, one of them fully reviewed via per-frame
    # state (Phase 3 strict bar — only "approved" counts toward
    # n_human_reviewed_datasets).
    rdir1, _ = runs_mod.create_run(
        project_id=pid, task="detection", prompt="t1",
        video_path="x.mp4", models={},
    )
    # Seed a one-frame per_frame.jsonl so derive_review_status has
    # something to count coverage against.
    pf1 = rdir1 / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    pf1.parent.mkdir(parents=True, exist_ok=True)
    pf1.write_text(
        json.dumps({"frame_idx": 0, "detections": []}) + "\n"
    )
    runs_mod.mark_completed(rdir1)
    runs_mod.set_frame_state(rdir1, 0, "confirmed_empty")

    rdir2, _ = runs_mod.create_run(
        project_id=pid, task="detection", prompt="t2",
        video_path="x.mp4", models={},
    )
    runs_mod.mark_completed(rdir2)

    # One running teacher (no mark_completed).
    runs_mod.create_run(
        project_id=pid, task="detection", prompt="t3",
        video_path="x.mp4", models={},
    )

    # One completed student.
    sdir, _ = runs_mod.create_student(
        project_id=pid,
        train_teacher_ids=[rdir1.name],
        eval_teacher_ids=[],
        task="detection",
        prompt="s1",
        models={"detect": "yolov8n"},
    )
    runs_mod.mark_student_completed(sdir)

    res = client.get("/projects")
    assert res.status_code == 200
    row = next(r for r in res.json()["projects"] if r["id"] == pid)
    assert row["n_teacher_datasets"] == 2
    assert row["n_human_reviewed_datasets"] == 1
    assert row["n_running"] == 1
    assert row["n_students"] == 1


# ---- Project gating on Teacher / Student endpoints ------------------------


def test_runs_list_404_for_missing_project(client: TestClient) -> None:
    res = client.get("/projects/proj_nope/runs")
    assert res.status_code == 404


def test_students_list_404_for_missing_project(client: TestClient) -> None:
    res = client.get("/projects/proj_nope/students")
    assert res.status_code == 404


def test_runs_list_returns_only_this_projects_runs(client: TestClient) -> None:
    """Runs in another project must not bleed into this project's list."""
    a = client.post(
        "/projects", json={"name": "a", "task": "detection", "prompts": ["x"]},
    ).json()
    b = client.post(
        "/projects", json={"name": "b", "task": "detection", "prompts": ["y"]},
    ).json()
    rdir_a, _ = runs_mod.create_run(
        project_id=a["id"], task="detection", prompt="ra",
        video_path="x.mp4", models={},
    )
    runs_mod.mark_completed(rdir_a)
    rdir_b, _ = runs_mod.create_run(
        project_id=b["id"], task="detection", prompt="rb",
        video_path="y.mp4", models={},
    )
    runs_mod.mark_completed(rdir_b)

    a_runs = client.get(f"/projects/{a['id']}/runs").json()["runs"]
    b_runs = client.get(f"/projects/{b['id']}/runs").json()["runs"]
    assert [r["id"] for r in a_runs] == [rdir_a.name]
    assert [r["id"] for r in b_runs] == [rdir_b.name]
    # And every manifest carries its project_id back.
    assert a_runs[0]["project_id"] == a["id"]
    assert b_runs[0]["project_id"] == b["id"]
