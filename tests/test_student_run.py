"""Phase 5: student-run helpers + endpoint validation.

Three surfaces under test:

1. The on-disk dataclasses + helpers in `pipeline.runs` (create, read,
   mark-completed, mark-failed, list, delete) — same shape as the existing
   teacher / student helpers but a third tier.
2. The `pipeline.student_run._resolve_input` validation (video path /
   teacher id existence + status).
3. The `/projects/{pid}/students/{sid}/runs[...]` endpoint surface — 404
   on missing student / run, 400 on validation failure.

Real inference is NOT tested here — that would require Ultralytics
weights and a real video, which is too slow + heavy for unit tests.
The inference loop has a manual verification step in the plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from pipeline import student_run
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
        name="Soccer ball",
        task="detection",
        prompts=["soccer ball"],
    )


@pytest.fixture
def student(project: runs_mod.Project, tmp_path: Path) -> runs_mod.StudentManifest:
    """A completed Student with a placeholder weights file.

    The weights file is empty bytes — the predict path is not exercised
    in these tests; we only need `best.pt` to exist so the
    `run_student_run_in_background` weight check passes. We allocate a
    minimal Teacher first because `create_student` requires at least one
    train_teacher_id.
    """
    tdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="ball",
        video_path=str(tmp_path / "x.mp4"),
        models={"detect": "groundingdino"},
    )
    runs_mod.mark_completed(tdir)
    rdir, manifest = runs_mod.create_student(
        project_id=project.id,
        train_teacher_ids=[tdir.name],
        eval_teacher_ids=[],
        task="detection",
        prompt="ball",
        models={"detect": "yolov8n"},
        architecture="yolov8n",
    )
    runs_mod.mark_student_completed(rdir)
    (rdir / "best.pt").write_bytes(b"")
    return runs_mod.read_student_manifest(rdir)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---- Disk helpers ----------------------------------------------------------


def test_create_project_creates_student_runs_subdir(runs_root: Path) -> None:
    p = runs_mod.create_project(
        name="X", task="detection", prompts=["a"]
    )
    pdir = runs_mod.project_dir(p.id)
    assert (pdir / runs_mod.STUDENT_RUNS_DIR).is_dir()


def test_create_student_run_writes_manifest(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    rdir, m = runs_mod.create_student_run(
        project_id=project.id,
        student_id=student.id,
        input_kind="video",
        input_ref="/tmp/x.mp4",
    )
    assert (rdir / runs_mod.MANIFEST_NAME).exists()
    assert (rdir / runs_mod.PREDICTIONS_DIR).is_dir()
    got = runs_mod.read_student_run_manifest(rdir)
    assert got.id == m.id
    assert got.student_id == student.id
    assert got.input_kind == "video"
    assert got.input_ref == "/tmp/x.mp4"
    assert got.status == "running"  # default before queue flips it


def test_create_student_run_unknown_input_kind(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    with pytest.raises(ValueError, match="unknown input_kind"):
        runs_mod.create_student_run(
            project_id=project.id,
            student_id=student.id,
            input_kind="bogus",
            input_ref="x",
        )


def test_create_student_run_missing_student_404(
    project: runs_mod.Project,
) -> None:
    with pytest.raises(FileNotFoundError):
        runs_mod.create_student_run(
            project_id=project.id,
            student_id="student_does_not_exist",
            input_kind="video",
            input_ref="/tmp/x.mp4",
        )


def test_mark_student_run_completed_writes_stats(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    rdir, _ = runs_mod.create_student_run(
        project_id=project.id,
        student_id=student.id,
        input_kind="video",
        input_ref="/tmp/x.mp4",
    )
    stats = runs_mod.StudentRunStats(
        n_frames=10, n_detections=5, avg_inference_ms=12.0,
        p50_inference_ms=10.0, p95_inference_ms=20.0,
    )
    m = runs_mod.mark_student_run_completed(rdir, stats)
    assert m.status == "completed"
    assert m.ended_at is not None
    got = runs_mod.read_student_run_stats(rdir)
    assert got is not None
    assert got.n_frames == 10
    assert got.map50 is None  # not populated for video runs


def test_mark_student_run_failed(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    rdir, _ = runs_mod.create_student_run(
        project_id=project.id,
        student_id=student.id,
        input_kind="video",
        input_ref="/tmp/x.mp4",
    )
    m = runs_mod.mark_student_run_failed(rdir, "test failure")
    assert m.status == "failed"
    assert m.error == "test failure"


def test_list_student_runs_newest_first(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    """The list helper sorts by `started_at` desc.

    We override `started_at` directly on disk because creating two runs
    in quick succession can produce identical timestamps (the IDs
    collide too — same second, same slug). For test purposes a manual
    edit is fine.
    """
    rdir1, m1 = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/a.mp4",
    )
    m1.started_at = "2025-01-01T00:00:00Z"
    runs_mod.write_student_run_manifest(rdir1, m1)

    # Force a different run id by slugifying differently — work around
    # the second-resolution timestamp collision.
    rdir2 = runs_mod.student_run_parent_dir(project.id, student.id) / "studentrun_20250102-000000_run"
    rdir2.mkdir()
    (rdir2 / runs_mod.PREDICTIONS_DIR).mkdir()
    m2 = runs_mod.StudentRunManifest(
        id=rdir2.name,
        student_id=student.id,
        project_id=project.id,
        input_kind="video",
        input_ref="/tmp/b.mp4",
        started_at="2025-02-01T00:00:00Z",
    )
    runs_mod.write_student_run_manifest(rdir2, m2)

    listed = runs_mod.list_student_runs(project.id, student.id)
    assert [m.id for m in listed] == [m2.id, m1.id]


def test_delete_student_run_removes_dir(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    rdir, _ = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    assert rdir.exists()
    runs_mod.delete_student_run(rdir)
    assert not rdir.exists()


def test_mark_stale_runs_sweeps_student_runs(
    project: runs_mod.Project, student: runs_mod.StudentManifest
) -> None:
    """Server-restart sweeper covers the new tier."""
    rdir, m = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    # Default is "running"; left as-is so the sweeper flips it.
    n = runs_mod.mark_stale_runs_failed()
    assert n >= 1
    got = runs_mod.read_student_run_manifest(rdir)
    assert got.status == "failed"


# ---- _resolve_input -------------------------------------------------------


def test_resolve_input_video_missing_path(project: runs_mod.Project) -> None:
    with pytest.raises(FileNotFoundError):
        student_run._resolve_input(
            project_id=project.id,
            input_kind="video",
            input_ref="/tmp/does-not-exist.mp4",
        )


def test_resolve_input_video_empty_ref(project: runs_mod.Project) -> None:
    with pytest.raises(ValueError, match="input_ref is required"):
        student_run._resolve_input(
            project_id=project.id, input_kind="video", input_ref=""
        )


def test_resolve_input_teacher_dataset_missing_teacher(
    project: runs_mod.Project,
) -> None:
    with pytest.raises(FileNotFoundError, match="no such teacher"):
        student_run._resolve_input(
            project_id=project.id,
            input_kind="teacher_dataset",
            input_ref="teacher_does_not_exist",
        )


def test_resolve_input_teacher_dataset_not_completed(
    project: runs_mod.Project, tmp_path: Path
) -> None:
    """Reject teachers still queued/running — the source video isn't
    guaranteed to be on disk yet, and the COCO labels for mAP definitely
    aren't."""
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="t",
        video_path=str(video),
        models={"detect": "groundingdino"},
    )
    # Default status is 'running'.
    with pytest.raises(ValueError, match="status is"):
        student_run._resolve_input(
            project_id=project.id,
            input_kind="teacher_dataset",
            input_ref=rdir.name,
        )


def test_resolve_input_teacher_dataset_returns_video_path(
    project: runs_mod.Project, tmp_path: Path
) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="t",
        video_path=str(video),
        models={"detect": "groundingdino"},
    )
    runs_mod.mark_completed(rdir)
    vp, tid = student_run._resolve_input(
        project_id=project.id,
        input_kind="teacher_dataset",
        input_ref=rdir.name,
    )
    assert vp == str(video)
    assert tid == rdir.name


# ---- Endpoint surface (validation only — no real inference) ---------------


def test_start_student_run_404_for_missing_student(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.post(
        f"/projects/{project.id}/students/missing/run",
        json={"input_kind": "video", "input_ref": "/tmp/x.mp4"},
    )
    assert res.status_code == 404


def test_start_student_run_400_for_missing_video(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    res = client.post(
        f"/projects/{project.id}/students/{student.id}/run",
        json={"input_kind": "video", "input_ref": "/tmp/does-not-exist.mp4"},
    )
    assert res.status_code == 400


def test_start_student_run_400_for_unknown_input_kind(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    res = client.post(
        f"/projects/{project.id}/students/{student.id}/run",
        json={"input_kind": "bogus", "input_ref": "x"},
    )
    # FastAPI validation rejects the Literal mismatch before our handler
    # runs, so this is a 422 from pydantic.
    assert res.status_code == 422


def test_list_student_runs_empty(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    res = client.get(f"/projects/{project.id}/students/{student.id}/runs")
    assert res.status_code == 200
    assert res.json() == {"runs": []}


def test_list_student_runs_404_for_missing_student(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.get(f"/projects/{project.id}/students/missing/runs")
    assert res.status_code == 404


def test_get_student_run_404(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    res = client.get(
        f"/projects/{project.id}/students/{student.id}/runs/missing"
    )
    assert res.status_code == 404


def test_delete_student_run(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    """Round-trip: create on disk, delete via API, confirm gone."""
    rdir, m = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    res = client.delete(
        f"/projects/{project.id}/students/{student.id}/runs/{m.id}"
    )
    assert res.status_code == 200
    assert not rdir.exists()


def test_get_student_run_round_trip(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    rdir, m = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    runs_mod.mark_student_run_completed(
        rdir,
        runs_mod.StudentRunStats(
            n_frames=3, n_detections=7, avg_inference_ms=11.0,
            p50_inference_ms=10.0, p95_inference_ms=15.0,
        ),
    )
    res = client.get(
        f"/projects/{project.id}/students/{student.id}/runs/{m.id}"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["manifest"]["id"] == m.id
    assert body["manifest"]["status"] == "completed"
    assert body["stats"]["n_frames"] == 3
    assert body["stats"]["map50"] is None


def test_student_run_overlay_404_when_not_written(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    rdir, m = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    res = client.get(
        f"/projects/{project.id}/students/{student.id}/runs/{m.id}/overlay.mp4"
    )
    assert res.status_code == 404


def test_student_run_labels_round_trip(
    project: runs_mod.Project,
    student: runs_mod.StudentManifest,
    client: TestClient,
) -> None:
    rdir, m = runs_mod.create_student_run(
        project_id=project.id, student_id=student.id,
        input_kind="video", input_ref="/tmp/x.mp4",
    )
    pf = rdir / runs_mod.PREDICTIONS_DIR / runs_mod.PER_FRAME_NAME
    pf.write_text(
        json.dumps({"frame_idx": 0, "detections": []}) + "\n"
        + json.dumps({"frame_idx": 1, "detections": [{"score": 0.9}]}) + "\n"
    )
    res = client.get(
        f"/projects/{project.id}/students/{student.id}/runs/{m.id}/labels"
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    lines = [l for l in res.text.split("\n") if l]
    assert len(lines) == 2
