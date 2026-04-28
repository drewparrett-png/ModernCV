"""Phase 3: per-frame review state — disk + endpoint coverage.

Two surfaces under test:

1. The disk-side `read_frame_states` / `write_frame_states` /
   `set_frame_state` / `unset_frame_state` helpers and the
   `derive_review_status` rule layered on top: "approved" iff every
   processed frame has a state entry; "in_progress" iff some but not
   all do; "unreviewed" otherwise.
2. The /projects/{pid}/runs/{id}/frame_states surface — GET, PUT,
   DELETE — with validation behavior (400 for bad shapes / out-of-range
   det indices) and round-trip persistence.
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


def _seed_per_frame(rdir: Path, frame_dets: dict[int, int]) -> None:
    """Write a minimal per_frame.jsonl with `frame_dets[fi]` placeholder
    detections per frame index. Used by tests that need
    `_count_processed_frames` and rejected_dets bounds-checking to work."""
    pf_path = rdir / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fi, n in frame_dets.items():
        rec = {
            "frame_idx": fi,
            "detections": [{"score": 0.9, "class_name": "x"} for _ in range(n)],
        }
        lines.append(json.dumps(rec))
    pf_path.write_text("\n".join(lines) + "\n")


def _make_completed_run(
    project_id: str,
    *,
    n_frames: int = 3,
    dets_per_frame: int = 1,
    prompt: str = "test prompt",
) -> Path:
    rdir, _ = runs_mod.create_run(
        project_id=project_id,
        task="detection",
        prompt=prompt,
        video_path="data/x.mp4",
        models={"detect": "groundingdino"},
    )
    _seed_per_frame(rdir, {i: dets_per_frame for i in range(n_frames)})
    runs_mod.mark_completed(rdir)
    return rdir


# ---- read/write_frame_states ----------------------------------------------


def test_read_frame_states_missing_file(tmp_path: Path) -> None:
    assert runs_mod.read_frame_states(tmp_path) == {}


def test_read_frame_states_malformed_json_is_empty(tmp_path: Path) -> None:
    (tmp_path / runs_mod.FRAME_STATES_NAME).write_text("{not valid json")
    assert runs_mod.read_frame_states(tmp_path) == {}


def test_read_frame_states_filters_unknown_states(tmp_path: Path) -> None:
    (tmp_path / runs_mod.FRAME_STATES_NAME).write_text(
        json.dumps(
            {
                "1": {"state": "curated", "rejected_dets": [0]},
                "2": {"state": "bogus"},
                "3": {"state": "confirmed_empty"},
            }
        )
    )
    states = runs_mod.read_frame_states(tmp_path)
    assert set(states.keys()) == {1, 3}
    assert states[1] == {"state": "curated", "rejected_dets": [0]}
    assert states[3] == {"state": "confirmed_empty", "rejected_dets": []}


def test_write_frame_states_omits_empty_rejected_dets(tmp_path: Path) -> None:
    runs_mod.write_frame_states(
        tmp_path,
        {
            5: {"state": "curated", "rejected_dets": []},
            7: {"state": "confirmed_empty", "rejected_dets": []},
        },
    )
    raw = json.loads((tmp_path / runs_mod.FRAME_STATES_NAME).read_text())
    # Curated with no rejections still serialises (no rejected_dets key);
    # confirmed_empty likewise drops the empty list.
    assert raw == {
        "5": {"state": "curated"},
        "7": {"state": "confirmed_empty"},
    }


# ---- set_frame_state validation -------------------------------------------


def test_set_frame_state_curated_with_rejected_dets(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    entry = runs_mod.set_frame_state(rdir, 1, "curated", rejected_dets=[0])
    assert entry == {"state": "curated", "rejected_dets": [0]}
    states = runs_mod.read_frame_states(rdir)
    assert states[1] == entry


def test_set_frame_state_rejected_dets_with_non_curated_raises(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    with pytest.raises(ValueError, match="rejected_dets only allowed"):
        runs_mod.set_frame_state(rdir, 1, "confirmed_empty", rejected_dets=[0])


def test_set_frame_state_invalid_state_raises(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id)
    with pytest.raises(ValueError, match="invalid state"):
        runs_mod.set_frame_state(rdir, 0, "approved")  # type: ignore[arg-type]


def test_set_frame_state_out_of_range_det_idx_raises(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    with pytest.raises(ValueError, match="out of range"):
        runs_mod.set_frame_state(rdir, 1, "curated", rejected_dets=[5])


def test_set_frame_state_unknown_frame_raises(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    with pytest.raises(ValueError, match="not found"):
        runs_mod.set_frame_state(rdir, 99, "curated", rejected_dets=[0])


# ---- approved_at auto-stamping --------------------------------------------


def test_approved_at_stamps_when_all_frames_covered(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=1)
    runs_mod.set_frame_state(rdir, 0, "confirmed_empty")
    runs_mod.set_frame_state(rdir, 1, "marked_missed")
    # Up to here, only 2 of 3 frames have state — manifest stays unstamped.
    assert runs_mod.read_manifest(rdir).approved_at is None
    runs_mod.set_frame_state(rdir, 2, "curated")
    assert runs_mod.read_manifest(rdir).approved_at is not None


def test_unset_frame_state_clears_approved_at(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=2, dets_per_frame=1)
    runs_mod.set_frame_state(rdir, 0, "confirmed_empty")
    runs_mod.set_frame_state(rdir, 1, "confirmed_empty")
    assert runs_mod.read_manifest(rdir).approved_at is not None
    runs_mod.unset_frame_state(rdir, 1)
    assert runs_mod.read_manifest(rdir).approved_at is None


# ---- derive_review_status -------------------------------------------------


def test_review_status_unreviewed_no_states(project: runs_mod.Project) -> None:
    rdir = _make_completed_run(project.id, n_frames=3)
    assert runs_mod.derive_review_status(rdir) == "unreviewed"


def test_review_status_in_progress_partial_coverage(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=1)
    runs_mod.set_frame_state(rdir, 0, "confirmed_empty")
    assert runs_mod.derive_review_status(rdir) == "in_progress"


def test_review_status_approved_full_coverage(
    project: runs_mod.Project,
) -> None:
    rdir = _make_completed_run(project.id, n_frames=2, dets_per_frame=1)
    runs_mod.set_frame_state(rdir, 0, "confirmed_empty")
    runs_mod.set_frame_state(rdir, 1, "marked_missed")
    assert runs_mod.derive_review_status(rdir) == "approved"


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


# ---- Endpoints -------------------------------------------------------------


def test_get_frame_states_empty_for_new_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    res = client.get(f"/projects/{project.id}/runs/{rdir.name}/frame_states")
    assert res.status_code == 200
    assert res.json() == {"frame_states": {}}


def test_put_frame_state_curated(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    res = client.put(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/1",
        json={"state": "curated", "rejected_dets": [0]},
    )
    assert res.status_code == 200
    assert res.json() == {"state": "curated", "rejected_dets": [0]}

    # Round-trip via GET.
    got = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states"
    ).json()
    assert got == {
        "frame_states": {"1": {"state": "curated", "rejected_dets": [0]}}
    }


def test_put_frame_state_rejected_dets_with_non_curated_400(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    res = client.put(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/1",
        json={"state": "confirmed_empty", "rejected_dets": [0]},
    )
    assert res.status_code == 400


def test_put_frame_state_out_of_range_400(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id, n_frames=3, dets_per_frame=2)
    res = client.put(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/1",
        json={"state": "curated", "rejected_dets": [99]},
    )
    assert res.status_code == 400


def test_put_frame_state_404_for_missing_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.put(
        f"/projects/{project.id}/runs/teacher_does_not_exist/frame_states/0",
        json={"state": "curated"},
    )
    assert res.status_code == 404


def test_delete_frame_state_204(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id, n_frames=2, dets_per_frame=1)
    client.put(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/0",
        json={"state": "confirmed_empty"},
    )
    res = client.delete(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/0"
    )
    assert res.status_code == 204
    got = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states"
    ).json()
    assert got["frame_states"] == {}


def test_delete_frame_state_no_op_on_missing_entry(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id)
    res = client.delete(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/0"
    )
    assert res.status_code == 204


def test_runs_list_serves_review_status_for_each(
    project: runs_mod.Project, client: TestClient
) -> None:
    a = _make_completed_run(project.id, n_frames=2, prompt="alpha")
    b = _make_completed_run(project.id, n_frames=2, prompt="beta")
    # `a` is partially reviewed, `b` is fully reviewed.
    runs_mod.set_frame_state(a, 0, "confirmed_empty")
    runs_mod.set_frame_state(b, 0, "confirmed_empty")
    runs_mod.set_frame_state(b, 1, "marked_missed")

    res = client.get(f"/projects/{project.id}/runs")
    assert res.status_code == 200
    statuses = {r["id"]: r["review_status"] for r in res.json()["runs"]}
    assert statuses[a.name] == "in_progress"
    assert statuses[b.name] == "approved"


def test_approved_at_stamped_via_endpoint(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_completed_run(project.id, n_frames=1, dets_per_frame=1)
    res = client.put(
        f"/projects/{project.id}/runs/{rdir.name}/frame_states/0",
        json={"state": "confirmed_empty"},
    )
    assert res.status_code == 200
    detail = client.get(f"/projects/{project.id}/runs/{rdir.name}").json()
    assert detail["manifest"]["review_status"] == "approved"
    assert detail["manifest"]["approved_at"] is not None
