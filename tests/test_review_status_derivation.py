"""Phase 0: dataset review state.

Two surfaces under test:

1. The derivation rule for `RunManifestModel.review_status` —
   `approved_at` wins; otherwise non-empty rejections file → "reviewed";
   otherwise → "unreviewed". Plus the cheap `has_any_rejections` helper
   that the computed field leans on.
2. The /runs/{id}/approve · /unapprove endpoints — 404 / 400 / no-op
   semantics from section 0.2 of `docs/evaluate.md`, and round-trip
   persistence (approve → reload manifest from disk → still approved).

Because `RUNS_DIR = Path("runs")` is relative, every test that touches
the filesystem uses `monkeypatch.chdir(tmp_path)` so the runs directory
lands in an isolated tmp workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app
from server.schemas import RunManifestModel


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated runs/ directory under tmp_path. Endpoints call
    `runs_mod.run_dir(id)` which resolves `Path("runs")` against the
    current working directory, so a chdir is the simplest hook."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_completed_run(runs_root: Path, prompt: str = "test prompt") -> Path:
    """Allocate a Teacher run dir that's already in the 'completed' state.

    The endpoints under test gate on `status == "completed"`, so building
    the canonical happy-path fixture this way means each test only has to
    spell out the bits that actually matter for it.

    `prompt` doubles as a uniqueness lever — `make_run_id` only goes to
    second precision, so two same-prompt runs in one test will collide
    on the directory name. Tests that need >1 run pass distinct prompts.
    """
    rdir, _ = runs_mod.create_run(
        task="detection",
        prompt=prompt,
        video_path="data/x.mp4",
        models={"detect": "groundingdino"},
        runs_root=runs_root,
    )
    runs_mod.mark_completed(rdir)
    return rdir


# ---- has_any_rejections (the cheap helper behind review_status) ------------


def test_has_any_rejections_missing_file(tmp_path: Path) -> None:
    """No rejections.json on disk ⇒ False. Most common state for a fresh
    run, and the path the helper short-circuits on first."""
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_empty_dict(tmp_path: Path) -> None:
    """`rejections.json` containing `{}` ⇒ False. Could happen if the
    user toggled and untoggled — `write_rejections` strips empty keys
    but the file itself stays."""
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text("{}")
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_only_empty_lists(tmp_path: Path) -> None:
    """Defensive: `{"5": []}` should still count as no rejections — the
    one in-memory list might happen if someone hand-edits the file."""
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text(json.dumps({"5": []}))
    assert runs_mod.has_any_rejections(tmp_path) is False


def test_has_any_rejections_populated(tmp_path: Path) -> None:
    """Real-world: at least one frame has at least one rejected
    detection ⇒ True. This is the only path that returns True."""
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text(
        json.dumps({"5": [0, 2], "12": [1]})
    )
    assert runs_mod.has_any_rejections(tmp_path) is True


def test_has_any_rejections_malformed_json_is_false(tmp_path: Path) -> None:
    """A corrupted rejections.json must not raise — same defensive
    behaviour as `read_rejections`."""
    (tmp_path / runs_mod.REJECTIONS_NAME).write_text("{not valid json")
    assert runs_mod.has_any_rejections(tmp_path) is False


# ---- review_status derivation ---------------------------------------------


def test_review_status_unreviewed_default(runs_root: Path) -> None:
    """Fresh completed run, no approval, no rejections ⇒ unreviewed.
    This is the default state every Learn run lives in until a human
    interacts with it."""
    rdir = _make_completed_run(runs_root)
    manifest = runs_mod.read_manifest(rdir)

    assert manifest.approved_at is None
    model = RunManifestModel(**manifest.__dict__)
    assert model.review_status == "unreviewed"


def test_review_status_reviewed_when_rejections_present(runs_root: Path) -> None:
    """Add a rejection without approving ⇒ reviewed (the amber state).
    Models the user who scrubbed labels but didn't bless the dataset
    yet."""
    rdir = _make_completed_run(runs_root)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)

    manifest = runs_mod.read_manifest(rdir)
    model = RunManifestModel(**manifest.__dict__)
    assert model.review_status == "reviewed"


def test_review_status_approved_overrides_rejections(runs_root: Path) -> None:
    """Approval beats rejection presence — once `approved_at` is set,
    that's the dataset state. (The user may have rejected some boxes,
    then later said "yes, fully reviewed".)"""
    rdir = _make_completed_run(runs_root)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)
    runs_mod.approve_run(rdir)

    manifest = runs_mod.read_manifest(rdir)
    model = RunManifestModel(**manifest.__dict__)
    assert model.review_status == "approved"
    assert model.approved_at is not None


def test_review_status_unapprove_falls_back_to_reviewed(runs_root: Path) -> None:
    """unapprove preserves rejection history — the user's curated kept
    labels don't vanish just because they retracted approval. So
    approved → unapproved should land back at "reviewed", not
    "unreviewed", whenever rejections exist."""
    rdir = _make_completed_run(runs_root)
    runs_mod.toggle_rejection(rdir, frame_idx=3, det_idx=0)
    runs_mod.approve_run(rdir)
    runs_mod.unapprove_run(rdir)

    manifest = runs_mod.read_manifest(rdir)
    assert manifest.approved_at is None
    assert RunManifestModel(**manifest.__dict__).review_status == "reviewed"


def test_review_status_unapprove_with_no_rejections_is_unreviewed(
    runs_root: Path,
) -> None:
    """The other unapprove branch: no rejections were ever recorded, so
    pulling approval drops the run all the way back to unreviewed."""
    rdir = _make_completed_run(runs_root)
    runs_mod.approve_run(rdir)
    runs_mod.unapprove_run(rdir)

    manifest = runs_mod.read_manifest(rdir)
    assert RunManifestModel(**manifest.__dict__).review_status == "unreviewed"


def test_legacy_manifest_without_approved_at_field_loads(runs_root: Path) -> None:
    """Acceptance criterion: old completed runs load with
    `approved_at = None`, `review_status = "unreviewed"`. Simulated by
    writing a manifest.json that pre-dates the new field."""
    rdir = runs_root / "teacher_legacy"
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
        # Note: NO approved_at — pre-Phase-0 schema.
    }
    (rdir / runs_mod.MANIFEST_NAME).write_text(json.dumps(legacy_manifest))

    manifest = runs_mod.read_manifest(rdir)
    assert manifest.approved_at is None
    assert RunManifestModel(**manifest.__dict__).review_status == "unreviewed"


# ---- Persistence: round-trip through disk ---------------------------------


def test_approve_persists_across_reread(runs_root: Path) -> None:
    """Acceptance: approval persists across server restart. Modeled here
    by calling `approve_run` then re-reading the manifest cold from
    disk — same on-disk JSON that a process restart would deserialise."""
    rdir = _make_completed_run(runs_root)
    runs_mod.approve_run(rdir)

    reread = runs_mod.read_manifest(rdir)
    assert reread.approved_at is not None

    raw = json.loads((rdir / runs_mod.MANIFEST_NAME).read_text())
    assert raw["approved_at"] == reread.approved_at


def test_approve_is_idempotent_does_not_drift_timestamp(runs_root: Path) -> None:
    """A double-click on Approve must not slide the timestamp forward.
    The first stamp is what the user sees in the tooltip; rewriting it
    on every call would silently lie about when they approved."""
    rdir = _make_completed_run(runs_root)
    first = runs_mod.approve_run(rdir)
    second = runs_mod.approve_run(rdir)

    assert first.approved_at is not None
    assert second.approved_at == first.approved_at


# ---- Endpoint contract: /approve and /unapprove ---------------------------


def test_approve_endpoint_happy_path(runs_root: Path, client: TestClient) -> None:
    """Approve a completed run ⇒ 200, manifest carries approved_at,
    review_status flips to "approved" on the wire (this is the field
    the GUI pill is going to bind to in Phase 1)."""
    rdir = _make_completed_run(runs_root)
    run_id = rdir.name

    res = client.post(f"/runs/{run_id}/approve")
    assert res.status_code == 200
    body = res.json()
    assert body["manifest"]["approved_at"] is not None
    assert body["manifest"]["review_status"] == "approved"


def test_approve_endpoint_404_for_missing_run(
    runs_root: Path, client: TestClient
) -> None:
    res = client.post("/runs/teacher_does_not_exist/approve")
    assert res.status_code == 404


def test_approve_endpoint_400_for_running_run(
    runs_root: Path, client: TestClient
) -> None:
    """A still-running Learn worker has nothing meaningful to approve —
    the dataset isn't finished. 400, not 404 (the run exists)."""
    rdir, _ = runs_mod.create_run(
        task="detection",
        prompt="still running",
        video_path="data/x.mp4",
        models={},
        runs_root=runs_root,
    )
    # Don't call mark_completed → status stays "running".

    res = client.post(f"/runs/{rdir.name}/approve")
    assert res.status_code == 400
    assert "completed" in res.json()["detail"]


def test_approve_endpoint_400_for_failed_run(
    runs_root: Path, client: TestClient
) -> None:
    """A failed run also can't be approved — the dataset wasn't finished
    successfully, so blessing it as ground truth is meaningless."""
    rdir, _ = runs_mod.create_run(
        task="detection",
        prompt="will fail",
        video_path="data/x.mp4",
        models={},
        runs_root=runs_root,
    )
    runs_mod.mark_failed(rdir, error="boom")

    res = client.post(f"/runs/{rdir.name}/approve")
    assert res.status_code == 400


def test_approve_endpoint_idempotent_on_already_approved(
    runs_root: Path, client: TestClient
) -> None:
    """Re-approve must be a no-op (200, unchanged timestamp). The
    helper-level idempotence test above guards the in-memory path; this
    one guards the endpoint surface the GUI will hit."""
    rdir = _make_completed_run(runs_root)
    run_id = rdir.name

    first = client.post(f"/runs/{run_id}/approve").json()
    second = client.post(f"/runs/{run_id}/approve").json()
    assert first["manifest"]["approved_at"] == second["manifest"]["approved_at"]


def test_unapprove_endpoint_clears_field(
    runs_root: Path, client: TestClient
) -> None:
    """Round-trip via the wire: approve → unapprove ⇒ approved_at is
    None, review_status falls back to unreviewed (no rejections in
    this fixture)."""
    rdir = _make_completed_run(runs_root)
    run_id = rdir.name

    client.post(f"/runs/{run_id}/approve")
    res = client.post(f"/runs/{run_id}/unapprove")

    assert res.status_code == 200
    body = res.json()
    assert body["manifest"]["approved_at"] is None
    assert body["manifest"]["review_status"] == "unreviewed"


def test_unapprove_endpoint_404_for_missing_run(client: TestClient) -> None:
    res = client.post("/runs/teacher_does_not_exist/unapprove")
    assert res.status_code == 404


def test_unapprove_endpoint_no_op_on_unapproved_run(
    runs_root: Path, client: TestClient
) -> None:
    """Unapproving an already-unapproved run is a quiet no-op (200), not
    a 4xx — keeps the GUI's optimistic-update path simple."""
    rdir = _make_completed_run(runs_root)
    res = client.post(f"/runs/{rdir.name}/unapprove")
    assert res.status_code == 200
    assert res.json()["manifest"]["approved_at"] is None


def test_runs_list_serves_review_status_for_each(
    runs_root: Path, client: TestClient
) -> None:
    """The /runs listing has to carry `review_status` for every run —
    Phase 1's pill in Learn-sidebar binds to it without a per-run
    follow-up GET."""
    a = _make_completed_run(runs_root, prompt="alpha")
    b = _make_completed_run(runs_root, prompt="beta")
    runs_mod.approve_run(b)
    runs_mod.toggle_rejection(a, frame_idx=0, det_idx=0)

    res = client.get("/runs")
    assert res.status_code == 200
    statuses = {r["id"]: r["review_status"] for r in res.json()["runs"]}
    assert statuses[a.name] == "reviewed"
    assert statuses[b.name] == "approved"
