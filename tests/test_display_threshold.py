"""Phase 2 tests — post-hoc display_threshold + PATCH endpoint.

The detector now persists every detection ≥ SCORE_FLOOR; the manifest's
`display_threshold` is the cutoff the GUI applies by default. Exercises
both layers:

  • `runs_mod.compute_stats_at_threshold` — re-derives RunStats from
    `per_frame.jsonl` at any threshold, leaving timing fields untouched
    and rebuilding the per-class breakdown from the filtered detections.
  • `PATCH /projects/{pid}/runs/{rid}` — persists a new threshold; the
    follow-up `GET` reflects both the new manifest field and the
    recomputed stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app


# ---- Fixtures -------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


@pytest.fixture()
def project(client: TestClient) -> str:
    project = runs_mod.create_project(
        name="Phase 2 test project",
        task="detection",
        prompts=["ball"],
    )
    return project.id


def _seed_completed_run(project_id: str, run_id: str) -> Path:
    """Lay out a fake completed Teacher run with a per_frame.jsonl that
    has detections spread across the score range (0.05 → 0.95) so a
    threshold sweep produces visibly different stats."""
    rdir = runs_mod.run_dir(project_id, run_id)
    (rdir / runs_mod.LABELS_DIR).mkdir(parents=True)
    manifest = runs_mod.RunManifest(
        id=run_id,
        task="detection",
        prompt="ball",
        video_path="/fake/video.mp4",
        started_at="2026-04-28T12:00:00Z",
        ended_at="2026-04-28T12:01:00Z",
        status="completed",
        models={"detect": "groundingdino"},
        # Default display_threshold is 0.30 — leave it on the manifest.
    )
    runs_mod.write_manifest(rdir, manifest)

    # Three frames, mix of high/medium/low scores.
    pf_path = rdir / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    with pf_path.open("w") as fp:
        for frame_idx, scores in enumerate(
            [
                [0.95, 0.55, 0.10],  # frame 0: 3 dets
                [0.40, 0.07],        # frame 1: 2 dets
                [0.20],              # frame 2: 1 det
            ]
        ):
            dets = [
                {
                    "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                    "score": s,
                    "class_id": 1,
                    "class_name": "ball",
                }
                for s in scores
            ]
            fp.write(json.dumps({"frame_idx": frame_idx, "detections": dets}) + "\n")

    # Base stats — written as if the detector emitted everything ≥ floor
    # (6 detections across 3 frames; timing dummy values).
    base_stats = runs_mod.RunStats(
        frames_processed=3,
        frames_with_detections=3,
        total_ms=300.0,
        avg_ms_per_frame=100.0,
        p50_ms_per_frame=100.0,
        p95_ms_per_frame=100.0,
        n_detections_total=6,
    )
    runs_mod.write_stats(rdir, base_stats)
    return rdir


# ---- compute_stats_at_threshold -------------------------------------------


def test_stats_at_floor_preserves_all_detections(
    client: TestClient, project: str
) -> None:
    """At a threshold below the lowest score, every detection survives."""
    _seed_completed_run(project, "teacher_a")
    rdir = runs_mod.run_dir(project, "teacher_a")
    stats = runs_mod.compute_stats_at_threshold(rdir, 0.05)
    assert stats is not None
    # 3 + 2 + 1 = 6 detections, every frame had at least one.
    assert stats.n_detections_total == 6
    assert stats.frames_with_detections == 3
    assert stats.detections_per_class["ball"]["n_detections"] == 6


def test_stats_at_high_threshold_drops_low_scores(
    client: TestClient, project: str
) -> None:
    """At threshold=0.50, only detections with score ≥ 0.50 survive."""
    _seed_completed_run(project, "teacher_a")
    rdir = runs_mod.run_dir(project, "teacher_a")
    stats = runs_mod.compute_stats_at_threshold(rdir, 0.50)
    assert stats is not None
    # Only 0.95 and 0.55 (frame 0) survive.
    assert stats.n_detections_total == 2
    assert stats.frames_with_detections == 1
    assert stats.detections_per_class["ball"]["n_detections"] == 2


def test_stats_threshold_preserves_timing_fields(
    client: TestClient, project: str
) -> None:
    """Timing percentiles are detector-side facts; sliding the threshold
    must not change them."""
    _seed_completed_run(project, "teacher_a")
    rdir = runs_mod.run_dir(project, "teacher_a")
    base = runs_mod.read_stats(rdir)
    sliced = runs_mod.compute_stats_at_threshold(rdir, 0.80)
    assert base is not None and sliced is not None
    assert sliced.frames_processed == base.frames_processed
    assert sliced.total_ms == base.total_ms
    assert sliced.avg_ms_per_frame == base.avg_ms_per_frame
    assert sliced.p50_ms_per_frame == base.p50_ms_per_frame
    assert sliced.p95_ms_per_frame == base.p95_ms_per_frame


# ---- API: GET ?threshold preview + PATCH persists -------------------------


def test_get_run_uses_manifest_threshold_by_default(
    client: TestClient, project: str
) -> None:
    """No `?threshold=` → server uses the manifest's display_threshold
    (0.30 in the seeded run). Five detections clear that bar."""
    _seed_completed_run(project, "teacher_a")
    res = client.get(f"/projects/{project}/runs/teacher_a")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["manifest"]["display_threshold"] == pytest.approx(0.30)
    # Scores ≥ 0.30: 0.95, 0.55, 0.40 → 3 detections across 2 frames.
    assert body["stats"]["n_detections_total"] == 3
    assert body["stats"]["frames_with_detections"] == 2


def test_get_run_with_threshold_query_does_not_persist(
    client: TestClient, project: str
) -> None:
    """`?threshold=0.60` overrides the response numbers but leaves the
    manifest untouched — the slider should be free to preview without
    side effects."""
    _seed_completed_run(project, "teacher_a")
    res = client.get(f"/projects/{project}/runs/teacher_a?threshold=0.60")
    assert res.status_code == 200, res.text
    body = res.json()
    # Manifest still at the default 0.30.
    assert body["manifest"]["display_threshold"] == pytest.approx(0.30)
    # Filtered numbers reflect the override (only 0.95 clears 0.60).
    assert body["stats"]["n_detections_total"] == 1


def test_patch_run_persists_display_threshold(
    client: TestClient, project: str
) -> None:
    """PATCH writes the manifest; the next GET reflects the new value
    AND returns stats filtered at it."""
    _seed_completed_run(project, "teacher_a")
    res = client.patch(
        f"/projects/{project}/runs/teacher_a",
        json={"display_threshold": 0.50},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["manifest"]["display_threshold"] == pytest.approx(0.50)
    assert body["stats"]["n_detections_total"] == 2  # 0.95 + 0.55

    # Second read confirms it stuck on disk.
    again = client.get(f"/projects/{project}/runs/teacher_a").json()
    assert again["manifest"]["display_threshold"] == pytest.approx(0.50)


def test_patch_run_rejects_out_of_range(client: TestClient, project: str) -> None:
    """display_threshold must be in [0, 1]. Pydantic 422s on either side."""
    _seed_completed_run(project, "teacher_a")
    for bad in (-0.1, 1.5):
        res = client.patch(
            f"/projects/{project}/runs/teacher_a",
            json={"display_threshold": bad},
        )
        assert res.status_code == 422, f"{bad}: {res.text}"


def test_patch_run_rejects_unknown_fields(client: TestClient, project: str) -> None:
    """The PATCH body uses extra='forbid' — surfacing typos as 422 instead
    of a silent no-op (mirrors how PATCH /projects/{pid} guards against
    `task`/`prompts` mutations)."""
    _seed_completed_run(project, "teacher_a")
    res = client.patch(
        f"/projects/{project}/runs/teacher_a",
        json={"display_threshold": 0.40, "status": "approved"},
    )
    assert res.status_code == 422


def test_patch_missing_run_404s(client: TestClient, project: str) -> None:
    res = client.patch(
        f"/projects/{project}/runs/does-not-exist",
        json={"display_threshold": 0.50},
    )
    assert res.status_code == 404
