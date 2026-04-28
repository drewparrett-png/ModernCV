"""Tests for the detection-breakdown stats added in Phase 1.

Two surfaces under test:

1. `compute_detection_breakdown` — pure function over per_frame.jsonl-shaped
   records. Must be:
     - additive (no overlap with the existing timing fields)
     - correct on a hand-crafted small input where I can hand-verify every
       number ("1 ball / 2 ball / 0 ball / 0 ball / 3 ball" → counts agree)
     - tolerant of empty / missing fields
2. `recompute_stats.recompute` — file-IO wrapper. Must:
     - leave timing fields alone if a stats.json already exists
     - rebuild the breakdown from the JSONL
     - work on a run dir that has *only* per_frame.jsonl (no prior stats)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import runs as runs_mod
from pipeline.recompute_stats import recompute


def test_compute_breakdown_simple_two_class():
    """5 frames, mixed class counts. Verifies every reported number."""
    records = [
        {"frame_idx": 0, "detections": [
            {"class_name": "ball", "score": 0.7},
            {"class_name": "player", "score": 0.9},
            {"class_name": "player", "score": 0.8},
        ]},
        {"frame_idx": 1, "detections": [
            {"class_name": "ball", "score": 0.6},
            {"class_name": "ball", "score": 0.5},
        ]},
        {"frame_idx": 2, "detections": []},                         # 0 dets frame
        {"frame_idx": 3, "detections": [
            {"class_name": "player", "score": 0.95},
        ]},
        {"frame_idx": 4, "detections": [
            {"class_name": "player", "score": 0.7},
            {"class_name": "player", "score": 0.7},
            {"class_name": "player", "score": 0.7},
            {"class_name": "ball", "score": 0.4},
        ]},
    ]
    out = runs_mod.compute_detection_breakdown(records, frames_processed=5)

    # Per-class — total ball=4 in 3 frames, player=6 in 3 frames
    pc = out["detections_per_class"]
    assert pc["ball"]["n_detections"] == 4
    assert pc["ball"]["frames_present"] == 3
    assert pc["ball"]["max_in_frame"] == 2
    assert pc["ball"]["avg_per_frame"] == pytest.approx(4 / 5)
    assert pc["ball"]["avg_per_present_frame"] == pytest.approx(4 / 3)

    assert pc["player"]["n_detections"] == 6
    assert pc["player"]["frames_present"] == 3
    assert pc["player"]["max_in_frame"] == 3
    assert pc["player"]["avg_per_frame"] == pytest.approx(6 / 5)

    # Per-frame total counts: [3, 2, 0, 1, 4]
    assert out["per_frame_count_min"] == 0
    assert out["per_frame_count_max"] == 4
    # avg = 10/5 = 2.0
    assert out["per_frame_count_avg"] == pytest.approx(2.0)

    # Histogram: {0:1, 1:1, 2:1, 3:1, 4:1}, JSON-stringified keys
    hist = out["per_frame_count_histogram"]
    assert hist == {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}


def test_compute_breakdown_empty_input_safe():
    """Zero records — every field defined, no division-by-zero."""
    out = runs_mod.compute_detection_breakdown([], frames_processed=0)
    assert out["detections_per_class"] == {}
    assert out["per_frame_count_min"] == 0
    assert out["per_frame_count_max"] == 0
    assert out["per_frame_count_avg"] == 0.0
    assert out["per_frame_count_histogram"] == {}


def test_compute_breakdown_handles_missing_class_name():
    """If class_name is missing on a det, fall back to class_id-derived name."""
    out = runs_mod.compute_detection_breakdown(
        [{"frame_idx": 0, "detections": [{"class_id": 7, "score": 0.5}]}],
        frames_processed=1,
    )
    assert "class_7" in out["detections_per_class"]


def test_recompute_preserves_timing_and_writes_breakdown(tmp_path: Path):
    """recompute() should not clobber timing fields from an existing
    stats.json; it should *add* the breakdown."""
    # Build a fake run dir
    rdir = tmp_path / "teacher_19700101-000000_test"
    (rdir / "labels").mkdir(parents=True)
    # Manifest is required by no read path here — we write only what
    # recompute touches, since recompute reads/writes stats.json + JSONL.
    (rdir / "labels" / runs_mod.PER_FRAME_NAME).write_text(
        "\n".join(json.dumps(r) for r in [
            {"frame_idx": 0, "detections": [{"class_name": "ball", "score": 0.5}]},
            {"frame_idx": 1, "detections": []},
            {"frame_idx": 2, "detections": [
                {"class_name": "ball", "score": 0.6},
                {"class_name": "ball", "score": 0.4},
            ]},
        ])
    )
    # Pre-existing timing stats — must survive recompute
    pre = runs_mod.RunStats(
        frames_processed=3,
        frames_with_detections=2,
        total_ms=999.0,
        avg_ms_per_frame=333.0,
        p50_ms_per_frame=320.0,
        p95_ms_per_frame=400.0,
        n_detections_total=3,
    )
    runs_mod.write_stats(rdir, pre)

    new = recompute(rdir.name, runs_root=tmp_path)

    # Timing fields untouched
    assert new.total_ms == 999.0
    assert new.avg_ms_per_frame == 333.0
    assert new.p50_ms_per_frame == 320.0

    # Breakdown filled in correctly
    assert new.detections_per_class["ball"]["n_detections"] == 3
    assert new.detections_per_class["ball"]["frames_present"] == 2
    assert new.per_frame_count_max == 2
    assert new.per_frame_count_min == 0


def test_recompute_works_without_existing_stats(tmp_path: Path):
    """Recompute should still succeed when stats.json doesn't exist —
    timing fields land at zero, breakdown is computed from JSONL."""
    rdir = tmp_path / "teacher_19700101-000001_nostats"
    (rdir / "labels").mkdir(parents=True)
    (rdir / "labels" / runs_mod.PER_FRAME_NAME).write_text(
        json.dumps({"frame_idx": 0, "detections": [
            {"class_name": "x", "score": 0.5}
        ]}) + "\n"
    )

    new = recompute(rdir.name, runs_root=tmp_path)
    assert new.frames_processed == 1
    assert new.n_detections_total == 1
    assert new.detections_per_class["x"]["n_detections"] == 1
    # Timing fields safely zero
    assert new.avg_ms_per_frame == 0.0
