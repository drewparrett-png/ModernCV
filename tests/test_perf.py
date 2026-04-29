"""Phase 6: pipeline.perf module — diagnostics + helper coverage.

Real RSS / MPS numbers are platform-dependent and out of scope for unit
tests; here we only assert that the helpers don't crash on this box and
that `append_diagnostics` round-trips through the on-disk JSONL file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import perf


def test_current_rss_mb_returns_a_number() -> None:
    """psutil is in pyproject so the value should always be present."""
    rss = perf.current_rss_mb()
    assert rss is None or rss > 0


def test_mps_allocator_stats_keys() -> None:
    stats = perf.mps_allocator_stats()
    # Always returns the two keys regardless of device.
    assert set(stats.keys()) == {"current_mb", "driver_mb"}


def test_empty_mps_cache_is_safe_on_non_mps_devices() -> None:
    # No assertion — just guarantees the call doesn't raise on CI / CPU.
    perf.empty_mps_cache()


def test_append_diagnostics_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records are appended to the JSONL file under the project root."""
    monkeypatch.chdir(tmp_path)
    perf.append_diagnostics(
        {"kind": "run_start", "run_id": "x", "rss_mb": 100.5}
    )
    perf.append_diagnostics(
        {"kind": "run_end", "run_id": "x", "rss_mb": 110.0}
    )
    p = tmp_path / "runs" / ".diagnostics" / "learn_perf.jsonl"
    assert p.exists()
    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["kind"] == "run_start"
    assert lines[1]["kind"] == "run_end"
    # Every record auto-stamps a timestamp so post-hoc analysis can
    # correlate against wall clock.
    assert "ts" in lines[0]
    assert "ts" in lines[1]


def test_append_diagnostics_swallows_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostics-write failure must not crash the caller.

    We point the file at a path whose parent already exists as a file,
    so `mkdir(exist_ok=True)` raises NotADirectoryError on append.
    """
    monkeypatch.chdir(tmp_path)
    # Pre-create `runs/.diagnostics` as a file so mkdir collides.
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / ".diagnostics").write_text("not a dir")
    perf.append_diagnostics({"kind": "run_start"})  # must not raise
