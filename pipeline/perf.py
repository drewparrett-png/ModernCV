"""Process + device memory probes (Phase 6).

Three callers in the codebase want to ask "how much memory are we using?"
without each one re-implementing the platform branches:

  • `pipeline.learn._execute_queued_job` logs RSS + MPS deltas around
    every Teacher run so the diagnostics file shows growth across the
    queue.
  • `pipeline.runner.run` emits an `avg_ms_last_100` sample so the
    reproducer can plot per-frame latency drift.
  • `notebooks/perf_reproducer.py` reads what those two write and asserts
    a flat-ness ratio between the first and last run.

`psutil` is a soft dependency: if it isn't installed the helpers return
`None`/`-1.0` rather than raising, so unit tests don't grow a new install
requirement just to import this module. `torch.mps` is similarly soft —
on CUDA or CPU boxes the MPS calls are no-ops.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


DIAGNOSTICS_DIR = Path("runs") / ".diagnostics"
LEARN_PERF_FILE = "learn_perf.jsonl"


def current_rss_mb() -> Optional[float]:
    """Resident-set size of the current process in megabytes.

    Returns None when `psutil` isn't importable so callers can render a
    blank field rather than crashing. We probe the *current* process,
    not the worker thread — RSS is process-level on every OS we ship on.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover — psutil is in pyproject
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("rss probe failed: %s", e)
        return None


def mps_allocator_stats() -> dict[str, Optional[float]]:
    """`torch.mps` allocator counters in megabytes when MPS is available.

    Returns a dict with `current_mb` and `driver_mb` keys, both `None`
    when MPS isn't the device. The two figures track different bookkeeping
    levels (Python-side allocations vs. driver-side reserve); growth in
    the driver number across runs is what we'd expect to see if the
    allocator never gets emptied.
    """
    out: dict[str, Optional[float]] = {"current_mb": None, "driver_mb": None}
    try:
        import torch
    except ImportError:  # pragma: no cover
        return out
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return out
    try:
        cur = torch.mps.current_allocated_memory()
        out["current_mb"] = float(cur) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        drv = torch.mps.driver_allocated_memory()
        out["driver_mb"] = float(drv) / (1024.0 * 1024.0)
    except Exception:
        pass
    return out


def now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _learn_perf_path() -> Path:
    return DIAGNOSTICS_DIR / LEARN_PERF_FILE


def append_diagnostics(record: dict[str, Any]) -> None:
    """Append one JSON record to `runs/.diagnostics/learn_perf.jsonl`.

    Best-effort: a write failure (read-only fs, permission denied, full
    disk) logs a warning and returns silently. The diagnostics file is
    a *signal*, not a contract — the user's run shouldn't fail because
    we couldn't write to it.
    """
    record.setdefault("ts", now_iso())
    p = _learn_perf_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fp:
            fp.write(json.dumps(record) + "\n")
    except OSError as e:
        log.warning("failed to write %s: %s", p, e)


def empty_mps_cache() -> None:
    """Idempotent best-effort `torch.mps.empty_cache()` call.

    Phase 6 fix: between Teacher runs the allocator can hold gigabytes of
    cached buffers from GroundingDINO's intermediate tensors. Emptying
    after each run is what stops per-frame latency from drifting up
    across the queue.
    """
    try:
        import torch
    except ImportError:
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return
    try:
        torch.mps.empty_cache()
    except Exception as e:  # pragma: no cover — defensive
        log.warning("mps.empty_cache() failed: %s", e)
