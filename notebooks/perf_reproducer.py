"""Phase 6 perf reproducer.

Queues a handful of short Teacher runs back-to-back against a known clip,
then loads `runs/.diagnostics/learn_perf.jsonl` and asserts that the per-
frame latency in the *last* run is within 15% of the *first* run.

This is NOT a unit test (real models + real video are too slow). It's a
runnable script you invoke locally after a fix:

    python notebooks/perf_reproducer.py path/to/clip.mp4

Output is a small per-run summary table plus a single PASS/FAIL line.
The diagnostics file is *not* cleared between invocations — if you want
a clean baseline, delete it manually first:

    rm runs/.diagnostics/learn_perf.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow this script to be run as `python notebooks/perf_reproducer.py`
# from the repo root without installing the package.
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline import perf, runs as runs_mod
from pipeline.learn import (
    ensure_learn_worker_started,
    run_learn_in_background,
    stop_learn_worker,
)


N_RUNS_DEFAULT = 5
DRIFT_TOLERANCE = 0.15  # 15% — first run vs last run


def _wait_for_completion(
    project_id: str, run_id: str, timeout_s: float = 600.0
) -> str:
    """Block until the manifest's status flips to completed/failed.

    Polls every second. Returns the final status. Raises on timeout —
    a 10-minute cap keeps a runaway run from blocking the script forever.
    """
    rdir = runs_mod.run_dir(project_id, run_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            m = runs_mod.read_manifest(rdir)
        except Exception:
            time.sleep(1.0)
            continue
        if m.status in ("completed", "failed"):
            return m.status
        time.sleep(1.0)
    raise TimeoutError(f"run {run_id} did not finish within {timeout_s}s")


def _load_diagnostics() -> list[dict]:
    p = perf.DIAGNOSTICS_DIR / perf.LEARN_PERF_FILE
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _summarize(records: list[dict], run_ids: list[str]) -> list[dict]:
    """For each run id, report its first and last `avg_ms_last_100`.

    First-vs-last comparison uses the *first* sample (warm-up burned in
    by the time the 100th frame finishes) so we're not penalised for
    model-load latency. Last sample reflects the per-frame steady state
    just before the run ended.
    """
    summary = []
    for rid in run_ids:
        windows = [
            r for r in records
            if r.get("kind") == "frame_window" and r.get("run_id") == rid
        ]
        if not windows:
            continue
        first = float(windows[0]["avg_ms_last_100"])
        last = float(windows[-1]["avg_ms_last_100"])
        rss_start = next(
            (r["rss_mb"] for r in records
             if r.get("kind") == "run_start" and r.get("run_id") == rid),
            None,
        )
        rss_end = next(
            (r["rss_mb"] for r in records
             if r.get("kind") == "run_end" and r.get("run_id") == rid),
            None,
        )
        summary.append(
            {
                "run_id": rid,
                "n_windows": len(windows),
                "first_ms": first,
                "last_ms": last,
                "rss_start_mb": rss_start,
                "rss_end_mb": rss_end,
            }
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=str, help="Path to a short test clip")
    ap.add_argument(
        "--n", type=int, default=N_RUNS_DEFAULT,
        help=f"Number of runs to queue (default {N_RUNS_DEFAULT})",
    )
    ap.add_argument(
        "--max-frames", type=int, default=300,
        help="Cap each run at N frames (default 300 — three windows per run)",
    )
    ap.add_argument(
        "--project-name", type=str, default="phase6-perf-repro",
    )
    args = ap.parse_args()

    if not Path(args.video).exists():
        ap.error(f"video not found: {args.video}")

    # One throwaway project per invocation. We don't delete it so the
    # user can poke at the diagnostics file by hand afterwards.
    project = runs_mod.create_project(
        name=args.project_name,
        task="detection",
        prompts=["soccer ball"],
    )
    print(f"created project {project.id}")

    ensure_learn_worker_started()
    run_ids: list[str] = []
    try:
        for i in range(args.n):
            m = run_learn_in_background(
                project_id=project.id,
                task="detection",
                video_path=args.video,
                prompt="soccer ball",
                max_frames=args.max_frames,
            )
            run_ids.append(m.id)
            print(f"queued run {i + 1}/{args.n}: {m.id}")

        for rid in run_ids:
            status = _wait_for_completion(project.id, rid)
            print(f"  {rid} → {status}")
    finally:
        stop_learn_worker()

    records = _load_diagnostics()
    summary = _summarize(records, run_ids)

    print("\n--- Per-run latency summary ---")
    print(f"{'run_id':50s}  {'first_ms':>9s}  {'last_ms':>9s}  {'rss_start':>9s}  {'rss_end':>9s}")
    for s in summary:
        print(
            f"{s['run_id']:50s}  {s['first_ms']:9.1f}  {s['last_ms']:9.1f}  "
            f"{s['rss_start_mb'] or 0:9.1f}  {s['rss_end_mb'] or 0:9.1f}"
        )

    if len(summary) < 2:
        print("\n[WARN] not enough samples to compare — need at least 2 runs.")
        return 2

    first_ms = summary[0]["first_ms"]
    last_ms = summary[-1]["last_ms"]
    ratio = last_ms / max(first_ms, 1e-6)
    print(f"\nlast / first per-frame ms ratio: {ratio:.3f}")
    if ratio <= 1.0 + DRIFT_TOLERANCE:
        print(f"PASS — drift within {DRIFT_TOLERANCE * 100:.0f}% tolerance")
        return 0
    print(f"FAIL — drift exceeds {DRIFT_TOLERANCE * 100:.0f}% tolerance")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
