"""Backfill detection-breakdown stats on existing Teacher runs.

Usage
-----
    # one run
    python -m pipeline.recompute_stats teacher_20260428-124802_soccer-ball-players

    # every teacher under runs/
    python -m pipeline.recompute_stats --all

    # also print the breakdown to stdout (handy for "step through to
    # understand"-style inspection)
    python -m pipeline.recompute_stats <run_id> --print

What it does
------------
Reads `labels/per_frame.jsonl` from the run dir and rewrites `stats.json` with
the new fields (`detections_per_class`, `per_frame_count_*`,
`per_frame_count_histogram`). Existing timing fields are preserved as-is —
we don't have ms/frame in the JSONL so we trust whatever is already in
`stats.json`. If `stats.json` is missing, timing fields fall back to zeros.

This is the way to retro-fit older runs that were created before the
breakdown stats existed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from . import runs as runs_mod

log = logging.getLogger(__name__)


def recompute(run_id: str, runs_root: Path = runs_mod.RUNS_DIR) -> runs_mod.RunStats:
    """Rebuild `stats.json` for one run. Returns the new RunStats."""
    rdir = runs_mod.run_dir(run_id, runs_root)
    if not rdir.exists():
        raise FileNotFoundError(f"run dir does not exist: {rdir}")

    # Preserve timing fields (we can't recompute them — JSONL has no ms).
    existing = runs_mod.read_stats(rdir)
    timing_kwargs: dict = {}
    if existing is not None:
        timing_kwargs = {
            "frames_processed": existing.frames_processed,
            "frames_with_detections": existing.frames_with_detections,
            "total_ms": existing.total_ms,
            "avg_ms_per_frame": existing.avg_ms_per_frame,
            "p50_ms_per_frame": existing.p50_ms_per_frame,
            "p95_ms_per_frame": existing.p95_ms_per_frame,
            "n_detections_total": existing.n_detections_total,
        }

    records = list(runs_mod.read_per_frame(rdir))

    # If the existing stats are missing or stale, recompute the count fields
    # from the JSONL itself so they at least agree with the breakdown.
    if not timing_kwargs or timing_kwargs.get("frames_processed", 0) == 0:
        n_frames = len(records)
        n_dets = sum(len(r.get("detections", [])) for r in records)
        n_with = sum(1 for r in records if r.get("detections"))
        timing_kwargs = {
            "frames_processed": n_frames,
            "frames_with_detections": n_with,
            "total_ms": timing_kwargs.get("total_ms", 0.0),
            "avg_ms_per_frame": timing_kwargs.get("avg_ms_per_frame", 0.0),
            "p50_ms_per_frame": timing_kwargs.get("p50_ms_per_frame", 0.0),
            "p95_ms_per_frame": timing_kwargs.get("p95_ms_per_frame", 0.0),
            "n_detections_total": n_dets,
        }

    breakdown = runs_mod.compute_detection_breakdown(
        records, frames_processed=timing_kwargs["frames_processed"]
    )
    stats = runs_mod.RunStats(**timing_kwargs, **breakdown)
    runs_mod.write_stats(rdir, stats)
    return stats


def _print_summary(run_id: str, stats: runs_mod.RunStats) -> None:
    print(f"\n=== {run_id} ===")
    print(
        f"frames: {stats.frames_processed} processed · "
        f"{stats.frames_with_detections} with dets · "
        f"{stats.n_detections_total} total dets"
    )
    print(
        "per-frame count: "
        f"min={stats.per_frame_count_min} "
        f"p50={stats.per_frame_count_p50} "
        f"p95={stats.per_frame_count_p95} "
        f"max={stats.per_frame_count_max} "
        f"avg={stats.per_frame_count_avg:.2f}"
    )
    if stats.per_frame_count_histogram:
        print("histogram (count → frames):")
        for k in sorted(stats.per_frame_count_histogram.keys(), key=int):
            v = stats.per_frame_count_histogram[k]
            bar = "█" * min(40, v)
            print(f"  {k:>3} | {bar} {v}")
    if stats.detections_per_class:
        # Widest class name for alignment
        w = max(len(c) for c in stats.detections_per_class) + 2
        print(f"\n{'class':<{w}}{'n':>6}{'frames':>9}{'max':>5}{'avg/f':>9}{'avg/p':>9}{'score':>8}")
        for cls in sorted(
            stats.detections_per_class,
            key=lambda c: stats.detections_per_class[c]["n_detections"],
            reverse=True,
        ):
            s = stats.detections_per_class[cls]
            print(
                f"{cls:<{w}}{s['n_detections']:>6}"
                f"{s['frames_present']:>9}{s['max_in_frame']:>5}"
                f"{s['avg_per_frame']:>9.2f}"
                f"{s['avg_per_present_frame']:>9.2f}"
                f"{s['score_avg']:>8.2f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline.recompute_stats",
        description="Backfill detection-breakdown stats on Teacher runs.",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("run_id", nargs="?", help="One run id, e.g. teacher_…")
    g.add_argument("--all", action="store_true", help="Recompute every teacher run")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=runs_mod.RUNS_DIR,
        help="Override the runs/ directory (default: ./runs)",
    )
    parser.add_argument(
        "--print",
        dest="print_summary",
        action="store_true",
        help="Print a human-readable breakdown after writing stats.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.all:
        ids = [
            p.name
            for p in args.runs_root.iterdir()
            if p.is_dir() and p.name.startswith("teacher_")
        ]
        if not ids:
            print(f"no teacher runs found under {args.runs_root}", file=sys.stderr)
            return 1
        ids.sort()
    else:
        ids = [args.run_id]

    rc = 0
    for rid in ids:
        try:
            stats = recompute(rid, runs_root=args.runs_root)
            log.info("recomputed stats for %s", rid)
            if args.print_summary:
                _print_summary(rid, stats)
        except Exception as e:
            log.error("FAILED %s: %s", rid, e)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
