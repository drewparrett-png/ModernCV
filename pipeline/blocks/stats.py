"""Stats block — derives metrics from tracks.

For v1: positions, simple per-track distance covered, possession proxy
(closest player to ball). For real distance/speed in meters we need the
calibration block to be wired (pixels → pitch coords); without it, stats
are in pixel-space which is fine for visualization but not for analysis.

Implementations:
    "basic"           — pixel-space positions and distances
    "calibrated"      — meter-space stats (requires homography from Calibrate)
    "team-cluster"    — KMeans on Re-ID embeddings to assign Team A / B / ref
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class StatsBlock(Block):
    kind: BlockKind = BlockKind.STATS

    def __init__(self, node_id: str, impl: str = "basic", params: dict | None = None):
        super().__init__(
            kind=BlockKind.STATS,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["basic", "calibrated", "team-cluster"]
