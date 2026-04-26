"""Re-ID block — appearance embeddings for each detection.

This is where DINOv3 plugs in. For each detected player crop, we run the
backbone and store the embedding on the Track. The Track block then uses
these embeddings as appearance features when matching across frames.

Implementations:
    "dinov3-vits16"  — small, fast on M-series (recommended)
    "dinov3-vitb16"  — base, slower but stronger features
    "dinov2-vits14"  — older but battle-tested fallback
    "osnet"          — classical Re-ID baseline for comparison
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class ReIDBlock(Block):
    kind: BlockKind = BlockKind.REID

    def __init__(self, node_id: str, impl: str = "dinov3-vits16", params: dict | None = None):
        super().__init__(
            kind=BlockKind.REID,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["dinov3-vits16", "dinov3-vitb16", "dinov2-vits14", "osnet"]
