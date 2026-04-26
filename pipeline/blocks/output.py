"""Output block — writes results.

Implementations:
    "overlay-mp4"  — render boxes/masks/IDs onto frames, write video file
    "json-tracks"  — dump tracks as JSON for downstream analysis
    "preview"      — yield frames to a websocket for live GUI preview
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class OutputBlock(Block):
    kind: BlockKind = BlockKind.OUTPUT

    def __init__(self, node_id: str, impl: str = "overlay-mp4", params: dict | None = None):
        super().__init__(
            kind=BlockKind.OUTPUT,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["overlay-mp4", "json-tracks", "preview"]
