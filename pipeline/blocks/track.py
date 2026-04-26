"""Track block — assigns persistent IDs across frames.

Consumes detections (and optionally Re-ID embeddings) and emits tracks. The
appearance feature from the Re-ID block is what lets us hold IDs through
occlusions and cross-overs — classical IoU-only trackers swap IDs constantly
on a soccer pitch.

Implementations:
    "bytetrack"   — fast, IoU-based; uses Re-ID embeddings if present
    "botsort"     — IoU + appearance, well-suited to people
    "ocsort"      — observation-centric, robust to occlusions
    "sam2-mask"   — mask propagation from SAM2 (no separate tracker needed)
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class TrackBlock(Block):
    kind: BlockKind = BlockKind.TRACK

    def __init__(self, node_id: str, impl: str = "bytetrack", params: dict | None = None):
        super().__init__(
            kind=BlockKind.TRACK,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["bytetrack", "botsort", "ocsort", "sam2-mask"]
