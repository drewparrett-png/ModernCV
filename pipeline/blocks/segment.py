"""Segment block — pixel-level masks for each detection.

Implementations:
    "sam2-tiny"   — Meta SAM2, video-native mask propagation (recommended)
    "sam2-small"  — bigger SAM2, better masks if MPS budget allows
    "mobilesam"   — light fallback, single-frame
    "fastsam"     — YOLO-based, fast but less accurate
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class SegmentBlock(Block):
    kind: BlockKind = BlockKind.SEGMENT

    def __init__(self, node_id: str, impl: str = "sam2-tiny", params: dict | None = None):
        super().__init__(
            kind=BlockKind.SEGMENT,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["sam2-tiny", "sam2-small", "mobilesam", "fastsam"]
