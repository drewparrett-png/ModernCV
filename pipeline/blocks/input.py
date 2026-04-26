"""Input block — reads frames from a video file.

Implementations:
    "opencv"   — cv2.VideoCapture, simplest path
    "ffmpeg"   — ffmpeg-python for codec-troublesome files
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class InputBlock(Block):
    kind: BlockKind = BlockKind.INPUT

    def __init__(self, node_id: str, impl: str = "opencv", params: dict | None = None):
        super().__init__(
            kind=BlockKind.INPUT,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )

    # The Input block is special: instead of `process(batch)`, the runner asks
    # it for a frame iterator. We'll formalize that interface when wiring the
    # first real implementation.
    def frames(self):
        raise NotImplementedError(
            f"InputBlock(impl={self.impl!r}) — wire video reader in pipeline.models.video"
        )

    def process(self, batch: FrameBatch) -> FrameBatch:  # pragma: no cover
        # InputBlock is a source; runner uses .frames() instead.
        return batch


AVAILABLE_IMPLS = ["opencv", "ffmpeg"]
