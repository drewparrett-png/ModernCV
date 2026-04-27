"""Input block — reads frames from a video file.

Implementations:
    "opencv"   — cv2.VideoCapture, simplest path
    "ffmpeg"   — ffmpeg-python for codec-troublesome files

Sources are special: they produce a stream of FrameBatches rather than
transforming one. So InputBlock exposes `frames()` instead of `process()`,
and delegates to a SourceAdapter.

An unwired source is fatal — you can't pass-through "no frames."
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from pipeline.blocks.base import Block, BlockKind, FrameBatch

if TYPE_CHECKING:
    from pipeline.models.adapters import SourceAdapter

log = logging.getLogger(__name__)


class InputBlock(Block):
    kind: BlockKind = BlockKind.INPUT

    def __init__(self, node_id: str, impl: str = "opencv", params: dict | None = None):
        super().__init__(
            kind=BlockKind.INPUT,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )

    def setup(self) -> None:
        # Override: sources use SourceAdapter, not Adapter.
        from pipeline.models.registry import make_adapter

        self._source: Optional["SourceAdapter"] = make_adapter(
            self.kind, self.impl, self.params
        )
        if self._source is None:
            raise RuntimeError(
                f"InputBlock(impl={self.impl!r}) has no registered SourceAdapter — "
                "wire one in pipeline/models/."
            )
        self._source.setup()

    def frames(self) -> Iterator[FrameBatch]:
        return self._source.frames()

    def process(self, batch: FrameBatch) -> FrameBatch:  # pragma: no cover
        # Sources don't transform.
        return batch

    def teardown(self) -> None:
        if getattr(self, "_source", None) is not None:
            self._source.teardown()


AVAILABLE_IMPLS = ["opencv", "ffmpeg"]
