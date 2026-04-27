"""Adapter base classes.

An *adapter* is the thing that actually does the work for a given (BlockKind,
impl) pair. The Block is the orchestration shell; the Adapter is the
implementation. Splitting them lets the GUI and runner stay completely
ignorant of model internals — they only know about Blocks.

Two flavors:

- `Adapter`         — for transformer stages (Detect, Segment, ReID, Track,
                      Stats, Output). Takes a FrameBatch, returns a
                      (possibly modified) FrameBatch.

- `SourceAdapter`   — for Input. Produces FrameBatch instances by yielding
                      from `frames()`. Doesn't take a batch as input.

Both have setup/teardown for one-time resource management (open files, load
weights, warm up MPS, etc.).
"""

from __future__ import annotations

from typing import Any, Iterator

from pipeline.blocks.base import FrameBatch


class Adapter:
    """Transformer adapter — read/write a FrameBatch in place."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def setup(self) -> None:
        """One-time init. Load weights, allocate GPU buffers, etc."""

    def process(self, batch: FrameBatch) -> FrameBatch:
        return batch

    def teardown(self) -> None:
        """Release resources."""


class SourceAdapter:
    """Source adapter — produces frames; does not consume a FrameBatch."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def setup(self) -> None:
        """Open the file/stream."""

    def frames(self) -> Iterator[FrameBatch]:
        raise NotImplementedError

    def teardown(self) -> None:
        """Release the file/stream."""
