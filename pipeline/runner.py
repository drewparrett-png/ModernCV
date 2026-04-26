"""Graph runner.

Walks a linear GraphSpec, instantiates one Block per node, then iterates
frames from the Input block and threads each FrameBatch through the chain.
This module is intentionally tiny — the work lives in the blocks and the
model adapters.
"""

from __future__ import annotations

from typing import Iterator

from pipeline.blocks import (
    Block,
    DetectBlock,
    InputBlock,
    OutputBlock,
    ReIDBlock,
    SegmentBlock,
    StatsBlock,
    TrackBlock,
)
from pipeline.blocks.base import BlockKind, FrameBatch
from pipeline.graph import GraphSpec, NodeSpec

_BLOCK_FOR_KIND: dict[BlockKind, type[Block]] = {
    BlockKind.INPUT: InputBlock,
    BlockKind.DETECT: DetectBlock,
    BlockKind.SEGMENT: SegmentBlock,
    BlockKind.REID: ReIDBlock,
    BlockKind.TRACK: TrackBlock,
    BlockKind.STATS: StatsBlock,
    BlockKind.OUTPUT: OutputBlock,
}


def build_block(spec: NodeSpec) -> Block:
    cls = _BLOCK_FOR_KIND[spec.kind]
    return cls(node_id=spec.id, impl=spec.impl, params=spec.params)


def run(graph: GraphSpec) -> Iterator[FrameBatch]:
    """Execute the graph. Yields one FrameBatch per processed frame.

    The caller can drive this from a CLI, a FastAPI streaming response, or a
    websocket pushing live previews to the GUI.
    """
    order = graph.linearize()
    if order[0].kind != BlockKind.INPUT:
        raise ValueError("first node must be Input")
    if order[-1].kind != BlockKind.OUTPUT:
        raise ValueError("last node must be Output")

    blocks = [build_block(n) for n in order]
    for b in blocks:
        b.setup()

    try:
        input_block: InputBlock = blocks[0]  # type: ignore[assignment]
        middle = blocks[1:-1]
        output_block = blocks[-1]
        for batch in input_block.frames():
            for b in middle:
                batch = b.process(batch)
            batch = output_block.process(batch)
            yield batch
    finally:
        for b in blocks:
            b.teardown()
