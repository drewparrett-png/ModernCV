"""Registry of available implementations per block kind.

Drives the dropdowns in the GUI. Keep in sync with the AVAILABLE_IMPLS
constants in each block module — or, better, import them directly.
"""

from __future__ import annotations

from pipeline.blocks import (
    detect as detect_block,
    input as input_block,
    output as output_block,
    reid as reid_block,
    segment as segment_block,
    stats as stats_block,
    track as track_block,
)
from pipeline.blocks.base import BlockKind

REGISTRY: dict[BlockKind, list[str]] = {
    BlockKind.INPUT: input_block.AVAILABLE_IMPLS,
    BlockKind.DETECT: detect_block.AVAILABLE_IMPLS,
    BlockKind.SEGMENT: segment_block.AVAILABLE_IMPLS,
    BlockKind.REID: reid_block.AVAILABLE_IMPLS,
    BlockKind.TRACK: track_block.AVAILABLE_IMPLS,
    BlockKind.STATS: stats_block.AVAILABLE_IMPLS,
    BlockKind.OUTPUT: output_block.AVAILABLE_IMPLS,
}


def list_all() -> dict[str, list[str]]:
    return {k.value: v for k, v in REGISTRY.items()}
