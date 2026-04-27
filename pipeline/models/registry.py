"""Two-layer registry.

`REGISTRY` advertises *what we plan to support* — drives the dropdowns in
the GUI. Stable; rarely changes.

`ADAPTERS` records *what is actually wired right now* — populated as we
implement each model. Adapters register themselves via `@register(...)`.

Keeping them separate lets the GUI advertise the full vision while pipelines
gracefully pass through unimplemented blocks during the build-up phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

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

if TYPE_CHECKING:
    from pipeline.models.adapters import Adapter, SourceAdapter

# ---- advertised impls (drives /blocks → GUI dropdowns) ---------------------

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


# ---- runtime adapters (populated by @register) -----------------------------

ADAPTERS: dict[tuple[BlockKind, str], type] = {}

T = TypeVar("T")


def register(kind: BlockKind, impl: str) -> Callable[[type[T]], type[T]]:
    """Decorator: associate an Adapter/SourceAdapter class with (kind, impl)."""

    def deco(cls: type[T]) -> type[T]:
        key = (kind, impl)
        if key in ADAPTERS:
            raise ValueError(f"adapter already registered for {key}: {ADAPTERS[key]}")
        ADAPTERS[key] = cls
        return cls

    return deco


def make_adapter(kind: BlockKind, impl: str, params: dict):
    """Instantiate an adapter for (kind, impl), or return None if unwired."""
    cls = ADAPTERS.get((kind, impl))
    if cls is None:
        return None
    return cls(params)


def is_wired(kind: BlockKind, impl: str) -> bool:
    return (kind, impl) in ADAPTERS
