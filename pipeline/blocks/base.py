"""Block base class and shared types.

A Block is one stage of the pipeline. It receives a FrameBatch (the rolling
context flowing through the graph), reads/writes specific fields on it, and
passes it on. Blocks are intentionally narrow: one job per block.

Implementations live in `pipeline.models.*` and are picked at runtime by name
from a registry. The block holds the "what" (Detect), the model adapter holds
the "how" (YOLOv8). If no adapter is registered for the chosen impl, the
block falls back to pass-through and logs a warning — this is what lets you
compose a partially-built pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from pipeline.models.adapters import Adapter

log = logging.getLogger(__name__)


class BlockKind(str, Enum):
    """High-level role of a block in the graph. Drives the GUI palette."""

    INPUT = "input"
    DETECT = "detect"
    SEGMENT = "segment"
    REID = "reid"
    TRACK = "track"
    STATS = "stats"
    OUTPUT = "output"


@dataclass
class Detection:
    """A single detection on a single frame."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """A persistent identity across frames."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    embedding: Optional[np.ndarray] = None  # DINOv3 re-id features


@dataclass
class FrameBatch:
    """The rolling state object passed between blocks.

    Each block reads what it needs and writes what it produced. Keeping it as
    a single object (rather than typed channels per block) keeps the runner
    simple and lets us iterate without churn.
    """

    frame_index: int
    image: np.ndarray  # HxWx3 uint8 BGR
    detections: list[Detection] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)  # binary HxW per detection
    tracks: list[Track] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    """Abstract base. Subclasses pin `kind` and may override `frames` (sources).

    `node_id`  — graph-level identifier from the GUI.
    `impl`     — name of the model adapter to use (e.g. "yolov8n").
    `params`   — implementation-specific kwargs from the GUI controls.

    The Block looks up an Adapter for (kind, impl) at setup time. If found,
    process() delegates to the adapter. If not found, process() passes the
    batch through unchanged and logs once.
    """

    kind: BlockKind
    node_id: str
    impl: str
    params: dict[str, Any] = field(default_factory=dict)

    def setup(self) -> None:
        # Lazy import to break the blocks ↔ models circular dependency.
        from pipeline.models.registry import make_adapter

        self._adapter: Optional["Adapter"] = make_adapter(self.kind, self.impl, self.params)
        if self._adapter is None:
            log.warning(
                "%s/%s has no registered adapter — passing through.",
                self.kind.value,
                self.impl,
            )
        else:
            self._adapter.setup()

    def process(self, batch: FrameBatch) -> FrameBatch:
        if self._adapter is None:
            return batch
        return self._adapter.process(batch)

    def teardown(self) -> None:
        if getattr(self, "_adapter", None) is not None:
            self._adapter.teardown()
