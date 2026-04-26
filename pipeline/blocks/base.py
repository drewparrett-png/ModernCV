"""Block base class and shared types.

A Block is one stage of the pipeline. It receives a FrameBatch (the rolling
context flowing through the graph), reads/writes specific fields on it, and
passes it on. Blocks are intentionally narrow: one job per block.

Implementations live in `pipeline.models.*` and are picked at runtime by name
from a registry. The block holds the "what" (Detect), the model adapter holds
the "how" (YOLOv8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


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
    """Abstract base. Subclasses override `process`.

    `node_id`  — graph-level identifier from the GUI.
    `impl`     — name of the model adapter to use (e.g. "yolov8n").
    `params`   — implementation-specific kwargs from the GUI controls.
    """

    kind: BlockKind
    node_id: str
    impl: str
    params: dict[str, Any] = field(default_factory=dict)

    def setup(self) -> None:
        """One-time initialization (load weights, warm up MPS, etc.)."""

    def process(self, batch: FrameBatch) -> FrameBatch:
        raise NotImplementedError(
            f"{type(self).__name__} (impl={self.impl!r}) is a stub."
        )

    def teardown(self) -> None:
        """Release resources."""
