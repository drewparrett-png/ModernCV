"""Pipeline blocks.

Each block is a stage in the video processing graph. A block declares the
"kind" of work it does (Input, Detect, Segment, ReID, Track, Stats, Output)
and one or more implementations (e.g. Detect can be YOLOv8 or RT-DETR).

The GUI surfaces blocks as nodes; the user picks an implementation per node
from a dropdown. The runner walks the graph and dispatches each node to the
chosen implementation.
"""

from pipeline.blocks.base import Block, BlockKind, FrameBatch
from pipeline.blocks.detect import DetectBlock
from pipeline.blocks.input import InputBlock
from pipeline.blocks.output import OutputBlock
from pipeline.blocks.reid import ReIDBlock
from pipeline.blocks.segment import SegmentBlock
from pipeline.blocks.stats import StatsBlock
from pipeline.blocks.track import TrackBlock

__all__ = [
    "Block",
    "BlockKind",
    "FrameBatch",
    "InputBlock",
    "DetectBlock",
    "SegmentBlock",
    "ReIDBlock",
    "TrackBlock",
    "StatsBlock",
    "OutputBlock",
]
