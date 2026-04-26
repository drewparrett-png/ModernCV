"""Detect block — finds players and the ball in each frame.

Implementations:
    "yolov8n"        — Ultralytics, fastest, good first run on M-series
    "yolov8s"        — slightly more accurate, still real-time on MPS
    "rtdetr"         — DETR-style, stronger on small objects (ball!)
    "groundingdino"  — text-promptable; great for iteration without fine-tune
"""

from __future__ import annotations

from pipeline.blocks.base import Block, BlockKind, FrameBatch


class DetectBlock(Block):
    kind: BlockKind = BlockKind.DETECT

    def __init__(self, node_id: str, impl: str = "yolov8n", params: dict | None = None):
        super().__init__(
            kind=BlockKind.DETECT,
            node_id=node_id,
            impl=impl,
            params=params or {},
        )


AVAILABLE_IMPLS = ["yolov8n", "yolov8s", "yolov11n", "rtdetr", "groundingdino"]
