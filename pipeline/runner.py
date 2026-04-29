"""Graph runner.

Walks a linear GraphSpec, instantiates one Block per node, then iterates
frames from the Input block and threads each FrameBatch through the chain.
This module is intentionally tiny — the work lives in the blocks and the
model adapters.

Phase 6: emits per-frame timing diagnostics (`avg_ms_last_100`) and
periodically drops the previous batch's references + runs `gc.collect()`
so MPS-cached intermediate tensors don't accumulate across long runs.
"""

from __future__ import annotations

import gc
import time
from typing import Iterator, Optional

from pipeline import perf
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


# Sample window for the rolling per-frame timing emitted to the
# diagnostics file. Phase 6's reproducer reads this to compare run #1
# to run #5; 100 is fine-grained enough to spot a slow drift without
# spamming the file.
PERF_SAMPLE_WINDOW = 100


def build_block(spec: NodeSpec) -> Block:
    cls = _BLOCK_FOR_KIND[spec.kind]
    return cls(node_id=spec.id, impl=spec.impl, params=spec.params)


def run(graph: GraphSpec, run_id: Optional[str] = None) -> Iterator[FrameBatch]:
    """Execute the graph. Yields one FrameBatch per processed frame.

    The caller can drive this from a CLI, a FastAPI streaming response, or a
    websocket pushing live previews to the GUI.

    `run_id` is optional — when present, every PERF_SAMPLE_WINDOW frames
    a diagnostics record is appended with the current rolling average
    so the Phase 6 reproducer can plot drift across queued runs.
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
        window: list[float] = []
        t_prev = time.perf_counter()
        n_frames = 0
        for batch in input_block.frames():
            for b in middle:
                batch = b.process(batch)
            batch = output_block.process(batch)
            yield batch
            # The for-loop will re-bind `batch` on the next iteration,
            # which drops the previous reference; explicit `del` here
            # releases it slightly earlier (before the next frame is
            # decoded) so MPS-cached intermediate tensors don't pile up
            # for the duration of an input-block read.
            del batch

            t_now = time.perf_counter()
            window.append((t_now - t_prev) * 1000.0)
            t_prev = t_now
            n_frames += 1
            if len(window) >= PERF_SAMPLE_WINDOW:
                if run_id is not None:
                    avg = sum(window) / len(window)
                    perf.append_diagnostics(
                        {
                            "kind": "frame_window",
                            "run_id": run_id,
                            "n_frames": n_frames,
                            "avg_ms_last_100": avg,
                            "rss_mb": perf.current_rss_mb(),
                        }
                    )
                window.clear()
                # gc.collect once per window — once-per-frame would tank
                # latency; once per 100 is amortised away.
                gc.collect()
    finally:
        for b in blocks:
            b.teardown()
