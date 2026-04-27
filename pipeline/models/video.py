"""Video I/O adapters.

Currently:
    Input  / "opencv"      — OpenCVVideoReader
    Output / "overlay-mp4" — OverlayMp4Writer

The writer also draws overlays (boxes, masks, IDs) when those fields are
populated on the FrameBatch. While middle blocks are still stubs, the input
frame passes through untouched and the output mp4 is visually identical to
the input — which is exactly the proof we want at this milestone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from pipeline.blocks.base import BlockKind, FrameBatch
from pipeline.models.adapters import Adapter, SourceAdapter
from pipeline.models.registry import register

log = logging.getLogger(__name__)


# ---- Input: OpenCV video reader -------------------------------------------


@register(BlockKind.INPUT, "opencv")
class OpenCVVideoReader(SourceAdapter):
    """Reads a video file with cv2.VideoCapture and yields FrameBatches.

    Params:
        path: str — path to the video file (relative to cwd or absolute).
        max_frames: int | None — optional cap for fast iteration during dev.
    """

    def setup(self) -> None:
        path = self.params.get("path")
        if not path:
            raise ValueError("OpenCVVideoReader requires params.path")
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.max_frames: Optional[int] = self.params.get("max_frames")
        log.info(
            "OpenCVVideoReader: %s — %.2f fps, %d frames",
            path,
            self.fps,
            self.frame_count,
        )

    def frames(self) -> Iterator[FrameBatch]:
        idx = 0
        while True:
            if self.max_frames is not None and idx >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            yield FrameBatch(
                frame_index=idx,
                image=frame,  # BGR HxWx3 uint8
                metadata={"fps": self.fps, "source_path": self.path},
            )
            idx += 1

    def teardown(self) -> None:
        if hasattr(self, "cap"):
            self.cap.release()


# ---- Output: mp4 writer with optional overlay drawing ----------------------


@register(BlockKind.OUTPUT, "overlay-mp4")
class OverlayMp4Writer(Adapter):
    """Encodes the (possibly overlaid) frames to an mp4.

    Params:
        path: str — output path (default "runs/out.mp4").
        draw_boxes: bool — draw detection bboxes if present (default True).
        draw_masks: bool — draw segmentation masks if present (default True).
        draw_tracks: bool — draw track IDs if present (default True).
    """

    def setup(self) -> None:
        self.path = self.params.get("path", "runs/out.mp4")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.draw_boxes = self.params.get("draw_boxes", True)
        self.draw_masks = self.params.get("draw_masks", True)
        self.draw_tracks = self.params.get("draw_tracks", True)
        # cv2.VideoWriter wants size up-front. We don't know dims until the
        # first frame, so create lazily inside process().
        self.writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

    def _ensure_writer(self, frame: np.ndarray, fps: float) -> None:
        if self.writer is not None:
            return
        h, w = frame.shape[:2]
        # mp4v is the most portable codec across platforms; quality is fine
        # for inspection. We can swap to H.264/avc1 later if needed.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, float(fps), (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {self.path}")
        log.info("OverlayMp4Writer: %s @ %.2f fps, %dx%d", self.path, fps, w, h)

    def process(self, batch: FrameBatch) -> FrameBatch:
        frame = batch.image
        fps = batch.metadata.get("fps", 30.0)
        self._ensure_writer(frame, fps)

        if self.draw_masks and batch.masks:
            frame = _draw_masks(frame, batch.masks)
        if self.draw_boxes and batch.detections:
            frame = _draw_detections(frame, batch.detections)
        if self.draw_tracks and batch.tracks:
            frame = _draw_tracks(frame, batch.tracks)

        self.writer.write(frame)
        self._frames_written += 1
        return batch

    def teardown(self) -> None:
        if self.writer is not None:
            self.writer.release()
            log.info("OverlayMp4Writer: wrote %d frames to %s", self._frames_written, self.path)


# ---- drawing helpers -------------------------------------------------------


def _draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.score:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def _draw_tracks(frame: np.ndarray, tracks) -> np.ndarray:
    img = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            img,
            f"#{t.track_id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )
    return img


def _draw_masks(frame: np.ndarray, masks) -> np.ndarray:
    img = frame.copy()
    overlay = np.zeros_like(img)
    for i, m in enumerate(masks):
        if m is None:
            continue
        # Cycle a small palette.
        colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 0)]
        color = colors[i % len(colors)]
        overlay[m.astype(bool)] = color
    return cv2.addWeighted(img, 1.0, overlay, 0.4, 0.0)
