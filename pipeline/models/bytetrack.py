"""ByteTrack adapter — multi-object tracking from detections.

A small in-tree implementation of ByteTrack (Zhang et al., ECCV 2022). The
algorithm is a thin layer over a constant-velocity Kalman filter plus IoU-
based association, so we vendor it here rather than pull a heavy third-party
dependency. Total surface is ~200 lines and matches the spirit of the paper.

Pipeline
--------
1. Predict each existing track's box with its Kalman filter.
2. Split incoming detections by score:
       high_dets : score ≥ track_thresh        (default 0.6)
       low_dets  : track_thresh > score ≥ low  (default 0.1)
3. *First association*: high_dets ↔ tracked tracks via IoU + Hungarian.
4. *Second association*: leftover tracked tracks ↔ low_dets — recovers
   barely-detected objects.
5. Unmatched high_dets become new tentative tracks (confirmed after
   `min_hits` consecutive frames).
6. Tracks unmatched for `max_age` frames are dropped.

What we *don't* do
------------------
- Camera motion compensation (BoT-SORT does this; for fixed-camera soccer
  footage the gain is marginal).
- Appearance features. The block consumes whatever embeddings the upstream
  Re-ID block stamped on the FrameBatch (none in v1) — when DinoV3 is
  wired we'll add a similarity term to the cost matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from pipeline.blocks.base import BlockKind, Detection, FrameBatch, Track
from pipeline.models.adapters import Adapter
from pipeline.models.registry import register

log = logging.getLogger(__name__)


# ---- box helpers -----------------------------------------------------------


def _xyxy_to_xyah(box: tuple[float, float, float, float]) -> np.ndarray:
    """Convert (x1,y1,x2,y2) → (cx, cy, aspect, h). Kalman state form."""
    x1, y1, x2, y2 = box
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w / h, h], dtype=np.float64)


def _xyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, a, h = state[:4]
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---- track ----------------------------------------------------------------


@dataclass
class _BTTrack:
    """Internal tracking record. Distinct from `pipeline.blocks.base.Track`,
    which is the externally-facing dataclass we emit per frame."""

    track_id: int
    class_id: int
    class_name: str
    kf: KalmanFilter
    age: int = 0  # frames since first seen
    hits: int = 1  # number of frames matched to a detection
    time_since_update: int = 0  # frames since last successful association
    confirmed: bool = False
    last_score: float = 1.0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.last_bbox = _xyah_to_xyxy(self.kf.x.flatten())

    def update_with(self, det: Detection) -> None:
        z = _xyxy_to_xyah(det.bbox_xyxy)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        self.last_score = det.score
        self.last_bbox = _xyah_to_xyxy(self.kf.x.flatten())


def _make_kf(initial_box: tuple[float, float, float, float]) -> KalmanFilter:
    """Constant-velocity Kalman in (cx, cy, aspect, h) state-space.

    State vector: [cx, cy, a, h, vcx, vcy, va, vh]. Measurement is the
    first four — position only. Aligns with the SORT/ByteTrack convention.
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1.0
    kf.F = np.eye(8)
    for i in range(4):
        kf.F[i, i + 4] = dt
    kf.H = np.zeros((4, 8))
    for i in range(4):
        kf.H[i, i] = 1.0
    # Observation noise — small for position, larger for aspect/height.
    kf.R[2:, 2:] *= 10.0
    # Process noise — let velocities drift a little.
    kf.P *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    z0 = _xyxy_to_xyah(initial_box)
    kf.x[:4, 0] = z0
    kf.x[4:, 0] = 0.0
    return kf


# ---- association ----------------------------------------------------------


def _associate(
    tracks: list[_BTTrack],
    detections: list[Detection],
    iou_thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian-match tracks↔detections by IoU.

    Returns (matches, unmatched_tracks, unmatched_dets), where matches is a
    list of (track_idx, det_idx). Pairs with IoU below `iou_thresh` are
    rejected after assignment so the Hungarian is allowed to find the
    globally-best partial assignment first.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            cost[ti, di] = 1.0 - _iou(t.last_bbox, d.bbox_xyxy)

    row, col = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for ti, di in zip(row.tolist(), col.tolist()):
        if 1.0 - cost[ti, di] >= iou_thresh:
            matches.append((ti, di))
            matched_tracks.add(ti)
            matched_dets.add(di)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


# ---- adapter --------------------------------------------------------------


@register(BlockKind.TRACK, "bytetrack")
class ByteTrackAdapter(Adapter):
    """Maintain stable IDs across a stream of FrameBatches.

    Params
    ------
    track_thresh : float = 0.6  — high-score gate for first association
    low_thresh   : float = 0.1  — keep low-score dets for second pass
    iou_thresh   : float = 0.3  — minimum IoU to accept a match
    max_age      : int   = 30   — drop tracks with no update for this many frames
    min_hits     : int   = 3    — frames before a tentative track is "confirmed"
    """

    def setup(self) -> None:
        self.track_thresh: float = float(self.params.get("track_thresh", 0.6))
        self.low_thresh: float = float(self.params.get("low_thresh", 0.1))
        self.iou_thresh: float = float(self.params.get("iou_thresh", 0.3))
        self.max_age: int = int(self.params.get("max_age", 30))
        self.min_hits: int = int(self.params.get("min_hits", 3))
        self._tracks: list[_BTTrack] = []
        self._next_id: int = 1

    def _new_track(self, det: Detection) -> _BTTrack:
        kf = _make_kf(det.bbox_xyxy)
        t = _BTTrack(
            track_id=self._next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            kf=kf,
            last_score=det.score,
            last_bbox=det.bbox_xyxy,
        )
        # When min_hits is 1 (e.g. single ball, you want it tracked instantly),
        # confirm immediately so the first frame already shows a track.
        if self.min_hits <= 1:
            t.confirmed = True
        self._next_id += 1
        return t

    def process(self, batch: FrameBatch) -> FrameBatch:
        # 1. predict
        for t in self._tracks:
            t.predict()

        dets = batch.detections
        high = [d for d in dets if d.score >= self.track_thresh]
        low = [
            d for d in dets if self.low_thresh <= d.score < self.track_thresh
        ]

        # 2. first association: high-score dets ↔ all tracks
        matches, unmatched_tracks, unmatched_high = _associate(
            self._tracks, high, self.iou_thresh
        )
        for ti, di in matches:
            self._tracks[ti].update_with(high[di])
            if (
                not self._tracks[ti].confirmed
                and self._tracks[ti].hits >= self.min_hits
            ):
                self._tracks[ti].confirmed = True

        # 3. second association: leftover tracks ↔ low-score dets
        leftover_tracks = [self._tracks[i] for i in unmatched_tracks]
        matches2, still_unmatched_tracks_local, _ = _associate(
            leftover_tracks, low, self.iou_thresh
        )
        for li, di in matches2:
            leftover_tracks[li].update_with(low[di])
            if (
                not leftover_tracks[li].confirmed
                and leftover_tracks[li].hits >= self.min_hits
            ):
                leftover_tracks[li].confirmed = True

        # 4. spawn new tentative tracks from unmatched high-score detections
        for di in unmatched_high:
            self._tracks.append(self._new_track(high[di]))

        # 5. cull stale tracks
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # 6. emit confirmed tracks for this frame
        out_tracks: list[Track] = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            if t.time_since_update > 0:
                # Track had no update this frame — skip emitting so the
                # overlay doesn't ghost-draw a phantom box. We still keep
                # the internal record so it can re-anchor next frame.
                continue
            out_tracks.append(
                Track(
                    track_id=t.track_id,
                    bbox_xyxy=t.last_bbox,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    embedding=None,
                )
            )
        batch.tracks = out_tracks
        return batch

    def teardown(self) -> None:
        self._tracks = []


# Tiny helper for tests / scripts: run the tracker over a sequence of det
# lists and return the track lists. Lets us validate the adapter without
# spinning up the full pipeline.
def run_offline(
    det_sequence: Iterable[list[Detection]],
    *,
    track_thresh: float = 0.6,
    iou_thresh: float = 0.3,
    min_hits: int = 1,
    max_age: int = 30,
) -> list[list[Track]]:
    a = ByteTrackAdapter(
        params={
            "track_thresh": track_thresh,
            "iou_thresh": iou_thresh,
            "min_hits": min_hits,
            "max_age": max_age,
        }
    )
    a.setup()
    out: list[list[Track]] = []
    try:
        for i, dets in enumerate(det_sequence):
            batch = FrameBatch(
                frame_index=i,
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                detections=list(dets),
            )
            a.process(batch)
            out.append(list(batch.tracks))
        return out
    finally:
        a.teardown()
