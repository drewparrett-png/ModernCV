"""Mask / polygon / prompt-raster helpers shared by prompting, export and pallet.

All coordinates are original-image pixels; masks are (H, W) bool or uint8.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.0015) -> Optional[list[list[float]]]:
    """Largest external contour of a mask as [[x, y], …], lightly simplified."""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = max(0.5, epsilon_frac * cv2.arcLength(c, True))
    c = cv2.approxPolyDP(c, eps, True)
    if len(c) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in c]


def mask_components(mask: np.ndarray, min_area: int = 16) -> list[np.ndarray]:
    """Split a mask into connected components (largest first)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    comps = [
        (stats[i, cv2.CC_STAT_AREA], labels == i)
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    comps.sort(key=lambda t: -t[0])
    return [m for _, m in comps]


def largest_component(mask: np.ndarray) -> np.ndarray:
    comps = mask_components(mask, min_area=1)
    return comps[0] if comps else np.zeros_like(mask, dtype=bool)


def polygon_to_mask(polygon: Iterable[Iterable[float]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.round(np.asarray(list(polygon), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(out, [pts], 1)
    return out.astype(bool)


def box_to_mask(bbox: Iterable[float], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    out = np.zeros((h, w), dtype=bool)
    xa, xb = int(max(0, math.floor(min(x1, x2)))), int(min(w, math.ceil(max(x1, x2))))
    ya, yb = int(max(0, math.floor(min(y1, y2)))), int(min(h, math.ceil(max(y1, y2))))
    out[ya:yb, xa:xb] = True
    return out


def annotation_mask(ann: dict, shape: tuple[int, int]) -> np.ndarray:
    if ann.get("polygon"):
        return polygon_to_mask(ann["polygon"], shape)
    return box_to_mask(ann["bbox"], shape)


def stroke_mask(points: list[list[float]], radius: float, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a brush stroke: round-capped polyline of width 2·radius."""
    out = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 2)
    r = max(1, int(round(radius)))
    if len(pts) == 1:
        cv2.circle(out, tuple(int(v) for v in pts[0]), r, 1, -1)
    else:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, 1, thickness=2 * r, lineType=cv2.LINE_8)
        for p in (pts[0], pts[-1]):
            cv2.circle(out, tuple(int(v) for v in p), r, 1, -1)
    return out.astype(bool)


def sample_stroke_points(points: list[list[float]], max_points: int = 8) -> list[list[float]]:
    """Evenly spaced samples along a polyline (by arc length), endpoints included.

    SAM treats each point as an independent click, so a long stroke
    across an object becomes a handful of well-spread positive clicks
    rather than hundreds of near-duplicates.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= 1:
        return pts.tolist()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-6:
        return [pts[0].tolist()]
    n = int(min(max_points, max(2, math.ceil(total / 25.0))))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for t in np.linspace(0, total, n):
        i = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1))
        f = 0.0 if seg[i] < 1e-9 else (t - cum[i]) / seg[i]
        out.append((pts[i] + f * (pts[i + 1] - pts[i])).tolist())
    return out


def mask_bbox(mask: np.ndarray) -> Optional[list[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def simplify_polygon(poly: np.ndarray, epsilon_frac: float = 0.0015) -> Optional[list[list[float]]]:
    """Simplify an (N, 2) float polygon from a model (e.g. `Results.masks.xy`)."""
    if poly is None or len(poly) < 3:
        return None
    c = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    eps = max(0.5, epsilon_frac * cv2.arcLength(c, True))
    c = cv2.approxPolyDP(c, eps, True)
    if len(c) < 3:
        return None
    return [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in c]
