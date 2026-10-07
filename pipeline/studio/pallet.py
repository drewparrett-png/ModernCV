"""Pal/DePal analysis: carton masks + depth → heights, layers, pick order.

Pipeline
--------
1. Depth (metres, z along the optical axis) from an RGB-D sensor upload or
   monocular Depth Anything V2 (metric-indoor).
2. Back-project every pixel with pinhole intrinsics: X = (u-cx)·Z/fx, …
3. Fit the *reference plane* with RANSAC:
     auto     background pixels (not inside any carton) — the pallet deck or
              floor in a top-down / oblique view;
     painted  pixels the user brushed onto the deck/floor;
     boxes    no background visible: the common normal of the carton top
              faces, with heights measured from the lowest carton top.
   The plane normal is oriented toward the camera so height = n·X + d > 0.
4. Per carton: the top face is the highest band of the (eroded) mask —
   robust to side faces visible in oblique views. Height = median of the
   top band; footprint = minAreaRect of the top band projected into the
   plane (length × width, yaw); tilt = angle between the top band's own
   plane and the reference plane.
5. Layers: gap clustering of heights (top layer = 1). Pick order: layers top
   down, rows within a layer. A carton is *blocked* if a higher carton's
   footprint overlaps its own.

Everything here is pure numpy/OpenCV on arrays so it is unit-testable
against the synthetic renderer's exact depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class PalletParams:
    layer_tol_m: float = 0.05
    top_band_m: Optional[float] = None  # None → auto from depth noise
    erode_px: int = 3
    plane_mode: str = "auto"  # auto | painted | boxes
    plane_thresh_m: Optional[float] = None  # RANSAC inlier distance; None → auto
    min_points: int = 40
    seed: int = 0


@dataclass
class Instance:
    id: str
    class_id: int
    mask: np.ndarray  # (H, W) bool
    box_only: bool = False


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    inlier_ratio: float
    rms_m: float
    n_points: int
    mode: str
    note: str = ""
    extra: dict = field(default_factory=dict)


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> dict:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def backproject(depth_m: np.ndarray, k: dict) -> np.ndarray:
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    z = depth_m.astype(np.float32)
    x = (us - k["cx"]) * z / k["fx"]
    y = (vs - k["cy"]) * z / k["fy"]
    return np.stack([x, y, z], axis=-1)


def surface_normals(points: np.ndarray, valid: np.ndarray, step: int = 3) -> np.ndarray:
    """Per-pixel unit normals from back-projected points (central differences).

    `step` px on each side trades noise for edge bleed; masks are eroded
    by at least this much before normals are trusted. Invalid → zeros.
    """
    h, w, _ = points.shape
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, step:-step] = points[:, 2 * step:] - points[:, : -2 * step]
    dy[step:-step, :] = points[2 * step:, :] - points[: -2 * step, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = (norm[..., 0] > 1e-9) & valid
    for sh in ((0, step), (0, -step), (step, 0), (-step, 0)):
        ok &= np.roll(valid, sh, axis=(0, 1))
    out = np.where(ok[..., None], n / np.maximum(norm, 1e-12), 0.0)
    return out.astype(np.float32)


def _plane_from_points(p: np.ndarray) -> tuple[np.ndarray, float]:
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, float(-n @ c)


def _orient_to_camera(n: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    # Camera at the origin: n·0 + d = d is the camera's height above the
    # plane, so d > 0 means n points toward the camera.
    return (n, d) if d > 0 else (-n, -d)


def fit_plane_ransac(
    pts: np.ndarray, thresh: float, iters: int = 400, seed: int = 0, max_points: int = 60000
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane → (unit normal toward camera, d, inlier mask over `pts`).

    Hypotheses are scored on a random subsample of at most `max_points`;
    the returned inlier mask always covers the full `pts` array.
    """
    rng = np.random.default_rng(seed)
    full = pts
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    if len(pts) < 3:
        raise ValueError("not enough points to fit a plane")
    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ s[0])
        count = int((np.abs(pts @ n + d) < thresh).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        raise ValueError("degenerate points — could not fit a plane")
    inl = np.abs(pts @ best_n + best_d) < thresh
    n, d = _plane_from_points(pts[inl])  # least-squares refine on inliers
    n, d = _orient_to_camera(n, d)
    return n, d, np.abs(full @ n + d) < thresh


def _auto_thresh(z: np.ndarray) -> float:
    # ~0.5 % of range, never under 8 mm: tight enough for a real RGB-D
    # sensor, loose enough for monocular depth noise at 2–4 m.
    return max(0.008, 0.005 * float(np.median(z)))


def fit_reference_plane(
    points: np.ndarray,
    valid: np.ndarray,
    union: np.ndarray,
    instances: list[Instance],
    params: PalletParams,
    painted: Optional[np.ndarray] = None,
) -> PlaneFit:
    z = points[..., 2][valid]
    thresh = params.plane_thresh_m or _auto_thresh(z)
    mode = params.plane_mode

    if mode == "painted":
        if painted is None or (painted & valid).sum() < 200:
            raise ValueError("paint at least a small patch of pallet deck / floor to use as the reference")
        sel = painted & valid
        n, d, inl = fit_plane_ransac(points[sel], thresh, seed=params.seed)
        res = np.abs(points[sel] @ n + d)
        return PlaneFit(n, d, float(inl.mean()), float(np.sqrt(np.mean(res[res < thresh] ** 2))),
                        int(sel.sum()), "painted", "fitted to the painted region")

    if mode == "auto":
        grow = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        bg = valid & ~grow
        if bg.sum() >= max(2000, 0.03 * valid.size):
            n, d, inl = fit_plane_ransac(points[bg], thresh, seed=params.seed)
            res = np.abs(points[bg] @ n + d)
            # Sanity: cartons must sit on the camera side of the plane. If most
            # carton pixels come out *below* it we latched onto a wall or the
            # stack face — fall back to the carton-top normal.
            if union.any():
                h_union = points[union & valid] @ n + d
                if np.median(h_union) > -thresh:
                    return PlaneFit(n, d, float(inl.mean()), float(np.sqrt(np.mean(res[res < thresh] ** 2))),
                                    int(bg.sum()), "auto", "largest plane among background pixels (deck / floor)")
            else:
                return PlaneFit(n, d, float(inl.mean()), float(np.sqrt(np.mean(res[res < thresh] ** 2))),
                                int(bg.sum()), "auto", "largest plane among background pixels")
        mode = "boxes"

    # mode == "boxes": common normal of carton tops, zero at the lowest top.
    normals = []
    tops = []
    for inst in instances:
        sel = inst.mask & valid
        if sel.sum() < params.min_points:
            continue
        try:
            n, d, inl = fit_plane_ransac(points[sel], thresh, iters=150, seed=params.seed)
        except ValueError:
            continue
        normals.append(n)
        tops.append(points[sel][inl])
    if not normals:
        raise ValueError("no background visible and no carton surfaces to fit a reference plane")
    nrm = np.median(np.stack(normals), axis=0)
    nrm /= np.linalg.norm(nrm)
    # Heights relative to the lowest carton top.
    d = -min(float(np.median(t @ nrm)) for t in tops)
    nrm, d = (nrm, d)
    return PlaneFit(nrm, d, 1.0, 0.0, int(sum(len(t) for t in tops)), "boxes",
                    "no background visible — heights are relative to the lowest carton top")


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # In-plane axes: project the camera x axis onto the plane so "u" reads
    # left→right in the image wherever possible.
    a = np.array([1.0, 0.0, 0.0])
    u = a - n * (a @ n)
    if np.linalg.norm(u) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
        u = a - n * (a @ n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _footprint(coords: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Trimmed minAreaRect of 2-D plane coords → (length, width, yaw°, corners)."""
    c = coords.mean(axis=0)
    cov = np.cov((coords - c).T) if len(coords) > 2 else np.eye(2)
    _, vecs = np.linalg.eigh(cov)
    proj = (coords - c) @ vecs
    lo, hi = np.percentile(proj, 1.5, axis=0), np.percentile(proj, 98.5, axis=0)
    keep = np.all((proj >= lo) & (proj <= hi), axis=1)
    pts = coords[keep] if keep.sum() >= 5 else coords
    rect = cv2.minAreaRect((pts * 1000.0).astype(np.float32).reshape(-1, 1, 2))
    (rw, rh) = rect[1]
    corners = cv2.boxPoints(rect) / 1000.0
    length, width = max(rw, rh) / 1000.0, min(rw, rh) / 1000.0
    angle = rect[2] if rw >= rh else rect[2] + 90.0
    yaw = ((angle + 90.0) % 180.0) - 90.0
    return float(length), float(width), float(yaw), corners


def analyze(
    depth_m: np.ndarray,
    k: dict,
    instances: list[Instance],
    params: Optional[PalletParams] = None,
    painted: Optional[np.ndarray] = None,
) -> dict:
    params = params or PalletParams()
    h, w = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m > 0.05)
    pts = backproject(np.where(valid, depth_m, 0), k)
    union = np.zeros((h, w), dtype=bool)
    for inst in instances:
        union |= inst.mask

    plane = fit_reference_plane(pts, valid, union, instances, params, painted)
    n, d = plane.normal, plane.d
    height = (pts @ n + d).astype(np.float32)
    height[~valid] = np.nan
    u_ax, v_ax = _plane_basis(n)
    zmed = float(np.median(depth_m[valid])) if valid.any() else 1.0
    band = params.top_band_m or max(0.02, 0.008 * zmed)

    # Upward-facing pixels: surface normal within ~35° of the plane normal.
    # A carton's top is the highest upward-facing surface inside its mask;
    # a mask with none (only side faces visible) has a hidden top.
    normals = surface_normals(pts, valid)
    facing_up = np.abs(normals @ n) > math.cos(math.radians(35))
    erode_px = max(params.erode_px, 3)
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    boxes: list[dict] = []
    for inst in instances:
        m = inst.mask
        er = cv2.erode(m.astype(np.uint8), kernel).astype(bool)
        if (er & valid).sum() >= params.min_points:
            m = er
        sel = m & valid
        flags: list[str] = []
        if inst.box_only:
            flags.append("box_only")
        if inst.mask[0, :].any() or inst.mask[-1, :].any() or inst.mask[:, 0].any() or inst.mask[:, -1].any():
            flags.append("touches_border")
        if sel.sum() < params.min_points:
            boxes.append({"id": inst.id, "class_id": inst.class_id, "height_m": None, "flags": flags + ["no_depth"]})
            continue
        hv = height[sel]
        up = facing_up[sel]
        no_top = up.sum() < max(params.min_points // 2, 0.02 * sel.sum())
        if no_top:
            # Nothing upward-facing: report the highest visible point as a
            # lower bound on the (hidden) top.
            ref = float(np.percentile(hv, 95))
            top = hv > ref - band
        else:
            ref = float(np.percentile(hv[up], 90))
            top = up & (hv > ref - band)
        top_h = float(np.median(hv[top]))
        p3 = pts[sel][top]
        coords = np.stack([p3 @ u_ax, p3 @ v_ax], axis=1)
        length, width, yaw, corners = _footprint(coords)
        tilt = None
        top_hidden = bool(no_top)
        if top_hidden:
            flags.append("top_hidden")
        elif len(p3) >= 10:
            tn, _ = _plane_from_points(p3)
            tilt = float(math.degrees(math.acos(min(1.0, abs(float(tn @ n))))))
            if tilt > 45:
                # The highest visible band is a *vertical* face: the carton's
                # top is hidden under another carton (oblique views). Its
                # true top is at least this high; keep it out of layering.
                top_hidden = True
                flags.append("top_hidden")
            elif tilt > 10:
                flags.append("tilted")
        if not top_hidden and width < 0.04:
            # A real carton top is at least a few cm wide in the plane. A
            # thinner band is the upper edge of a side face peeking out under
            # another carton: its points are nearly collinear, so the tilt
            # test above can't see that it's vertical.
            top_hidden = True
            flags.append("top_hidden")
        ys, xs = np.nonzero(sel)
        top_px = np.zeros((h, w), dtype=bool)
        top_px[ys[top], xs[top]] = True
        cy_, cx_ = ys[top].mean(), xs[top].mean()
        if len(p3) < 4 * params.min_points:
            flags.append("few_points")
        boxes.append(
            {
                "id": inst.id,
                "class_id": inst.class_id,
                "height_m": round(top_h, 4),
                "height_is_lower_bound": top_hidden,
                "dims_m": None if top_hidden else [round(length, 4), round(width, 4)],
                "yaw_deg": None if top_hidden else round(yaw, 1),
                "tilt_deg": None if tilt is None else round(tilt, 1),
                "centroid_px": [round(float(cx_), 1), round(float(cy_), 1)],
                "plane_xy_m": [round(float(coords[:, 0].mean()), 4), round(float(coords[:, 1].mean()), 4)],
                "footprint_m": None if top_hidden else corners.round(4).tolist(),
                "top_area_frac": round(float(top.mean()), 3),
                "n_points": int(len(p3)),
                "flags": flags,
                "_top_px": top_px,
            }
        )

    measured = [b for b in boxes if b.get("height_m") is not None and not b.get("height_is_lower_bound")]
    measured.sort(key=lambda b: -b["height_m"])
    layers: list[list[dict]] = []
    for b in measured:
        if layers and abs(np.mean([x["height_m"] for x in layers[-1]]) - b["height_m"]) <= params.layer_tol_m:
            layers[-1].append(b)
        else:
            layers.append([b])
    layer_rows = []
    for li, members in enumerate(layers, start=1):
        hs = [x["height_m"] for x in members]
        for x in members:
            x["layer"] = li
        layer_rows.append({"layer": li, "n": len(members), "mean_height_m": round(float(np.mean(hs)), 4),
                           "min_height_m": round(float(min(hs)), 4), "max_height_m": round(float(max(hs)), 4)})

    # Blocked: a higher carton's footprint overlaps this one's.
    for b in measured:
        b["blocked"] = False
        for o in measured:
            if o is b or o["height_m"] <= b["height_m"] + params.layer_tol_m:
                continue
            inter, _ = cv2.intersectConvexConvex(
                np.asarray(b["footprint_m"], np.float32) * 1000, np.asarray(o["footprint_m"], np.float32) * 1000
            )
            area = b["dims_m"][0] * b["dims_m"][1] * 1e6
            if area > 0 and inter / area > 0.1:
                b["blocked"] = True
                break

    # Pick order: rows along the pallet's own grid, not the camera's axes.
    # Carton edges share the pallet orientation modulo 90°, so the circular
    # mean of 4·yaw recovers it; rotating plane coordinates by it keeps rows
    # straight on a pallet that sits rotated in the image.
    yaws = [math.radians(b["yaw_deg"]) for b in measured if b.get("yaw_deg") is not None]
    grid = 0.0
    if yaws:
        grid = math.atan2(np.mean(np.sin(4 * np.array(yaws))), np.mean(np.cos(4 * np.array(yaws)))) / 4
    cg, sg = math.cos(grid), math.sin(grid)
    order = 1
    for li, members in enumerate(layers, start=1):
        def aligned(b: dict) -> tuple[float, float]:
            x, y = b["plane_xy_m"]
            return x * cg + y * sg, -x * sg + y * cg

        # Rows (far → near), then left → right, in grid-aligned coordinates.
        rows = sorted(members, key=lambda b: (-round(aligned(b)[1] / 0.15), aligned(b)[0]))
        for b in rows:
            b["pick_order"] = order
            order += 1

    cam_tilt = float(math.degrees(math.acos(min(1.0, abs(float(n[2]))))))
    for b in boxes:
        b.pop("_top_px", None)

    # Wall-vs-floor check. Without gravity, a wall facing the camera and a
    # floor seen from above look alike (normal ≈ optical axis). A floor/deck
    # reaches the bottom of the frame; a back wall in a side view doesn't.
    suspect_wall = False
    if plane.mode == "auto" and cam_tilt < 40:
        thresh = params.plane_thresh_m or _auto_thresh(depth_m[valid])
        inl = valid & (np.abs(np.nan_to_num(height, nan=1e9)) < thresh) & ~union
        bottom = inl[int(h * 0.85):, :]
        bottom_valid = (valid & ~union)[int(h * 0.85):, :].sum()
        if bottom_valid > 0 and bottom.sum() / bottom_valid < 0.15:
            suspect_wall = True
    n_hidden = sum(1 for b in boxes if b.get("height_is_lower_bound"))
    side_view = len(boxes) >= 3 and n_hidden >= 0.75 * len(boxes) and len(measured) <= 2
    return {
        "plane": {
            "mode": plane.mode,
            "note": plane.note,
            "normal": [round(float(v), 5) for v in n],
            "d_m": round(float(d), 4),
            "inlier_ratio": round(plane.inlier_ratio, 3),
            "rms_m": round(plane.rms_m, 4),
            "n_points": plane.n_points,
            "camera_tilt_deg": round(cam_tilt, 1),
            "camera_height_m": round(float(d), 3),
            "suspect_wall": suspect_wall,
        },
        "boxes": boxes,
        "layers": layer_rows,
        "summary": {
            "n_boxes": len(boxes),
            "n_measured": len(measured),
            "n_layers": len(layers),
            "max_height_m": round(max((b["height_m"] for b in measured), default=0.0), 4),
            "n_blocked": sum(1 for b in measured if b.get("blocked")),
            "n_top_hidden": sum(1 for b in boxes if b.get("height_is_lower_bound")),
            "side_view": side_view,
            "top_band_m": round(band, 4),
        },
        "_height_map": height,
    }


def colorize(values: np.ndarray, lo: float, hi: float, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Turbo colour map of a float map → BGR uint8 (invalid pixels black).

    `lo` maps to the cold end and `hi` to the warm end; pass them reversed
    (lo > hi) to invert the ramp, e.g. near depth = warm.
    """
    span = hi - lo
    if abs(span) < 1e-6:
        span = 1e-6
    v = np.nan_to_num((values - lo) / span, nan=0.0)
    img = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask = np.isfinite(values) if valid is None else valid & np.isfinite(values)
    img[~mask] = 0
    return img


# ---- Orchestration (store + models) -------------------------------------

MONO_CAVEAT = (
    "Monocular depth: relative ordering of carton heights is usually right, but absolute heights "
    "can be off by 10–30 cm (on our synthetic test pallets, Depth Anything V2 Base with a known "
    "camera height gave ~17 cm mean error, rank correlation ~0.9). Use an RGB-D depth map for "
    "real Pal/DePal decisions."
)


def _jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def run_analysis(store, image_id: str, req: dict) -> dict:
    """Full Pal/DePal analysis for one stored image (see module docstring)."""
    from pipeline.studio import engines
    from pipeline.studio.geometry import annotation_mask, stroke_mask
    from pipeline.studio.infer import run_predict
    from pipeline.studio.store import Invalid

    rec = store.get_image(image_id)
    img = store.read_image(image_id)
    h, w = img.shape[:2]
    warnings: list[str] = []

    # 1) Instances.
    src = req.get("instances") or {"kind": "annotations"}
    kind = src.get("kind", "annotations")
    class_filter = set(int(c) for c in src.get("class_ids") or [])
    if kind == "annotations":
        anns = store.annotations(image_id)
    elif kind == "suggestions":
        min_score = float(src.get("min_score") or 0)
        anns = [s for s in store.suggestions(image_id)["items"] if (s.get("score") or 1) >= min_score]
    elif kind == "predict":
        if not src.get("model"):
            raise Invalid("choose a model to detect cartons with")
        anns = run_predict(store, src["model"], img, src.get("params") or {})["detections"]
        for i, a in enumerate(anns):
            a["id"] = f"p{i}"
    else:
        raise Invalid(f"unknown instance source {kind!r}")
    if class_filter:
        anns = [a for a in anns if int(a.get("class_id", -1)) in class_filter]
    if not anns:
        raise Invalid("no cartons to analyse — label, accept suggestions or pick a model first")
    instances = [
        Instance(id=str(a.get("id", i)), class_id=int(a.get("class_id", -1)), mask=annotation_mask(a, (h, w)),
                 box_only=not a.get("polygon"))
        for i, a in enumerate(anns)
    ]
    if any(i.box_only for i in instances):
        warnings.append(
            "Some instances are boxes, not masks — their heights mix carton and background pixels. "
            "Segment them (SAM / a -seg model) for reliable heights."
        )

    # 2) Depth.
    dreq = req.get("depth") or {}
    source = dreq.get("source", "auto")
    depth_model = dreq.get("model", "da2-metric-indoor-b")
    if source in ("auto", "sensor") and rec.get("has_depth"):
        depth = store.read_depth_m(image_id)
        depth_label = "sensor"
    elif source == "sensor":
        raise Invalid("this image has no uploaded depth map")
    else:
        cache = store.cache_dir / "depth" / f"{image_id}_{depth_model}.npy"
        if cache.exists():
            depth = np.load(cache)
        else:
            with engines.INFER_LOCK:
                depth = engines.estimate_depth_m(img, depth_model)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, depth)
        depth_label = f"mono:{depth_model}"
        warnings.append(MONO_CAVEAT)
    depth = depth.astype(np.float32)

    # 3) Intrinsics.
    if rec.get("intrinsics") and req.get("use_image_intrinsics", True):
        k = rec["intrinsics"]
        k_source = "image"
    else:
        k = intrinsics_from_hfov(w, h, float(req.get("hfov_deg") or 60.0))
        k_source = f"assumed {float(req.get('hfov_deg') or 60.0):g}° horizontal FOV"

    # 4) Geometry.
    preq = req.get("plane") or {}
    params = PalletParams(
        layer_tol_m=float(req.get("layer_tol_m") or 0.05),
        plane_mode=preq.get("mode", "auto"),
        erode_px=int(req.get("erode_px", 3)),
    )
    painted = None
    if params.plane_mode == "painted":
        painted = np.zeros((h, w), dtype=bool)
        for s in preq.get("strokes") or []:
            painted |= stroke_mask(s["points"], float(s.get("radius", 10)), (h, w))
    try:
        res = analyze(depth, k, instances, params, painted)
        cal = req.get("known_camera_height_m")
        if cal:
            scale = float(cal) / max(1e-6, res["plane"]["camera_height_m"])
            depth = depth * scale
            res = analyze(depth, k, instances, params, painted)
            res["calibration"] = {"known_camera_height_m": float(cal), "depth_scale": round(scale, 4)}
    except ValueError as e:
        raise Invalid(str(e)) from e
    if res["plane"]["mode"] == "boxes":
        warnings.append(res["plane"]["note"])
    if res["plane"].get("suspect_wall"):
        warnings.append(
            "The auto reference plane looks like a wall facing the camera, not the floor or pallet deck "
            "(it never reaches the bottom of the image). If the wall reads as 0 cm in the height map, set "
            "Reference plane → Painted and paint the floor/deck, or shoot from above."
        )
    if res["summary"].get("side_view"):
        warnings.append(
            "Most cartons show only side faces — this looks like a side view. Heights need the carton tops "
            "in view: mount the camera above the pallet (top-down or ~30° oblique)."
        )
    if res["summary"]["n_top_hidden"]:
        warnings.append(
            f"{res['summary']['n_top_hidden']} carton(s) show only a side face (top hidden under another "
            "carton) — their heights are lower bounds and they are left out of the layer count."
        )

    # 5) Visuals.
    hm = res.pop("_height_map")
    max_h = res["summary"]["max_height_m"] or float(np.nanpercentile(hm, 99))
    vis_hi = max(0.05, max_h * 1.05)
    hcol = colorize(hm, 0.0, vis_hi)
    height_vis = cv2.addWeighted(img, 0.45, hcol, 0.55, 0)
    valid = np.isfinite(depth) & (depth > 0.05)
    lo, hi = (np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1))
    depth_vis = colorize(np.where(valid, depth, np.nan), float(hi), float(lo))  # near = warm

    polys = {str(a.get("id", i)): (a.get("polygon") or _box_poly(a["bbox"])) for i, a in enumerate(anns)}
    for b in res["boxes"]:
        b["polygon"] = polys.get(b["id"])
    res.update(
        image_id=image_id,
        depth_source=depth_label,
        intrinsics={**{kk: round(float(v), 2) for kk, v in k.items()}, "source": k_source},
        instance_source=kind,
        height_range_m=[0.0, round(vis_hi, 3)],
        depth_range_m=[round(float(lo), 3), round(float(hi), 3)],
        height_vis=_jpeg_b64(height_vis),
        depth_vis=_jpeg_b64(depth_vis),
        warnings=warnings,
    )
    return res


def _box_poly(bb: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bb
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
