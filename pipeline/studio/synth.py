"""Synthetic pallet scenes — ray-cast cartons on a EUR pallet with exact depth.

Why this exists
---------------
Pal/DePal questions ("can we segment the cartons?", "which ones are on the
top layer?", "how tall is the stack?") need images *with ground truth* to
check the geometry end to end. Public pallet imagery with depth is scarce,
so this module renders scenes we fully control:

  • an RGB image (cardboard albedo + tape + labels, Lambert shading, hard
    shadows, sensor noise),
  • an exact z-depth map in millimetres — the same format an RGB-D camera
    (RealSense, Zivid, Photoneo…) would hand us as a 16-bit PNG,
  • the camera intrinsics, and
  • per-carton visible masks with their true top heights and layer index.

The renderer is a vectorised numpy ray caster: one ray per pixel, slab
intersection against yawed cuboids, plus a floor plane. A 960×720 scene
with ~30 cuboids renders in a couple of seconds on an M-series Mac, which
is fast enough to generate a small training set on demand.

World frame: Z up, floor at z=0, pallet centred on the origin with its
long side along X. Camera frame: x right, y down, z forward (OpenCV), so
back-projection in `pipeline.studio.pallet` is the textbook
X = (u - cx) * Z / fx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pipeline.studio.geometry import mask_to_polygon

# EUR pallet: 1200 × 800 × 144 mm.
PALLET_L = 1.2
PALLET_W = 0.8
PALLET_H = 0.144
DECK_THICK = 0.022

# Carton SKUs (length, width, height) in metres.
SKUS: list[tuple[float, float, float]] = [
    (0.40, 0.30, 0.25),
    (0.40, 0.40, 0.30),
    (0.60, 0.40, 0.30),
    (0.30, 0.20, 0.20),
    (0.40, 0.27, 0.22),
    (0.30, 0.30, 0.28),
]

CARTON_ALBEDOS = np.array(
    [
        [0.72, 0.55, 0.36],
        [0.66, 0.50, 0.33],
        [0.76, 0.60, 0.40],
        [0.62, 0.47, 0.30],
        [0.80, 0.66, 0.46],  # light kraft
    ]
)


@dataclass
class Cuboid:
    center: np.ndarray  # (3,) world
    half: np.ndarray  # (3,) half extents in the cuboid's local frame
    yaw: float  # rotation about world Z, radians
    kind: str  # "carton" | "pallet"
    albedo: np.ndarray  # (3,) RGB in 0..1
    layer: int = 0  # 1-based stacking layer for cartons (1 = bottom)
    has_tape: bool = False
    label: Optional[tuple[float, float, float, float]] = None  # top-face label rect (x0,y0,x1,y1) local

    @property
    def top_z(self) -> float:
        return float(self.center[2] + self.half[2])


@dataclass
class SynthScene:
    bgr: np.ndarray  # (H, W, 3) uint8
    depth_mm: np.ndarray  # (H, W) uint16, z-depth along the optical axis
    intrinsics: dict  # fx, fy, cx, cy
    instances: list[dict]  # visible cartons, see `render_pallet_scene`
    meta: dict = field(default_factory=dict)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _intersect_cuboid(
    origins: np.ndarray, dirs: np.ndarray, box: Cuboid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slab test for N rays vs one yawed cuboid.

    `origins` may be (3,) (shared camera centre) or (N, 3) (shadow rays).
    Returns (t, local_normal_axis, local_hit) where t is +inf for misses.
    """
    rinv = _rot_z(-box.yaw)
    o = (origins - box.center) @ rinv.T
    d = dirs @ rinv.T
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    inv = 1.0 / d
    t1 = (-box.half - o) * inv
    t2 = (box.half - o) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tnear = tmin.max(axis=1)
    tfar = tmax.min(axis=1)
    hit = (tnear <= tfar) & (tnear > 1e-6)
    t = np.where(hit, tnear, np.inf)
    axis = tmin.argmax(axis=1)
    local_hit = o + tnear[:, None] * d if o.ndim == 2 else o[None, :] + tnear[:, None] * d
    return t, axis, local_hit


def _build_pallet(rng: np.random.Generator) -> list[Cuboid]:
    wood = np.array([0.70, 0.58, 0.40]) * rng.uniform(0.9, 1.08)
    parts: list[Cuboid] = []
    # Top deck: 5 boards running along X.
    board_w = 0.145
    ys = np.linspace(-PALLET_W / 2 + board_w / 2, PALLET_W / 2 - board_w / 2, 5)
    for y in ys:
        parts.append(
            Cuboid(
                center=np.array([0.0, y, PALLET_H - DECK_THICK / 2]),
                half=np.array([PALLET_L / 2, board_w / 2, DECK_THICK / 2]),
                yaw=0.0,
                kind="pallet",
                albedo=wood * rng.uniform(0.92, 1.05),
            )
        )
    # 9 blocks under the deck.
    block_h = PALLET_H - DECK_THICK
    for x in (-PALLET_L / 2 + 0.07, 0.0, PALLET_L / 2 - 0.07):
        for y in (-PALLET_W / 2 + 0.05, 0.0, PALLET_W / 2 - 0.05):
            parts.append(
                Cuboid(
                    center=np.array([x, y, block_h / 2]),
                    half=np.array([0.07, 0.05, block_h / 2]),
                    yaw=0.0,
                    kind="pallet",
                    albedo=wood * rng.uniform(0.8, 0.95),
                )
            )
    return parts


def _layer_grid(sku: tuple[float, float, float], gap: float) -> tuple[float, float, int, int]:
    """Pick the orientation of `sku` that packs the most cartons per layer."""
    lx, ly, _ = sku
    best = None
    for a, b in ((lx, ly), (ly, lx)):
        nx = int((PALLET_L + 1e-9) // (a + gap))
        ny = int((PALLET_W + 1e-9) // (b + gap))
        if best is None or nx * ny > best[2] * best[3]:
            best = (a, b, nx, ny)
    assert best is not None
    return best


def _build_stack(rng: np.random.Generator, n_full_layers: int, top_keep: float) -> list[Cuboid]:
    cartons: list[Cuboid] = []
    z = PALLET_H
    gap = rng.uniform(0.004, 0.012)
    base_sku = SKUS[rng.integers(len(SKUS))]
    layers = n_full_layers + 1  # + one partial top layer
    for li in range(layers):
        # Mostly-uniform pallets, with the odd mixed layer.
        sku = base_sku if rng.random() < 0.75 else SKUS[rng.integers(len(SKUS))]
        a, b, nx, ny = _layer_grid(sku, gap)
        h = sku[2]
        is_top = li == layers - 1
        x0 = -(nx * (a + gap) - gap) / 2 + a / 2
        y0 = -(ny * (b + gap) - gap) / 2 + b / 2
        placed = 0
        for ix in range(nx):
            for iy in range(ny):
                if is_top and rng.random() > top_keep:
                    continue
                # Partial top layers sometimes mix carton heights — the case
                # where "relative height" actually matters for pick order.
                hh = h
                if is_top and rng.random() < 0.25:
                    hh = float(rng.choice([s[2] for s in SKUS]))
                jitter = rng.uniform(-0.006, 0.006, size=2)
                albedo = CARTON_ALBEDOS[rng.integers(len(CARTON_ALBEDOS))] * rng.uniform(0.93, 1.06)
                label = None
                if rng.random() < 0.35:
                    lw, lh = 0.10, 0.07
                    cx = rng.uniform(-a / 2 + lw / 2 + 0.02, a / 2 - lw / 2 - 0.02)
                    cy = rng.uniform(-b / 2 + lh / 2 + 0.02, b / 2 - lh / 2 - 0.02)
                    if abs(cy) < 0.04:  # keep clear of the tape
                        cy = 0.06 if b / 2 > 0.1 else cy
                    label = (cx - lw / 2, cy - lh / 2, cx + lw / 2, cy + lh / 2)
                cartons.append(
                    Cuboid(
                        center=np.array(
                            [x0 + ix * (a + gap) + jitter[0], y0 + iy * (b + gap) + jitter[1], z + hh / 2]
                        ),
                        half=np.array([a / 2, b / 2, hh / 2]),
                        yaw=float(rng.uniform(-0.03, 0.03)),
                        kind="carton",
                        albedo=np.clip(albedo, 0, 1),
                        layer=li + 1,
                        has_tape=rng.random() < 0.85,
                        label=label,
                    )
                )
                placed += 1
        z += h
        if placed == 0 and is_top:
            break
    return cartons


def _value_noise(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    small = rng.uniform(-1, 1, size=(cells, int(cells * w / h) + 1)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def render_pallet_scene(
    seed: int = 0,
    width: int = 960,
    height: int = 720,
    hfov_deg: Optional[float] = None,
    tilt_deg: Optional[float] = None,
    n_full_layers: Optional[int] = None,
) -> SynthScene:
    """Render one pallet scene.

    Every argument left as `None` is randomised from `seed`, so
    `render_pallet_scene(seed=k)` is a reproducible scene generator.

    Returned `instances` (one per *visible* carton, ≥400 px):
        mask            (H, W) bool visible region
        polygon         [[x, y], …] largest contour of the mask, pixels
        bbox            [x1, y1, x2, y2]
        top_height_m    carton top above the floor
        height_above_deck_m
        layer           1-based stacking layer (1 = sits on the pallet)
        size_m          [length, width, height]
        yaw_deg
    """
    rng = np.random.default_rng(seed)
    hfov = math.radians(hfov_deg if hfov_deg is not None else rng.uniform(55, 70))
    tilt = math.radians(tilt_deg if tilt_deg is not None else rng.uniform(0, 32))
    az = rng.uniform(-math.pi, math.pi) if tilt > 1e-3 else -math.pi / 2
    layers = n_full_layers if n_full_layers is not None else int(rng.integers(1, 4))

    pallet = _build_pallet(rng)
    cartons = _build_stack(rng, layers, top_keep=float(rng.uniform(0.3, 0.85)))
    objects = pallet + cartons
    stack_top = max((c.top_z for c in cartons), default=PALLET_H)

    # Camera: look at the middle of the stack from far enough away that the
    # whole pallet fits in frame.
    target = np.array([0.0, 0.0, stack_top * 0.5])
    fx = (width / 2) / math.tan(hfov / 2)
    fy = fx
    cx, cy = width / 2, height / 2
    vfov = 2 * math.atan((height / 2) / fy)
    radius = math.hypot(PALLET_L, PALLET_W) / 2 * 1.05
    dist = max(radius / math.tan(min(hfov, vfov) / 2), stack_top + 0.9) + rng.uniform(0.2, 0.6)
    cam = target + dist * np.array(
        [math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)]
    )
    z_cam = (target - cam) / np.linalg.norm(target - cam)
    x_cam = np.array([-math.sin(az), math.cos(az), 0.0])
    x_cam = x_cam - z_cam * float(x_cam @ z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    r_cw = np.stack([x_cam, y_cam, z_cam], axis=1)  # camera → world

    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1).reshape(-1, 3)
    dirs = d_cam @ r_cw.T  # unnormalised: t == z-depth

    n = dirs.shape[0]
    t_best = np.full(n, np.inf)
    obj_id = np.full(n, -1, dtype=np.int32)  # -1 floor/background
    normal_w = np.zeros((n, 3))
    local_best = np.zeros((n, 3))
    axis_best = np.zeros(n, dtype=np.int64)

    # Floor.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dirs[:, 2] < -1e-9, -cam[2] / dirs[:, 2], np.inf)
    t_best = t_floor.copy()
    normal_w[:] = [0, 0, 1]

    for i, ob in enumerate(objects):
        t, axis, local = _intersect_cuboid(cam, dirs, ob)
        closer = t < t_best
        if not closer.any():
            continue
        t_best[closer] = t[closer]
        obj_id[closer] = i
        local_best[closer] = local[closer]
        axis_best[closer] = axis[closer]
        # Local normal: ±unit on the entering axis, facing the ray.
        rinv = _rot_z(-ob.yaw)
        d_local = dirs[closer] @ rinv.T
        ax = axis[closer]
        nl = np.zeros((int(closer.sum()), 3))
        nl[np.arange(len(ax)), ax] = -np.sign(d_local[np.arange(len(ax)), ax])
        normal_w[closer] = nl @ _rot_z(ob.yaw).T

    valid = np.isfinite(t_best)
    pts = cam[None, :] + np.where(valid, t_best, 0)[:, None] * dirs

    # ---- Shading --------------------------------------------------------
    light = np.array([0.35, -0.45, 1.0])
    light /= np.linalg.norm(light)
    lambert = np.clip(normal_w @ light, 0, 1)
    in_shadow = np.zeros(n, dtype=bool)
    shadow_origin = pts + normal_w * 1e-4
    light_dirs = np.broadcast_to(light, (n, 3))
    for ob in objects:
        t, _, _ = _intersect_cuboid(shadow_origin, light_dirs, ob)
        in_shadow |= np.isfinite(t)

    h, w = height, width
    albedo = np.zeros((n, 3))
    floor_noise = (0.05 * _value_noise(rng, h, w, 6) + 0.025 * _value_noise(rng, h, w, 60)).reshape(-1)
    is_floor = obj_id < 0
    albedo[is_floor] = (np.array([0.52, 0.52, 0.50]) + floor_noise[is_floor, None])

    grain = rng.normal(0, 0.025, size=n)
    for i, ob in enumerate(objects):
        sel = obj_id == i
        if not sel.any():
            continue
        loc = local_best[sel]
        col = np.tile(ob.albedo, (int(sel.sum()), 1))
        if ob.kind == "pallet":
            col *= (1 + 0.06 * np.sin(loc[:, 0] * 90 + i) + grain[sel] * 1.5)[:, None]
        else:
            col *= (1 + grain[sel])[:, None]
            nz = normal_w[sel][:, 2] > 0.9  # top face
            if ob.has_tape:
                across = loc[:, 1] if ob.half[0] >= ob.half[1] else loc[:, 0]
                tape = nz & (np.abs(across) < 0.026)
                col[tape] = np.array([0.82, 0.70, 0.50]) * 1.04
            if ob.label is not None:
                x0, y0, x1, y1 = ob.label
                lab = nz & (loc[:, 0] > x0) & (loc[:, 0] < x1) & (loc[:, 1] > y0) & (loc[:, 1] < y1)
                col[lab] = [0.93, 0.93, 0.90]
            # Darken a thin band at every face edge so neighbouring cartons
            # read as separate objects (real cartons have visible seams).
            to_edge = ob.half[None, :] - np.abs(loc)
            to_edge[np.arange(len(loc)), axis_best[sel]] = np.inf  # the face's own axis
            col[to_edge.min(axis=1) < 0.005] *= 0.72
        albedo[sel] = col

    shade = 0.38 + 0.62 * lambert * (~in_shadow)
    rgb = np.clip(albedo * shade[:, None], 0, 1).reshape(h, w, 3)
    # Gentle vignette + sensor noise + optics blur.
    vig = 1 - 0.18 * (((us - cx) / cx) ** 2 + ((vs - cy) / cy) ** 2) / 2
    rgb = np.clip(rgb * vig[..., None] + rng.normal(0, 0.012, size=rgb.shape), 0, 1)
    img = (rgb * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_m = np.where(valid, t_best, 0).reshape(h, w)
    depth_mm = np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16)

    # ---- Ground-truth instances ----------------------------------------
    ids = obj_id.reshape(h, w)
    instances: list[dict] = []
    n_pallet = len(pallet)
    for k, c in enumerate(cartons):
        mask = ids == (n_pallet + k)
        if mask.sum() < 400:
            continue
        poly = mask_to_polygon(mask)
        if poly is None:
            continue
        ys_, xs_ = np.nonzero(mask)
        instances.append(
            {
                "mask": mask,
                "polygon": poly,
                "bbox": [float(xs_.min()), float(ys_.min()), float(xs_.max() + 1), float(ys_.max() + 1)],
                "top_height_m": round(c.top_z, 4),
                "height_above_deck_m": round(c.top_z - PALLET_H, 4),
                "layer": c.layer,
                "size_m": [round(2 * float(v), 4) for v in c.half],
                "yaw_deg": round(math.degrees(c.yaw), 2),
            }
        )

    return SynthScene(
        bgr=bgr,
        depth_mm=depth_mm,
        intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        instances=instances,
        meta={
            "seed": seed,
            "tilt_deg": round(math.degrees(tilt), 2),
            "hfov_deg": round(math.degrees(hfov), 2),
            "camera_height_m": round(float(cam[2]), 3),
            "pallet_deck_height_m": PALLET_H,
            "n_cartons": len(cartons),
            "n_visible": len(instances),
            "stack_top_m": round(stack_top, 4),
        },
    )
