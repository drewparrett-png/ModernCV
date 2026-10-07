"""Pal/DePal geometry against the synthetic renderer's exact depth."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.studio.pallet import Instance, PalletParams, analyze, fit_plane_ransac, intrinsics_from_hfov
from pipeline.studio.synth import render_pallet_scene


@pytest.fixture(scope="module")
def scenes():
    # Small renders keep the suite fast; seed 3 is near top-down, seed 1 oblique.
    return {s: render_pallet_scene(seed=s, width=480, height=360) for s in (1, 3)}


def _instances(sc) -> list[Instance]:
    return [Instance(id=str(i), class_id=0, mask=g["mask"]) for i, g in enumerate(sc.instances)]


def test_ransac_recovers_a_noisy_plane() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(4000, 2))
    z = 2.5 + 0.1 * xy[:, 0] + rng.normal(0, 0.002, 4000)
    pts = np.column_stack([xy, z])
    outliers = rng.uniform(-1, 3, size=(800, 3))
    n, d, inl = fit_plane_ransac(np.vstack([pts, outliers]), thresh=0.01)
    assert d > 0  # oriented toward the camera at the origin
    assert abs(abs(n[2]) - 1 / np.sqrt(1 + 0.1**2)) < 0.01
    assert inl[:4000].mean() > 0.95


@pytest.mark.parametrize("seed", [1, 3])
def test_visible_top_heights_are_exact_with_sensor_depth(scenes, seed: int) -> None:
    sc = scenes[seed]
    depth = sc.depth_mm.astype(np.float32) / 1000.0
    res = analyze(depth, sc.intrinsics, _instances(sc), PalletParams())
    assert res["plane"]["mode"] == "auto"
    assert res["plane"]["camera_height_m"] == pytest.approx(sc.meta["camera_height_m"], abs=0.01)
    measured = [b for b in res["boxes"] if b["height_m"] is not None and not b["height_is_lower_bound"]]
    assert measured
    for b in measured:
        gt = sc.instances[int(b["id"])]["top_height_m"]
        assert b["height_m"] == pytest.approx(gt, abs=0.01)
    # Layer 1 is the highest; pick order starts on it.
    first = min(measured, key=lambda b: b["pick_order"])
    assert first["layer"] == 1
    assert first["height_m"] == max(b["height_m"] for b in measured)


def test_hidden_tops_are_flagged_not_layered(scenes) -> None:
    sc = scenes[1]  # oblique: lower cartons show only side faces
    depth = sc.depth_mm.astype(np.float32) / 1000.0
    res = analyze(depth, sc.intrinsics, _instances(sc), PalletParams())
    hidden = [b for b in res["boxes"] if b.get("height_is_lower_bound")]
    for b in hidden:
        assert "top_hidden" in b["flags"]
        assert b.get("layer") is None
        assert b["height_m"] <= sc.instances[int(b["id"])]["top_height_m"] + 0.01


def test_boxes_mode_measures_relative_to_lowest_top(scenes) -> None:
    sc = scenes[3]
    depth = sc.depth_mm.astype(np.float32) / 1000.0
    res = analyze(depth, sc.intrinsics, _instances(sc), PalletParams(plane_mode="boxes"))
    hs = [b["height_m"] for b in res["boxes"] if b["height_m"] is not None and not b["height_is_lower_bound"]]
    gts = [sc.instances[int(b["id"])]["top_height_m"] for b in res["boxes"]
           if b["height_m"] is not None and not b["height_is_lower_bound"]]
    assert min(hs) == pytest.approx(0.0, abs=0.01)
    assert max(hs) - min(hs) == pytest.approx(max(gts) - min(gts), abs=0.015)


def test_intrinsics_from_hfov() -> None:
    k = intrinsics_from_hfov(640, 480, 90.0)
    assert k["fx"] == pytest.approx(320.0)
    assert (k["cx"], k["cy"]) == (320.0, 240.0)
