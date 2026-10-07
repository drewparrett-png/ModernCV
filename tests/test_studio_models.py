"""Model-backed Studio checks — the Ultralytics pitfalls we work around.

Skipped unless the weights are already in data/weights/ (no downloads in
CI). Each test renders a synthetic pallet so expectations are exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.studio import engines, prompting
from pipeline.studio.geometry import polygon_to_mask
from pipeline.studio.infer import run_predict
from pipeline.studio.store import StudioStore
from pipeline.studio.synth import render_pallet_scene

needs = lambda *names: pytest.mark.skipif(  # noqa: E731
    not all(engines.is_cached(n) for n in names), reason=f"weights not cached: {names}"
)


@pytest.fixture(scope="module")
def scene():
    return render_pallet_scene(seed=3, width=640, height=480)


@pytest.fixture()
def store(tmp_path: Path, scene) -> tuple[StudioStore, str]:
    st = StudioStore(tmp_path / "studio")
    st.set_classes([{"name": "carton"}])
    rec = st.add_image_array(scene.bgr, "synthetic.jpg", "synthetic")
    return st, rec["id"]


@needs("yoloe-11s-seg.pt")
def test_mask_visual_prompt_finds_other_cartons(store, scene) -> None:
    """Brush-stroke (mask) visual prompts — upstream only accepts boxes."""
    st, iid = store
    g = scene.instances[0]
    ys, xs = np.nonzero(g["mask"])
    stroke = {"type": "stroke", "points": [[float(xs.min() + 5), float(ys.mean())], [float(xs.max() - 5), float(ys.mean())]],
              "radius": 6, "class_id": 0, "polarity": 1}
    out = prompting.visual_prompt_detect(st, [{"image_id": iid, "prompts": [stroke]}], scope="image", image_id=iid,
                                         family="11", size="s", conf=0.2)
    assert out["n_examples"] == 1
    assert out["counts"][iid] >= 3  # several other cartons found from one stroke


@needs("sam2.1_t.pt")
def test_sam_box_prompt_matches_carton(store, scene) -> None:
    st, iid = store
    g = max(scene.instances, key=lambda i: i["mask"].sum())
    res = prompting.sam_segment(st, iid, [{"type": "box", "bbox": g["bbox"], "class_id": 0, "polarity": 1}])
    pm = polygon_to_mask(res["polygon"], g["mask"].shape)
    iou = (pm & g["mask"]).sum() / (pm | g["mask"]).sum()
    assert iou > 0.75


@needs("sam2.1_t.pt")
def test_sam_granularity_orders_by_area(store, scene) -> None:
    st, iid = store
    g = max(scene.instances, key=lambda i: i["mask"].sum())
    ys, xs = np.nonzero(g["mask"])
    stroke = {"type": "stroke", "points": [[float(xs.mean()), float(ys.mean())]], "radius": 3, "class_id": 0, "polarity": 1}
    areas = [prompting.sam_segment(st, iid, [stroke], granularity=g_)["area"] for g_ in ("fine", "medium", "coarse")]
    assert areas == sorted(areas)


@needs("yolo26n.pt")
def test_end2end_toggle_switches_heads(store) -> None:
    """NMS mode needs its own instance: e2e inference fuses the o2m head away."""
    st, iid = store
    img = st.read_image(iid)
    spec = {"kind": "yolo26", "task": "detect", "size": "n"}
    a = run_predict(st, spec, img, {"conf": 0.01, "end2end": True})
    b = run_predict(st, spec, img, {"conf": 0.01, "end2end": False})
    c = run_predict(st, spec, img, {"conf": 0.01})
    assert a["end2end"] is True and b["end2end"] is False and c["end2end"] is True
    assert len(a["detections"]) == len(c["detections"])
