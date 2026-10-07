"""Draft prompts (boxes / brush strokes) → masks, labels and suggestions.

A *prompt* is what the user draws on the canvas before committing labels:

    {"type": "box",    "bbox": [x1, y1, x2, y2], "class_id": int, "polarity": 1}
    {"type": "stroke", "points": [[x, y], …], "radius": px, "class_id": int, "polarity": 1 | -1}

Positive strokes paint "this", negative strokes (erase tool) paint "not
this". Three things can be done with a draft:

  • `sam_segment`      — the whole draft describes ONE object → SAM 2.1 mask.
  • `paint_to_annotations` — use the painted pixels / boxes directly as labels.
  • `visual_prompt_detect` — the draft (plus, optionally, existing labels)
    are *examples*; YOLOE finds every similar object in target images.

Text-prompt, prompt-free and model-assist detection share the same
post-processing (`_finalise_items`): dataset-class mapping, optional SAM
refinement of each box, and de-duplication against existing labels.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from pipeline.studio import engines
from pipeline.studio.geometry import (
    annotation_mask,
    box_iou,
    box_to_mask,
    largest_component,
    mask_bbox,
    mask_components,
    mask_to_polygon,
    sample_stroke_points,
    stroke_mask,
)
from pipeline.studio.infer import results_to_json, run_predict
from pipeline.studio.store import Invalid, StudioStore

log = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[int, int], None]]


def _split(prompts: list[dict]) -> tuple[list[dict], list[dict]]:
    pos = [p for p in prompts if int(p.get("polarity", 1)) >= 0]
    neg = [p for p in prompts if int(p.get("polarity", 1)) < 0]
    return pos, neg


def prompt_mask(p: dict, shape: tuple[int, int]) -> np.ndarray:
    if p.get("type") == "box":
        return box_to_mask(p["bbox"], shape)
    if p.get("type") == "stroke":
        if not p.get("points"):
            raise Invalid("stroke prompt has no points")
        return stroke_mask(p["points"], float(p.get("radius", 8)), shape)
    raise Invalid(f"unknown prompt type {p.get('type')!r}")


def negative_mask(prompts: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=bool)
    for p in prompts:
        if int(p.get("polarity", 1)) < 0:
            out |= prompt_mask(p, shape)
    return out


# ---- SAM: one object from the whole draft --------------------------------


GRANULARITIES = ("auto", "fine", "medium", "coarse")


def sam_segment(
    store: StudioStore,
    image_id: str,
    prompts: list[dict],
    sam_model: str = "sam2.1_t",
    granularity: str = "auto",
) -> Optional[dict]:
    """Segment the single object the draft describes.

    Box → SAM box prompt (first positive box wins; SAM takes one box per
    object). Positive strokes → evenly spaced positive clicks; negative
    strokes → negative clicks, and their painted area is also cut out of
    the final mask so "erase" means erase.

    Granularity: a box pins down the object, so "auto" takes SAM's single
    best mask. Strokes/clicks are ambiguous (one carton vs the whole stack),
    so SAM's three hypotheses (≈ part / object / whole) are ranked by area.
    "auto" takes the smallest one that contains the whole stroke — paint
    across one carton and you get that carton; paint across its top and
    side and you get the full carton rather than one face. "fine",
    "medium" and "coarse" pick the smallest / middle / largest hypothesis
    that touches the painted pixels — an explicit override.
    """
    if granularity not in GRANULARITIES:
        raise Invalid(f"granularity must be one of {GRANULARITIES}")
    if not prompts:
        raise Invalid("draw a box or paint a stroke first")
    img = store.read_image(image_id)
    shape = img.shape[:2]
    pos, neg = _split(prompts)
    box = next((p["bbox"] for p in pos if p.get("type") == "box"), None)
    points: list[list[float]] = []
    labels: list[int] = []
    for p in pos:
        if p.get("type") == "stroke":
            for q in sample_stroke_points(p["points"], max_points=6):
                points.append(q)
                labels.append(1)
    for p in neg:
        if p.get("type") == "stroke":
            for q in sample_stroke_points(p["points"], max_points=4):
                points.append(q)
                labels.append(0)
        elif p.get("type") == "box":
            x1, y1, x2, y2 = p["bbox"]
            points.append([(x1 + x2) / 2, (y1 + y2) / 2])
            labels.append(0)
    if box is None and not any(lab == 1 for lab in labels):
        raise Invalid("SAM needs at least one positive box or stroke")

    has_strokes = any(p.get("type") == "stroke" for p in pos)
    multimask = granularity != "auto" or (box is None and has_strokes)
    with engines.INFER_LOCK:
        session = engines.sam_session(sam_model)
        cands = session.candidates(
            image_id, img, bbox=box, points=points or None, labels=labels or None, multimask=multimask
        )
    if not cands:
        return None
    if not multimask:
        mask, score = max(cands, key=lambda c: c[1])
    else:
        cands = sorted(cands, key=lambda c: int(c[0].sum()))
        if granularity == "auto":
            centre = [q for p in pos if p.get("type") == "stroke" for q in sample_stroke_points(p["points"], 64)]
            xy = np.clip(np.round(np.asarray(centre)).astype(int), 0, [shape[1] - 1, shape[0] - 1])

            def cover(m: np.ndarray) -> float:
                return float(m[xy[:, 1], xy[:, 0]].mean())

            ok = [c for c in cands if cover(c[0]) >= 0.9]
            mask, score = ok[0] if ok else max(cands, key=lambda c: cover(c[0]))
        else:
            painted = np.zeros(shape, dtype=bool)
            for p in pos:
                if p.get("type") == "stroke":
                    painted |= prompt_mask(p, shape)
            if painted.any():
                touching = [c for c in cands if (c[0] & painted).any()]
                cands = touching or cands
            pick = {"fine": 0, "coarse": len(cands) - 1}.get(granularity, len(cands) // 2)
            mask, score = cands[pick]
    mask = mask.copy()

    mask &= ~negative_mask(prompts, shape)
    if box is not None:
        # SAM occasionally bleeds outside a box prompt; the box is a hard limit.
        mask &= box_to_mask([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], shape)
    mask = largest_component(mask)
    poly = mask_to_polygon(mask)
    if poly is None:
        return None
    class_id = next((int(p["class_id"]) for p in pos if p.get("class_id") is not None), None)
    return {
        "class_id": class_id,
        "polygon": poly,
        "bbox": mask_bbox(mask),
        "score": round(score, 4),
        "area": int(mask.sum()),
        "source": f"sam:{sam_model}",
    }


# ---- Paint directly into labels ------------------------------------------


def paint_to_annotations(store: StudioStore, image_id: str, prompts: list[dict]) -> list[dict]:
    """Boxes → box labels; strokes → mask labels (one per connected blob).

    Negative strokes cut holes in the painted masks. Strokes are grouped by
    class so painting two separate cartons with the same class yields two
    instances, while one carton painted with several overlapping strokes
    yields one.
    """
    rec = store.get_image(image_id)
    shape = (rec["height"], rec["width"])
    pos, _ = _split(prompts)
    neg = negative_mask(prompts, shape)
    out: list[dict] = []
    by_class: dict[int, np.ndarray] = {}
    for p in pos:
        cid = p.get("class_id")
        if cid is None:
            raise Invalid("every prompt needs a class to become a label")
        cid = int(cid)
        if p.get("type") == "box":
            out.append({"class_id": cid, "bbox": list(map(float, p["bbox"])), "polygon": None, "source": "manual"})
        else:
            m = prompt_mask(p, shape)
            by_class[cid] = by_class.get(cid, np.zeros(shape, bool)) | m
    for cid, m in by_class.items():
        for comp in mask_components(m & ~neg, min_area=25):
            poly = mask_to_polygon(comp)
            if poly:
                out.append({"class_id": cid, "polygon": poly, "bbox": mask_bbox(comp), "source": "brush"})
    return out


# ---- Shared post-processing ----------------------------------------------


def _class_lookup(store: StudioStore) -> dict[str, int]:
    return {c["name"].lower(): int(c["id"]) for c in store.classes()}


def _finalise_items(
    store: StudioStore,
    image_id: str,
    img: np.ndarray,
    dets: list[dict],
    source: str,
    refine_with_sam: bool,
    sam_model: str,
    dedupe_iou: float = 0.7,
) -> list[dict]:
    """Map predictor classes onto dataset classes, refine, de-duplicate."""
    lookup = _class_lookup(store)
    existing = store.annotations(image_id)
    items = []
    for d in dets:
        cid = d.get("dataset_class_id")
        if cid is None:
            cid = lookup.get(str(d.get("class_name", "")).lower(), -1)
        item = {
            "class_id": int(cid),
            "class_name": d.get("class_name", ""),
            "bbox": d["bbox"],
            "polygon": d.get("polygon"),
            "score": d.get("score"),
            "source": source,
        }
        if refine_with_sam:
            with engines.INFER_LOCK:
                session = engines.sam_session(sam_model)
                m, _ = session.segment(image_id, img, bbox=d["bbox"])
            m &= box_to_mask([d["bbox"][0] - 2, d["bbox"][1] - 2, d["bbox"][2] + 2, d["bbox"][3] + 2], img.shape[:2])
            poly = mask_to_polygon(largest_component(m))
            if poly:
                item["polygon"] = poly
                item["source"] = f"{source}+sam"
        # Skip anything already labelled (same class, heavy overlap) — the
        # reference image re-detects its own examples, and re-running a
        # prompt shouldn't pile duplicates into review.
        if any(
            int(a["class_id"]) == item["class_id"] and box_iou(a["bbox"], item["bbox"]) >= dedupe_iou
            for a in existing
        ):
            continue
        items.append(item)
    return items


def _targets(store: StudioStore, scope: str, image_id: Optional[str], image_ids: Optional[list[str]]) -> list[str]:
    if scope == "image":
        if not image_id:
            raise Invalid("scope 'image' needs image_id")
        store.get_image(image_id)
        return [image_id]
    recs = store.list_images()
    if scope == "all":
        return [r["id"] for r in recs]
    if scope == "unlabeled":
        return [r["id"] for r in recs if r["n_annotations"] == 0 and not r.get("negative")]
    if scope == "ids":
        known = {r["id"] for r in recs}
        return [i for i in image_ids or [] if i in known]
    raise Invalid(f"unknown scope {scope!r}")


def _store_suggestions(
    store: StudioStore, per_image: dict[str, list[dict]], source: str
) -> dict[str, int]:
    counts = {}
    for iid, items in per_image.items():
        data = store.set_suggestions(iid, items, source)
        counts[iid] = len(data["items"])
    return counts


# ---- YOLOE visual prompts: "find more like this" -------------------------


def _reference_masks(
    store: StudioStore, ref: dict
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    iid = ref["image_id"]
    img = store.read_image(iid)
    shape = img.shape[:2]
    prompts = ref.get("prompts") or []
    neg = negative_mask(prompts, shape)
    masks: list[np.ndarray] = []
    cls: list[int] = []
    for p in _split(prompts)[0]:
        if p.get("class_id") is None:
            raise Invalid("visual prompts need a class")
        m = prompt_mask(p, shape) & ~neg
        if m.sum() >= 16:
            masks.append(m)
            cls.append(int(p["class_id"]))
    if ref.get("use_annotations"):
        wanted = set(ref.get("annotation_ids") or [])
        for a in store.annotations(iid):
            if wanted and a["id"] not in wanted:
                continue
            masks.append(annotation_mask(a, shape))
            cls.append(int(a["class_id"]))
    return img, masks, cls


def visual_prompt_detect(
    store: StudioStore,
    refs: list[dict],
    scope: str = "image",
    image_id: Optional[str] = None,
    image_ids: Optional[list[str]] = None,
    family: str = "11",
    size: str = "s",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    refine_with_sam: bool = False,
    sam_model: str = "sam2.1_t",
    progress: ProgressFn = None,
) -> dict:
    """Detect objects similar to the examples in `refs` across target images.

    Each ref is {"image_id", "prompts": [...], "use_annotations": bool,
    "annotation_ids": [...]}. Visual-prompt embeddings are computed per
    reference image and averaged per class (then re-normalised), so
    examples from several images make a sturdier "concept" than one.
    """
    import torch
    import torch.nn.functional as F

    sums: dict[int, Any] = {}
    counts: dict[int, int] = {}
    n_examples = 0
    with engines.INFER_LOCK:
        model = engines.load_yoloe(family, size)
        for ref in refs:
            img, masks, cls = _reference_masks(store, ref)
            if not masks:
                continue
            n_examples += len(masks)
            uniq, vpe = engines.yoloe_visual_embeddings(
                model, img, np.stack(masks).astype(np.uint8), np.array(cls), imgsz=imgsz
            )
            for j, c in enumerate(uniq):
                v = vpe[0, j].float().cpu()
                sums[c] = v if c not in sums else sums[c] + v
                counts[c] = counts.get(c, 0) + 1
        if not sums:
            raise Invalid("no usable examples — draw a box or paint a stroke (or pick labelled images)")
        class_ids = sorted(sums)
        names_by_id = {int(c["id"]): c["name"] for c in store.classes()}
        names = [names_by_id.get(c, f"class{c}") for c in class_ids]
        pe = torch.stack([F.normalize(sums[c] / counts[c], dim=-1) for c in class_ids])[None]
        p = next(model.model.parameters())
        engines.yoloe_set_visual_classes(model, names, pe.to(device=p.device, dtype=p.dtype))

        targets = _targets(store, scope, image_id, image_ids)
        per_image: dict[str, list[dict]] = {}
        for k, tid in enumerate(targets):
            img = store.read_image(tid)
            r = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=engines.device(), verbose=False)[0]
            dets = results_to_json(r)["detections"]
            for d in dets:
                d["dataset_class_id"] = class_ids[d["class_id"]] if d["class_id"] < len(class_ids) else -1
                d["class_name"] = names[d["class_id"]] if d["class_id"] < len(names) else d["class_name"]
            per_image[tid] = _finalise_items(
                store, tid, img, dets, f"yoloe-{family}{size}:visual", refine_with_sam, sam_model
            )
            if progress:
                progress(k + 1, len(targets))

    counts_out = _store_suggestions(store, per_image, f"yoloe-{family}{size}:visual")
    return {
        "engine": engines.yoloe_name(family, size).removesuffix(".pt"),
        "n_examples": n_examples,
        "classes": [{"id": c, "name": n} for c, n in zip(class_ids, names)],
        "counts": counts_out,
    }


# ---- Text / prompt-free / model assist -----------------------------------


def detect_to_suggestions(
    store: StudioStore,
    spec: dict,
    scope: str = "image",
    image_id: Optional[str] = None,
    image_ids: Optional[list[str]] = None,
    params: Optional[dict] = None,
    refine_with_sam: bool = False,
    sam_model: str = "sam2.1_t",
    assign_class_id: Optional[int] = None,
    progress: ProgressFn = None,
) -> dict:
    """Run any predictor over target images and store results as suggestions.

    `assign_class_id` forces every detection onto one dataset class — handy
    when a COCO model's "suitcase" is really your "carton".
    """
    targets = _targets(store, scope, image_id, image_ids)
    source = {
        "yoloe-text": f"yoloe-{spec.get('family', '26')}{spec.get('size', 's')}:text",
        "yoloe-pf": f"yoloe-{spec.get('family', '26')}{spec.get('size', 's')}:prompt-free",
        "yolo": f"yolo{spec.get('family', '26')}{spec.get('size', 'n')}-{spec.get('task', 'detect')}",
        "yolo26": f"yolo26{spec.get('size', 'n')}-{spec.get('task', 'detect')}",
        "trained": f"model:{spec.get('model_id')}",
    }.get(str(spec.get("kind")), str(spec.get("kind")))
    per_image: dict[str, list[dict]] = {}
    total = 0
    for k, tid in enumerate(targets):
        img = store.read_image(tid)
        pred = run_predict(store, spec, img, params)
        dets = pred["detections"]
        if assign_class_id is not None:
            for d in dets:
                d["dataset_class_id"] = int(assign_class_id)
        per_image[tid] = _finalise_items(store, tid, img, dets, source, refine_with_sam, sam_model)
        total += len(per_image[tid])
        if progress:
            progress(k + 1, len(targets))
    counts = _store_suggestions(store, per_image, source)
    return {"engine": source, "counts": counts, "n_total": total}


def accept_suggestions(
    store: StudioStore,
    image_id: str,
    ids: Optional[list[str]] = None,
    min_score: float = 0.0,
    class_id: Optional[int] = None,
    create_missing_classes: bool = True,
) -> dict:
    """Promote suggestions to annotations.

    Suggestions whose class doesn't exist in the dataset yet (text prompt
    "pallet" before a "pallet" class exists) get the class created, or are
    forced onto `class_id` when given.
    """
    with store.lock:
        sugg = store.suggestions(image_id)["items"]
        chosen = [
            s for s in sugg
            if (ids is None or s["id"] in set(ids)) and (s.get("score") is None or s["score"] >= min_score)
        ]
        new_anns = []
        for s in chosen:
            cid = class_id if class_id is not None else s["class_id"]
            if cid is None or int(cid) < 0 or int(cid) not in store.class_ids():
                if class_id is None and create_missing_classes and s.get("class_name"):
                    cid = store.ensure_class(s["class_name"])["id"]
                else:
                    continue
            new_anns.append(
                {
                    "class_id": int(cid),
                    "bbox": s["bbox"],
                    "polygon": s.get("polygon"),
                    "source": s.get("source") or "suggestion",
                    "score": s.get("score"),
                }
            )
        anns = store.add_annotations(image_id, new_anns)
        remaining = store.clear_suggestions(image_id, [s["id"] for s in chosen])
    return {"annotations": anns, "suggestions": remaining, "accepted": len(new_anns)}
