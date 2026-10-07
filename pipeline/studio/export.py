"""Studio annotations → Ultralytics YOLO dataset (detect / segment / OBB).

Output layout (what `model.train(data=…/data.yaml)` expects):

    out/
      data.yaml
      images/{train,val,test}/<iid>.<ext>    hard links to the store (copy fallback)
      labels/{train,val,test}/<iid>.txt

Which images go in:
  • images with ≥1 annotation, and
  • images explicitly marked `negative` (background-only examples).
Unlabelled images are left out — the dataset is also where test images
live, and silently training on them as "nothing here" would teach the
model to miss objects.

Label formats (all normalised to [0, 1]):
  detect   cls cx cy w h
  segment  cls x1 y1 x2 y2 …            (box-only labels → rectangle polygon)
  obb      cls x1 y1 x2 y2 x3 y3 x4 y4  (polygon → minAreaRect corners)

Class ids are remapped to a contiguous 0..K-1 range in the order of the
store's class list; `names` in data.yaml records the mapping.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.studio.store import StudioStore, auto_split

TASKS = ("detect", "segment", "obb")


@dataclass
class ExportSummary:
    root: Path
    data_yaml: Path
    names: list[str]
    class_map: dict[int, int]  # store class id → contiguous id
    counts: dict[str, int] = field(default_factory=dict)  # images per split
    instances: int = 0
    boxes_as_polygons: int = 0  # segment/obb: box-only labels converted
    warnings: list[str] = field(default_factory=list)


def _norm(v: float, size: int) -> float:
    return min(1.0, max(0.0, v / size))


def label_line(task: str, cls: int, ann: dict, w: int, h: int) -> tuple[str, bool]:
    """One YOLO label line. Returns (line, was_box_converted)."""
    x1, y1, x2, y2 = ann["bbox"]
    poly = ann.get("polygon")
    if task == "detect":
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return f"{cls} {_norm(cx, w):.6f} {_norm(cy, h):.6f} {_norm(x2 - x1, w):.6f} {_norm(y2 - y1, h):.6f}", False
    converted = not poly
    pts = np.asarray(poly if poly else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if task == "obb":
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        pts = cv2.boxPoints(rect)
    coords = " ".join(f"{_norm(float(x), w):.6f} {_norm(float(y), h):.6f}" for x, y in pts)
    return f"{cls} {coords}", converted


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_yolo_dataset(
    store: StudioStore,
    out: Path,
    task: str,
    val_pct: int = 20,
    class_ids: Optional[list[int]] = None,
) -> ExportSummary:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)

    # `class_ids` fixes the class order — a trained model's output index i
    # means class_ids[i], so re-exporting for validation must keep it.
    by_id = {int(c["id"]): c for c in store.classes()}
    if class_ids is None:
        ordered = list(by_id)
    else:
        missing = [c for c in class_ids if int(c) not in by_id]
        if missing:
            raise ValueError(f"class id(s) {missing} no longer exist in this dataset")
        ordered = [int(c) for c in class_ids]
    class_map = {cid: i for i, cid in enumerate(ordered)}
    names = [by_id[cid]["name"] for cid in ordered]
    summary = ExportSummary(
        root=out, data_yaml=out / "data.yaml", names=names, class_map=class_map,
        counts={"train": 0, "val": 0, "test": 0},
    )

    chosen: list[tuple[dict, str, list[dict]]] = []
    for rec in store.list_images():
        anns = [a for a in store.annotations(rec["id"]) if int(a["class_id"]) in class_map]
        if not anns and not rec.get("negative"):
            continue
        split = rec["split"] if rec["split"] in ("train", "val", "test") else auto_split(rec["id"], val_pct)
        chosen.append((rec, split, anns))

    # Small datasets: make sure both train and val exist. With a single
    # labelled image we validate on the training image — numbers are then
    # optimistic, which the warning says out loud.
    splits = [s for _, s, _ in chosen]
    if chosen and "train" not in splits:
        rec, _, anns = chosen[0]
        chosen[0] = (rec, "train", anns)
    splits = [s for _, s, _ in chosen]
    if chosen and "val" not in splits:
        auto_idx = [i for i, (r, s, _) in enumerate(chosen) if s == "train" and r["split"] == "auto"]
        if len(auto_idx) >= 2:
            i = auto_idx[-1]
            chosen[i] = (chosen[i][0], "val", chosen[i][2])
        else:
            summary.warnings.append(
                "No validation images — validating on the training set, so metrics will be optimistic. "
                "Label a few more images (or set some to 'val')."
            )

    for rec, split, anns in chosen:
        src = store.images_dir / rec["file"]
        _link_or_copy(src, out / "images" / split / rec["file"])
        lines = []
        for a in anns:
            line, conv = label_line(task, class_map[int(a["class_id"])], a, rec["width"], rec["height"])
            lines.append(line)
            summary.boxes_as_polygons += int(conv)
        lbl = out / "labels" / split / f"{Path(rec['file']).stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        summary.counts[split] += 1
        summary.instances += len(lines)

    has_val = summary.counts["val"] > 0
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        f"val: images/{'val' if has_val else 'train'}",
    ]
    if summary.counts["test"]:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: {_yaml_str(n)}" for i, n in enumerate(names)]
    out.mkdir(parents=True, exist_ok=True)
    summary.data_yaml.write_text("\n".join(yaml_lines) + "\n")

    if task != "detect" and summary.boxes_as_polygons:
        summary.warnings.append(
            f"{summary.boxes_as_polygons} box-only label(s) were converted to rectangles for {task} — "
            "refine them with SAM for better masks."
        )
    if not chosen:
        summary.warnings.append("No labelled images yet.")
    return summary


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
