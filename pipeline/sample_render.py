"""Side-by-side prediction comparison images for a Student's eval set.

After per-eval-teacher mAP is computed, this module renders N held-out
frames with the Student's predictions in red and the Teacher's ground
truth in green. The image (single frame, both sets of boxes) lands at
`student_dir/samples/<eval_teacher_id>/<frame>.jpg`. The frontend's
SamplePredictionsGrid serves them via the /samples endpoint so the
user can spot *why* mAP is low instead of staring at a single number.

We do NOT render anything during training itself — those frames don't
exist on disk yet. Eval is the right time: by then the YOLO eval dir
already contains both `images/val/*.jpg` (raw frames) and
`labels/val/*.txt` (YOLO-format ground truth).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Number of comparison frames per eval teacher. Six fits a 3-column
# thumbnail grid without a scrollbar at typical detail-pane widths;
# more would be diminishing returns since they're all from the same
# clip.
N_SAMPLES_PER_EVAL = 6

# BGR colors (cv2's native order). Greens for ground truth, reds for
# the Student's predictions — matches the legend the frontend renders.
GT_COLOR = (34, 139, 34)        # forest green
PRED_COLOR = (60, 60, 220)      # red-ish


def render_eval_samples(
    *,
    weights: Path,
    eval_dir: Path,
    class_names: list[str],
    out_dir: Path,
    n_samples: int = N_SAMPLES_PER_EVAL,
) -> list[str]:
    """Render up to `n_samples` comparison images and return the
    sorted list of basenames written.

    `eval_dir` is the per-eval-teacher YOLO dataset root (the one
    `prepare_eval_dataset` builds — has `images/val/` and `labels/val/`
    subdirs). `weights` is the trained Student's `.pt` file.

    Failures are logged and skipped — sample rendering is a value-add,
    not a hard requirement; one corrupt frame shouldn't break the rest.
    """
    images_dir = eval_dir / "images" / "val"
    labels_dir = eval_dir / "labels" / "val"
    if not images_dir.exists():
        log.info("no images/val under %s — skipping samples", eval_dir)
        return []

    jpgs = sorted(images_dir.glob("*.jpg"))[:n_samples]
    if not jpgs:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — heavy modules.
    import cv2
    import numpy as np  # noqa: F401  used implicitly via cv2 / ultralytics
    from ultralytics import YOLO

    from pipeline.students.yolo import _pick_device

    model = YOLO(str(weights))
    device = _pick_device()

    # Warmup pass — kernel compilation hits this one and skews all
    # subsequent predict() calls' boxes-on-disk timing if you trust
    # them for anything. We don't, but it costs nothing.
    try:
        model.predict(str(jpgs[0]), device=device, verbose=False)
    except Exception:
        pass

    written: list[str] = []
    for jpg in jpgs:
        try:
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            h, w = img.shape[:2]

            # 1) Ground truth from YOLO label file. YOLO format: one line
            #    per box, "class_id cx cy w h" all normalized to [0, 1].
            label_path = labels_dir / f"{jpg.stem}.txt"
            if label_path.exists():
                for line in label_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, bw, bh = (float(p) for p in parts[1:5])
                    except ValueError:
                        continue
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
                    cname = (
                        class_names[cid]
                        if 0 <= cid < len(class_names)
                        else f"class_{cid}"
                    )
                    _put_label(img, cname, x1, y1, GT_COLOR)

            # 2) Student predictions.
            results = model.predict(str(jpg), device=device, verbose=False)
            if results:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = (
                        boxes.xyxy.cpu().numpy()
                        if hasattr(boxes.xyxy, "cpu")
                        else boxes.xyxy
                    )
                    conf = (
                        boxes.conf.cpu().numpy()
                        if hasattr(boxes.conf, "cpu")
                        else boxes.conf
                    )
                    cls = (
                        boxes.cls.cpu().numpy()
                        if hasattr(boxes.cls, "cpu")
                        else boxes.cls
                    )
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
                        cid = int(cls[i])
                        cname = (
                            class_names[cid]
                            if 0 <= cid < len(class_names)
                            else f"class_{cid}"
                        )
                        _put_label(
                            img,
                            f"{cname} {float(conf[i]):.2f}",
                            x1,
                            y2,
                            PRED_COLOR,
                        )

            out_path = out_dir / jpg.name
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            written.append(jpg.name)
        except Exception as e:
            log.warning("sample render failed for %s: %s", jpg, e)
            continue

    return written


def _put_label(img, text: str, x: int, y: int, color: tuple) -> None:
    """Draw a small text label with a filled background. Used so labels
    stay readable on busy frames where a thin outlined box's text would
    blend into the scene."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 2
    bx1 = x
    by1 = max(0, y - th - 2 * pad)
    bx2 = x + tw + 2 * pad
    by2 = by1 + th + 2 * pad
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(
        img,
        text,
        (bx1 + pad, by2 - pad),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
