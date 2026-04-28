"""RT-DETR student trainer (Phase 2.1).

Ultralytics dispatches RT-DETR through the same `YOLO()` class that drives
the YOLOv8 family — same `.train()` / `.val()` / `.predict()` surface,
same metric extraction, same on-disk weights layout. So `YoloTrainer`'s
body works for RT-DETR as-is; the only thing that changes is the upstream
checkpoint name, hence a one-line subclass.

Skipping `rtdetr-x` for now — too heavy for a laptop test loop. If
someone later wants it, `@register("rtdetr-x") class RTDETRExtra(YoloTrainer):
base_model = "rtdetr-x.pt"` is the entire patch.

Kwarg compatibility note
------------------------
`YoloTrainer` calls `model.predict(path, device=…, verbose=False)`,
`model.val(data=…, device=…, verbose=False, plots=False, save_json=False)`,
and `model.train(data=…, epochs=…, imgsz=…, device=…, project=…,
name=…, exist_ok=True, verbose=True, plots=False)`. RT-DETR's Ultralytics
wrapper accepts all of these — they're framework-level args, not
augmentation flags. No method overrides needed.
"""

from __future__ import annotations

from pipeline.students.registry import register
from pipeline.students.yolo import YoloTrainer


@register("rtdetr-l")
class RTDETRLarge(YoloTrainer):
    """RT-DETR-L. Ultralytics dispatches RT-DETR through the same YOLO()
    class, so the trainer body in YoloTrainer Just Works — only the
    base_model differs. Skipping rtdetr-x for now (too heavy for a
    laptop test loop).
    """

    base_model = "rtdetr-l.pt"
