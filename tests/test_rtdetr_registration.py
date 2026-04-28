"""Registration tests for the RT-DETR student trainer (Phase 2.1 / 2.3).

The trainer body is `YoloTrainer`'s body — Ultralytics dispatches RT-DETR
through the same `YOLO()` class, so all the interesting code already has
test coverage in `test_student_registry.py`. What's worth testing here
is the wiring: the `@register("rtdetr-l")` line plus the import in
`pipeline.students.__init__` are what make `make_trainer("rtdetr-l")`
actually find the class on app boot.

These tests deliberately do NOT instantiate Ultralytics' `YOLO("rtdetr-l.pt")`
— that triggers a weight download (~62 MB) on first run, which is slow
and fails in offline CI. The construction surface we exercise is the
trainer wrapper, not the underlying framework.
"""

from __future__ import annotations

# Importing the package fires the `@register(...)` side effects in
# both `pipeline.students.yolo` and `pipeline.students.rtdetr` so the
# registry is populated before `make_trainer` is called below.
import pipeline.students  # noqa: F401
from pipeline.students import make_trainer
from pipeline.students.registry import list_trainers
from pipeline.students.rtdetr import RTDETRLarge
from pipeline.students.yolo import YoloTrainer


def test_rtdetr_l_in_list_trainers() -> None:
    """`list_trainers()` is what the GUI dropdown reads and what the
    `OptimizeRequest` validator queries. If `rtdetr-l` is missing here
    the architecture <select> won't show it and explicit requests will
    422 — both regressions the spec's Phase 2 acceptance test catches."""
    names = list_trainers()
    assert "rtdetr-l" in names
    # Sanity: pre-existing yolov8n must still be there too. A registry
    # error that wiped existing entries would also fail this.
    assert "yolov8n" in names


def test_make_trainer_returns_rtdetr_large() -> None:
    """`make_trainer("rtdetr-l")` must return an `RTDETRLarge` instance
    with the correct upstream checkpoint name. The checkpoint string is
    what `YOLO(self.base_model)` consumes inside `train()` / `eval()` /
    `time_inference()`; getting it wrong silently swaps in YOLOv8's
    weights instead, which would train without erroring but produce
    nonsense numbers under the rtdetr-l label."""
    trainer = make_trainer("rtdetr-l")
    assert isinstance(trainer, RTDETRLarge)
    assert trainer.base_model == "rtdetr-l.pt"
    # The `@register` decorator stamps the registered name onto the
    # class so instances satisfy the `StudentTrainer` Protocol's
    # `name: str` attribute.
    assert trainer.name == "rtdetr-l"


def test_rtdetr_inherits_yolotrainer_methods() -> None:
    """The whole reason `RTDETRLarge` is a one-line subclass is that
    Ultralytics dispatches RT-DETR through the same `YOLO()` class, so
    `YoloTrainer.train` / `eval` / `time_inference` work as-is. If
    something refactored `RTDETRLarge` to override one of these, this
    test fails loudly — at which point the override needs its own
    coverage. (Subclass-level overrides are fine; this asserts the
    *inheritance chain*, not method identity.)"""
    trainer = make_trainer("rtdetr-l")
    assert isinstance(trainer, YoloTrainer)
    # Every method on the StudentTrainer Protocol must be reachable.
    assert callable(getattr(trainer, "train", None))
    assert callable(getattr(trainer, "eval", None))
    assert callable(getattr(trainer, "time_inference", None))
