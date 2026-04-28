"""Student trainers — the dispatchable training backends.

Each module under here owns one architecture's `train` / `eval` /
`time_inference` triple and registers itself with the registry on
import. `pipeline.optimize` imports this package, which is what
guarantees the `@register(...)` side-effects fire before
`make_trainer(name)` is called.

Public surface
--------------
Importers should pull from this top-level module rather than reaching
into submodules — keeps the rest of the codebase free of trainer-specific
imports.

  • `make_trainer(name)`   — instantiate a registered trainer.
  • `list_trainers()`      — names available right now (drives the GUI
                             dropdown and the validator).
  • `register(name)`       — decorator for new trainer classes (used by
                             the trainer modules themselves; not normally
                             called by application code).
  • `StudentTrainer`,
    `TrainResult`,
    `INFERENCE_TIMING_SAMPLES` — the contract types.

Adding a new trainer file: drop it under `pipeline/students/`, add an
`@register("…") class …(YoloTrainer): base_model = "…"` block, and add
a `from . import <module>  # noqa: F401` line below so the import-time
side-effect fires. The yolo trainer module lands in commit 1.2.
"""

from __future__ import annotations

from pipeline.students.base import (
    INFERENCE_TIMING_SAMPLES,
    StudentTrainer,
    TrainResult,
)
from pipeline.students.registry import (
    TRAINERS,
    list_trainers,
    make_trainer,
    register,
)

__all__ = [
    "INFERENCE_TIMING_SAMPLES",
    "StudentTrainer",
    "TRAINERS",
    "TrainResult",
    "list_trainers",
    "make_trainer",
    "register",
]
