"""Student-trainer base types.

Defines the small, uniform interface every student trainer (yolov8n,
rtdetr-l, dinov3-detr, …) must satisfy so `pipeline.optimize` can stay
architecture-agnostic. The orchestration layer just calls
`make_trainer(name)` and drives the same three methods regardless of
which framework is doing the actual work underneath.

Why a Protocol rather than an ABC
---------------------------------
The trainers we have today (Ultralytics YOLO, RT-DETR) are framework
classes whose method shapes are dictated by upstream — making them
inherit a custom ABC would force a wrapper layer. A `Protocol` lets a
plain class with the right method signatures count as a `StudentTrainer`
for type-checking purposes without forcing a runtime base class. The
`@register("name")` decorator is the actual hook the registry uses;
the Protocol is purely a typing aid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol


# Frames sampled for inference timing. Pulled into `base.py` so trainers
# don't have to import `pipeline.distill` (the original home) — the value
# itself stays in distill.py for back-compat re-export, but trainers
# pick it up from here.
INFERENCE_TIMING_SAMPLES = 30


@dataclass
class TrainResult:
    """What a successful `train()` call returns.

    `weights_path` is the on-disk artefact downstream code (eval, inference
    timing, the GUI's "model size" display) consumes. `train_seconds` is
    the wall-clock time including all framework overhead — what the user
    feels.

    `epoch_metrics` is reserved for the future learning-curve plot. One
    dict per epoch, at minimum `{"epoch": int, "map50": float}`. May be
    empty for architectures whose framework doesn't expose per-epoch
    metrics cleanly — Phase 1 ships with this empty for YOLO since
    `train_yolo` doesn't currently surface them, and the consumer renders
    "—" rather than failing.
    """

    weights_path: Path
    train_seconds: float
    epoch_metrics: list[dict] = field(default_factory=list)


class StudentTrainer(Protocol):
    """The contract `pipeline.optimize` calls.

    Implementations register themselves via `@register(name)` from
    `pipeline.students.registry`. The registry returns the *class*; the
    orchestration layer instantiates it (with optional params) before
    invoking `train` / `eval` / `time_inference`.

    All methods are keyword-only on the call site so adding a new
    parameter (e.g. `device`) doesn't reshuffle positional args.
    """

    name: str

    def train(
        self,
        *,
        data_yaml: Path,
        student_dir: Path,
        epochs: int,
        imgsz: int,
        progress: Optional[Callable[[int, int], None]],
    ) -> TrainResult:
        """Train on the prepared YOLO-format dataset and return weights.

        `progress(epoch, total_epochs)` is invoked after each epoch when
        the framework supports a callback. It must never raise into the
        trainer — implementations should swallow exceptions from the
        callback so a buggy GUI hook can't crash a long training run.
        """
        ...

    def eval(self, *, weights: Path, data_yaml: Path) -> tuple[float, float]:
        """Compute (map50, map50_95) for `weights` on `data_yaml`.

        Used for per-eval-teacher transferability scoring. Returns
        `(0.0, 0.0)` when the framework can't produce a number rather
        than raising — one bad eval teacher shouldn't tank the whole run.
        """
        ...

    def time_inference(
        self,
        *,
        weights: Path,
        sample_image_dir: Path,
        n_samples: int = INFERENCE_TIMING_SAMPLES,
    ) -> tuple[float, float, float]:
        """Run inference on up to `n_samples` JPGs and return
        (avg_ms, p50_ms, p95_ms).

        The first inference call MUST be discarded as a warmup pass —
        kernel compilation, JIT warm-up and weight-to-device transfer
        all land on call #1 and would skew the median otherwise. The
        returned numbers reflect what the user actually feels at runtime.
        """
        ...

    def predict(
        self,
        *,
        weights: Path,
        frames: Iterable[Any],
    ) -> Iterator[list[dict]]:
        """Stream per-frame predictions over a frame iterator (Phase 5).

        `frames` yields HxWx3 uint8 BGR `np.ndarray` images. The trainer
        loads the model once and yields one list of detection dicts per
        frame — same shape as `pipeline.runs.detection_to_dict` produces:
        `{bbox_xyxy: [x1,y1,x2,y2], score: float, class_id: int, class_name: str}`.

        Used by `pipeline.student_run` to drive a Student against a video
        end-to-end. Implementations should NOT keep the full prediction
        list in memory — yield one frame's worth at a time so memory
        stays flat across long videos.
        """
        ...
