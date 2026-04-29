"""Ultralytics-backed student trainers (YOLOv8 family).

`YoloTrainer` ports the existing `train_yolo` / `eval_yolo` /
`time_inference` bodies from `pipeline.distill` into a single class
parameterised by `base_model`. Registering a new size variant is then
a one-line subclass with a different `base_model` string — see the
`@register(...)` blocks at the bottom.

Why one class with subclasses
-----------------------------
The training/eval/timing bodies are byte-identical across yolov8n /s/m
— only the upstream weight checkpoint changes. We could pass the model
name as a constructor arg, but the registry's contract is "name → zero-arg
class", and shipping three classes makes the registration declarative
and grep-able ("which sizes do we support?" → grep `class YoloV8` here).

RT-DETR (Phase 2) reuses this base class as-is — Ultralytics dispatches
RT-DETR through the same `YOLO()` class internally — so the only Phase 2
addition is a new file with `@register("rtdetr-l") class …(YoloTrainer):
base_model = "rtdetr-l.pt"`.
"""

from __future__ import annotations

import logging
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from pipeline.students.base import (
    INFERENCE_TIMING_SAMPLES,
    StudentTrainer,
    TrainResult,
)
from pipeline.students.registry import register

log = logging.getLogger(__name__)


def _pick_device() -> str:
    """Best available torch device for Ultralytics on this box.

    On Apple Silicon (the project's primary target) MPS is dramatically
    faster than CPU — Ultralytics auto-detects it but we set it explicitly
    so the progress logs show the right thing.

    Lifted verbatim from `pipeline.distill._pick_device` so the dispatcher
    path produces numerically identical numbers to the pre-Phase-1 code
    path. `pipeline.distill` re-exports this for callers that historically
    imported it from there (a thin shim during the transition).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dep
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloTrainer:
    """Ultralytics YOLO/RT-DETR trainer.

    Subclasses set `base_model` to the upstream checkpoint name
    (`"yolov8n.pt"`, `"rtdetr-l.pt"`, …). The shared bodies below are
    framework-agnostic enough that subclassing is one line of code.

    Implements the `StudentTrainer` Protocol (`name`, `train`, `eval`,
    `time_inference`); `name` is stamped on by the `@register` decorator
    so subclasses don't need to set it manually.
    """

    # Set by subclasses; the type-checker would complain about it being
    # unset on the base class otherwise. The base class itself is never
    # instantiated — it's an abstract-ish trainer.
    base_model: str = ""

    # Stamped by `@register(name)` at import time. Declared here for
    # type-checkers that look at the class body.
    name: str = ""

    def train(
        self,
        *,
        data_yaml: Path,
        student_dir: Path,
        epochs: int,
        imgsz: int,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> TrainResult:
        """Train Ultralytics on the prepared dataset; return weights + timing.

        Body ported verbatim from `pipeline.distill.train_yolo` — same
        device pick, same callback, same log-capture setup, same fallback
        from `best.pt` → `last.pt`. The only difference is the return
        type (`TrainResult` instead of a `(Path, float)` tuple).

        `progress(epoch, total_epochs)` is invoked after each epoch via
        an Ultralytics callback so the GUI's progress bar tracks training
        rather than freezing for the entire `train_seconds` duration.
        """
        from ultralytics import YOLO  # imported lazily — heavy module

        device = _pick_device()
        log.info(
            "Distill: training %s on %s, %d epochs, imgsz=%d",
            self.base_model, device, epochs, imgsz,
        )

        model = YOLO(self.base_model)

        if progress:
            # Ultralytics calls each callback with the trainer object; we
            # peek at trainer.epoch (0-indexed) to report 1-based progress.
            def _on_epoch_end(trainer):
                try:
                    progress(int(trainer.epoch) + 1, int(epochs))
                except Exception as e:  # pragma: no cover — never let progress kill training
                    log.warning("progress callback failed: %s", e)

            model.add_callback("on_train_epoch_end", _on_epoch_end)

        # Ultralytics' `project` arg is interpreted relative to the current
        # CWD for filesystem ops, but newer versions (8.3+) can also resolve
        # it against SETTINGS["runs_dir"] in some code paths. Pass an
        # absolute path so output ends up exactly where we look for it on
        # success and error reporting stays accurate.
        project_dir = (student_dir / "ultralytics").resolve()
        project_dir.mkdir(parents=True, exist_ok=True)

        # Tee Ultralytics' chatter into a per-run log file so post-mortems
        # don't have to scroll the uvicorn console. We attach a FileHandler
        # to the "ultralytics" logger (catches LOGGER.warning/info), and
        # also redirect stdout/stderr (catches print() and tqdm). The two
        # streams may interleave, but for debugging "training silently
        # produced no weights" that's exactly what we want.
        log_path = student_dir / "train.log"
        ult_logger = logging.getLogger("ultralytics")
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        ult_logger.addHandler(fh)

        t0 = time.perf_counter()
        try:
            with open(log_path, "a") as logf, redirect_stdout(logf), redirect_stderr(logf):
                model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    imgsz=imgsz,
                    device=device,
                    project=str(project_dir),
                    name="train",
                    exist_ok=True,
                    verbose=True,  # log everything; we capture it to train.log
                    plots=False,  # save the disk space; we don't render Ultralytics plots in the UI
                )
        finally:
            ult_logger.removeHandler(fh)
            fh.close()
        elapsed = time.perf_counter() - t0

        # Locate best.pt — Ultralytics writes it into project/name/weights/best.pt.
        best = project_dir / "train" / "weights" / "best.pt"
        if not best.exists():
            # Fallback: last.pt is usually present even if best.pt save was suppressed.
            last = project_dir / "train" / "weights" / "last.pt"
            if not last.exists():
                raise RuntimeError(
                    f"Ultralytics produced no weights under {best.parent} — "
                    f"training may have failed silently. See {log_path} for the "
                    "captured stdout/stderr from the trainer."
                )
            best = last

        # Copy to a stable, predictable location at the student dir root so
        # downstream callers don't have to know about the ultralytics subtree.
        out = student_dir / "best.pt"
        shutil.copy2(best, out)
        log.info("Distill: training done in %.1fs; weights -> %s", elapsed, out)
        return TrainResult(weights_path=out, train_seconds=elapsed)

    def eval(
        self, *, weights: Path, data_yaml: Path
    ) -> tuple[float, float]:
        """Compute (map50, map50_95) — body ported from `eval_yolo`.

        Used per eval teacher to populate the per-eval-teacher
        transferability table. Ultralytics' `model.val()` returns a
        `DetMetrics` object with `.box.map50` (mAP@0.5) and `.box.map`
        (the unsuffixed `.map` IS map@0.5:0.95 in Ultralytics' API).
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
        res = model.val(
            data=str(data_yaml),
            device=_pick_device(),
            verbose=False,
            plots=False,
            save_json=False,
        )
        map50 = float(getattr(res.box, "map50", 0.0) or 0.0)
        map5095 = float(getattr(res.box, "map", 0.0) or 0.0)
        return map50, map5095

    def time_inference(
        self,
        *,
        weights: Path,
        sample_image_dir: Path,
        n_samples: int = INFERENCE_TIMING_SAMPLES,
    ) -> tuple[float, float, float]:
        """Body ported from `pipeline.distill.time_inference`.

        Returns (avg_ms, p50_ms, p95_ms). The first inference call is
        discarded as a warmup pass — kernel compilation, JIT warm-up,
        weight-to-device transfer all land on call #1 and would skew
        the median otherwise. Subsequent calls are what the user feels.
        """
        from ultralytics import YOLO

        jpgs = sorted(sample_image_dir.rglob("*.jpg"))
        if not jpgs:
            log.warning("no images for timing under %s — skipping", sample_image_dir)
            return 0.0, 0.0, 0.0
        sample = jpgs[:n_samples]

        model = YOLO(str(weights))
        device = _pick_device()

        # Warmup pass — discarded.
        model.predict(str(sample[0]), device=device, verbose=False)

        timings: list[float] = []
        for p in sample:
            t0 = time.perf_counter()
            model.predict(str(p), device=device, verbose=False)
            timings.append((time.perf_counter() - t0) * 1000.0)

        if not timings:
            return 0.0, 0.0, 0.0
        timings.sort()
        avg = sum(timings) / len(timings)
        p50 = timings[len(timings) // 2]
        p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
        return avg, p50, p95

    def predict(
        self,
        *,
        weights: Path,
        frames: Iterable[Any],
    ) -> Iterator[list[dict]]:
        """Phase 5: stream detections over a frame iterator.

        Loads the Ultralytics model once and calls `model.predict()` per
        frame. We pass numpy arrays directly (BGR uint8) — Ultralytics
        accepts them and produces `Results` with `.boxes.xyxy` /
        `.boxes.conf` / `.boxes.cls` and a `.names` map for class names.

        Each yielded list mirrors `runs.detection_to_dict`'s output so the
        caller can write straight to `predictions/per_frame.jsonl` without
        another conversion pass.
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
        device = _pick_device()
        names = getattr(model, "names", None) or {}

        for frame in frames:
            results = model.predict(frame, device=device, verbose=False)
            dets: list[dict] = []
            if not results:
                yield dets
                continue
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                yield dets
                continue
            # `.xyxy` / `.conf` / `.cls` are torch tensors; pull to numpy
            # for cheap iteration without forcing the whole batch through
            # tensor ops.
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            for i in range(len(xyxy)):
                cid = int(cls[i])
                cname = (
                    names.get(cid, f"class_{cid}") if isinstance(names, dict) else
                    (names[cid] if 0 <= cid < len(names) else f"class_{cid}")
                )
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                dets.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "score": float(conf[i]),
                        "class_id": cid,
                        "class_name": str(cname),
                    }
                )
            yield dets


# ---- Registered size variants ---------------------------------------------
#
# Each `@register("…")` line populates `pipeline.students.registry.TRAINERS`
# at import time so `make_trainer(name)` can find the class. Adding a new
# size is a one-liner here.


@register("yolov8n")
class YoloV8Nano(YoloTrainer):
    """Smallest YOLOv8 variant — the project default. ~3M params,
    fastest to train and infer; what Phase 1's "no behaviour change"
    acceptance gate is measured against."""

    base_model = "yolov8n.pt"


@register("yolov8s")
class YoloV8Small(YoloTrainer):
    """Slightly larger — ~11M params. Better accuracy on hard scenes
    (small objects, motion blur) at ~2× the inference cost of nano."""

    base_model = "yolov8s.pt"


@register("yolov8m")
class YoloV8Medium(YoloTrainer):
    """Mid-size — ~25M params. The largest size we expose by default;
    larger variants (l, x) belong on a server, not the user's laptop."""

    base_model = "yolov8m.pt"


# Tell type-checkers the base class satisfies the Protocol — done at the
# bottom so the explicit assertion can reference fully-defined symbols.
_protocol_check: type[StudentTrainer] = YoloTrainer  # type: ignore[type-abstract]
