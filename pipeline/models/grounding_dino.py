"""GroundingDINO adapter — text-promptable detection.

Wraps HuggingFace's hosted `IDEA-Research/grounding-dino-tiny` (or `-base`)
checkpoint via `AutoProcessor` + `AutoModelForZeroShotObjectDetection`.
That HF path avoids the messy `groundingdino-py` source install (which
needs a CUDA build) and works on Apple Silicon out of the box.

Block params consumed
---------------------
    prompt          : str   — the text query, e.g. "soccer ball"
                              (multi-class is supported via "ball . player")
    box_threshold   : float — keep boxes with score ≥ this  (default 0.30)
    text_threshold  : float — text-token confidence floor   (default 0.25)
    model_id        : str   — HF repo id, default "IDEA-Research/grounding-dino-tiny"
                              ("…-base" for the larger Swin-B variant)

MPS notes
---------
A handful of attention ops in the model don't have MPS kernels; we set
`PYTORCH_ENABLE_MPS_FALLBACK=1` *before* torch is imported so unsupported
ops fall back to CPU silently. Latency penalty is small relative to the
total per-frame cost.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Optional

# Set MPS fallback before any torch import — once torch is loaded the env
# variable is read and cached. Belt-and-braces in case another module imported
# torch already; no harm setting it twice.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pipeline.blocks.base import BlockKind, Detection, FrameBatch
from pipeline.model_cache import is_hf_cached
from pipeline.models.adapters import Adapter
from pipeline.models.registry import register

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _pick_device() -> str:
    """MPS on M-series, else CUDA, else CPU. Local import keeps adapter file
    importable in environments without torch (e.g. doc generation)."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_prompt(prompt: str) -> str:
    """GroundingDINO wants prompts as lowercase, period-separated phrases.

    The model's tokenizer treats "." as the delimiter between target classes:
        "soccer ball"          → one class
        "ball . player"        → two classes
    Most users will type a single phrase; we just lowercase + ensure a
    trailing period so the tokenizer doesn't trim weirdly.
    """
    p = prompt.strip().lower().rstrip(".").strip()
    return p + "." if p else ""


def _normalize_chips(raw: Any) -> list[str]:
    """Coerce whatever the caller gave us — a list of phrases or a single
    pre-joined string — into a clean list of *chips* (one user-intended
    class per element, lowercase, trimmed, no trailing period).

    Accepted shapes:
        ["soccer ball", "player"]   → ["soccer ball", "player"]
        "soccer ball"               → ["soccer ball"]
        "soccer ball . player"      → ["soccer ball", "player"]   (legacy)
        ""                          → []
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        # Split on '.' so legacy single-string callers that already used
        # GroundingDINO's class-delimiter convention keep working.
        parts = raw.split(".")
    elif isinstance(raw, (list, tuple)):
        parts = []
        for x in raw:
            if x is None:
                continue
            # Each element may itself contain a "." separator if the user
            # pasted a legacy multi-class prompt into a single chip.
            parts.extend(str(x).split("."))
    else:
        parts = [str(raw)]
    chips: list[str] = []
    for p in parts:
        p = p.strip().lower()
        if p:
            chips.append(p)
    # Dedupe while preserving order — the chip-snap matcher below picks the
    # first match, so ordering matters and duplicates would be dead weight.
    seen: set[str] = set()
    out: list[str] = []
    for c in chips:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _snap_label_to_chip(raw_label: str, chips: list[str]) -> Optional[str]:
    """Map a model-emitted label substring back to one of the user's chips.

    The HF post-processor returns whichever consecutive tokens cleared the
    `text_threshold` for a given query — for a prompt like
    "soccer ball . player ." the per-detection label may come back as
    "soccer", "ball", "soccer ball", "player", "." or "". The user, having
    typed "soccer ball" as one phrase, expects EVERY ball-ish detection to
    surface as `soccer ball`, not as a separate `soccer` class.

    Strategy:
      1. Strip whitespace and punctuation from the label.
      2. Compare the cleaned label to each chip in two directions:
            - chip contains label  (model emitted a partial token span)
            - label contains chip  (rare but possible if model overshoots)
         Score each match by length of overlap.
      3. Return the chip with the longest overlap. Ties → the chip that
         appears first in the user's list (stable, predictable).
      4. Empty / pure-punctuation labels → None (drop the detection).

    Returning None is a deliberate "kill it" signal: junk-label detections
    are noise and shouldn't be displayed alongside real classes.
    """
    if not chips:
        return None
    cleaned = raw_label.strip().strip(".").strip().lower()
    if not cleaned:
        return None
    best: Optional[str] = None
    best_overlap = 0
    for chip in chips:
        overlap = 0
        if cleaned in chip:
            overlap = len(cleaned)
        elif chip in cleaned:
            overlap = len(chip)
        else:
            # Word-level intersection as a last resort: handles cases where
            # the model returns a stemmed or reordered token set.
            label_words = set(cleaned.split())
            chip_words = set(chip.split())
            shared = label_words & chip_words
            if shared:
                overlap = sum(len(w) for w in shared)
        if overlap > best_overlap:
            best_overlap = overlap
            best = chip
    return best


@register(BlockKind.DETECT, "groundingdino")
class GroundingDINOAdapter(Adapter):
    """Text-prompted detection per frame.

    Loads model + processor once in `setup()` (this is the expensive bit —
    several hundred MB the first time, then cache hits). `process()` takes
    a single FrameBatch, runs one forward pass, and writes the resulting
    boxes onto `batch.detections`.

    We accept the per-frame batch-of-one cost rather than batching N frames
    so the pipeline runner stays a simple iterator. If we need more speed
    later, the right move is a separate batched-detect block, not surgery
    in this adapter.
    """

    def setup(self) -> None:
        # Local imports — keep the test/lint-time cost off this file at import.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_id: str = self.params.get("model_id", DEFAULT_MODEL_ID)
        self.box_threshold: float = float(self.params.get("box_threshold", 0.30))
        self.text_threshold: float = float(self.params.get("text_threshold", 0.25))
        # Prompt may arrive as either:
        #   params.prompts  : list[str]  — preferred; one chip per class
        #   params.prompt   : str        — legacy; single phrase or "."-joined
        # _normalize_chips collapses both into a clean list of user chips.
        # We keep both attrs around: `chips` drives the snap-back labeling,
        # `prompt` is the joined string we feed the model.
        raw_prompts = self.params.get("prompts", self.params.get("prompt", ""))
        self.chips: list[str] = _normalize_chips(raw_prompts)
        if not self.chips:
            raise ValueError(
                "GroundingDINO adapter requires non-empty prompts — pass "
                "params.prompts as a list (e.g. ['soccer ball', 'player']) "
                "or params.prompt as a single phrase."
            )
        # Joined prompt with one trailing period per chip — this is the
        # tokenizer-friendly format GroundingDINO expects.
        self.prompt: str = " . ".join(self.chips) + " ."

        # Resize controls.
        #
        # The HF GroundingDinoImageProcessor defaults to
        #   {do_resize: True, size: {shortest_edge: 800, longest_edge: 1333}}
        # which silently shrinks 1920x1080 footage to ~1333x750. For small
        # objects (a soccer ball is ~10–15 px at 1080p) that's the difference
        # between detected and missed.
        #
        # Two knobs the GUI / Learn request can set:
        #   full_resolution     : bool  — pass the source frame to the model
        #                                 with NO resize (do_resize=False).
        #                                 do_pad still runs so the Swin
        #                                 backbone gets multiple-of-32 sides.
        #   resize_longest_edge : int   — keep resize on but override the
        #                                 longest edge (e.g. 2000). Ignored
        #                                 when full_resolution=True.
        self.full_resolution: bool = bool(self.params.get("full_resolution", False))
        self.resize_longest_edge: Optional[int] = self.params.get("resize_longest_edge")
        if self.resize_longest_edge is not None:
            self.resize_longest_edge = int(self.resize_longest_edge)

        self.device = _pick_device()
        # Local-first load. If the model is already in the HF cache, pass
        # `local_files_only=True` so neither call issues a HEAD against
        # huggingface.co to check freshness — that HEAD is the source of
        # "transient DNS blip kills my Learn run" failures. On a cache
        # miss, fall through to the default networked path which will
        # download once and populate the cache for next time.
        cached = is_hf_cached(self.model_id)
        load_kwargs: dict[str, Any] = {"local_files_only": True} if cached else {}
        log.info(
            "GroundingDINO: loading %s on %s (cached=%s, prompt=%r, "
            "full_resolution=%s, resize_longest_edge=%s)",
            self.model_id,
            self.device,
            cached,
            self.prompt,
            self.full_resolution,
            self.resize_longest_edge,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, **load_kwargs)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id, **load_kwargs
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        # Cache the imported torch module so process() doesn't re-import per frame.
        self._torch = torch
        # Pre-compute the per-call kwargs we'll feed the processor on every
        # frame. Keeps process() tight; avoids re-deciding shape per frame.
        self._processor_call_kwargs: dict[str, Any] = {}
        if self.full_resolution:
            # Skip resize entirely. Padding is still required — the Swin
            # backbone window-attends on multiples of 32 — but HF's pad path
            # tolerates arbitrary input sizes when do_resize is False.
            self._processor_call_kwargs["do_resize"] = False
        elif self.resize_longest_edge is not None:
            le = self.resize_longest_edge
            # Match HF's convention: shortest_edge ~ 0.6 * longest_edge keeps
            # the same aspect-fit math their default uses (800/1333 ≈ 0.60).
            se = max(1, int(le * 800 / 1333))
            self._processor_call_kwargs["size"] = {
                "shortest_edge": se,
                "longest_edge": le,
            }

    def process(self, batch: FrameBatch) -> FrameBatch:
        from PIL import Image

        torch = self._torch
        # OpenCV gives us BGR; HF processors expect RGB PIL.
        rgb = batch.image[:, :, ::-1]
        pil = Image.fromarray(rgb)
        h, w = batch.image.shape[:2]

        inputs = self.processor(
            images=pil,
            text=self.prompt,
            return_tensors="pt",
            **self._processor_call_kwargs,
        ).to(self.device)
        if log.isEnabledFor(logging.DEBUG):
            # Useful when sanity-checking that full_resolution actually
            # made it to the model. `pixel_values` is the resized+padded
            # tensor the backbone sees.
            pv = inputs.get("pixel_values")
            if pv is not None:
                log.debug(
                    "frame %d: source %dx%d, model input %s",
                    batch.frame_index,
                    w,
                    h,
                    tuple(pv.shape),
                )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: returns boxes in xyxy at the *original image* resolution.
        # The HF API drifted across versions:
        #   - older: post_process_grounded_object_detection(outputs, input_ids,
        #            box_threshold=…, text_threshold=…, target_sizes=…)
        #   - newer: post_process_grounded_object_detection(outputs,
        #            threshold=…, text_threshold=…, target_sizes=…)
        # We sniff the signature once and call accordingly.
        results = self._post_process(outputs, inputs.input_ids, h, w)[0]

        detections: list[Detection] = []
        # `labels` is a list of strings (the matched text spans). The HF
        # post-processor returns the contiguous token span that cleared
        # text_threshold per-query — so a query for "soccer ball" can
        # come back as "soccer", "ball", "soccer ball", or even ".".
        # We snap each label back to one of the user's chips so the
        # downstream UI shows the class the user actually asked for.
        # Detections whose label can't be matched to any chip are dropped
        # (they're either pure-punctuation noise or off-topic firings).
        dropped_labels = 0
        for box, score, raw_label in zip(
            results["boxes"].tolist(),
            results["scores"].tolist(),
            results.get("labels", [""] * len(results["boxes"])),
        ):
            x1, y1, x2, y2 = box
            class_name = _snap_label_to_chip(raw_label, self.chips)
            if class_name is None:
                dropped_labels += 1
                continue
            detections.append(
                Detection(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    class_id=_class_id_for(class_name),
                    class_name=class_name,
                )
            )

        if dropped_labels and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "frame %d: dropped %d unmatched-label detection(s)",
                batch.frame_index,
                dropped_labels,
            )

        batch.detections = detections
        return batch

    def teardown(self) -> None:
        # Drop references so weights can be GC'd between runs.
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]

    def _post_process(self, outputs: Any, input_ids: Any, h: int, w: int) -> Any:
        """Call post_process_grounded_object_detection with whichever kwargs
        the installed transformers version accepts.

        We cache the resolved kwargs after the first call so we only pay the
        introspection cost once per run.
        """
        kwargs = getattr(self, "_pp_kwargs", None)
        if kwargs is None:
            sig = inspect.signature(
                self.processor.post_process_grounded_object_detection
            )
            params = sig.parameters
            built: dict[str, Any] = {
                "target_sizes": [(h, w)],
                "text_threshold": self.text_threshold,
            }
            # box-score kwarg renamed `box_threshold` → `threshold` around
            # transformers 4.51. Pick whichever exists.
            if "threshold" in params:
                built["threshold"] = self.box_threshold
            elif "box_threshold" in params:
                built["box_threshold"] = self.box_threshold
            else:  # pragma: no cover — neither kwarg present, future-proofing
                log.warning(
                    "post_process_grounded_object_detection has no recognized "
                    "score-threshold kwarg; using defaults."
                )
            self._pp_takes_input_ids = "input_ids" in params
            # Cache only the static kwargs; target_sizes varies per call.
            self._pp_kwargs = {k: v for k, v in built.items() if k != "target_sizes"}
            kwargs = self._pp_kwargs

        call_kwargs = dict(kwargs)
        call_kwargs["target_sizes"] = [(h, w)]
        if getattr(self, "_pp_takes_input_ids", False):
            return self.processor.post_process_grounded_object_detection(
                outputs, input_ids=input_ids, **call_kwargs
            )
        return self.processor.post_process_grounded_object_detection(
            outputs, **call_kwargs
        )


# Stable class-id assignment so per-class downstream code (filtering,
# coloring, COCO export) sees consistent numbering within a single run.
_CLASS_ID_MAP: dict[str, int] = {}


def _class_id_for(name: str) -> int:
    if name not in _CLASS_ID_MAP:
        _CLASS_ID_MAP[name] = len(_CLASS_ID_MAP) + 1
    return _CLASS_ID_MAP[name]


def reset_class_ids() -> None:
    """Reset the class-id counter (call between runs to keep IDs deterministic)."""
    _CLASS_ID_MAP.clear()


# Param schema — useful for the GUI later. Not consumed yet.
PARAM_SCHEMA: dict[str, Any] = {
    "prompt": {"type": "string", "default": "", "description": "Text query"},
    "box_threshold": {"type": "number", "default": 0.30, "min": 0.0, "max": 1.0},
    "text_threshold": {"type": "number", "default": 0.25, "min": 0.0, "max": 1.0},
    "model_id": {"type": "string", "default": DEFAULT_MODEL_ID},
}
