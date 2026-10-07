"""Model handles for Studio: YOLO26, YOLOE-26, SAM 2.1 and Depth Anything V2.

Everything heavy is imported lazily so `import pipeline.studio` stays cheap
for the API process and the test suite.

Weights live in `data/weights/` (gitignored). Ultralytics' downloader
writes to whatever absolute path it is given, so asking for
`data/weights/yolo26n-seg.pt` fetches the release asset straight there.

Threading model
---------------
`INFER_LOCK` serialises every forward pass in the API process. MPS command
queues are not safe to drive from several threads at once, and the YOLOE /
SAM handles carry per-request state (class embeddings, cached image
features). Training runs in its own subprocess and never takes the lock.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "data" / "weights"

INFER_LOCK = threading.RLock()

TASKS = ("detect", "segment", "classify", "pose", "obb")
TASK_SUFFIX = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose", "obb": "-obb"}
SIZES = ("n", "s", "m", "l", "x")

SAM_MODELS = {
    "sam2.1_t": "SAM 2.1 tiny — fastest, good for interactive use",
    "sam2.1_s": "SAM 2.1 small",
    "sam2.1_b": "SAM 2.1 base+ — sharper edges, ~3× slower",
}

DEPTH_MODELS = {
    "da2-metric-indoor-s": (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "Depth Anything V2 · metric indoor · small",
    ),
    "da2-metric-indoor-b": (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "Depth Anything V2 · metric indoor · base (slower, sharper)",
    ),
}

TEXT_ENCODER_WEIGHTS = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}


# Optional packages Ultralytics would normally pip-install on first use
# (this project's venv is uv-managed and has no pip, so that auto-install
# fails mid-request). Checked up front so the error says what to install.
OPTIONAL_DEPS = {
    "clip": ("git+https://github.com/ultralytics/CLIP.git", "YOLOE text prompts"),
    "lap": ("lap>=0.5.12", "video tracking (ByteTrack / BoT-SORT)"),
}


def missing_dep(module: str) -> Optional[str]:
    """None when importable, else a human-readable install hint."""
    import importlib.util

    if importlib.util.find_spec(module) is not None:
        return None
    spec, purpose = OPTIONAL_DEPS[module]
    return f"{purpose} needs the '{module}' package: uv pip install --python .venv/bin/python \"{spec}\""


def yolo26_name(task: str, size: str) -> str:
    if task not in TASK_SUFFIX:
        raise ValueError(f"unknown task {task!r}")
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}")
    return f"yolo26{size}{TASK_SUFFIX[task]}.pt"


# Open-vocabulary YOLOE comes in two families. YOLOE-26 is the YOLO26-based
# release; YOLOE-11 is kept because, in our carton tests (real pallet photos
# and synthetic renders), its *visual-prompt* branch scored stacked cartons
# at 0.6–0.95 where every YOLOE-26 size stayed below 0.25. Text prompting
# was weak for cartons in both families.
YOLOE_FAMILIES = ("26", "11")
YOLOE_SIZES = ("s", "m", "l")


def yoloe_name(family: str, size: str, prompt_free: bool = False) -> str:
    if family not in YOLOE_FAMILIES:
        raise ValueError(f"unknown YOLOE family {family!r}")
    if size not in YOLOE_SIZES:
        raise ValueError(f"unknown YOLOE size {size!r}")
    return f"yoloe-{family}{size}-seg{'-pf' if prompt_free else ''}.pt"


def device() -> str:
    from pipeline.students.yolo import _pick_device

    return _pick_device()


def is_cached(name: str) -> bool:
    return (WEIGHTS_DIR / name).exists()


def ensure_weights(name: str) -> Path:
    """Path to `name` under WEIGHTS_DIR, downloading the release asset if needed."""
    path = WEIGHTS_DIR / name
    if path.exists():
        return path
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    log.info("studio: downloading %s", name)
    out = Path(attempt_download_asset(str(path)))
    if not out.exists():
        raise RuntimeError(f"could not download weights {name!r}")
    return out


class _LRU:
    """Tiny LRU so switching between a handful of models doesn't reload them."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: "OrderedDict[Any, Any]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any, factory: Callable[[], Any]) -> Any:
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        obj = factory()
        with self.lock:
            self.items[key] = obj
            self.items.move_to_end(key)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
        return obj

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


_MODELS = _LRU(6)


def clear_cache() -> None:
    _MODELS.clear()
    _SAM_SESSIONS.clear()


# ---- YOLO26 / trained models --------------------------------------------


def load_yolo(path: Path, nms_head: bool = False) -> Any:
    """Any Ultralytics checkpoint or export (pt, onnx, torchscript, …).

    `nms_head=True` returns a separate instance of a YOLO26 checkpoint with
    its one-to-many head (+ NMS) active instead of the NMS-free end-to-end
    head. It must be a separate instance: the first end-to-end forward
    pass fuses the one-to-many branch away, after which switching back is
    impossible.
    """
    from ultralytics import YOLO

    path = Path(path)
    key = ("yolo", str(path.resolve()), path.stat().st_mtime, nms_head)

    def _factory() -> Any:
        model = YOLO(str(path))
        inner = getattr(model, "model", None)
        if nms_head and inner is not None and hasattr(inner, "end2end"):
            inner.end2end = False
        return model

    return _MODELS.get(key, _factory)


# ---- YOLOE-26: text / visual / prompt-free ------------------------------


def load_yoloe(family: str, size: str, prompt_free: bool = False) -> Any:
    from ultralytics import YOLOE

    name = yoloe_name(family, size, prompt_free)
    return _MODELS.get(("yoloe", name), lambda: YOLOE(str(ensure_weights(name))))


_TEXT_ENCODERS: dict[str, Any] = {}


def yoloe_set_text_classes(model: Any, names: list[str]) -> None:
    """Point a YOLOE model at text-prompt classes.

    Bypasses `YOLOE.set_classes`, which (a) skips the update when the new
    names equal the current ones — wrong after a visual prompt installed
    embeddings under the same names — and (b) rebuilds the MobileCLIP text
    encoder on every call (~4 s). We keep one CPU encoder per variant and
    load it from WEIGHTS_DIR instead of the process CWD.
    """
    hint = missing_dep("clip")
    if hint:
        raise RuntimeError(hint)
    import torch
    from ultralytics.nn.text_model import MobileCLIPTS

    inner = model.model
    variant = getattr(inner, "text_model", "mobileclip:blt")
    weight = TEXT_ENCODER_WEIGHTS.get(variant)
    head = inner.model[-1]
    p = next(head.parameters())
    if weight is None:  # unknown encoder — fall back to Ultralytics' own path
        pe = inner.get_text_pe(list(names))
    else:
        enc = _TEXT_ENCODERS.get(variant)
        if enc is None:
            enc = _TEXT_ENCODERS[variant] = MobileCLIPTS(
                torch.device("cpu"), weight=str(ensure_weights(weight))
            )
        with torch.inference_mode():
            feats = enc.encode_text(enc.tokenize(list(names))).detach()
            feats = feats.reshape(1, len(names), -1).to(device=p.device, dtype=p.dtype)
            pe = head.get_tpe(feats)
    inner.set_classes(list(names), pe)
    model.predictor = None


def _mask_vp_predictor_cls() -> type:
    """YOLOEVPSegPredictor that accepts 2-D mask prompts.

    Upstream letterboxes mask prompts with the image LetterBox, which
    returns (H, W, 1) for single-channel input and then fails inside
    `LoadVisualPrompt.get_visuals`. We letterbox the masks ourselves with
    the exact offsets upstream uses for box prompts.
    """
    import cv2
    from ultralytics.data.augment import LoadVisualPrompt
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    class MaskVPPredictor(YOLOEVPSegPredictor):
        def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
            nh, nw = round(src_shape[0] * gain), round(src_shape[1] * gain)
            top = round((dst_shape[0] - nh) / 2 - 0.1)
            left = round((dst_shape[1] - nw) / 2 - 0.1)
            out = np.zeros((len(masks), dst_shape[0], dst_shape[1]), np.uint8)
            for i, mk in enumerate(masks):
                out[i, top : top + nh, left : left + nw] = cv2.resize(
                    mk.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
                )
            return LoadVisualPrompt().get_visuals(category, dst_shape, None, out)

    return MaskVPPredictor


def yoloe_visual_embeddings(
    model: Any, image_bgr: np.ndarray, masks: np.ndarray, cls: np.ndarray, imgsz: int = 640
) -> tuple[list[int], Any]:
    """Visual-prompt embeddings for the classes present in one reference image.

    `masks` is (N, H, W) uint8 at image resolution (boxes pre-rasterised to
    filled rectangles), `cls` the (N,) global class index per mask. Returns
    (sorted unique class indices, tensor (1, len(unique), D)).
    """
    pred_cls = _mask_vp_predictor_cls()
    uniq = sorted({int(c) for c in cls})
    local = np.array([uniq.index(int(c)) for c in cls])
    inner = model.model
    inner.model[-1].nc = len(uniq)
    inner.names = [f"object{i}" for i in range(len(uniq))]
    pred = pred_cls(
        overrides={
            "task": "segment",
            "mode": "predict",
            "save": False,
            "verbose": False,
            "batch": 1,
            "device": device(),
            "imgsz": imgsz,
        }
    )
    pred.set_prompts({"masks": masks, "cls": local})
    pred.setup_model(model=inner, verbose=False)
    vpe = pred.get_vpe(image_bgr)
    return uniq, vpe.detach()


def yoloe_set_visual_classes(model: Any, names: list[str], vpe: Any) -> None:
    model.model.set_classes(list(names), vpe)
    model.predictor = None


# ---- SAM 2.1 ------------------------------------------------------------


class SamSession:
    """SAM 2.1 predictor that keeps the last image's encoder features.

    Interactive prompting re-segments the *same* image dozens of times as
    the user adds strokes; the image encoder is ~90 % of SAM's cost, so we
    encode once per image and only re-run the prompt decoder.
    """

    def __init__(self, name: str):
        from ultralytics.models.sam import SAM2Predictor

        self.name = name
        self.pred = SAM2Predictor(
            overrides={
                "conf": 0.0,
                "task": "segment",
                "mode": "predict",
                "imgsz": 1024,
                "model": str(ensure_weights(f"{name}.pt")),
                "device": device(),
                "save": False,
                "verbose": False,
            }
        )
        self.image_key: Optional[str] = None

    def segment(
        self,
        image_key: str,
        image_bgr: np.ndarray,
        bbox: Optional[list[float]] = None,
        points: Optional[list[list[float]]] = None,
        labels: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, float]:
        """One object from a box and/or labelled points → (bool mask, score)."""
        cands = self.candidates(image_key, image_bgr, bbox, points, labels, multimask=False)
        if not cands:
            return np.zeros(image_bgr.shape[:2], dtype=bool), 0.0
        return max(cands, key=lambda c: c[1])

    def candidates(
        self,
        image_key: str,
        image_bgr: np.ndarray,
        bbox: Optional[list[float]] = None,
        points: Optional[list[list[float]]] = None,
        labels: Optional[list[int]] = None,
        multimask: bool = True,
    ) -> list[tuple[np.ndarray, float]]:
        """SAM's mask hypotheses for one prompt set.

        With `multimask=True` SAM returns its three granularities (roughly
        part / object / whole) — the ambiguity a single click or stroke on a
        carton in a stack can't resolve on its own.
        """
        if image_key != self.image_key:
            self.pred.set_image(image_bgr)
            self.image_key = image_key
        kwargs: dict[str, Any] = {}
        if bbox is not None:
            kwargs["bboxes"] = [list(map(float, bbox))]
        if points:
            kwargs["points"] = [[list(map(float, p)) for p in points]]
            kwargs["labels"] = [list(map(int, labels or [1] * len(points)))]
        if not kwargs:
            raise ValueError("SAM needs a box or at least one point")
        results = self.pred(multimask_output=multimask, **kwargs)
        r = results[0]
        if r.masks is None or len(r.masks.data) == 0:
            return []
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(r.masks.data))
        masks = r.masks.data.cpu().numpy().astype(bool)
        return [(masks[i], float(scores[i])) for i in range(len(masks))]


_SAM_SESSIONS = _LRU(1)


def sam_session(name: str = "sam2.1_t") -> SamSession:
    if name not in SAM_MODELS:
        raise ValueError(f"unknown SAM model {name!r}")
    return _SAM_SESSIONS.get(name, lambda: SamSession(name))


# ---- Depth Anything V2 --------------------------------------------------


def load_depth(key: str = "da2-metric-indoor-s") -> tuple[Any, Any]:
    if key not in DEPTH_MODELS:
        raise ValueError(f"unknown depth model {key!r}")
    repo = DEPTH_MODELS[key][0]

    def _factory() -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(repo).to(device()).eval()
        return proc, model

    return _MODELS.get(("depth", key), _factory)


def estimate_depth_m(image_bgr: np.ndarray, key: str = "da2-metric-indoor-s") -> np.ndarray:
    """Monocular metric depth (metres, H×W float32) for a BGR image."""
    import torch
    from PIL import Image

    proc, model = load_depth(key)
    rgb = Image.fromarray(image_bgr[:, :, ::-1])
    with torch.inference_mode():
        inputs = proc(images=rgb, return_tensors="pt").to(device())
        out = model(**inputs)
        depth = proc.post_process_depth_estimation(out, target_sizes=[(rgb.height, rgb.width)])[0][
            "predicted_depth"
        ]
    return depth.float().cpu().numpy().astype(np.float32)


# ---- Catalog --------------------------------------------------------------


def catalog() -> dict:
    """What the GUI can offer, and which weights are already on disk."""
    yolo = {
        task: [{"size": s, "weights": yolo26_name(task, s), "cached": is_cached(yolo26_name(task, s))} for s in SIZES]
        for task in TASKS
    }
    yoloe = [
        {
            "family": f,
            "size": s,
            "weights": yoloe_name(f, s),
            "cached": is_cached(yoloe_name(f, s)),
            "pf_cached": is_cached(yoloe_name(f, s, True)),
        }
        for f in YOLOE_FAMILIES
        for s in YOLOE_SIZES
    ]
    sam = [{"id": k, "label": v, "cached": is_cached(f"{k}.pt")} for k, v in SAM_MODELS.items()]
    depth = [{"id": k, "label": v[1]} for k, v in DEPTH_MODELS.items()]
    deps = {m: missing_dep(m) for m in OPTIONAL_DEPS}
    return {"device": device(), "yolo26": yolo, "yoloe": yoloe, "sam": sam, "depth": depth, "missing_deps": deps}
