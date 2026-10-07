"""On-disk Studio dataset: classes, images, annotations, suggestions, depth.

Layout under `runs/projects/{pid}/studio/`:

    classes.json              [{id, name, color}]
    images.json               {"images": [ImageRecord, …]}  (insertion order)
    images/{iid}.{ext}        original bytes (jpg/png/webp kept as-is)
    depth/{iid}.png           optional 16-bit z-depth in millimetres
    annotations/{iid}.json    [Annotation, …]
    suggestions/{iid}.json    {"source", "created_at", "items": [Suggestion, …]}
    cache/thumbs/             derived thumbnails (safe to delete)

Annotation (all coordinates in original-image pixels):

    {"id": "a_…", "class_id": int, "bbox": [x1, y1, x2, y2],
     "polygon": [[x, y], …] | null, "source": str, "score": float | null}

A polygon, when present, is authoritative and the bbox is derived from it.
Suggestions have the same shape plus `class_name` (the predictor's label,
used to map onto a dataset class when accepted).

Every read-modify-write goes through a per-root `RLock` and files are
replaced atomically, so the GUI's rapid-fire saves can't interleave into a
torn JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from pipeline import runs as runs_mod

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#469990", "#dcbeff", "#9a6324", "#800000",
    "#808000", "#000075", "#fabed4", "#aaffc3",
]

# Formats browsers can display directly are stored verbatim; anything else
# OpenCV can decode (bmp, tiff, …) is re-encoded to PNG on the way in.
WEB_EXTS = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png", ".webp": ".webp"}
SPLITS = ("auto", "train", "val", "test")


class NotFound(KeyError):
    """Unknown image / class id. The API layer maps this to a 404."""


class Invalid(ValueError):
    """Bad input (undecodable image, unknown class, …). Mapped to 400/422."""


_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def check_id(value: str, what: str = "id") -> str:
    """Reject ids that could escape their directory when joined into a path.

    Ids arrive in JSON bodies (model ids, export names, …); a "../x" or an
    absolute path would otherwise walk out of the project via pathlib.
    """
    v = str(value)
    if not _ID_RE.match(v) or v in (".", ".."):
        raise Invalid(f"invalid {what}: {value!r}")
    return v


def studio_root(project_id: str, runs_root: Path = runs_mod.RUNS_DIR) -> Path:
    return runs_mod.project_dir(project_id, runs_root) / "studio"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    key = str(root.resolve())
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = _LOCKS[key] = threading.RLock()
        return lk


def auto_split(image_id: str, val_pct: int = 20) -> str:
    """Deterministic train/val assignment for images left on `auto`.

    Hash-based so an image keeps its split across exports — re-training
    after labelling a few more images doesn't silently shuffle what the
    model validated on.
    """
    h = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) % 100
    return "val" if h < val_pct else "train"


class StudioStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = _lock_for(self.root)

    @classmethod
    def for_project(cls, project_id: str, runs_root: Path = runs_mod.RUNS_DIR) -> "StudioStore":
        return cls(studio_root(project_id, runs_root))

    # ---- paths ----------------------------------------------------------

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def ann_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def sugg_dir(self) -> Path:
        return self.root / "suggestions"

    @property
    def depth_dir(self) -> Path:
        return self.root / "depth"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def tracks_dir(self) -> Path:
        return self.root / "tracks"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    # ---- classes --------------------------------------------------------

    def classes(self) -> list[dict]:
        return _read_json(self.root / "classes.json", [])

    def class_ids(self) -> set[int]:
        return {int(c["id"]) for c in self.classes()}

    def set_classes(self, incoming: list[dict]) -> list[dict]:
        """Replace the class list.

        Entries with an `id` keep it (rename / recolour); entries without
        one are new and get the next free id. Classes that disappear take
        their annotations with them — the GUI confirms before sending that.
        """
        with self.lock:
            current = {int(c["id"]): c for c in self.classes()}
            next_id = max(current.keys(), default=-1) + 1
            out: list[dict] = []
            seen_names: set[str] = set()
            for c in incoming:
                name = str(c.get("name", "")).strip()
                if not name:
                    raise Invalid("class names must be non-empty")
                if name.lower() in seen_names:
                    raise Invalid(f"duplicate class name {name!r}")
                seen_names.add(name.lower())
                cid = c.get("id")
                if cid is None or int(cid) not in current:
                    cid = next_id
                    next_id += 1
                cid = int(cid)
                color = c.get("color") or PALETTE[cid % len(PALETTE)]
                out.append({"id": cid, "name": name, "color": color})
            removed = set(current) - {c["id"] for c in out}
            _atomic_write_json(self.root / "classes.json", out)
            if removed:
                for rec in self.list_images():
                    anns = self.annotations(rec["id"])
                    kept = [a for a in anns if int(a["class_id"]) not in removed]
                    if len(kept) != len(anns):
                        self._write_annotations(rec["id"], kept)
            return out

    def ensure_class(self, name: str) -> dict:
        """Find a class by (case-insensitive) name, creating it if needed."""
        name = name.strip()
        with self.lock:
            for c in self.classes():
                if c["name"].lower() == name.lower():
                    return c
            classes = self.classes() + [{"name": name}]
            out = self.set_classes(classes)
            return next(c for c in out if c["name"].lower() == name.lower())

    # ---- images ---------------------------------------------------------

    def _index(self) -> list[dict]:
        return _read_json(self.root / "images.json", {"images": []})["images"]

    def _write_index(self, images: list[dict]) -> None:
        _atomic_write_json(self.root / "images.json", {"images": images})

    def list_images(self) -> list[dict]:
        return self._index()

    def get_image(self, iid: str) -> dict:
        for rec in self._index():
            if rec["id"] == iid:
                return rec
        raise NotFound(iid)

    def image_path(self, iid: str) -> Path:
        return self.images_dir / self.get_image(iid)["file"]

    def read_image(self, iid: str) -> np.ndarray:
        img = cv2.imread(str(self.image_path(iid)), cv2.IMREAD_COLOR)
        if img is None:
            raise Invalid(f"image {iid} is unreadable on disk")
        return img

    def add_image_bytes(self, data: bytes, filename: str, source: str = "upload") -> dict:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise Invalid(f"{filename}: not a decodable image")
        ext = WEB_EXTS.get(Path(filename).suffix.lower())
        iid = new_id("img")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if ext is None:
            ext = ".png"
            cv2.imwrite(str(self.images_dir / f"{iid}{ext}"), img)
        else:
            (self.images_dir / f"{iid}{ext}").write_bytes(data)
        return self._register(iid, f"{iid}{ext}", filename, img.shape, source)

    def add_image_array(
        self, bgr: np.ndarray, filename: str, source: str, extra: Optional[dict] = None
    ) -> dict:
        iid = new_id("img")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.images_dir / f"{iid}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return self._register(iid, f"{iid}.jpg", filename, bgr.shape, source, extra)

    def _register(
        self, iid: str, file: str, filename: str, shape: tuple, source: str, extra: Optional[dict] = None
    ) -> dict:
        rec = {
            "id": iid,
            "file": file,
            "filename": filename,
            "width": int(shape[1]),
            "height": int(shape[0]),
            "split": "auto",
            "source": source,
            "added_at": now_iso(),
            "has_depth": False,
            "intrinsics": None,
            "negative": False,
            "n_annotations": 0,
            "n_suggestions": 0,
        }
        if extra:
            rec.update(extra)
        with self.lock:
            images = self._index()
            images.append(rec)
            self._write_index(images)
        return rec

    def update_image(self, iid: str, **fields: Any) -> dict:
        allowed = {"split", "intrinsics", "negative", "filename"}
        bad = set(fields) - allowed
        if bad:
            raise Invalid(f"cannot update image fields {sorted(bad)}")
        if "split" in fields and fields["split"] not in SPLITS:
            raise Invalid(f"split must be one of {SPLITS}")
        if fields.get("intrinsics") is not None:
            fields["intrinsics"] = _check_intrinsics(fields["intrinsics"])
        with self.lock:
            images = self._index()
            for rec in images:
                if rec["id"] == iid:
                    rec.update(fields)
                    self._write_index(images)
                    return rec
        raise NotFound(iid)

    def _patch_record(self, iid: str, **fields: Any) -> None:
        with self.lock:
            images = self._index()
            for rec in images:
                if rec["id"] == iid:
                    rec.update(fields)
                    self._write_index(images)
                    return
        raise NotFound(iid)

    def delete_image(self, iid: str) -> None:
        with self.lock:
            rec = self.get_image(iid)
            images = [r for r in self._index() if r["id"] != iid]
            self._write_index(images)
            for p in (
                self.images_dir / rec["file"],
                self.ann_dir / f"{iid}.json",
                self.sugg_dir / f"{iid}.json",
                self.depth_dir / f"{iid}.png",
            ):
                p.unlink(missing_ok=True)
            for sub in ("thumbs", "depth"):  # derived caches (thumbnails, monocular depth)
                d = self.cache_dir / sub
                if d.exists():
                    for t in d.glob(f"{iid}_*"):
                        t.unlink(missing_ok=True)

    def thumbnail(self, iid: str, size: int = 160) -> Path:
        size = int(max(32, min(size, 512)))
        out = self.cache_dir / "thumbs" / f"{iid}_{size}.jpg"
        if out.exists():
            return out
        img = self.read_image(iid)
        h, w = img.shape[:2]
        scale = size / max(h, w)
        th = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out

    # ---- depth ----------------------------------------------------------

    def set_depth(self, iid: str, depth_mm: np.ndarray, intrinsics: Optional[dict] = None) -> dict:
        rec = self.get_image(iid)
        if depth_mm.ndim != 2:
            raise Invalid("depth must be a single-channel image")
        if depth_mm.shape != (rec["height"], rec["width"]):
            # RGB-D rigs often deliver depth at a different resolution than
            # colour; nearest-neighbour keeps edges from bleeding between
            # foreground and background depths.
            depth_mm = cv2.resize(
                depth_mm, (rec["width"], rec["height"]), interpolation=cv2.INTER_NEAREST
            )
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.depth_dir / f"{iid}.png"), depth_mm.astype(np.uint16))
        fields: dict[str, Any] = {"has_depth": True}
        if intrinsics is not None:
            fields["intrinsics"] = _check_intrinsics(intrinsics)
        self._patch_record(iid, **fields)
        return self.get_image(iid)

    def set_depth_bytes(self, iid: str, data: bytes, filename: str, scale_to_mm: Optional[float] = None) -> dict:
        """Accept a 16-bit PNG/TIFF or a `.npy` in metres.

        `scale_to_mm` converts raw PNG/TIFF units to millimetres (e.g. 0.25
        for a sensor reporting quarter-millimetres). `.npy` is always metres;
        the scale doesn't apply to it.
        """
        name = filename.lower()
        if name.endswith(".npy"):
            import io

            arr = np.load(io.BytesIO(data), allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float64).squeeze()
            mm = np.clip(np.nan_to_num(arr) * 1000.0, 0, 65535).astype(np.uint16)
        else:
            raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise Invalid(f"{filename}: not a decodable depth image")
            if raw.ndim == 3:
                raise Invalid(
                    f"{filename}: depth must be single-channel (16-bit PNG in mm) — "
                    "got a colour image. Colourised depth visualisations can't be inverted."
                )
            if raw.dtype == np.uint8:
                raise Invalid(f"{filename}: 8-bit depth has too little range; export 16-bit PNG in mm")
            scale = 1.0 if scale_to_mm is None else scale_to_mm
            mm = np.clip(raw.astype(np.float64) * scale, 0, 65535).astype(np.uint16)
        return self.set_depth(iid, mm)

    def read_depth_m(self, iid: str) -> Optional[np.ndarray]:
        p = self.depth_dir / f"{iid}.png"
        if not p.exists():
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / 1000.0

    def delete_depth(self, iid: str) -> dict:
        (self.depth_dir / f"{iid}.png").unlink(missing_ok=True)
        self._patch_record(iid, has_depth=False)
        return self.get_image(iid)

    # ---- annotations ----------------------------------------------------

    def annotations(self, iid: str) -> list[dict]:
        return _read_json(self.ann_dir / f"{iid}.json", [])

    def _write_annotations(self, iid: str, anns: list[dict]) -> None:
        _atomic_write_json(self.ann_dir / f"{iid}.json", anns)
        self._patch_record(iid, n_annotations=len(anns))

    def set_annotations(self, iid: str, anns: list[dict]) -> list[dict]:
        with self.lock:
            rec = self.get_image(iid)
            valid = self.class_ids()
            out = [normalize_annotation(a, rec["width"], rec["height"], valid) for a in anns]
            self._write_annotations(iid, out)
            return out

    def add_annotations(self, iid: str, anns: list[dict]) -> list[dict]:
        with self.lock:
            return self.set_annotations(iid, self.annotations(iid) + list(anns))

    # ---- suggestions ----------------------------------------------------

    def suggestions(self, iid: str) -> dict:
        return _read_json(self.sugg_dir / f"{iid}.json", {"source": None, "created_at": None, "items": []})

    def set_suggestions(self, iid: str, items: list[dict], source: str) -> dict:
        with self.lock:
            rec = self.get_image(iid)
            norm = []
            for it in items:
                try:
                    s = normalize_annotation(it, rec["width"], rec["height"], None)
                except Invalid:
                    continue  # degenerate prediction (sub-pixel box) — not worth reviewing
                s["class_name"] = str(it.get("class_name") or "")
                norm.append(s)
            data = {"source": source, "created_at": now_iso(), "items": norm}
            _atomic_write_json(self.sugg_dir / f"{iid}.json", data)
            self._patch_record(iid, n_suggestions=len(norm))
            return data

    def clear_suggestions(self, iid: str, ids: Optional[list[str]] = None) -> dict:
        """Drop all suggestions, or only the given ids (accepted/rejected)."""
        with self.lock:
            cur = self.suggestions(iid)
            items = [] if ids is None else [s for s in cur["items"] if s["id"] not in set(ids)]
            cur["items"] = items
            _atomic_write_json(self.sugg_dir / f"{iid}.json", cur)
            self._patch_record(iid, n_suggestions=len(items))
            return cur

    # ---- dataset stats --------------------------------------------------

    def stats(self) -> dict:
        classes = self.classes()
        per_class = {c["id"]: 0 for c in classes}
        splits = {"train": 0, "val": 0, "test": 0}
        labeled = negatives = boxes_only = 0
        images = self.list_images()
        for rec in images:
            anns = self.annotations(rec["id"])
            if anns:
                labeled += 1
            elif rec.get("negative"):
                negatives += 1
            else:
                continue
            split = rec["split"] if rec["split"] != "auto" else auto_split(rec["id"])
            splits[split] += 1
            for a in anns:
                per_class[int(a["class_id"])] = per_class.get(int(a["class_id"]), 0) + 1
                if not a.get("polygon"):
                    boxes_only += 1
        return {
            "n_images": len(images),
            "n_labeled": labeled,
            "n_negative": negatives,
            "n_unlabeled": len(images) - labeled - negatives,
            "splits": splits,
            "instances_per_class": {str(k): v for k, v in per_class.items()},
            "n_instances": sum(per_class.values()),
            "n_box_only": boxes_only,
        }

    def wipe(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def studio_summary(project_id: str, runs_root: Path = runs_mod.RUNS_DIR) -> dict[str, int]:
    """Counters for the project picker: Studio images and trained models."""
    root = studio_root(project_id, runs_root)
    n_images = len(_read_json(root / "images.json", {"images": []})["images"])
    n_models = 0
    models = root / "models"
    if models.exists():
        for m in models.glob("*/model.json"):
            try:
                n_models += json.loads(m.read_text()).get("status") == "completed"
            except json.JSONDecodeError:
                continue
    return {"n_studio_images": n_images, "n_studio_models": n_models}


def _check_intrinsics(k: dict) -> dict:
    try:
        out = {key: float(k[key]) for key in ("fx", "fy", "cx", "cy")}
    except (KeyError, TypeError, ValueError) as e:
        raise Invalid(f"intrinsics need numeric fx, fy, cx, cy: {e}") from e
    if out["fx"] <= 0 or out["fy"] <= 0:
        raise Invalid("focal lengths must be positive")
    return out


def normalize_annotation(
    a: dict, width: int, height: int, valid_class_ids: Optional[set[int]]
) -> dict:
    """Validate + canonicalise one annotation (or suggestion).

    `valid_class_ids=None` skips the class check — suggestions may carry a
    predictor class id that isn't (yet) a dataset class.
    """
    try:
        cid = int(a.get("class_id", -1))
    except (TypeError, ValueError) as e:
        raise Invalid(f"bad class_id {a.get('class_id')!r}") from e
    if valid_class_ids is not None and cid not in valid_class_ids:
        raise Invalid(f"unknown class_id {cid}")

    poly = a.get("polygon")
    polygon = None
    if poly:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            pts[:, 0] = np.clip(pts[:, 0], 0, width)
            pts[:, 1] = np.clip(pts[:, 1], 0, height)
            polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]

    if polygon is not None:
        arr = np.asarray(polygon)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
    else:
        bb = a.get("bbox")
        if not bb or len(bb) != 4:
            raise Invalid("annotation needs a bbox or a polygon with ≥3 points")
        x1, y1, x2, y2 = (float(v) for v in bb)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0, width))
    x2 = float(np.clip(x2, 0, width))
    y1 = float(np.clip(y1, 0, height))
    y2 = float(np.clip(y2, 0, height))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise Invalid("annotation is smaller than one pixel")

    score = a.get("score")
    return {
        "id": str(a.get("id") or new_id("a")),
        "class_id": cid,
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "polygon": polygon,
        "source": str(a.get("source") or "manual"),
        "score": None if score is None else round(float(score), 4),
    }
