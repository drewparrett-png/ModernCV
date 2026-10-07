"""YOLO26 training queue, validation, export and benchmarking for Studio.

Jobs run one at a time (like the Teacher/Student queues) but each in its
own **subprocess** (`python -m pipeline.studio.train_job <model_dir>`):

  • a crash inside MPS / a dataloader worker can't take the API down,
  • Cancel is a plain `terminate()`,
  • the child writes `progress.json` after every epoch, so the GUI polls
    files rather than sharing Python state.

`models/{mid}/`:
    model.json        manifest (config, status, dataset summary, metrics)
    progress.json     epoch / total / last metrics / stage
    dataset/          snapshot exported when the job was queued
    ultralytics/      Ultralytics run dir (results.csv, plots, weights)
    best.pt           copied out on success
    exports/          ONNX / TorchScript / CoreML … artefacts
    train.log         child stdout + stderr
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from pipeline.studio import engines
from pipeline.studio.export import export_yolo_dataset
from pipeline.studio.store import Invalid, NotFound, StudioStore, check_id, now_iso

log = logging.getLogger(__name__)

TRAIN_TASKS = ("detect", "segment", "obb")

# Ultralytics hyper-parameters the GUI may override. Anything else in the
# request's `augment` block is rejected rather than silently forwarded.
AUGMENT_KEYS = {
    "mosaic", "close_mosaic", "fliplr", "flipud", "degrees", "translate", "scale", "shear",
    "perspective", "hsv_h", "hsv_s", "hsv_v", "mixup", "cutmix", "copy_paste", "multi_scale",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "patience": 25,
    "optimizer": "auto",
    "lr0": None,
    "cos_lr": False,
    "freeze": None,
    "val_pct": 20,
    "seed": 0,
    "workers": 2,
    "cache": False,
    "augment": {},
}


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_manifest(mdir: Path) -> dict:
    return json.loads((mdir / "model.json").read_text())


def model_dir(store: StudioStore, mid: str) -> Path:
    """Directory of model `mid`, validated; NotFound if it doesn't exist."""
    mdir = store.models_dir / check_id(mid, "model id")
    if not (mdir / "model.json").exists():
        raise NotFound(mid)
    return mdir


def write_manifest(mdir: Path, m: dict) -> None:
    _write_json(mdir / "model.json", m)


def read_progress(mdir: Path) -> Optional[dict]:
    try:
        return json.loads((mdir / "progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def results_csv(mdir: Path) -> Path:
    return mdir / "ultralytics" / "train" / "results.csv"


def read_curve(mdir: Path) -> list[dict]:
    p = results_csv(mdir)
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    clean[k.strip()] = float(v)
                except (TypeError, ValueError):
                    clean[k.strip()] = v
            rows.append(clean)
    return rows


def _model_id(task: str, size: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"m_{stamp}_{size}-{task}"


def base_weights_label(task: str, size: str, base: str) -> str:
    if base == "scratch":
        return f"yolo26{size}{engines.TASK_SUFFIX[task]}.yaml"
    if base.startswith("model:"):
        return base
    return engines.yolo26_name(task, size)


def start_training(store: StudioStore, req: dict) -> dict:
    task = req.get("task", "detect")
    size = req.get("size", "n")
    base = req.get("base", "pretrained")
    if task not in TRAIN_TASKS:
        raise Invalid(f"task must be one of {TRAIN_TASKS}")
    if size not in engines.SIZES:
        raise Invalid(f"size must be one of {engines.SIZES}")
    if base not in ("pretrained", "scratch") and not str(base).startswith("model:"):
        raise Invalid("base must be 'pretrained', 'scratch' or 'model:<id>'")
    if str(base).startswith("model:"):
        parent = store.models_dir / check_id(str(base)[6:], "base model id")
        if not (parent / "best.pt").exists():
            raise Invalid(f"can't fine-tune from {base}: it has no trained weights")
        if read_manifest(parent)["task"] != task:
            raise Invalid(f"{base} was trained for a different task")

    cfg = {**DEFAULT_CONFIG, **{k: v for k, v in (req.get("config") or {}).items() if k in DEFAULT_CONFIG}}
    bad = set(cfg.get("augment") or {}) - AUGMENT_KEYS
    if bad:
        raise Invalid(f"unknown augmentation keys {sorted(bad)}")
    cfg["epochs"] = int(max(1, min(int(cfg["epochs"]), 1000)))
    cfg["imgsz"] = int(max(64, min(int(cfg["imgsz"]), 2048)))
    cfg["batch"] = int(max(1, min(int(cfg["batch"]), 128)))

    class_ids = [int(c["id"]) for c in store.classes()]
    if not class_ids:
        raise Invalid("add at least one class and label some images first")

    mid = _model_id(task, size)
    mdir = store.models_dir / mid
    mdir.mkdir(parents=True, exist_ok=True)
    summary = export_yolo_dataset(store, mdir / "dataset", task, val_pct=int(cfg["val_pct"]), class_ids=class_ids)
    if summary.counts["train"] == 0:
        shutil.rmtree(mdir, ignore_errors=True)
        raise Invalid("no labelled images to train on")

    manifest = {
        "id": mid,
        "name": req.get("name") or f"yolo26{size}-{task} · {cfg['epochs']}ep",
        "task": task,
        "size": size,
        "base": base,
        "base_weights": base_weights_label(task, size, base),
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "config": cfg,
        "dataset": {
            "class_ids": class_ids,
            "names": summary.names,
            "counts": summary.counts,
            "instances": summary.instances,
            "warnings": summary.warnings,
        },
        "metrics": None,
        "per_class": None,
        "exports": [],
    }
    write_manifest(mdir, manifest)
    _write_json(mdir / "progress.json", {"stage": "queued", "epoch": 0, "epochs": cfg["epochs"], "updated_at": now_iso()})
    ensure_worker_started()
    _QUEUE.put(mdir)
    return manifest


def list_models(store: StudioStore) -> list[dict]:
    out = []
    if not store.models_dir.exists():
        return out
    for d in sorted(store.models_dir.iterdir(), reverse=True):
        if (d / "model.json").exists():
            m = read_manifest(d)
            m["progress"] = read_progress(d)
            out.append(m)
    return out


def model_detail(store: StudioStore, mid: str) -> dict:
    mdir = model_dir(store, mid)
    m = read_manifest(mdir)
    m["progress"] = read_progress(mdir)
    m["curve"] = read_curve(mdir)
    plots_dir = mdir / "ultralytics" / "train"
    m["plots"] = sorted(p.name for p in plots_dir.glob("*.png")) if plots_dir.exists() else []
    return m


def delete_model(store: StudioStore, mid: str) -> None:
    mdir = model_dir(store, mid)
    cancel(store, mid)
    shutil.rmtree(mdir, ignore_errors=True)


# ---- Worker -------------------------------------------------------------

_QUEUE: "queue.Queue[Optional[Path]]" = queue.Queue()
_WORKER: Optional[threading.Thread] = None
_WORKER_LOCK = threading.Lock()
_CURRENT: dict[str, Any] = {"mdir": None, "proc": None}


def ensure_worker_started() -> None:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(target=_worker_loop, name="studio-train", daemon=True)
        _WORKER.start()


def stop_worker(timeout: float = 2.0) -> None:
    """Stop the queue thread but leave a running trainer alone.

    A dev reload (`uvicorn --reload`) restarts only the API process; the
    trainer keeps going and the next process re-adopts it (`recover_jobs`).
    Ctrl-C in a terminal still stops it: the trainer shares the terminal's
    process group and receives the SIGINT directly.
    """
    _QUEUE.put(None)
    if _WORKER is not None:
        _WORKER.join(timeout=timeout)


def _worker_loop() -> None:
    while True:
        mdir = _QUEUE.get()
        if mdir is None:
            return
        try:
            _run_job(mdir)
        except Exception as e:  # pragma: no cover — defensive
            log.exception("studio training job crashed: %s", mdir.name)
            try:
                m = read_manifest(mdir)
                m.update(status="failed", error=str(e), finished_at=now_iso())
                write_manifest(mdir, m)
            except Exception:
                pass


def _run_job(mdir: Path) -> None:
    if not (mdir / "model.json").exists():
        return  # deleted while queued
    # One trainer at a time — including one adopted from a previous server.
    while any(_trainer_alive(p) for p in list(_ADOPTED)):
        time.sleep(2)
    m = read_manifest(mdir)
    if m["status"] != "queued":
        return  # cancelled while queued
    m.update(status="running", started_at=now_iso())
    write_manifest(mdir, m)
    log_path = mdir / "train.log"
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pipeline.studio.train_job", str(mdir.resolve())],
            cwd=str(engines.REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        _CURRENT.update(mdir=mdir, proc=proc)
        m = read_manifest(mdir)
        m["pid"] = proc.pid
        write_manifest(mdir, m)
        rc = proc.wait()
        _CURRENT.update(mdir=None, proc=None)
    if not (mdir / "model.json").exists():
        return
    m = read_manifest(mdir)
    if m["status"] == "running":  # child died without recording an outcome
        tail = _tail(log_path, 25)
        m.update(status="failed", finished_at=now_iso(), error=f"trainer exited with code {rc}\n{tail}")
        write_manifest(mdir, m)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _tail(path: Path, n: int) -> str:
    """Last `n` log lines, without ANSI codes or tqdm carriage-return frames."""
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    lines = [_ANSI.sub("", ln.split("\r")[-1]).rstrip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln][-n:])


def cancel(store: StudioStore, mid: str) -> dict:
    mdir = model_dir(store, mid)
    m = read_manifest(mdir)
    if m["status"] in ("queued", "running"):
        m.update(status="cancelled", finished_at=now_iso(), error="cancelled by user")
        write_manifest(mdir, m)
        cur = _CURRENT.get("mdir")
        proc = _CURRENT.get("proc")
        if cur is not None and Path(cur).resolve() == mdir.resolve() and proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        elif _trainer_alive(m.get("pid")):  # adopted from a previous server process
            os.kill(int(m["pid"]), signal.SIGTERM)
    return m


def _trainer_alive(pid: Optional[int]) -> bool:
    """Is `pid` still one of our training subprocesses?"""
    if not pid:
        return False
    try:
        import psutil

        proc = psutil.Process(int(pid))
        return proc.is_running() and "pipeline.studio.train_job" in " ".join(proc.cmdline())
    except Exception:
        return False


_ADOPTED: set[int] = set()


def _watch_adopted(mdir: Path, pid: int) -> None:
    """Follow a trainer started by a previous server process until it exits."""
    while _trainer_alive(pid):
        time.sleep(2)
    _ADOPTED.discard(pid)
    try:
        m = read_manifest(mdir)
    except FileNotFoundError:
        return
    if m["status"] == "running":  # the child normally records its own outcome
        m.update(status="failed", finished_at=now_iso(),
                 error=f"trainer exited without a result\n{_tail(mdir / 'train.log', 25)}")
        write_manifest(mdir, m)


def recover_jobs(runs_root: Path) -> dict[str, int]:
    """Reconcile jobs left behind by a previous server process (e.g. a dev reload).

    • running + trainer subprocess still alive → adopt it (keep "running",
      watch it on a thread; the next queued job waits for it),
    • running + process gone → failed,
    • queued → re-queued (the in-memory queue died with the old process).
    """
    counts = {"adopted": 0, "failed": 0, "requeued": 0}
    for mfile in sorted(Path(runs_root).glob("projects/*/studio/models/*/model.json")):
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        mdir = mfile.parent
        if m.get("status") == "running":
            pid = m.get("pid")
            if _trainer_alive(pid):
                _ADOPTED.add(int(pid))
                threading.Thread(target=_watch_adopted, args=(mdir, int(pid)), daemon=True,
                                 name=f"studio-adopt-{pid}").start()
                counts["adopted"] += 1
            else:
                m.update(status="failed", finished_at=now_iso(), error="interrupted — the server restarted")
                _write_json(mfile, m)
                counts["failed"] += 1
        elif m.get("status") == "queued":
            ensure_worker_started()
            _QUEUE.put(mdir)
            counts["requeued"] += 1
    return counts


def mark_stale_jobs_failed(runs_root: Path) -> int:
    """Back-compat wrapper: number of jobs that could not be recovered."""
    return recover_jobs(runs_root)["failed"]


# ---- Validation ---------------------------------------------------------


def validate(store: StudioStore, mid: str, split: str = "val", conf: Optional[float] = None, iou: float = 0.7) -> dict:
    """Re-validate a trained model against the dataset's *current* labels."""
    mdir = model_dir(store, mid)
    m = read_manifest(mdir)
    if not (mdir / "best.pt").exists():
        raise Invalid("model has no weights yet")
    try:
        summary = export_yolo_dataset(store, mdir / "val_dataset", m["task"], val_pct=int(m["config"]["val_pct"]),
                                      class_ids=m["dataset"]["class_ids"])
    except ValueError as e:
        raise Invalid(str(e)) from e
    split_key = split if split in ("train", "val", "test") else "val"
    if summary.counts.get(split_key, 0) == 0 and split_key != "val":
        raise Invalid(f"no labelled images in split {split_key!r}")
    from ultralytics import YOLO

    with engines.INFER_LOCK:
        model = YOLO(str(mdir / "best.pt"))
        kw: dict[str, Any] = {
            # plots=True: Ultralytics only accumulates the confusion matrix
            # when plotting is on (the PNGs land in val_runs/, harmless).
            "data": str(summary.data_yaml), "split": split_key, "device": engines.device(),
            "plots": True, "verbose": False, "imgsz": int(m["config"]["imgsz"]), "iou": iou,
            "project": str(mdir / "val_runs"), "name": "val", "exist_ok": True,
        }
        if conf is not None:
            kw["conf"] = conf
        res = model.val(**kw)
    return metrics_to_json(res, m["dataset"]["names"], m["task"])


def metrics_to_json(res: Any, names: list[str], task: str) -> dict:
    rd = {k: round(float(v), 4) for k, v in (getattr(res, "results_dict", {}) or {}).items()}
    out: dict[str, Any] = {"summary": rd, "per_class": [], "speed": {}, "confusion": None}
    speed = getattr(res, "speed", None) or {}
    out["speed"] = {k: round(float(v), 2) for k, v in speed.items() if v is not None}
    heads = [("box", getattr(res, "box", None))]
    if task == "segment":
        heads.append(("mask", getattr(res, "seg", None)))
    per: dict[int, dict] = {}
    for head_name, head in heads:
        if head is None:
            continue
        idx = getattr(head, "ap_class_index", None)
        ap50 = getattr(head, "ap50", None)
        ap = getattr(head, "ap", None)
        p = getattr(head, "p", None)
        r = getattr(head, "r", None)
        if idx is None or ap50 is None:
            continue
        for pos, cid in enumerate(list(idx)):
            cid = int(cid)
            row = per.setdefault(cid, {"class_id": cid, "class_name": names[cid] if cid < len(names) else str(cid)})
            row[f"{head_name}_map50"] = round(float(ap50[pos]), 4)
            if ap is not None and pos < len(ap):
                row[f"{head_name}_map50_95"] = round(float(ap[pos]), 4)
            if p is not None and pos < len(p):
                row[f"{head_name}_precision"] = round(float(p[pos]), 4)
            if r is not None and pos < len(r):
                row[f"{head_name}_recall"] = round(float(r[pos]), 4)
    out["per_class"] = [per[k] for k in sorted(per)]
    cm = getattr(res, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        out["confusion"] = {"labels": list(names) + ["background"], "matrix": cm.matrix.astype(int).tolist()}
    return out


# ---- Export + benchmark ---------------------------------------------------

EXPORT_FORMATS: dict[str, dict[str, Any]] = {
    "onnx": {"label": "ONNX", "module": "onnx", "note": "Portable; runs in ONNX Runtime (CPU / CoreML EP)."},
    "torchscript": {"label": "TorchScript", "module": None, "note": "Self-contained PyTorch graph."},
    "coreml": {"label": "CoreML", "module": "coremltools", "note": "Apple Neural Engine on macOS / iOS."},
    "openvino": {"label": "OpenVINO", "module": "openvino", "note": "Intel CPUs / iGPUs (industrial PCs)."},
    "ncnn": {"label": "NCNN", "module": "ncnn", "note": "Mobile / ARM edge devices."},
    "tflite": {"label": "TFLite", "module": "tensorflow", "note": "Android / Coral. Heavy TensorFlow dependency."},
}


def export_formats() -> list[dict]:
    import importlib.util

    out = []
    for key, f in EXPORT_FORMATS.items():
        mod = f["module"]
        ok = mod is None or importlib.util.find_spec(mod) is not None
        out.append({"id": key, "label": f["label"], "available": ok, "note": f["note"] if ok else f"needs `{mod}` installed"})
    return out


def export_model(
    store: StudioStore, mid: str, fmt: str, imgsz: Optional[int] = None, half: bool = False,
    int8: bool = False, dynamic: bool = False, nms: bool = False,
) -> dict:
    if fmt not in EXPORT_FORMATS:
        raise Invalid(f"unknown export format {fmt!r}")
    avail = {f["id"]: f for f in export_formats()}
    if not avail[fmt]["available"]:
        raise Invalid(f"{EXPORT_FORMATS[fmt]['label']} export {avail[fmt]['note']}")
    mdir = model_dir(store, mid)
    m = read_manifest(mdir)
    if not (mdir / "best.pt").exists():
        raise Invalid("model has no weights yet")
    from ultralytics import YOLO

    work = mdir / "export_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copy2(mdir / "best.pt", work / "best.pt")
    t0 = time.perf_counter()
    model = YOLO(str(work / "best.pt"))
    kw: dict[str, Any] = {"format": fmt, "imgsz": int(imgsz or m["config"]["imgsz"]), "device": "cpu"}
    if half:
        kw["half"] = True
    if int8:
        kw["int8"] = True
        kw["data"] = str(mdir / "dataset" / "data.yaml")
    if dynamic:
        kw["dynamic"] = True
    if nms:
        kw["nms"] = True
    produced = Path(model.export(**kw))
    secs = time.perf_counter() - t0
    exports = mdir / "exports"
    exports.mkdir(exist_ok=True)
    suffix = "".join(p for p in ["-fp16" if half else "", "-int8" if int8 else "", "-dyn" if dynamic else "", "-nms" if nms else ""])
    target_name = f"{produced.stem}{suffix}{produced.suffix}" if produced.is_file() else f"{produced.name}{suffix}"
    target = exports / target_name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    _export_zip_path(mdir, target_name).unlink(missing_ok=True)  # stale download archive
    shutil.move(str(produced), str(target))
    shutil.rmtree(work, ignore_errors=True)
    size = target.stat().st_size if target.is_file() else sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    entry = {"format": fmt, "file": target.name, "size_bytes": size, "seconds": round(secs, 1), "created_at": now_iso(),
             "imgsz": kw["imgsz"], "half": half, "int8": int8, "dynamic": dynamic, "nms": nms}
    m = read_manifest(mdir)
    m["exports"] = [e for e in m.get("exports", []) if e["file"] != target.name] + [entry]
    write_manifest(mdir, m)
    return entry


def export_download_path(store: StudioStore, mid: str, file: str) -> Path:
    """File to send for an export; directories (CoreML, OpenVINO) get zipped."""
    mdir = model_dir(store, mid)
    target = (mdir / "exports" / file).resolve()
    if mdir.resolve() not in target.parents or not target.exists():
        raise NotFound(file)
    if target.is_file():
        return target
    # make_archive appends ".zip" to the base name; don't use with_suffix,
    # which would turn "best.mlpackage" into "best.zip".
    zip_path = _export_zip_path(mdir, target.name)
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(zip_path.parent / target.name), "zip", root_dir=str(target.parent), base_dir=target.name)
    return zip_path


def _export_zip_path(mdir: Path, name: str) -> Path:
    return mdir / "exports" / ".zips" / f"{name}.zip"


def benchmark(store: StudioStore, mid: str, n_images: int = 20) -> list[dict]:
    """Latency of best.pt (MPS + CPU) and every export on dataset images."""
    import numpy as np
    from ultralytics import YOLO

    mdir = model_dir(store, mid)
    m = read_manifest(mdir)
    if not (mdir / "best.pt").exists():
        raise Invalid("model has no weights yet")
    imgs = [store.read_image(r["id"]) for r in store.list_images()[: max(2, n_images)]]
    if not imgs:
        raise Invalid("add some images to benchmark on")
    imgsz = int(m["config"]["imgsz"])
    candidates: list[tuple[str, Path, str]] = [("PyTorch", mdir / "best.pt", engines.device())]
    if engines.device() != "cpu":
        candidates.append(("PyTorch", mdir / "best.pt", "cpu"))
    for e in m.get("exports", []):
        p = mdir / "exports" / e["file"]
        if p.exists() and e["format"] in ("onnx", "torchscript", "coreml", "openvino"):
            candidates.append((EXPORT_FORMATS[e["format"]]["label"], p, "cpu"))
    rows = []
    with engines.INFER_LOCK:
        for label, path, dev in candidates:
            try:
                model = YOLO(str(path), task=m["task"])
                # Warm up once per distinct input shape: MPS (and some
                # exporters) compile kernels per shape, and those one-off
                # costs would otherwise land in p95.
                seen: set[tuple[int, ...]] = set()
                for im in imgs:
                    if im.shape not in seen:
                        seen.add(im.shape)
                        model.predict(im, imgsz=imgsz, device=dev, verbose=False)
                times = []
                for im in imgs:
                    t0 = time.perf_counter()
                    model.predict(im, imgsz=imgsz, device=dev, verbose=False)
                    times.append((time.perf_counter() - t0) * 1000)
                arr = np.array(times)
                rows.append({
                    "runtime": label, "artifact": path.name, "device": dev,
                    "p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "mean_ms": round(float(arr.mean()), 2),
                    "fps": round(1000.0 / float(arr.mean()), 1),
                    "n": len(times),
                })
            except Exception as e:  # one broken runtime shouldn't sink the table
                rows.append({"runtime": label, "artifact": path.name, "device": dev, "error": str(e)[:300]})
    return rows
