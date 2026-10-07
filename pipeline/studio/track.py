"""Video tracking jobs — Ultralytics Track mode (ByteTrack / BoT-SORT).

`tracks/{tid}/`:
    track.json      manifest: request, status, stats
    progress.json   frames done / total
    overlay.mp4     H.264 (browser-playable) annotated video
    tracks.jsonl    one line per processed frame: {"frame", "detections": […]}

Optional *counting line* — the conveyor use case: a horizontal (axis "y")
or vertical (axis "x") line at a fraction of the frame; a track is counted
once when its box centre crosses it, per direction.

Jobs run sequentially on a daemon thread. Each frame takes `INFER_LOCK`
only for its own forward pass, so interactive prompting stays responsive
while a long video tracks in the background.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from pipeline.studio import engines
from pipeline.studio.infer import predict_kwargs, results_to_json
from pipeline.studio.store import Invalid, NotFound, StudioStore, check_id, new_id, now_iso

log = logging.getLogger(__name__)

DATA_DIR = engines.REPO_ROOT / "data"
TRACKERS = ("bytetrack", "botsort")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def resolve_video(path: str) -> Path:
    """Only videos under data/ may be read (same rule as GET /videos)."""
    p = Path(path)
    p = (engines.REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if DATA_DIR.resolve() not in p.parents or not p.is_file():
        raise Invalid(f"video not found under data/: {path}")
    return p


def start_track(store: StudioStore, req: dict) -> dict:
    hint = engines.missing_dep("lap")
    if hint:
        raise Invalid(hint)
    video = resolve_video(str(req.get("video_path", "")))
    tracker = req.get("tracker", "bytetrack")
    if tracker not in TRACKERS:
        raise Invalid(f"tracker must be one of {TRACKERS}")
    spec = req.get("model") or {}
    if spec.get("kind") not in ("yolo26", "trained", "yoloe-text"):
        raise Invalid("tracking needs a YOLO26, trained or YOLOE text-prompt model")
    line = req.get("count_line")
    if line is not None:
        if line.get("axis") not in ("x", "y") or not (0.0 < float(line.get("pos", -1)) < 1.0):
            raise Invalid("count_line needs axis 'x'|'y' and 0 < pos < 1")
    tid = new_id("trk")
    tdir = store.tracks_dir / tid
    tdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": tid,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "video_path": str(req.get("video_path")),
        "tracker": tracker,
        "model": spec,
        "params": req.get("params") or {},
        "stride": int(max(1, int(req.get("stride") or 1))),
        "max_frames": int(req.get("max_frames") or 0),
        "count_line": line,
        "stats": None,
    }
    _write_json(tdir / "track.json", manifest)
    ensure_worker_started()
    _QUEUE.put((store.root, tdir))
    return track_detail(store, tid)


def recover_jobs(runs_root: Path) -> dict[str, int]:
    """Reconcile tracking jobs left by a previous server process.

    Unlike training, tracking runs in-process, so a restart kills it:
    running → failed ("interrupted"), queued → re-queued.
    """
    counts = {"failed": 0, "requeued": 0}
    for mfile in sorted(Path(runs_root).glob("projects/*/studio/tracks/*/track.json")):
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        tdir = mfile.parent
        if m.get("status") == "running":
            m.update(status="failed", finished_at=now_iso(), error="interrupted — the server restarted")
            _write_json(mfile, m)
            counts["failed"] += 1
        elif m.get("status") == "queued":
            ensure_worker_started()
            _QUEUE.put((tdir.parent.parent, tdir))
            counts["requeued"] += 1
        elif m.get("status") == "cancelled":  # cancelled while queued, never cleaned up
            shutil.rmtree(tdir, ignore_errors=True)
    return counts


def list_tracks(store: StudioStore) -> list[dict]:
    if not store.tracks_dir.exists():
        return []
    out = []
    for d in sorted(store.tracks_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "track.json").exists():
            out.append(track_detail(store, d.name))
    return out


def track_detail(store: StudioStore, tid: str) -> dict:
    tdir = store.tracks_dir / tid
    if not (tdir / "track.json").exists():
        raise NotFound(tid)
    m = json.loads((tdir / "track.json").read_text())
    try:
        m["progress"] = json.loads((tdir / "progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        m["progress"] = None
    m["has_video"] = (tdir / "overlay.mp4").exists() and m["status"] == "completed"
    return m


def delete_track(store: StudioStore, tid: str) -> None:
    tdir = store.tracks_dir / tid
    if not tdir.exists():
        raise NotFound(tid)
    m = json.loads((tdir / "track.json").read_text())
    if m["status"] in ("queued", "running"):
        m["status"] = "cancelled"
        _write_json(tdir / "track.json", m)
        _CANCELLED.add(str(tdir.resolve()))
        return
    shutil.rmtree(tdir, ignore_errors=True)


_QUEUE: "queue.Queue[Optional[tuple[Path, Path]]]" = queue.Queue()
_WORKER: Optional[threading.Thread] = None
_LOCK = threading.Lock()
_CANCELLED: set[str] = set()


def ensure_worker_started() -> None:
    global _WORKER
    with _LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(target=_loop, name="studio-track", daemon=True)
        _WORKER.start()


def stop_worker(timeout: float = 2.0) -> None:
    _QUEUE.put(None)
    if _WORKER is not None:
        _WORKER.join(timeout=timeout)


def _loop() -> None:
    while True:
        job = _QUEUE.get()
        if job is None:
            return
        root, tdir = job
        try:
            _run(StudioStore(root), tdir)
        except Exception as e:
            log.exception("track job failed: %s", tdir.name)
            try:
                m = json.loads((tdir / "track.json").read_text())
                m.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=now_iso())
                _write_json(tdir / "track.json", m)
            except Exception:
                pass


def _load_track_model(store: StudioStore, spec: dict) -> Any:
    # Fresh instance per job: the tracker state lives on the predictor and
    # must not leak between jobs or into the shared prompt models.
    from ultralytics import YOLO, YOLOE

    kind = spec["kind"]
    if kind == "yolo26":
        return YOLO(str(engines.ensure_weights(engines.yolo26_name(spec.get("task", "detect"), spec.get("size", "n")))))
    if kind == "trained":
        p = store.models_dir / check_id(str(spec.get("model_id")), "model id") / "best.pt"
        if not p.exists():
            raise Invalid("trained model has no weights")
        return YOLO(str(p))
    model = YOLOE(str(engines.ensure_weights(engines.yoloe_name(str(spec.get("family", "26")), str(spec.get("size", "s"))))))
    engines.yoloe_set_text_classes(model, [str(c) for c in spec.get("classes") or []] or ["object"])
    return model


class _Writer:
    """H.264 via an ffmpeg pipe (browser-playable); mp4v fallback."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path = path
        self.proc = None
        self.cv = None
        ff = shutil.which("ffmpeg")
        if ff:
            self.proc = subprocess.Popen(
                [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE,
            )
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        elif self.cv is not None:
            self.cv.release()


def _run(store: StudioStore, tdir: Path) -> None:
    if not (tdir / "track.json").exists():
        return
    m = json.loads((tdir / "track.json").read_text())
    if m["status"] == "cancelled":  # cancelled while still queued
        shutil.rmtree(tdir, ignore_errors=True)
        _CANCELLED.discard(str(tdir.resolve()))
        return
    if m["status"] != "queued":
        return
    m.update(status="running", started_at=now_iso())
    _write_json(tdir / "track.json", m)

    video = resolve_video(m["video_path"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise Invalid(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = m["stride"]
    n_target = total // stride if total else 0
    if m["max_frames"]:
        n_target = min(n_target, m["max_frames"]) if n_target else m["max_frames"]

    with engines.INFER_LOCK:
        model = _load_track_model(store, m["model"])
    kw = predict_kwargs(m["params"], allow_end2end=m["model"]["kind"] != "yoloe-text")
    kw["persist"] = True
    kw["tracker"] = f"{m['tracker']}.yaml"

    line = m.get("count_line")
    last_side: dict[int, int] = {}
    counted: dict[int, str] = {}
    counts = {"forward": 0, "backward": 0}
    ids_by_class: dict[str, set[int]] = {}
    writer: Optional[_Writer] = None
    t0 = time.perf_counter()
    infer_ms: list[float] = []
    done = 0
    idx = -1
    with open(tdir / "tracks.jsonl", "w") as jf:
        while True:
            if str(tdir.resolve()) in _CANCELLED:
                break
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            if n_target and done >= n_target:
                break
            t1 = time.perf_counter()
            with engines.INFER_LOCK:
                r = model.track(frame, **kw)[0]
            infer_ms.append((time.perf_counter() - t1) * 1000)
            res = results_to_json(r, polygons=False)
            dets = res["detections"]
            jf.write(json.dumps({"frame": idx, "detections": dets}) + "\n")

            vis = r.plot(line_width=2, font_size=12)
            hh, ww = vis.shape[:2]
            for d in dets:
                tid = d.get("track_id")
                if tid is None:
                    continue
                ids_by_class.setdefault(d["class_name"], set()).add(tid)
                if line:
                    x1, y1, x2, y2 = d["bbox"]
                    c = (y1 + y2) / 2 / hh if line["axis"] == "y" else (x1 + x2) / 2 / ww
                    side = 1 if c >= float(line["pos"]) else -1
                    prev = last_side.get(tid)
                    if prev is not None and prev != side and tid not in counted:
                        direction = "forward" if side > 0 else "backward"
                        counted[tid] = direction
                        counts[direction] += 1
                    last_side[tid] = side
            if line:
                if line["axis"] == "y":
                    y = int(float(line["pos"]) * hh)
                    cv2.line(vis, (0, y), (ww, y), (0, 255, 255), 2)
                else:
                    x = int(float(line["pos"]) * ww)
                    cv2.line(vis, (x, 0), (x, hh), (0, 255, 255), 2)
                cv2.putText(vis, f"count {counts['forward'] + counts['backward']}  (+{counts['forward']} / -{counts['backward']})",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, f"count {counts['forward'] + counts['backward']}  (+{counts['forward']} / -{counts['backward']})",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            # Keep overlays light: cap width at 1280 and keep even dims for yuv420p.
            if ww > 1280:
                s = 1280 / ww
                vis = cv2.resize(vis, (1280, int(round(hh * s))), interpolation=cv2.INTER_AREA)
            vh, vw = vis.shape[:2]
            vis = vis[: vh - vh % 2, : vw - vw % 2]
            if writer is None:
                writer = _Writer(tdir / "overlay.mp4", vis.shape[1], vis.shape[0], fps / stride)
            writer.write(vis)
            done += 1
            if done % 5 == 0:
                _write_json(tdir / "progress.json", {"done": done, "total": n_target, "updated_at": now_iso()})
    cap.release()
    if writer is not None:
        writer.close()

    m = json.loads((tdir / "track.json").read_text())
    if m["status"] == "cancelled":
        shutil.rmtree(tdir, ignore_errors=True)
        _CANCELLED.discard(str(tdir.resolve()))
        return
    secs = time.perf_counter() - t0
    m.update(
        status="completed",
        finished_at=now_iso(),
        stats={
            "frames": done,
            "seconds": round(secs, 1),
            "fps": round(done / secs, 1) if secs > 0 else None,
            "mean_infer_ms": round(float(np.mean(infer_ms)), 1) if infer_ms else None,
            "unique_tracks": len(set().union(*ids_by_class.values())) if ids_by_class else 0,
            "unique_by_class": {k: len(v) for k, v in sorted(ids_by_class.items())},
            "line_counts": counts if line else None,
        },
    )
    _write_json(tdir / "track.json", m)
    _write_json(tdir / "progress.json", {"done": done, "total": n_target or done, "updated_at": now_iso()})
