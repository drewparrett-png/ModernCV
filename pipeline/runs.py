"""Teacher run directory format.

A "Teacher run" is everything produced by one Learn-mode invocation:
the prompt, the source video, the labels, the visualization, and timing
stats. This module is the single source of truth for how those artifacts
are laid out on disk and what the manifest looks like.

Layout
------
    runs/teacher_<timestamp>_<slug>/
        manifest.json        # task, prompt, video, models, status, timing
        overlay.mp4          # visualization (boxes/masks rendered)
        stats.json           # frames_processed, ms/frame, n_detections, …
        labels/
            coco.json        # standard COCO format for downstream training
            per_frame.jsonl  # one line per frame — UI-fast slice

The manifest is the source of truth for status. A run with
`status: "completed"` is what unlocks Optimize mode.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
MANIFEST_NAME = "manifest.json"
STATS_NAME = "stats.json"
PROGRESS_NAME = "progress.json"
REJECTIONS_NAME = "rejections.json"
OVERLAY_NAME = "overlay.mp4"
LABELS_DIR = "labels"
COCO_NAME = "coco.json"
PER_FRAME_NAME = "per_frame.jsonl"


# ---- Manifest --------------------------------------------------------------


@dataclass
class RunManifest:
    """The on-disk description of a Teacher run.

    Persisted as `manifest.json` at the root of the run directory. Keep this
    small and stable — UI lists deserialize it, and downstream tools depend
    on the field names.
    """

    id: str
    task: str  # "detection" | "segmentation"
    prompt: str
    video_path: str  # relative to project root
    started_at: str  # ISO 8601 UTC
    ended_at: Optional[str] = None
    status: str = "running"  # "running" | "completed" | "failed"
    models: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    # ISO 8601 UTC timestamp set when the run is marked "Approved as ground
    # truth" via /runs/{id}/approve. None means not approved (review_status
    # is then derived from rejection presence). Default None so legacy
    # manifests load unchanged.
    approved_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class RunProgress:
    """Live progress for a running Teacher run.

    Persisted as `progress.json` and overwritten on each update — frontend
    polls /runs/{id} and gets this back. We update at most every N frames
    (see `PerFrameWriter.flush_period` in learn.py) so we don't burn IO.
    """

    stage: str = "starting"  # "starting" | "loading_models" | "running" | "finalizing"
    message: str = ""
    current_frame: int = 0
    total_frames: int = 0  # 0 when unknown (e.g. infinite stream)
    frames_with_detections: int = 0
    started_at: str = ""
    updated_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class RunStats:
    """Timing + count summary for a Teacher run.

    Optional sub-stats (per-block timing, per-class counts) can be added
    over time without breaking older readers — keep new fields optional.

    Detection breakdown
    -------------------
    `detections_per_class`: nested dict, one entry per class_name. Shape:
        {
          "soccer ball": {
            "n_detections": 18,                # total dets of this class
            "frames_present": 12,              # frames with ≥1 of this class
            "max_in_frame": 3,                 # most of this class in one frame
            "avg_per_frame": 0.36,             # n_detections / frames_processed
            "avg_per_present_frame": 1.5,      # n_detections / frames_present
            "score_avg": 0.42,                 # mean confidence across all dets
            "score_p50": 0.38,                 # median confidence
          },
          "player": { ... }
        }
    `per_frame_count_*` describe the distribution of *total* detections per
    frame (across all classes) — answers "how many objects do we typically
    find in a frame?".
    `per_frame_count_histogram`: count (as string key for JSON) → frames.
    """

    frames_processed: int = 0
    frames_with_detections: int = 0
    total_ms: float = 0.0
    avg_ms_per_frame: float = 0.0
    p50_ms_per_frame: float = 0.0
    p95_ms_per_frame: float = 0.0
    n_detections_total: int = 0

    # ---- detection breakdown (added phase 1) ------------------------------
    detections_per_class: dict[str, dict] = field(default_factory=dict)
    per_frame_count_min: int = 0
    per_frame_count_p50: int = 0
    per_frame_count_p95: int = 0
    per_frame_count_max: int = 0
    per_frame_count_avg: float = 0.0
    per_frame_count_histogram: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---- Student-run manifests ------------------------------------------------


@dataclass
class StudentManifest:
    """Distillation run — a small Student trained on Teacher COCO labels.

    Lives at `runs/student_<id>/manifest.json`. Carries two lists back to the
    Teachers that produced it:

      • `train_teacher_ids` — the Teachers whose curated `labels/coco.json`
        files get merged and used as ground truth during training.
      • `eval_teacher_ids`  — held-out Teachers used to measure the Student's
        transferability (mAP on clips/conditions it never saw at train
        time). May be empty.

    The two lists may overlap (a Teacher in both columns is "in-distribution
    eval" — a sanity check that the Student matches its Teacher on data it
    trained on, not a real generalization signal). The UI warns when this
    happens.

    Legacy on-disk format
    ---------------------
    Earlier Student dirs had a single `teacher_id: str` field. The reader
    promotes that to `train_teacher_ids=[teacher_id]`, `eval_teacher_ids=[]`
    so old runs still list cleanly without a separate migration script.
    """

    id: str
    train_teacher_ids: list[str]
    eval_teacher_ids: list[str]
    task: str  # mirrors the teachers' task (all train teachers must share it)
    prompt: str  # carried for display; pulled from the first train teacher
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"  # "running" | "completed" | "failed"
    models: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    # Confidence-band thresholds used to bucket frames at training time.
    # Persisted on the manifest so the run is reproducible from manifest.json
    # alone — i.e. someone reading the file later can answer "what filter
    # produced this Student's training set?" without grepping logs. All
    # defaulted so older manifests without these keys still load.
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class StudentStats:
    """Distilled-model evaluation summary. Filled in once the trainer runs.

    `map50` / `map50_95` are the **mean** across all eval teachers — useful
    for a single headline number but lossy. The full per-clip story is in
    `per_eval_teacher`: one entry per eval teacher with that teacher's
    own n_images / n_annotations / mAPs. That's what makes "transferability"
    actually informative — you can see "great on clip A, falls apart on
    clip B" instead of an opaque mean.

    If `eval_teacher_ids` was empty at distillation time, `per_eval_teacher`
    is `[]` and `map50`/`map50_95` are 0.0 — the trainer skipped the eval
    pass entirely.
    """

    train_images: int = 0
    train_annotations: int = 0
    train_seconds: float = 0.0
    epochs: int = 0
    map50: float = 0.0  # mean of per-eval-teacher map50; 0 if no eval teachers
    map50_95: float = 0.0
    avg_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    model_size_mb: float = 0.0
    # Per-eval-teacher breakdown. Each entry: {teacher_id, n_images,
    # n_annotations, map50, map50_95}. Empty list = no eval teachers.
    per_eval_teacher: list[dict[str, Any]] = field(default_factory=list)
    # ---- Phase 0.5/0.6 frame-bucket breakdown ----------------------------
    # All defaulted to safe values so existing stats.json files (no
    # bucketing fields) keep loading without a migration script.
    n_positive_frames: int = 0
    n_uncertain_dropped: int = 0
    n_true_negative_frames: int = 0
    # One entry per train teacher:
    # {"teacher_id": str, "positive": int, "uncertain": int, "true_negative": int}
    per_teacher_buckets: list[dict[str, Any]] = field(default_factory=list)
    # Thresholds the trainer actually used. Stamped on the stats so the
    # detail card can render "frame buckets at t_high=0.35, t_low=0.15"
    # without re-reading the manifest.
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---- Helpers ---------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Turn 'Find the soccer ball!' into 'find-the-soccer-ball' for filesystem use."""
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not s:
        s = "run"
    return s[:max_len]


def make_run_id(prompt: str, now: Optional[datetime] = None) -> str:
    """Generate a sortable, descriptive run id: teacher_<utc>_<slug>."""
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    return f"teacher_{ts}_{slugify(prompt)}"


def make_student_id(prompt: str, now: Optional[datetime] = None) -> str:
    """Sortable id for a Student run: student_<utc>_<slug>."""
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    return f"student_{ts}_{slugify(prompt)}"


def run_dir(run_id: str, runs_root: Path = RUNS_DIR) -> Path:
    return runs_root / run_id


# ---- Create / write --------------------------------------------------------


def create_run(
    task: str,
    prompt: str,
    video_path: str,
    models: dict[str, str],
    runs_root: Path = RUNS_DIR,
) -> tuple[Path, RunManifest]:
    """Allocate a fresh run directory and write an initial 'running' manifest.

    Returns (run_dir_path, manifest). The caller should fill in stats and
    flip status to 'completed' (or 'failed') when done — see
    `mark_completed` / `mark_failed`.
    """
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"unknown task: {task!r}")

    run_id = make_run_id(prompt)
    rdir = run_dir(run_id, runs_root)
    rdir.mkdir(parents=True, exist_ok=False)
    (rdir / LABELS_DIR).mkdir()

    manifest = RunManifest(
        id=run_id,
        task=task,
        prompt=prompt,
        video_path=video_path,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        models=models,
    )
    write_manifest(rdir, manifest)
    log.info("created run %s", rdir)
    return rdir, manifest


def write_manifest(rdir: Path, manifest: RunManifest) -> None:
    (rdir / MANIFEST_NAME).write_text(manifest.to_json())


def write_stats(rdir: Path, stats: RunStats) -> None:
    (rdir / STATS_NAME).write_text(stats.to_json())


def write_progress(rdir: Path, progress: RunProgress) -> None:
    """Atomic-ish progress write — write to a temp file then rename so a
    polling reader never sees a half-written JSON."""
    tmp = rdir / (PROGRESS_NAME + ".tmp")
    tmp.write_text(progress.to_json())
    tmp.replace(rdir / PROGRESS_NAME)


def read_progress(rdir: Path) -> Optional[RunProgress]:
    p = rdir / PROGRESS_NAME
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        # Reader raced the writer; the rename in `write_progress` should
        # prevent this, but be defensive.
        return None
    known = {f for f in RunProgress.__dataclass_fields__}
    return RunProgress(**{k: v for k, v in raw.items() if k in known})


def mark_completed(rdir: Path, stats: Optional[RunStats] = None) -> RunManifest:
    manifest = read_manifest(rdir)
    manifest.status = "completed"
    manifest.ended_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_manifest(rdir, manifest)
    if stats is not None:
        write_stats(rdir, stats)
    return manifest


def mark_failed(rdir: Path, error: str) -> RunManifest:
    manifest = read_manifest(rdir)
    manifest.status = "failed"
    manifest.error = error
    manifest.ended_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_manifest(rdir, manifest)
    return manifest


def approve_run(rdir: Path) -> RunManifest:
    """Stamp `approved_at` on the manifest. No-op if already approved.

    Caller (the /runs/{id}/approve endpoint) is responsible for the
    "must be completed" 400 — this helper trusts its input.
    """
    manifest = read_manifest(rdir)
    if manifest.approved_at is not None:
        return manifest
    manifest.approved_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_manifest(rdir, manifest)
    return manifest


def unapprove_run(rdir: Path) -> RunManifest:
    """Clear `approved_at`. No-op if not currently approved."""
    manifest = read_manifest(rdir)
    if manifest.approved_at is None:
        return manifest
    manifest.approved_at = None
    write_manifest(rdir, manifest)
    return manifest


# ---- Read / list -----------------------------------------------------------


def read_manifest(rdir: Path) -> RunManifest:
    raw = json.loads((rdir / MANIFEST_NAME).read_text())
    # Tolerant of unknown fields so we can evolve the schema additively.
    known = {f for f in RunManifest.__dataclass_fields__}
    return RunManifest(**{k: v for k, v in raw.items() if k in known})


def read_stats(rdir: Path) -> Optional[RunStats]:
    p = rdir / STATS_NAME
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    known = {f for f in RunStats.__dataclass_fields__}
    return RunStats(**{k: v for k, v in raw.items() if k in known})


_STALE_INTERRUPT_MSG = (
    "Run interrupted — server restarted before it could finish. "
    "Delete this run and start a new one."
)


_STALE_STATES = ("running", "queued")


def mark_stale_runs_failed(runs_root: Path = RUNS_DIR) -> int:
    """At server startup, sweep any 'running' or 'queued' runs and flip
    them to failed.

    Rationale: Learn / Optimize workers run as *daemon threads* on the
    server process, and the Teacher queue is purely in-memory. uvicorn
    --reload (or any process restart) kills both daemon threads and the
    queue contents instantly, but the manifest on disk still says
    'running' or 'queued'. The UI polls and shows a forever-spinning
    state with no progress updates because there's no live worker.

    Calling this on startup is correct because at that exact moment, by
    definition, no in-flight worker can have survived from before the
    restart — so every 'running' / 'queued' status on disk is stale.
    Both teacher runs (RunManifest) and student runs (StudentManifest)
    are swept.

    Returns the number of runs that were patched.
    """
    if not runs_root.exists():
        return 0
    n = 0
    for p in runs_root.iterdir():
        if not p.is_dir() or not (p / MANIFEST_NAME).exists():
            continue
        # Teacher run? (id prefix "teacher_")
        if p.name.startswith("teacher_"):
            try:
                m = read_manifest(p)
                if m.status in _STALE_STATES:
                    mark_failed(p, _STALE_INTERRUPT_MSG)
                    n += 1
            except Exception as e:  # pragma: no cover — corrupted manifest
                log.warning("could not patch teacher %s: %s", p, e)
        elif p.name.startswith("student_"):
            try:
                m = read_student_manifest(p)
                if m.status in _STALE_STATES:
                    mark_student_failed(p, _STALE_INTERRUPT_MSG)
                    n += 1
            except Exception as e:  # pragma: no cover
                log.warning("could not patch student %s: %s", p, e)
    if n:
        log.info("marked %d stale running/queued runs as failed at startup", n)
    return n


def list_runs(runs_root: Path = RUNS_DIR) -> list[RunManifest]:
    """Teacher-run manifests on disk, newest first. Filters to dirs whose
    name starts with 'teacher_' so Student dirs don't bleed into the
    Teacher list."""
    if not runs_root.exists():
        return []
    out: list[RunManifest] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("teacher_"):
            continue
        if not (p / MANIFEST_NAME).exists():
            continue
        try:
            out.append(read_manifest(p))
        except Exception as e:  # pragma: no cover — corrupted manifest, skip
            log.warning("skipping %s: %s", p, e)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out


def read_student_manifest(rdir: Path) -> StudentManifest:
    """Deserialize a Student manifest, with legacy-format compatibility.

    Older Students (pre multi-teacher) stored a single `teacher_id: str`.
    We promote that field on read so the rest of the codebase only ever
    sees the new shape:

        teacher_id: "T123"  →  train_teacher_ids=["T123"], eval_teacher_ids=[]

    The first time such a manifest is *written* (e.g. via mark_student_failed)
    it'll be persisted in the new shape, so the next read is a straight
    pass-through.
    """
    raw = json.loads((rdir / MANIFEST_NAME).read_text())
    # Legacy promotion: single teacher_id → train list of one.
    if "teacher_id" in raw and "train_teacher_ids" not in raw:
        raw["train_teacher_ids"] = [raw["teacher_id"]]
        raw["eval_teacher_ids"] = []
    raw.setdefault("train_teacher_ids", [])
    raw.setdefault("eval_teacher_ids", [])
    known = {f for f in StudentManifest.__dataclass_fields__}
    return StudentManifest(**{k: v for k, v in raw.items() if k in known})


def write_student_manifest(rdir: Path, manifest: StudentManifest) -> None:
    (rdir / MANIFEST_NAME).write_text(manifest.to_json())


def read_student_stats(rdir: Path) -> Optional[StudentStats]:
    p = rdir / STATS_NAME
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    known = {f for f in StudentStats.__dataclass_fields__}
    return StudentStats(**{k: v for k, v in raw.items() if k in known})


def write_student_stats(rdir: Path, stats: StudentStats) -> None:
    (rdir / STATS_NAME).write_text(stats.to_json())


def list_students(runs_root: Path = RUNS_DIR) -> list[StudentManifest]:
    if not runs_root.exists():
        return []
    out: list[StudentManifest] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("student_"):
            continue
        if not (p / MANIFEST_NAME).exists():
            continue
        try:
            out.append(read_student_manifest(p))
        except Exception as e:  # pragma: no cover
            log.warning("skipping student %s: %s", p, e)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out


def create_student(
    *,
    train_teacher_ids: list[str],
    eval_teacher_ids: list[str],
    task: str,
    prompt: str,
    models: dict[str, str],
    t_high: float = 0.35,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    runs_root: Path = RUNS_DIR,
) -> tuple[Path, StudentManifest]:
    """Allocate a fresh Student dir + initial 'running' manifest.

    Caller (the trainer) is responsible for writing stats and flipping
    status to 'completed'/'failed' via `mark_student_completed` /
    `mark_student_failed`.

    `train_teacher_ids` must be non-empty — a Student with no training data
    is meaningless. `eval_teacher_ids` may be empty, in which case the
    Student is trained but not evaluated for transferability (the trainer
    will skip the held-out mAP step).

    The confidence-band thresholds (`t_high`, `t_low`,
    `treat_empty_as_negative`) are recorded on the manifest at creation
    time so the run is reproducible from manifest.json alone.
    """
    if not train_teacher_ids:
        raise ValueError("train_teacher_ids must contain at least one teacher")

    student_id = make_student_id(prompt)
    rdir = runs_root / student_id
    rdir.mkdir(parents=True, exist_ok=False)

    manifest = StudentManifest(
        id=student_id,
        train_teacher_ids=list(train_teacher_ids),
        eval_teacher_ids=list(eval_teacher_ids),
        task=task,
        prompt=prompt,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        models=models,
        t_high=t_high,
        t_low=t_low,
        treat_empty_as_negative=treat_empty_as_negative,
    )
    write_student_manifest(rdir, manifest)
    log.info(
        "created student %s (train: %s, eval: %s)",
        rdir,
        train_teacher_ids,
        eval_teacher_ids or "—",
    )
    return rdir, manifest


def mark_student_completed(rdir: Path, stats: Optional[StudentStats] = None) -> StudentManifest:
    manifest = read_student_manifest(rdir)
    manifest.status = "completed"
    manifest.ended_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_student_manifest(rdir, manifest)
    if stats is not None:
        write_student_stats(rdir, stats)
    return manifest


def mark_student_failed(rdir: Path, error: str) -> StudentManifest:
    manifest = read_student_manifest(rdir)
    manifest.status = "failed"
    manifest.error = error
    manifest.ended_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_student_manifest(rdir, manifest)
    return manifest


# ---- Per-frame labels (jsonl) ---------------------------------------------


class PerFrameWriter:
    """Append-only writer for `labels/per_frame.jsonl`.

    Each line is a small JSON object: {"frame_idx": int, "detections": [...],
    "masks": [...]}. The Inspector fetches this file once and slices
    client-side — small files, no decode-on-seek.
    """

    def __init__(self, rdir: Path):
        self.path = rdir / LABELS_DIR / PER_FRAME_NAME
        self._fp = self.path.open("w")

    def write(self, frame_idx: int, detections: list[dict], masks: list[dict] | None = None) -> None:
        rec: dict[str, Any] = {"frame_idx": frame_idx, "detections": detections}
        if masks is not None:
            rec["masks"] = masks
        self._fp.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "PerFrameWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def detection_to_dict(det: Any) -> dict:
    """Serialize a Detection dataclass to a JSON-safe dict for per_frame.jsonl."""
    return {
        "bbox_xyxy": list(det.bbox_xyxy),
        "score": float(det.score),
        "class_id": int(det.class_id),
        "class_name": det.class_name,
    }


# ---- Detection-breakdown statistics ---------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on a *pre-sorted* list. Returns 0.0 on empty.

    We use the same convention as the existing p50/p95 timing code in
    learn.py (`sorted_ms[min(n - 1, int(n * 0.95))]`) so timing and detection
    percentiles agree on edge cases.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(n - 1, int(n * q))
    return float(sorted_values[idx])


def compute_detection_breakdown(
    per_frame_records: Iterable[dict],
    *,
    frames_processed: Optional[int] = None,
) -> dict[str, Any]:
    """Compute per-class counts + per-frame count distribution.

    Single source of truth — used by `learn.py` to fill in stats during a
    fresh run, and by `recompute_stats` to backfill older runs from
    `labels/per_frame.jsonl`.

    Parameters
    ----------
    per_frame_records:
        Iterable of dicts with shape `{"frame_idx": int, "detections": [...]}`.
        Detections need `class_name` and `score`.
    frames_processed:
        Total frame count to use in the `avg_per_frame` denominator. If
        omitted, we count records as they're consumed. Pass this explicitly
        when the run skipped frames with zero detections (none currently do,
        but the field-level signal is cleaner this way).

    Returns
    -------
    A dict ready to be merged into RunStats kwargs.
    """
    # Per-class accumulators
    per_class_counts: dict[str, int] = {}
    per_class_frames_present: dict[str, int] = {}
    per_class_max_in_frame: dict[str, int] = {}
    per_class_scores: dict[str, list[float]] = {}

    # Per-frame total-count series (for percentiles + histogram)
    per_frame_total: list[int] = []
    histogram: dict[int, int] = {}

    n_frames = 0
    for rec in per_frame_records:
        n_frames += 1
        dets = rec.get("detections", []) or []
        per_frame_total.append(len(dets))
        histogram[len(dets)] = histogram.get(len(dets), 0) + 1

        # Per-class slice for *this* frame
        per_frame_class: dict[str, int] = {}
        for d in dets:
            cls = d.get("class_name") or f"class_{d.get('class_id', '?')}"
            per_class_counts[cls] = per_class_counts.get(cls, 0) + 1
            per_frame_class[cls] = per_frame_class.get(cls, 0) + 1
            score = d.get("score")
            if score is not None:
                per_class_scores.setdefault(cls, []).append(float(score))

        for cls, c in per_frame_class.items():
            per_class_frames_present[cls] = per_class_frames_present.get(cls, 0) + 1
            if c > per_class_max_in_frame.get(cls, 0):
                per_class_max_in_frame[cls] = c

    if frames_processed is None:
        frames_processed = n_frames
    denom_frames = max(1, frames_processed)

    # Build per-class summary dict
    detections_per_class: dict[str, dict] = {}
    for cls, n in per_class_counts.items():
        scores = sorted(per_class_scores.get(cls, []))
        present = per_class_frames_present.get(cls, 0)
        detections_per_class[cls] = {
            "n_detections": int(n),
            "frames_present": int(present),
            "max_in_frame": int(per_class_max_in_frame.get(cls, 0)),
            "avg_per_frame": float(n) / denom_frames,
            "avg_per_present_frame": (float(n) / present) if present else 0.0,
            "score_avg": (sum(scores) / len(scores)) if scores else 0.0,
            "score_p50": _percentile(scores, 0.5),
        }

    # Per-frame distribution
    sorted_counts = sorted(per_frame_total)
    if sorted_counts:
        avg_count = sum(sorted_counts) / len(sorted_counts)
        cmin = int(sorted_counts[0])
        cmax = int(sorted_counts[-1])
        cp50 = int(_percentile(sorted_counts, 0.5))
        cp95 = int(_percentile(sorted_counts, 0.95))
    else:
        avg_count = 0.0
        cmin = cmax = cp50 = cp95 = 0

    # JSON keys must be strings — coerce histogram keys
    hist_str: dict[str, int] = {str(k): int(v) for k, v in sorted(histogram.items())}

    return {
        "detections_per_class": detections_per_class,
        "per_frame_count_min": cmin,
        "per_frame_count_p50": cp50,
        "per_frame_count_p95": cp95,
        "per_frame_count_max": cmax,
        "per_frame_count_avg": float(avg_count),
        "per_frame_count_histogram": hist_str,
    }


def read_per_frame(rdir: Path) -> Iterable[dict]:
    """Yield parsed records from `labels/per_frame.jsonl`. Skips blank lines."""
    p = rdir / LABELS_DIR / PER_FRAME_NAME
    if not p.exists():
        return
    with p.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover — corrupt line
                log.warning("skipping malformed line in %s: %s", p, e)


# ---- COCO export -----------------------------------------------------------


# ---- Rejections (human curation) ------------------------------------------


def has_any_rejections(rdir: Path) -> bool:
    """Cheap "is anything rejected here?" check for `review_status` derivation.

    Avoids parsing the full rejection map — `review_status` is computed on
    every /runs listing entry, so a 50-run directory must not pay 50 full
    JSON parses. Short-circuits on:

      1. File missing → False.
      2. File present but empty / "{}" → False.
      3. File parseable as a dict with at least one frame whose value is
         a non-empty list → True.

    Anything malformed (bad JSON, non-dict root) is treated as "no
    rejections" — same defensive behavior as `read_rejections`.
    """
    p = rdir / REJECTIONS_NAME
    if not p.exists():
        return False
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False
    for v in raw.values():
        if isinstance(v, list) and len(v) > 0:
            return True
    return False


def read_rejections(rdir: Path) -> dict[int, list[int]]:
    """Read curated rejections.

    On disk: `{"<frame_idx>": [det_idx, det_idx, …], …}` (str keys for JSON).
    In memory: int keys, sorted-ish lists. Missing file → empty dict.
    """
    p = rdir / REJECTIONS_NAME
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        log.warning("rejections.json was malformed at %s; treating as empty", p)
        return {}
    out: dict[int, list[int]] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = sorted({int(x) for x in v})
        except (TypeError, ValueError):
            continue
    return out


def write_rejections(rdir: Path, rejections: dict[int, list[int]]) -> None:
    """Atomic-ish write — temp file + rename so a polling reader never sees
    half-written JSON."""
    serializable = {str(k): sorted(set(int(x) for x in v)) for k, v in rejections.items() if v}
    tmp = rdir / (REJECTIONS_NAME + ".tmp")
    tmp.write_text(json.dumps(serializable, indent=2))
    tmp.replace(rdir / REJECTIONS_NAME)


def toggle_rejection(rdir: Path, frame_idx: int, det_idx: int) -> dict[int, list[int]]:
    """Flip the rejected state for one detection. Returns the updated map.

    Used by `POST /runs/{id}/rejections/toggle`. Idempotent in the sense that
    two consecutive toggles return to the original state.
    """
    state = read_rejections(rdir)
    bucket = set(state.get(frame_idx, []))
    if det_idx in bucket:
        bucket.discard(det_idx)
    else:
        bucket.add(det_idx)
    if bucket:
        state[frame_idx] = sorted(bucket)
    else:
        state.pop(frame_idx, None)
    write_rejections(rdir, state)
    return state


def filter_detections_by_rejections(
    per_frame: Iterable[dict],
    rejections: dict[int, list[int]],
) -> Iterable[dict]:
    """Helper for the eventual Optimize trainer. Yields per-frame records
    with rejected detection indices removed. Doesn't change schema; just
    drops elements from `detections` and reindexes nothing — caller should
    treat detection-position as opaque."""
    for rec in per_frame:
        idx = int(rec.get("frame_idx", -1))
        rejected = set(rejections.get(idx, []))
        if not rejected:
            yield rec
            continue
        kept = [d for i, d in enumerate(rec.get("detections", [])) if i not in rejected]
        new_rec = dict(rec)
        new_rec["detections"] = kept
        yield new_rec


# ---- Delete a run ---------------------------------------------------------


def delete_run(rdir: Path) -> None:
    """Recursively remove a run directory. Caller is responsible for stopping
    any in-flight worker that might still be writing to it."""
    import shutil

    if not rdir.exists():
        return
    shutil.rmtree(rdir)


def write_coco(
    rdir: Path,
    *,
    prompt: str,
    image_records: Iterable[dict],
    annotation_records: Iterable[dict],
    category_names: Iterable[str],
) -> None:
    """Write a minimal but valid COCO-format file.

    Categories are derived from class names seen in the run. Image records
    are {id, file_name, width, height}. Annotation records are
    {id, image_id, category_id, bbox: [x,y,w,h], score, segmentation?, area,
    iscrowd}.
    """
    categories = [
        {"id": i + 1, "name": name, "supercategory": "thing"}
        for i, name in enumerate(category_names)
    ]
    coco = {
        "info": {"description": f"ModernCV teacher run — prompt: {prompt!r}"},
        "licenses": [],
        "images": list(image_records),
        "annotations": list(annotation_records),
        "categories": categories,
    }
    (rdir / LABELS_DIR / COCO_NAME).write_text(json.dumps(coco))
