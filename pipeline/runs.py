"""Teacher / Student / Project directory format.

A "Project" is the top-level container: it pins one (task, prompts) tuple
and owns every Teacher and Student run produced under it. A "Teacher run"
is everything produced by one Learn-mode invocation under a project.

Layout
------
    runs/
      projects/
        <project_id>/
          project.json
          teachers/
            teacher_<timestamp>_<slug>/
              manifest.json        # task, prompt, video, models, status, timing
              overlay.mp4          # visualization (boxes/masks rendered)
              stats.json           # frames_processed, ms/frame, n_detections, …
              labels/
                coco.json
                per_frame.jsonl
          students/
            student_<timestamp>_<slug>/
              manifest.json
              stats.json
              ...

The project's `project.json` is the source of truth for (task, prompts).
Both fields are LOCKED at creation — Teacher runs inside a project inherit
them. The manifest is the source of truth for run status; a run with
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
PROJECTS_DIR = "projects"
PROJECT_FILE = "project.json"
TEACHERS_DIR = "teachers"
STUDENTS_DIR = "students"
STUDENT_RUNS_DIR = "student_runs"
MANIFEST_NAME = "manifest.json"
STATS_NAME = "stats.json"
PROGRESS_NAME = "progress.json"
FRAME_STATES_NAME = "frame_states.json"
OVERLAY_NAME = "overlay.mp4"
LABELS_DIR = "labels"
PREDICTIONS_DIR = "predictions"
COCO_NAME = "coco.json"
PER_FRAME_NAME = "per_frame.jsonl"
CROPS_DIR = "crops"


# ---- Project --------------------------------------------------------------


@dataclass
class Project:
    """Container metadata. Persisted as `project.json` at the project root.

    `task` and `prompts` are immutable post-creation — runs reference them
    by inheritance, and mutating them after a Student trains would break
    the comparability of past runs. Only `name` is editable.
    """

    id: str
    name: str
    task: str  # "detection" | "segmentation"
    prompts: list[str]
    created_at: str  # ISO 8601 UTC

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def make_project_id(name: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    return f"proj_{ts}_{slugify(name)}"


def projects_root(runs_root: Path = RUNS_DIR) -> Path:
    return runs_root / PROJECTS_DIR


def project_dir(project_id: str, runs_root: Path = RUNS_DIR) -> Path:
    return projects_root(runs_root) / project_id


def teachers_dir(project_id: str, runs_root: Path = RUNS_DIR) -> Path:
    return project_dir(project_id, runs_root) / TEACHERS_DIR


def students_dir(project_id: str, runs_root: Path = RUNS_DIR) -> Path:
    return project_dir(project_id, runs_root) / STUDENTS_DIR


def student_runs_root(project_id: str, runs_root: Path = RUNS_DIR) -> Path:
    """Phase 5: project-level top-level dir holding all student-run dirs.

    On disk: `runs/projects/<pid>/student_runs/<sid>/<rid>/`. The middle
    `<sid>` partition keeps a Student's runs grouped without forcing
    callers to scan every run when listing one Student's history.
    """
    return project_dir(project_id, runs_root) / STUDENT_RUNS_DIR


def write_project(pdir: Path, project: Project) -> None:
    (pdir / PROJECT_FILE).write_text(project.to_json())


def read_project(pdir: Path) -> Project:
    raw = json.loads((pdir / PROJECT_FILE).read_text())
    known = {f for f in Project.__dataclass_fields__}
    return Project(**{k: v for k, v in raw.items() if k in known})


def create_project(
    *,
    name: str,
    task: str,
    prompts: list[str],
    runs_root: Path = RUNS_DIR,
) -> Project:
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"unknown task: {task!r}")
    if not prompts:
        raise ValueError("prompts must contain at least one entry")

    project_id = make_project_id(name)
    pdir = project_dir(project_id, runs_root)
    pdir.mkdir(parents=True, exist_ok=False)
    (pdir / TEACHERS_DIR).mkdir()
    (pdir / STUDENTS_DIR).mkdir()
    (pdir / STUDENT_RUNS_DIR).mkdir()

    project = Project(
        id=project_id,
        name=name,
        task=task,
        prompts=list(prompts),
        created_at=_now_iso(),
    )
    write_project(pdir, project)
    log.info("created project %s", pdir)
    return project


def project_summary_counts(project_id: str, runs_root: Path = RUNS_DIR) -> dict[str, int]:
    """Compute counters used by the project picker:
        n_running, n_teacher_datasets, n_human_reviewed_datasets, n_students.

    Walks the project's teachers/ + students/ once. Cheap relative to the
    Learn / Optimize work that produced the runs.
    """
    n_running = 0
    n_teacher = 0
    n_reviewed = 0
    n_students = 0

    tdir = teachers_dir(project_id, runs_root)
    if tdir.exists():
        for p in tdir.iterdir():
            if not p.is_dir() or not (p / MANIFEST_NAME).exists():
                continue
            try:
                m = read_manifest(p)
            except Exception:
                continue
            if m.status in _STALE_STATES:
                n_running += 1
            if m.status == "completed":
                n_teacher += 1
                # Phase 3: strict bar — only fully-reviewed runs count. A
                # run is "approved" iff every processed frame has an explicit
                # per-frame state entry in frame_states.json (in_progress
                # and unreviewed don't count toward the project total).
                if derive_review_status(p) == "approved":
                    n_reviewed += 1

    sdir = students_dir(project_id, runs_root)
    if sdir.exists():
        for p in sdir.iterdir():
            if not p.is_dir() or not (p / MANIFEST_NAME).exists():
                continue
            try:
                m = read_student_manifest(p)
            except Exception:
                continue
            if m.status in _STALE_STATES:
                n_running += 1
            if m.status == "completed":
                n_students += 1

    return {
        "n_running": n_running,
        "n_teacher_datasets": n_teacher,
        "n_human_reviewed_datasets": n_reviewed,
        "n_students": n_students,
    }


def list_projects(runs_root: Path = RUNS_DIR) -> list[Project]:
    root = projects_root(runs_root)
    if not root.exists():
        return []
    out: list[Project] = []
    for p in root.iterdir():
        if not p.is_dir() or not (p / PROJECT_FILE).exists():
            continue
        try:
            out.append(read_project(p))
        except Exception as e:  # pragma: no cover — corrupt project.json
            log.warning("skipping project %s: %s", p, e)
    out.sort(key=lambda p: p.created_at, reverse=True)
    return out


def rename_project(project_id: str, name: str, runs_root: Path = RUNS_DIR) -> Project:
    pdir = project_dir(project_id, runs_root)
    if not (pdir / PROJECT_FILE).exists():
        raise FileNotFoundError(f"no such project: {project_id}")
    p = read_project(pdir)
    p.name = name
    write_project(pdir, p)
    return p


def delete_project(project_id: str, runs_root: Path = RUNS_DIR) -> None:
    import shutil

    pdir = project_dir(project_id, runs_root)
    if not pdir.exists():
        return
    shutil.rmtree(pdir)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    # ISO 8601 UTC timestamp stamped by `set_frame_state` the moment the
    # last unreviewed frame flips, and cleared by `unset_frame_state` if
    # the run drops back below 100% coverage. Phase 3 derives
    # `review_status` from `frame_states.json` directly; this field is
    # surfaced to the GUI as "approved at <timestamp>" but is not the
    # source of truth for the status itself. Default None so legacy
    # manifests load unchanged.
    approved_at: Optional[str] = None
    # Phase 2: post-hoc score filter applied to per_frame.jsonl at read time.
    # The detector persists everything ≥ SCORE_FLOOR (0.05); this is the
    # cutoff the GUI shows by default and what `compute_stats_at_threshold`
    # uses when no `?threshold=` override is passed.
    display_threshold: float = 0.30

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
    # Optional epoch-pace fields. Populated by the Optimize trainer's
    # on_train_epoch_end callback — None for Teacher runs and during
    # the prep / eval / timing phases of a training run. Lets the GUI
    # render a determinate progress bar with ETA instead of an
    # indeterminate spinner once the first epoch lands.
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    epoch_seconds_avg: Optional[float] = None

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
    # produced this Student's training set?" without grepping logs. Phase 2
    # renamed `t_high` → `export_threshold`; default tracks
    # `RunManifest.display_threshold`.
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False
    # Trainer architecture (Phase 1.3). Defaulted to "yolov8n" so old
    # manifests without this key load as the only architecture that
    # existed before the dispatcher. Future runs persist whatever the
    # GUI sent — `yolov8s`, `rtdetr-l`, etc.
    architecture: str = "yolov8n"

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
    # {"teacher_id": str, "positive": int, "uncertain": int, "true_negative": int,
    #  "n_frames_curated": int, "n_frames_confirmed_empty": int,
    #  "n_frames_marked_missed": int, "n_frames_unreviewed_used": int}
    per_teacher_buckets: list[dict[str, Any]] = field(default_factory=list)
    # Thresholds the trainer actually used. Stamped on the stats so the
    # detail card can render "frame buckets at export_threshold=0.30,
    # t_low=0.15" without re-reading the manifest.
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False
    # ---- Phase 3 review-source counters --------------------------------
    # All defaulted to 0 so old stats.json files (pre-Phase-3) keep
    # loading. See server/schemas.py::StudentStatsModel for semantics.
    n_frames_curated: int = 0
    n_frames_confirmed_empty: int = 0
    n_frames_marked_missed: int = 0
    n_frames_unreviewed_used: int = 0
    # ---- Phase 2.2 comparability fields ---------------------------------
    # Stamped so the Phase 3 compare view can flag mismatches across
    # students. All defaulted so old stats.json files (without these
    # keys) keep loading without a migration script.
    #   • `imgsz`       — image size the trainer trained / evaluated at.
    #     Different sizes invalidate latency comparisons.
    #   • `device`      — what the trainer actually ran on
    #     ("cuda" | "mps" | "cpu" | ""). Empty string for legacy runs
    #     where it wasn't recorded.
    #   • `inference_warmup_discarded` — whether `time_inference` dropped
    #     a warmup pass before sampling. Recorded as a fact so future
    #     trainers that *don't* discard a warmup are visibly different
    #     in compare-view tooltips. YoloTrainer always discards.
    imgsz: int = 640
    device: str = ""
    inference_warmup_discarded: bool = True

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


def run_dir(project_id: str, run_id: str, runs_root: Path = RUNS_DIR) -> Path:
    """Path to a Teacher run directory under its project."""
    return teachers_dir(project_id, runs_root) / run_id


def student_dir(project_id: str, student_id: str, runs_root: Path = RUNS_DIR) -> Path:
    """Path to a Student run directory under its project."""
    return students_dir(project_id, runs_root) / student_id


# ---- Create / write --------------------------------------------------------


def create_run(
    project_id: str,
    task: str,
    prompt: str,
    video_path: str,
    models: dict[str, str],
    runs_root: Path = RUNS_DIR,
) -> tuple[Path, RunManifest]:
    """Allocate a fresh run directory under the given project and write an
    initial 'running' manifest.

    `task` and `prompt` are typically supplied by the caller from the
    project's locked metadata. Validation here is defensive — the project
    is the source of truth.

    Returns (run_dir_path, manifest). The caller should fill in stats and
    flip status to 'completed' (or 'failed') when done — see
    `mark_completed` / `mark_failed`.
    """
    if task not in {"detection", "segmentation"}:
        raise ValueError(f"unknown task: {task!r}")
    pdir = project_dir(project_id, runs_root)
    if not (pdir / PROJECT_FILE).exists():
        raise FileNotFoundError(f"no such project: {project_id}")

    run_id = make_run_id(prompt)
    rdir = run_dir(project_id, run_id, runs_root)
    rdir.mkdir(parents=True, exist_ok=False)
    (rdir / LABELS_DIR).mkdir()

    manifest = RunManifest(
        id=run_id,
        task=task,
        prompt=prompt,
        video_path=video_path,
        started_at=_now_iso(),
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
    manifest.ended_at = _now_iso()
    write_manifest(rdir, manifest)
    if stats is not None:
        write_stats(rdir, stats)
    return manifest


def mark_failed(rdir: Path, error: str) -> RunManifest:
    manifest = read_manifest(rdir)
    manifest.status = "failed"
    manifest.error = error
    manifest.ended_at = _now_iso()
    write_manifest(rdir, manifest)
    return manifest


def set_display_threshold(rdir: Path, threshold: float) -> RunManifest:
    """Persist a new `display_threshold` on the run's manifest.

    Validation is at the API layer (Pydantic clamps to [0, 1]); this helper
    trusts its input so the in-process callers don't have to re-validate.
    """
    manifest = read_manifest(rdir)
    manifest.display_threshold = float(threshold)
    write_manifest(rdir, manifest)
    return manifest


def _count_processed_frames(rdir: Path) -> int:
    """How many frames did the run actually produce labels for.

    Used by `derive_review_status` to decide whether the user has covered
    every frame with an explicit state. Source order:

      1. `progress.json::frames_processed` if present (cheapest).
      2. `stats.json::frames_processed` if present.
      3. Fallback: count non-blank lines in `per_frame.jsonl`.

    Returns 0 when none of the above are available — `derive_review_status`
    treats 0 as "no frames yet, nothing to review".
    """
    progress_path = rdir / PROGRESS_NAME
    if progress_path.exists():
        try:
            raw = json.loads(progress_path.read_text())
            n = int(raw.get("current_frame", 0))
            if n > 0:
                return n
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    stats_path = rdir / STATS_NAME
    if stats_path.exists():
        try:
            raw = json.loads(stats_path.read_text())
            n = int(raw.get("frames_processed", 0))
            if n > 0:
                return n
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    pf_path = rdir / LABELS_DIR / PER_FRAME_NAME
    if pf_path.exists():
        n = 0
        with pf_path.open("r") as fp:
            for line in fp:
                if line.strip():
                    n += 1
        return n

    return 0


def derive_review_status(rdir: Path) -> str:
    """Three-state human-review summary computed from disk (Phase 3).

    Source of truth is `frame_states.json` together with the run's
    processed-frame count:

      • "approved"    — every processed frame has a state entry.
      • "in_progress" — at least one frame has a state, but not all.
      • "unreviewed"  — no states set (or no frames at all).

    `manifest.approved_at` is a *derived* timestamp stamped by
    `set_frame_state` when coverage flips to 100% and cleared by
    `unset_frame_state` when it drops below — it's not consulted here, the
    file content is.
    """
    states = read_frame_states(rdir)
    if not states:
        return "unreviewed"
    n_frames = _count_processed_frames(rdir)
    if n_frames > 0 and len(states) >= n_frames:
        return "approved"
    return "in_progress"


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
    """At server startup, sweep all projects' Teacher + Student runs and
    flip any in 'running' / 'queued' status to 'failed'.

    Rationale: Learn / Optimize workers run as *daemon threads* on the
    server process, and the Teacher queue is purely in-memory. uvicorn
    --reload (or any process restart) kills both daemon threads and the
    queue contents instantly, but the manifest on disk still says
    'running' or 'queued'. The UI polls and shows a forever-spinning
    state with no progress updates because there's no live worker.

    Calling this on startup is correct because at that exact moment, by
    definition, no in-flight worker can have survived from before the
    restart — so every 'running' / 'queued' status on disk is stale.

    Returns the number of runs that were patched.
    """
    root = projects_root(runs_root)
    if not root.exists():
        return 0
    n = 0
    for pdir in root.iterdir():
        if not pdir.is_dir() or not (pdir / PROJECT_FILE).exists():
            continue
        tdir = pdir / TEACHERS_DIR
        if tdir.exists():
            for p in tdir.iterdir():
                if not p.is_dir() or not (p / MANIFEST_NAME).exists():
                    continue
                try:
                    m = read_manifest(p)
                    if m.status in _STALE_STATES:
                        mark_failed(p, _STALE_INTERRUPT_MSG)
                        n += 1
                except Exception as e:  # pragma: no cover
                    log.warning("could not patch teacher %s: %s", p, e)
        sdir = pdir / STUDENTS_DIR
        if sdir.exists():
            for p in sdir.iterdir():
                if not p.is_dir() or not (p / MANIFEST_NAME).exists():
                    continue
                try:
                    m = read_student_manifest(p)
                    if m.status in _STALE_STATES:
                        mark_student_failed(p, _STALE_INTERRUPT_MSG)
                        n += 1
                except Exception as e:  # pragma: no cover
                    log.warning("could not patch student %s: %s", p, e)
        # Phase 5: student_runs/<sid>/<rid>/manifest.json — same shape, just
        # one more level of nesting because runs partition by student id.
        srdir = pdir / STUDENT_RUNS_DIR
        if srdir.exists():
            for sub in srdir.iterdir():
                if not sub.is_dir():
                    continue
                for p in sub.iterdir():
                    if not p.is_dir() or not (p / MANIFEST_NAME).exists():
                        continue
                    try:
                        sm = read_student_run_manifest(p)
                        if sm.status in _STALE_STATES:
                            mark_student_run_failed(p, _STALE_INTERRUPT_MSG)
                            n += 1
                    except Exception as e:  # pragma: no cover
                        log.warning("could not patch student_run %s: %s", p, e)
    if n:
        log.info("marked %d stale running/queued runs as failed at startup", n)
    return n


def list_runs(project_id: str, runs_root: Path = RUNS_DIR) -> list[RunManifest]:
    """Teacher-run manifests under one project, newest first."""
    tdir = teachers_dir(project_id, runs_root)
    if not tdir.exists():
        return []
    out: list[RunManifest] = []
    for p in tdir.iterdir():
        if not p.is_dir() or not (p / MANIFEST_NAME).exists():
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


def list_students(project_id: str, runs_root: Path = RUNS_DIR) -> list[StudentManifest]:
    """Student manifests under one project, newest first."""
    sdir = students_dir(project_id, runs_root)
    if not sdir.exists():
        return []
    out: list[StudentManifest] = []
    for p in sdir.iterdir():
        if not p.is_dir() or not (p / MANIFEST_NAME).exists():
            continue
        try:
            out.append(read_student_manifest(p))
        except Exception as e:  # pragma: no cover
            log.warning("skipping student %s: %s", p, e)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out


def create_student(
    *,
    project_id: str,
    train_teacher_ids: list[str],
    eval_teacher_ids: list[str],
    task: str,
    prompt: str,
    models: dict[str, str],
    export_threshold: float = 0.30,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    architecture: str = "yolov8n",
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

    The confidence-band thresholds (`export_threshold`, `t_low`,
    `treat_empty_as_negative`) are recorded on the manifest at creation
    time so the run is reproducible from manifest.json alone.
    """
    if not train_teacher_ids:
        raise ValueError("train_teacher_ids must contain at least one teacher")
    pdir = project_dir(project_id, runs_root)
    if not (pdir / PROJECT_FILE).exists():
        raise FileNotFoundError(f"no such project: {project_id}")

    student_id = make_student_id(prompt)
    rdir = student_dir(project_id, student_id, runs_root)
    rdir.mkdir(parents=True, exist_ok=False)

    manifest = StudentManifest(
        id=student_id,
        train_teacher_ids=list(train_teacher_ids),
        eval_teacher_ids=list(eval_teacher_ids),
        task=task,
        prompt=prompt,
        started_at=_now_iso(),
        models=models,
        export_threshold=export_threshold,
        t_low=t_low,
        treat_empty_as_negative=treat_empty_as_negative,
        architecture=architecture,
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
    manifest.ended_at = _now_iso()
    write_student_manifest(rdir, manifest)
    if stats is not None:
        write_student_stats(rdir, stats)
    return manifest


def mark_student_failed(rdir: Path, error: str) -> StudentManifest:
    manifest = read_student_manifest(rdir)
    manifest.status = "failed"
    manifest.error = error
    manifest.ended_at = _now_iso()
    write_student_manifest(rdir, manifest)
    return manifest


# ---- Student-run (Phase 5) ------------------------------------------------


@dataclass
class StudentRunManifest:
    """A trained Student running inference against an arbitrary input.

    Lives at `runs/projects/<pid>/student_runs/<sid>/<rid>/manifest.json`.
    `input_kind` discriminates the union: `"video"` means `input_ref` is
    a path under `data/`, `"teacher_dataset"` means `input_ref` is a
    Teacher run id within the same project — the Teacher's source video
    becomes the input and its COCO labels become ground truth for mAP.
    """

    id: str  # "studentrun_<ts>_<slug>"
    student_id: str
    project_id: str
    input_kind: str  # "video" | "teacher_dataset"
    input_ref: str
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"  # "queued" | "running" | "completed" | "failed"
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class StudentRunStats:
    """Inference summary for one Student run.

    `n_frames` is what was processed end-to-end. Latency numbers are
    measured per-frame around the trainer's `predict` call so they're
    comparable to the timing the Student's own training stats reported.
    `map50` / `map50_95` are populated only when the input was a teacher
    dataset (we have ground truth); video-input runs leave them at 0.0
    and the GUI hides those rows.
    """

    n_frames: int = 0
    n_detections: int = 0
    avg_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    map50: Optional[float] = None
    map50_95: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def make_student_run_id(now: Optional[datetime] = None) -> str:
    """Sortable id for a student-run: studentrun_<utc>_<slug>.

    The slug is just `run` — student-run dirs are partitioned under the
    student id already, and the timestamp suffices for ordering. Adding
    a per-input slug would invite confusion when two runs against the
    same input dir collide on the same second.
    """
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    return f"studentrun_{ts}_run"


def student_run_parent_dir(
    project_id: str, student_id: str, runs_root: Path = RUNS_DIR
) -> Path:
    """All runs for one Student live here."""
    return student_runs_root(project_id, runs_root) / student_id


def student_run_dir(
    project_id: str,
    student_id: str,
    run_id: str,
    runs_root: Path = RUNS_DIR,
) -> Path:
    """Path to one student-run dir."""
    return student_run_parent_dir(project_id, student_id, runs_root) / run_id


def write_student_run_manifest(rdir: Path, manifest: StudentRunManifest) -> None:
    (rdir / MANIFEST_NAME).write_text(manifest.to_json())


def read_student_run_manifest(rdir: Path) -> StudentRunManifest:
    raw = json.loads((rdir / MANIFEST_NAME).read_text())
    known = {f for f in StudentRunManifest.__dataclass_fields__}
    return StudentRunManifest(**{k: v for k, v in raw.items() if k in known})


def write_student_run_stats(rdir: Path, stats: StudentRunStats) -> None:
    (rdir / STATS_NAME).write_text(stats.to_json())


def read_student_run_stats(rdir: Path) -> Optional[StudentRunStats]:
    p = rdir / STATS_NAME
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    known = {f for f in StudentRunStats.__dataclass_fields__}
    return StudentRunStats(**{k: v for k, v in raw.items() if k in known})


def create_student_run(
    *,
    project_id: str,
    student_id: str,
    input_kind: str,
    input_ref: str,
    runs_root: Path = RUNS_DIR,
) -> tuple[Path, StudentRunManifest]:
    """Allocate a fresh student-run dir + initial 'running' manifest.

    The PREDICTIONS_DIR is created up-front so writers can append without
    worrying about whether their parent exists. We do NOT create the
    `overlay.mp4` placeholder — the OverlayMp4Writer is lazy about that.
    """
    if input_kind not in {"video", "teacher_dataset"}:
        raise ValueError(f"unknown input_kind: {input_kind!r}")

    sdir = student_dir(project_id, student_id, runs_root)
    if not (sdir / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"no such student: {student_id}")

    run_id = make_student_run_id()
    parent = student_run_parent_dir(project_id, student_id, runs_root)
    parent.mkdir(parents=True, exist_ok=True)
    rdir = parent / run_id
    rdir.mkdir(parents=True, exist_ok=False)
    (rdir / PREDICTIONS_DIR).mkdir()

    manifest = StudentRunManifest(
        id=run_id,
        student_id=student_id,
        project_id=project_id,
        input_kind=input_kind,
        input_ref=input_ref,
        started_at=_now_iso(),
    )
    write_student_run_manifest(rdir, manifest)
    return rdir, manifest


def mark_student_run_completed(
    rdir: Path, stats: Optional[StudentRunStats] = None
) -> StudentRunManifest:
    manifest = read_student_run_manifest(rdir)
    manifest.status = "completed"
    manifest.ended_at = _now_iso()
    write_student_run_manifest(rdir, manifest)
    if stats is not None:
        write_student_run_stats(rdir, stats)
    return manifest


def mark_student_run_failed(rdir: Path, error: str) -> StudentRunManifest:
    manifest = read_student_run_manifest(rdir)
    manifest.status = "failed"
    manifest.error = error
    manifest.ended_at = _now_iso()
    write_student_run_manifest(rdir, manifest)
    return manifest


def list_student_runs(
    project_id: str, student_id: str, runs_root: Path = RUNS_DIR
) -> list[StudentRunManifest]:
    """Manifests for one Student's runs, newest first.

    Returns an empty list if the student has no runs (or the dir doesn't
    exist) rather than raising — the GUI calls this on every Student
    detail open and a missing dir is the common case.
    """
    parent = student_run_parent_dir(project_id, student_id, runs_root)
    if not parent.exists():
        return []
    out: list[StudentRunManifest] = []
    for p in parent.iterdir():
        if not p.is_dir() or not (p / MANIFEST_NAME).exists():
            continue
        try:
            out.append(read_student_run_manifest(p))
        except Exception as e:  # pragma: no cover — corrupt manifest
            log.warning("could not read student-run manifest at %s: %s", p, e)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out


def delete_student_run(rdir: Path) -> None:
    """Recursively remove a student-run directory."""
    import shutil

    if not rdir.exists():
        return
    shutil.rmtree(rdir)


# ---- Per-frame labels (jsonl) ---------------------------------------------


class PerFrameWriter:
    """Append-only writer for `<labels|predictions>/per_frame.jsonl`.

    Each line is a small JSON object: {"frame_idx": int, "detections": [...],
    "masks": [...]}. The Inspector fetches this file once and slices
    client-side — small files, no decode-on-seek.

    `subdir` lets Phase 5 student-runs reuse this writer against
    `predictions/per_frame.jsonl` without forking the body. The default
    `LABELS_DIR` keeps Teacher behavior unchanged.
    """

    def __init__(self, rdir: Path, subdir: str = LABELS_DIR):
        self.path = rdir / subdir / PER_FRAME_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
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


def compute_stats_at_threshold(rdir: Path, threshold: float) -> Optional[RunStats]:
    """Re-derive `RunStats` from per_frame.jsonl with `score >= threshold`.

    The on-disk `stats.json` carries the *base* numbers — everything at or
    above the detector's `SCORE_FLOOR`. Phase 2 added a post-hoc filter
    knob; the API filters detections at request time so the user can drag
    the inspector slider without rerunning the detector.

    Returns None when the run has no `stats.json` yet (still in progress)
    or no `per_frame.jsonl` to filter from. Timing fields are pulled from
    the base stats unchanged — the detector's per-frame ms doesn't shift
    with a confidence cutoff.
    """
    base = read_stats(rdir)
    if base is None:
        return None
    pf_path = rdir / LABELS_DIR / PER_FRAME_NAME
    if not pf_path.exists():
        return base

    def _filtered_records() -> Iterable[dict]:
        for rec in read_per_frame(rdir):
            dets = rec.get("detections", []) or []
            kept = [d for d in dets if float(d.get("score", 0.0)) >= threshold]
            new_rec = dict(rec)
            new_rec["detections"] = kept
            yield new_rec

    breakdown = compute_detection_breakdown(
        _filtered_records(), frames_processed=base.frames_processed
    )

    n_detections_total = sum(
        cls["n_detections"] for cls in breakdown["detections_per_class"].values()
    )
    frames_with_detections = sum(
        n for k, n in breakdown["per_frame_count_histogram"].items() if int(k) > 0
    )

    return RunStats(
        frames_processed=base.frames_processed,
        frames_with_detections=frames_with_detections,
        total_ms=base.total_ms,
        avg_ms_per_frame=base.avg_ms_per_frame,
        p50_ms_per_frame=base.p50_ms_per_frame,
        p95_ms_per_frame=base.p95_ms_per_frame,
        n_detections_total=n_detections_total,
        **breakdown,
    )


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


# ---- Per-frame review state (human curation) ------------------------------

_VALID_FRAME_STATES = ("curated", "confirmed_empty", "marked_missed")


def _frame_states_path(rdir: Path) -> Path:
    return rdir / FRAME_STATES_NAME


def read_frame_states(rdir: Path) -> dict[int, dict]:
    """Read per-frame review state.

    On disk: `{"<frame_idx>": {"state": "...", "rejected_dets": [...]}, …}`
    (str keys for JSON; `rejected_dets` omitted unless the entry has any).
    In memory: int keys → entry dict. Missing file or malformed JSON →
    empty dict.

    Filters out entries with unknown states (defensive — never let a bad
    on-disk value propagate to distill or the GUI).
    """
    p = _frame_states_path(rdir)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        log.warning("frame_states.json was malformed at %s; treating as empty", p)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            fi = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        state = v.get("state")
        if state not in _VALID_FRAME_STATES:
            continue
        entry: dict = {"state": state}
        rejected = v.get("rejected_dets")
        if state == "curated" and isinstance(rejected, list):
            try:
                entry["rejected_dets"] = sorted({int(x) for x in rejected})
            except (TypeError, ValueError):
                entry["rejected_dets"] = []
        else:
            entry["rejected_dets"] = []
        out[fi] = entry
    return out


def write_frame_states(rdir: Path, states: dict[int, dict]) -> None:
    """Atomic-ish write of frame_states.json.

    Empty `rejected_dets` lists are stripped from the on-disk shape so the
    file stays compact (`{"5": {"state": "confirmed_empty"}}` vs the
    longer form). The reader normalizes both shapes.
    """
    serializable: dict[str, dict] = {}
    for fi, entry in states.items():
        state = entry.get("state")
        if state not in _VALID_FRAME_STATES:
            continue
        out_entry: dict = {"state": state}
        rejected = entry.get("rejected_dets") or []
        if state == "curated" and rejected:
            out_entry["rejected_dets"] = sorted({int(x) for x in rejected})
        serializable[str(int(fi))] = out_entry
    tmp = rdir / (FRAME_STATES_NAME + ".tmp")
    tmp.write_text(json.dumps(serializable, indent=2))
    tmp.replace(_frame_states_path(rdir))


def _frame_det_count(rdir: Path, frame_idx: int) -> Optional[int]:
    """Look up how many detections a frame has, for rejected_dets bounds-checking.

    Returns None if the frame isn't found in `per_frame.jsonl` (caller treats
    that as "frame doesn't exist" → 404 / ValueError). Streams the file —
    cheap on small runs, fine on large ones since we short-circuit.
    """
    pf_path = rdir / LABELS_DIR / PER_FRAME_NAME
    if not pf_path.exists():
        return None
    with pf_path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(rec.get("frame_idx", -1)) == frame_idx:
                return len(rec.get("detections") or [])
    return None


def _stamp_approved_at_if_complete(rdir: Path, states: dict[int, dict]) -> None:
    """When the user finishes the last frame, write the timestamp.

    Mirrors the inverse op in `unset_frame_state`: as soon as coverage
    drops below 100%, `approved_at` clears. Together they keep the
    timestamp in sync with `derive_review_status() == "approved"` without
    requiring callers to recompute it.
    """
    n_frames = _count_processed_frames(rdir)
    fully_covered = n_frames > 0 and len(states) >= n_frames
    try:
        manifest = read_manifest(rdir)
    except Exception:
        return
    if fully_covered and manifest.approved_at is None:
        manifest.approved_at = _now_iso()
        write_manifest(rdir, manifest)
    elif not fully_covered and manifest.approved_at is not None:
        manifest.approved_at = None
        write_manifest(rdir, manifest)


def set_frame_state(
    rdir: Path,
    frame_idx: int,
    state: str,
    rejected_dets: Optional[list[int]] = None,
) -> dict:
    """Set (or update) the review entry for one frame.

    Validates:
      • `state` is one of `_VALID_FRAME_STATES`.
      • `rejected_dets` is only allowed when `state == "curated"`.
      • Each rejected det index is in range for that frame's per_frame.jsonl
        entry; raising prevents stale UI state from persisting bogus indices.

    On success, writes frame_states.json and re-stamps `approved_at` on the
    manifest if the run just hit full coverage. Returns the persisted entry.

    Raises `ValueError` for any validation failure — the API layer maps it
    to 400.
    """
    if state not in _VALID_FRAME_STATES:
        raise ValueError(f"invalid state {state!r}; expected one of {_VALID_FRAME_STATES}")
    if state != "curated" and rejected_dets:
        raise ValueError(
            f"rejected_dets only allowed with state='curated', not {state!r}"
        )

    cleaned: list[int] = []
    if state == "curated" and rejected_dets:
        det_count = _frame_det_count(rdir, frame_idx)
        if det_count is None:
            raise ValueError(f"frame {frame_idx} not found in per_frame.jsonl")
        for x in rejected_dets:
            try:
                idx = int(x)
            except (TypeError, ValueError) as e:
                raise ValueError(f"invalid det_idx {x!r}: {e}") from e
            if not (0 <= idx < det_count):
                raise ValueError(
                    f"det_idx {idx} out of range for frame {frame_idx} "
                    f"(has {det_count} detections)"
                )
            cleaned.append(idx)
        cleaned = sorted(set(cleaned))

    entry: dict = {"state": state, "rejected_dets": cleaned}
    states = read_frame_states(rdir)
    states[int(frame_idx)] = entry
    write_frame_states(rdir, states)
    _stamp_approved_at_if_complete(rdir, states)
    return entry


def unset_frame_state(rdir: Path, frame_idx: int) -> None:
    """Remove the review entry for one frame.

    No-op if the frame had no entry. Clears `approved_at` if removing the
    entry takes the run below full coverage.
    """
    states = read_frame_states(rdir)
    if int(frame_idx) not in states:
        return
    states.pop(int(frame_idx), None)
    write_frame_states(rdir, states)
    _stamp_approved_at_if_complete(rdir, states)


# ---- Detections ordering + crops (Phase 4) -------------------------------


@dataclass
class DetectionRow:
    """One detection in a run, flattened across frames for crop-flip review.

    `accepted` is derived from `frame_states.json`: a detection is rejected
    iff its frame's state is `curated` and its index is in `rejected_dets`.
    Frames with state `confirmed_empty` or `marked_missed` keep the
    detection as `accepted=true` here — those whole-frame verdicts are
    expressed elsewhere (the inspector pills) and aren't a per-detection
    accept/reject signal.
    """

    frame_idx: int
    det_idx: int
    class_name: str
    score: float
    accepted: bool


def iter_detection_rows(rdir: Path) -> list[DetectionRow]:
    """Flatten a run's per-frame detections into one ordered list.

    Reads `per_frame.jsonl` once and `frame_states.json` once, then walks
    each frame's detections in original `det_idx` order. Returned rows are
    in the natural file order (frame asc, det_idx asc); callers sort.
    """
    states = read_frame_states(rdir)
    rows: list[DetectionRow] = []
    for rec in read_per_frame(rdir):
        try:
            fi = int(rec.get("frame_idx", -1))
        except (TypeError, ValueError):
            continue
        if fi < 0:
            continue
        entry = states.get(fi)
        rejected = (
            set(entry.get("rejected_dets", []))
            if entry and entry.get("state") == "curated"
            else set()
        )
        for di, d in enumerate(rec.get("detections") or []):
            try:
                score = float(d.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            cls = d.get("class_name") or f"class_{d.get('class_id', '?')}"
            rows.append(
                DetectionRow(
                    frame_idx=fi,
                    det_idx=di,
                    class_name=str(cls),
                    score=score,
                    accepted=di not in rejected,
                )
            )
    return rows


def detection_bbox_xyxy(
    rdir: Path, frame_idx: int, det_idx: int
) -> Optional[tuple[float, float, float, float]]:
    """Return the bbox (x1, y1, x2, y2) for a specific detection, or None
    if the frame or det index is missing.

    Streams the per-frame file to avoid loading the whole thing for one
    crop. The two-level lookup matches what the crop endpoint does.
    """
    pf_path = rdir / LABELS_DIR / PER_FRAME_NAME
    if not pf_path.exists():
        return None
    with pf_path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(rec.get("frame_idx", -1)) != frame_idx:
                continue
            dets = rec.get("detections") or []
            if not (0 <= det_idx < len(dets)):
                return None
            bbox = dets[det_idx].get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return None
            try:
                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (TypeError, ValueError):
                return None
    return None


def crops_dir(rdir: Path) -> Path:
    """Path to the on-disk crop cache for this run."""
    return rdir / CROPS_DIR


def detection_crop_path(rdir: Path, frame_idx: int, det_idx: int, pad: int) -> Path:
    """Where a single (frame, det, pad) crop is cached.

    Pad goes in the filename so changing the slider invalidates the cache
    naturally (different file). Padding is bucketed to a positive integer
    by the endpoint.
    """
    return crops_dir(rdir) / f"{frame_idx}_{det_idx}_p{pad}.jpg"


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
