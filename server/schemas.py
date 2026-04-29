"""Pydantic schemas for the FastAPI surface.

These describe the JSON contract between the GUI and the runner. Keep them
narrow — the graph spec is simple, and adding fields piecemeal is fine.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ---- Projects --------------------------------------------------------------


class ProjectCreateRequest(BaseModel):
    """Body of `POST /projects`.

    `task` and `prompts` are LOCKED at creation. Once a project owns runs,
    mutating either would invalidate cross-run comparability — so the
    `PATCH /projects/{pid}` endpoint refuses to touch them.
    """

    name: str = Field(min_length=1, max_length=80)
    task: Literal["detection", "segmentation"]
    prompts: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _normalize_prompts(self) -> "ProjectCreateRequest":
        # Strip + dedupe while preserving order. Reject empty / whitespace
        # entries so "[ '' ]" doesn't sneak past min_length=1.
        seen: set[str] = set()
        cleaned: list[str] = []
        for p in self.prompts:
            s = p.strip()
            if not s:
                raise ValueError("prompt entries cannot be empty")
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        self.prompts = cleaned
        return self


class ProjectPatchRequest(BaseModel):
    """Body of `PATCH /projects/{pid}`. Only `name` is mutable.

    `extra="forbid"` is what enforces the lock — sending `task` or
    `prompts` here yields a 422 with a clear message, instead of being
    silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)


class ProjectModel(BaseModel):
    id: str
    name: str
    task: Literal["detection", "segmentation"]
    prompts: list[str]
    created_at: str  # ISO 8601 UTC


class ProjectSummaryModel(BaseModel):
    """Lightweight row for the project picker.

    The four counters are computed by walking the project directory once
    per `GET /projects` call. With dozens of teachers per project that's
    cheap; if we ever blow past a few hundred runs per project we'll cache
    the counts on `project.json` and update on write.
    """

    id: str
    name: str
    task: Literal["detection", "segmentation"]
    prompts: list[str]
    created_at: str
    n_running: int = 0
    n_teacher_datasets: int = 0
    n_human_reviewed_datasets: int = 0
    n_students: int = 0


class ProjectsResponse(BaseModel):
    projects: list[ProjectSummaryModel] = Field(default_factory=list)


# ---- Graph editor (legacy) -------------------------------------------------


class NodeSpecModel(BaseModel):
    id: str
    kind: str  # "input" | "detect" | "segment" | "reid" | "track" | "stats" | "output"
    impl: str
    params: dict[str, Any] = Field(default_factory=dict)


class GraphSpecModel(BaseModel):
    nodes: list[NodeSpecModel]
    edges: list[tuple[str, str]]


class RunRequest(BaseModel):
    graph: GraphSpecModel
    # Input video path is carried in the Input node's params for now; this
    # field exists for future overrides (e.g. uploaded blob path).
    video_path: str | None = None


class BlockKindInfo(BaseModel):
    kind: str
    impls: list[str]


class BlocksResponse(BaseModel):
    blocks: list[BlockKindInfo]


# ---- Learn / Optimize / Inspector ------------------------------------------


class LearnRequest(BaseModel):
    """Inputs from the Learn wizard.

    Project-scoped (Phase 1): `task` and `prompts` come from the parent
    project, NOT from this request — so the request only carries the
    per-run knobs. Posted to `POST /projects/{pid}/learn`.
    """

    video_path: str
    # Per-task model overrides — optional; defaults are picked by the backend
    # so the user can stay in pure "fill three fields and go" mode.
    detect_impl: Optional[str] = None
    segment_impl: Optional[str] = None
    reid_impl: Optional[str] = None
    track_impl: Optional[str] = None
    max_frames: Optional[int] = None  # cap for fast iteration
    # Phase 2: detector confidence is no longer a run-time knob. The
    # detector runs at a fixed low floor and the GUI filters post-hoc via
    # the manifest's `display_threshold`.


class RunManifestModel(BaseModel):
    """Mirror of pipeline.runs.RunManifest for API output.

    Project-scoped (Phase 1): `project_id` identifies the parent project.
    `review_status` is computed by the endpoint at response time (it
    needs disk access — `has_any_rejections` — and the pydantic model
    stays a pure data carrier).
    """

    id: str
    project_id: str
    task: str
    prompt: str
    video_path: str
    started_at: str
    ended_at: Optional[str] = None
    status: str
    models: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    approved_at: Optional[str] = None
    # Phase 3: tightened bar — "approved" requires every processed frame to
    # carry an explicit per-frame state (curated / confirmed_empty /
    # marked_missed). "in_progress" means at least one frame has a state
    # but not all of them. Status is *derived* from `frame_states.json` +
    # `progress.json`; `approved_at` is the timestamp stamped when coverage
    # flips to 100%.
    review_status: Literal["unreviewed", "in_progress", "approved"] = "unreviewed"
    n_frames_reviewed: int = 0
    n_frames_total: int = 0
    # Phase 2: post-hoc score filter the GUI applies by default. The
    # detector persists every detection at SCORE_FLOOR=0.05; the inspector
    # slider PATCHes this field to change the canonical view without
    # rerunning the detector.
    display_threshold: float = 0.30


class RunPatchRequest(BaseModel):
    """Body of `PATCH /projects/{pid}/runs/{rid}`. Phase 2 surfaces a
    single mutable field — the post-hoc display threshold."""

    model_config = ConfigDict(extra="forbid")

    display_threshold: float = Field(ge=0.0, le=1.0)


class PerClassStatsModel(BaseModel):
    """Per-class slice of the detection breakdown. See RunStats docs."""

    n_detections: int = 0
    frames_present: int = 0
    max_in_frame: int = 0
    avg_per_frame: float = 0.0
    avg_per_present_frame: float = 0.0
    score_avg: float = 0.0
    score_p50: float = 0.0


class RunStatsModel(BaseModel):
    frames_processed: int = 0
    frames_with_detections: int = 0
    total_ms: float = 0.0
    avg_ms_per_frame: float = 0.0
    p50_ms_per_frame: float = 0.0
    p95_ms_per_frame: float = 0.0
    n_detections_total: int = 0

    # ---- detection breakdown ---------------------------------------------
    detections_per_class: dict[str, PerClassStatsModel] = Field(default_factory=dict)
    per_frame_count_min: int = 0
    per_frame_count_p50: int = 0
    per_frame_count_p95: int = 0
    per_frame_count_max: int = 0
    per_frame_count_avg: float = 0.0
    # JSON keys are str; values are frame counts.
    per_frame_count_histogram: dict[str, int] = Field(default_factory=dict)


class RunProgressModel(BaseModel):
    stage: str = "starting"
    message: str = ""
    current_frame: int = 0
    total_frames: int = 0
    frames_with_detections: int = 0
    started_at: str = ""
    updated_at: str = ""


class RunDetail(BaseModel):
    manifest: RunManifestModel
    stats: Optional[RunStatsModel] = None
    progress: Optional[RunProgressModel] = None


class RunsResponse(BaseModel):
    runs: list[RunManifestModel]


# ---- Per-frame review state (Phase 3) -------------------------------------


FrameState = Literal["curated", "confirmed_empty", "marked_missed"]


class FrameStateEntry(BaseModel):
    """One frame's review verdict.

    `curated` — the user looked at the model's detections and approved
    what's left after `rejected_dets` are dropped.
    `confirmed_empty` — the user confirmed the frame is genuinely empty
    (forced true negative for the Student trainer).
    `marked_missed` — the user noticed the model missed something; the
    labels here can't be trusted, so distill drops the frame entirely.

    `rejected_dets` is only meaningful with `state == "curated"` and is
    enforced as empty otherwise by the endpoint.
    """

    state: FrameState
    rejected_dets: list[int] = Field(default_factory=list)


class FrameStatesResponse(BaseModel):
    """Map of frame_idx (str — JSON key) → entry.

    String keys for transport; `RunInspector` re-keys with `String(idx)`
    on the GUI side.
    """

    frame_states: dict[str, FrameStateEntry] = Field(default_factory=dict)


class DetectionRowModel(BaseModel):
    """One row of the run-level detections list (Phase 4 crop-flip review).

    `accepted` is derived: false iff the frame's state is `curated` AND the
    det index is in `rejected_dets`. Whole-frame verdicts (`confirmed_empty`,
    `marked_missed`) are not surfaced here — they aren't per-detection
    accept/reject signals.
    """

    frame_idx: int
    det_idx: int
    class_name: str
    score: float
    accepted: bool


class DetectionsResponse(BaseModel):
    """Body of `GET /projects/{pid}/runs/{rid}/detections`.

    `total` is the unfiltered count of all detections at the floor — it
    powers the "47 / 312" position-in-list indicator on the GUI side and
    stays stable across pagination.
    """

    detections: list[DetectionRowModel] = Field(default_factory=list)
    total: int = 0


class PutFrameStateRequest(BaseModel):
    """Body of `PUT /projects/{pid}/runs/{rid}/frame_states/{frame_idx}`.

    `rejected_dets` is `None` (not [] — that's an explicit empty list, ie
    "curated with no rejections") so we can distinguish "client didn't
    send the field" from "client wants to clear all rejections" if it
    matters. In practice clients always send a list with `state="curated"`.
    """

    model_config = ConfigDict(extra="forbid")

    state: FrameState
    rejected_dets: Optional[list[int]] = None


# ---- Optimize / Students ---------------------------------------------------


class OptimizeRequest(BaseModel):
    """Request to start a Student distillation run.

    `train_teacher_ids` (required, non-empty): the Teachers whose curated
    COCO labels get merged into the Student's training set.
    `eval_teacher_ids` (optional): Teachers held out for transferability
    evaluation — the Student is scored against their labels but never sees
    them at train time. May overlap with the train set (in-distribution
    sanity check); the GUI warns when it does.

    Confidence-band fields (`export_threshold`, `t_low`,
    `treat_empty_as_negative`) drive the frame-bucket filter in
    `pipeline.distill.prepare_yolo_dataset`. See `docs/student-training.md`
    Phase 0 for the full motivation; the short version is "drop frames the
    teacher was unsure about so we don't train the Student to suppress
    detections it should be making". Phase 2 renamed `t_high` to
    `export_threshold` to make the role explicit — it's the cutoff at
    which a Teacher detection becomes a Student training label.
    """

    train_teacher_ids: list[str] = Field(default_factory=list)
    eval_teacher_ids: list[str] = Field(default_factory=list)
    detect_impl: Optional[str] = None
    segment_impl: Optional[str] = None
    track_impl: Optional[str] = None
    epochs: int = 50
    # Confidence bands. `export_threshold` is the cutoff at which a teacher
    # detection becomes a student training label (and a frame becomes
    # "positive"); `t_low` floors the uncertain band. Default mirrors
    # `RunManifest.display_threshold` so a Student trained immediately
    # after a Teacher reflects what the user is looking at in the inspector.
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False
    # Trainer architecture (Phase 1.3). Default `yolov8n` reproduces the
    # only architecture the project supported before the dispatcher landed.
    # Validated at model_validator time against the live registry so
    # bogus values 422 immediately rather than spinning up a worker that
    # crashes inside `make_trainer`.
    architecture: str = "yolov8n"

    @model_validator(mode="after")
    def _t_low_le_export_threshold(self) -> "OptimizeRequest":
        if self.t_low > self.export_threshold:
            raise ValueError(
                f"t_low ({self.t_low}) must be <= export_threshold "
                f"({self.export_threshold}) — the uncertain band "
                "[t_low, export_threshold) would otherwise be empty/inverted."
            )
        return self

    @model_validator(mode="after")
    def _architecture_is_registered(self) -> "OptimizeRequest":
        """Reject unknown architectures at request time so the GUI sees
        a 422 inline, not a "Student failed" row a few seconds later.

        We import lazily so this module stays importable even when the
        registry hasn't been populated yet (e.g. test collection before
        the optimize worker has touched the package). The list of valid
        names is read live so newly-registered trainers light up without
        a schema change.
        """
        from pipeline.students import list_trainers

        available = list_trainers()
        if self.architecture not in available:
            raise ValueError(
                f"unknown architecture {self.architecture!r} — "
                f"available: {available or '<none registered>'}"
            )
        return self


class StudentManifestModel(BaseModel):
    id: str
    project_id: str
    train_teacher_ids: list[str] = Field(default_factory=list)
    eval_teacher_ids: list[str] = Field(default_factory=list)
    task: str
    prompt: str
    started_at: str
    ended_at: Optional[str] = None
    status: str
    models: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    # Confidence-band thresholds for the bucketing pass — persisted on the
    # manifest so a Student is reproducible from manifest.json alone.
    # Phase 2 renamed `t_high` → `export_threshold`; default tracks
    # `RunManifest.display_threshold`.
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False
    # Trainer architecture (Phase 1.3). Defaulted to `yolov8n` so old
    # manifest.json files without this key load as the only architecture
    # that existed before the dispatcher.
    architecture: str = "yolov8n"


class PerEvalTeacherStat(BaseModel):
    """One row of the Student's per-eval-teacher transferability table."""

    teacher_id: str
    n_images: int = 0
    n_annotations: int = 0
    map50: float = 0.0
    map50_95: float = 0.0
    # Set when the trainer skipped this teacher (missing source video, etc.).
    error: Optional[str] = None


class PerTrainTeacherBucket(BaseModel):
    """One row of the Student's per-train-teacher frame-bucket breakdown.

    Mirrors the dict shape produced by `pipeline.distill.prepare_yolo_dataset`
    so old `stats.json` files (pre-Phase-0.6) that stored this as a plain
    dict still round-trip without migration.

    Phase 3 added the four `n_frames_*` review-source counters so the
    Student detail card can show "where the training signal came from" —
    same total as positive+uncertain+true_negative, sliced by review state
    instead.
    """

    teacher_id: str
    positive: int = 0
    uncertain: int = 0
    true_negative: int = 0
    n_frames_curated: int = 0
    n_frames_confirmed_empty: int = 0
    n_frames_marked_missed: int = 0
    n_frames_unreviewed_used: int = 0


class StudentStatsModel(BaseModel):
    train_images: int = 0
    train_annotations: int = 0
    train_seconds: float = 0.0
    epochs: int = 0
    # Mean across per_eval_teacher; 0.0 if no eval teachers were configured.
    map50: float = 0.0
    map50_95: float = 0.0
    avg_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    model_size_mb: float = 0.0
    per_eval_teacher: list[PerEvalTeacherStat] = Field(default_factory=list)
    # ---- frame-bucket breakdown (Phase 0.5/0.6) --------------------------
    # All defaulted so old stats.json files without these keys still load.
    n_positive_frames: int = 0
    n_uncertain_dropped: int = 0
    n_true_negative_frames: int = 0
    per_teacher_buckets: list[PerTrainTeacherBucket] = Field(default_factory=list)
    # ---- Phase 3 review-source counters ---------------------------------
    # Same training-frame total as positive+uncertain+true_negative, but
    # sliced by what drove each frame into the trainer:
    #   curated         — frame state set, train labels = score-filtered
    #                     dets minus the user's rejected_dets list
    #   confirmed_empty — frame state set, forced true negative
    #   marked_missed   — frame state set, frame DROPPED (not in trainer)
    #   unreviewed_used — no frame state, fell through to threshold bucketing
    #                     and ended up in positive or true_negative
    n_frames_curated: int = 0
    n_frames_confirmed_empty: int = 0
    n_frames_marked_missed: int = 0
    n_frames_unreviewed_used: int = 0
    # Stamp the thresholds used by the trainer so the detail card shows
    # "buckets at export_threshold=0.30" without re-reading the manifest.
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False
    # ---- Phase 2.2 comparability fields (mirror of StudentStats) --------
    # Defaulted so old stats.json files without these keys still load
    # cleanly. The Phase 3 compare view will surface mismatches; this
    # phase only persists them.
    imgsz: int = 640
    device: str = ""
    inference_warmup_discarded: bool = True


class StudentDetail(BaseModel):
    manifest: StudentManifestModel
    stats: Optional[StudentStatsModel] = None
    progress: Optional[RunProgressModel] = None


class StudentsResponse(BaseModel):
    students: list[StudentManifestModel]


class ArchitecturesResponse(BaseModel):
    """Names of every registered student trainer (Phase 1.4).

    Drives the architecture `<select>` in the New Student form. Sorted
    so the GUI dropdown order is stable across reloads.
    """

    architectures: list[str] = Field(default_factory=list)


# ---- Student-run (Phase 5) -------------------------------------------------


class StudentRunRequest(BaseModel):
    """Body of `POST /projects/{pid}/students/{sid}/run`.

    `input_kind`:
      - `video` — `input_ref` is a path under `data/`. The Student runs
        inference on the file and persists predictions + an overlay.
      - `teacher_dataset` — `input_ref` is a Teacher run id within the
        same project. The Teacher's source video becomes the input and
        its COCO labels become ground truth for mAP scoring.
    """

    model_config = ConfigDict(extra="forbid")

    input_kind: Literal["video", "teacher_dataset"]
    input_ref: str = Field(min_length=1)


class StudentRunModel(BaseModel):
    """Manifest projection of a single student-run."""

    id: str
    student_id: str
    project_id: str
    input_kind: str
    input_ref: str
    started_at: str
    ended_at: Optional[str] = None
    status: str
    error: Optional[str] = None


class StudentRunStatsModel(BaseModel):
    """Inference summary for a completed student-run.

    `map50` / `map50_95` are populated only for `teacher_dataset` runs;
    video-input runs have no ground truth and these fields stay `None`.
    """

    n_frames: int = 0
    n_detections: int = 0
    avg_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    map50: Optional[float] = None
    map50_95: Optional[float] = None


class StudentRunDetail(BaseModel):
    """`GET /projects/{pid}/students/{sid}/runs/{rid}` response shape."""

    manifest: StudentRunModel
    stats: Optional[StudentRunStatsModel] = None
    progress: Optional[RunProgressModel] = None


class StudentRunsResponse(BaseModel):
    runs: list[StudentRunModel] = Field(default_factory=list)


# ---- Preview-buckets endpoint ---------------------------------------------


class PreviewBucketsRequest(BaseModel):
    """Live-preview request for the New Student form (Phase 0.4).

    Same threshold semantics as `OptimizeRequest` — running the same
    classification pass that `prepare_yolo_dataset` uses, but without
    extracting any frames so it's cheap enough to run on every keystroke
    in the GUI.
    """

    teacher_ids: list[str] = Field(default_factory=list)
    export_threshold: float = 0.30
    t_low: float = 0.15
    treat_empty_as_negative: bool = False

    @model_validator(mode="after")
    def _t_low_le_export_threshold(self) -> "PreviewBucketsRequest":
        if self.t_low > self.export_threshold:
            raise ValueError(
                f"t_low ({self.t_low}) must be <= export_threshold "
                f"({self.export_threshold}) — the uncertain band "
                "[t_low, export_threshold) would otherwise be empty/inverted."
            )
        return self


class PreviewBucketsAggregate(BaseModel):
    positive: int = 0
    uncertain: int = 0
    true_negative: int = 0
    n_classes: int = 0
    class_names: list[str] = Field(default_factory=list)


class PreviewBucketsPerTeacher(BaseModel):
    teacher_id: str
    positive: int = 0
    uncertain: int = 0
    true_negative: int = 0


class PreviewBucketsResponse(BaseModel):
    aggregate: PreviewBucketsAggregate
    per_teacher: list[PreviewBucketsPerTeacher] = Field(default_factory=list)


# ---- Model cache status ----------------------------------------------------


class CacheStatusModel(BaseModel):
    impl: str
    known: bool
    cached: bool
    estimated_bytes: int = 0
    model_id: Optional[str] = None


class CacheStatusResponse(BaseModel):
    impls: list[CacheStatusModel]
