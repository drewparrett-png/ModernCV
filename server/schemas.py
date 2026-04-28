"""Pydantic schemas for the FastAPI surface.

These describe the JSON contract between the GUI and the runner. Keep them
narrow — the graph spec is simple, and adding fields piecemeal is fine.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


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
    """Inputs from the Learn wizard. Backend assembles the graph from these."""

    task: str  # "detection" | "segmentation"
    # Either:
    #   prompts : list[str]  — preferred; one chip per class. Labels on
    #                          detections will be one of these exact strings.
    #   prompt  : str        — legacy single-phrase input; still accepted.
    # If both are provided, `prompts` wins.
    prompt: Optional[str] = None
    prompts: Optional[list[str]] = None
    video_path: str
    # Per-task model overrides — optional; defaults are picked by the backend
    # so the user can stay in pure "fill three fields and go" mode.
    detect_impl: Optional[str] = None
    segment_impl: Optional[str] = None
    reid_impl: Optional[str] = None
    track_impl: Optional[str] = None
    max_frames: Optional[int] = None  # cap for fast iteration
    # Detector confidence knobs. Both default to None ⇒ backend uses the
    # adapter's built-in defaults (0.30 / 0.25 for GroundingDINO). Useful
    # to lower for small-object prompts ("soccer ball") that score below
    # the default threshold.
    box_threshold: Optional[float] = None
    text_threshold: Optional[float] = None
    # When True, the GroundingDINO adapter passes frames to the model with
    # NO resize — the source resolution is preserved. Default False keeps the
    # HF processor defaults (shortest 800 / longest 1333), which is faster
    # but blurs small objects (a 10-px ball at 1080p halves to 5 px).
    full_resolution: Optional[bool] = None


class RunManifestModel(BaseModel):
    """Mirror of pipeline.runs.RunManifest for API output."""

    id: str
    task: str
    prompt: str
    video_path: str
    started_at: str
    ended_at: Optional[str] = None
    status: str
    models: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    approved_at: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def review_status(self) -> Literal["unreviewed", "reviewed", "approved"]:
        """Three-state human-review summary, derived at serialise time.

        - "approved": `approved_at` is populated (one-click approval stamp).
        - "reviewed": no approval stamp, but at least one rejection on disk.
        - "unreviewed": neither — a fresh, untouched run.

        The rejection check uses `has_any_rejections` which short-circuits
        on file existence + a single non-empty list, so this stays cheap
        enough to serialise on every /runs entry. If 50-run lists ever show
        latency, cache the bool in the manifest at write time.
        """
        from pipeline import runs as runs_mod

        if self.approved_at is not None:
            return "approved"
        if runs_mod.has_any_rejections(runs_mod.run_dir(self.id)):
            return "reviewed"
        return "unreviewed"


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


class ApproveResponse(BaseModel):
    """Response for /runs/:id/approve and /runs/:id/unapprove. Only the
    manifest can change — stats/progress aren't touched — so the wire
    surface stays minimal."""

    manifest: RunManifestModel


class RejectionsResponse(BaseModel):
    """Curated rejections for one run.

    Keyed by frame index (string in JSON; React parses back to number) →
    list of detection indices within that frame's per_frame.jsonl entry.
    """

    rejections: dict[str, list[int]] = Field(default_factory=dict)


class RejectToggleRequest(BaseModel):
    frame_idx: int
    det_idx: int


# ---- Optimize / Students ---------------------------------------------------


class OptimizeRequest(BaseModel):
    """Request to start a Student distillation run.

    `train_teacher_ids` (required, non-empty): the Teachers whose curated
    COCO labels get merged into the Student's training set.
    `eval_teacher_ids` (optional): Teachers held out for transferability
    evaluation — the Student is scored against their labels but never sees
    them at train time. May overlap with the train set (in-distribution
    sanity check); the GUI warns when it does.

    Confidence-band fields (`t_high`, `t_low`, `treat_empty_as_negative`)
    drive the frame-bucket filter in `pipeline.distill.prepare_yolo_dataset`.
    See `docs/student-training.md` Phase 0 for the full motivation; the
    short version is "drop frames the teacher was unsure about so we don't
    train the Student to suppress detections it should be making".
    """

    train_teacher_ids: list[str] = Field(default_factory=list)
    eval_teacher_ids: list[str] = Field(default_factory=list)
    detect_impl: Optional[str] = None
    segment_impl: Optional[str] = None
    track_impl: Optional[str] = None
    epochs: int = 50
    # Confidence bands. Defaults match the spec (`docs/student-training.md`).
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False

    @model_validator(mode="after")
    def _t_low_le_t_high(self) -> "OptimizeRequest":
        if self.t_low > self.t_high:
            raise ValueError(
                f"t_low ({self.t_low}) must be <= t_high ({self.t_high}) — "
                "the uncertain band [t_low, t_high) would otherwise be empty/inverted."
            )
        return self


class StudentManifestModel(BaseModel):
    id: str
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
    # Defaulted so old manifest.json files without these keys still load.
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False


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
    """

    teacher_id: str
    positive: int = 0
    uncertain: int = 0
    true_negative: int = 0


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
    # Stamp the thresholds used by the trainer so the detail card shows
    # "buckets at t_high=0.35" without re-reading the manifest.
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False


class StudentDetail(BaseModel):
    manifest: StudentManifestModel
    stats: Optional[StudentStatsModel] = None
    progress: Optional[RunProgressModel] = None


class StudentsResponse(BaseModel):
    students: list[StudentManifestModel]


# ---- Preview-buckets endpoint ---------------------------------------------


class PreviewBucketsRequest(BaseModel):
    """Live-preview request for the New Student form (Phase 0.4).

    Same threshold semantics as `OptimizeRequest` — running the same
    classification pass that `prepare_yolo_dataset` uses, but without
    extracting any frames so it's cheap enough to run on every keystroke
    in the GUI.
    """

    teacher_ids: list[str] = Field(default_factory=list)
    t_high: float = 0.35
    t_low: float = 0.15
    treat_empty_as_negative: bool = False

    @model_validator(mode="after")
    def _t_low_le_t_high(self) -> "PreviewBucketsRequest":
        if self.t_low > self.t_high:
            raise ValueError(
                f"t_low ({self.t_low}) must be <= t_high ({self.t_high}) — "
                "the uncertain band [t_low, t_high) would otherwise be empty/inverted."
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
