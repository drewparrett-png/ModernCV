export type BlockKind =
  | "input"
  | "detect"
  | "segment"
  | "reid"
  | "track"
  | "stats"
  | "output";

export const BLOCK_KINDS: BlockKind[] = [
  "input",
  "detect",
  "segment",
  "reid",
  "track",
  "stats",
  "output",
];

export interface BlockKindInfo {
  kind: BlockKind;
  impls: string[];
}

export interface BlocksResponse {
  blocks: BlockKindInfo[];
}

export interface NodeSpec {
  id: string;
  kind: BlockKind;
  impl: string;
  params: Record<string, unknown>;
}

export interface GraphSpec {
  nodes: NodeSpec[];
  edges: [string, string][];
}

export interface RunRequest {
  graph: GraphSpec;
  video_path: string | null;
}

export interface RunResponse {
  frames_processed: number;
  error: string | null;
  graph_node_count: number;
}

export interface BlockNodeData {
  kind: BlockKind;
  impl: string;
  params: Record<string, unknown>;
}

export interface VideosResponse {
  videos: string[];
  data_dir: string;
  count: number;
}

// ---- Projects (Phase 1) -------------------------------------------------

export type Task = "detection" | "segmentation";

export interface Project {
  id: string;
  name: string;
  task: Task;
  prompts: string[];
  created_at: string;
}

export interface ProjectSummary extends Project {
  n_running: number;
  n_teacher_datasets: number;
  n_human_reviewed_datasets: number;
  n_students: number;
}

export interface ProjectsResponse {
  projects: ProjectSummary[];
}

export interface ProjectCreateRequest {
  name: string;
  task: Task;
  prompts: string[];
}

// ---- Modes / Learn / Runs ------------------------------------------------

export type Mode = "learn" | "optimize";

/** Project-scoped Learn request (Phase 1). `task` and `prompts` come from
 *  the parent project, NOT from this body. Phase 2 dropped the per-run
 *  detector threshold knobs — every detection ≥ SCORE_FLOOR is persisted
 *  and filtered post-hoc by `RunManifest.display_threshold`. */
export interface LearnRequest {
  video_path: string;
  detect_impl?: string;
  segment_impl?: string;
  reid_impl?: string;
  track_impl?: string;
  max_frames?: number;
}

/** Phase 3: a run is "approved" only when every processed frame carries an
 *  explicit per-frame state. "in_progress" means at least one but not all
 *  frames have been reviewed. */
export type ReviewStatus = "unreviewed" | "in_progress" | "approved";

/** Per-frame review verdict (Phase 3). See server/schemas.py FrameStateEntry. */
export type FrameState = "curated" | "confirmed_empty" | "marked_missed";

export interface FrameStateEntry {
  state: FrameState;
  /** Detection indices the user rejected on this frame. Only meaningful
   *  when `state === "curated"` (server returns []) for the other two. */
  rejected_dets: number[];
}

/** Map keyed by String(frame_idx) → entry. Frames absent from the map are
 *  unreviewed. */
export type FrameStatesMap = Record<string, FrameStateEntry>;

/** Phase 4: one row of the run-level flat detection list, used by
 *  CropReview. `accepted` is derived server-side from frame_states. */
export interface DetectionRow {
  frame_idx: number;
  det_idx: number;
  class_name: string;
  score: number;
  accepted: boolean;
}

export interface DetectionsResponse {
  detections: DetectionRow[];
  total: number;
}

export interface RunManifest {
  id: string;
  project_id: string;
  task: Task;
  prompt: string;
  video_path: string;
  started_at: string;
  ended_at: string | null;
  status: "queued" | "running" | "completed" | "failed";
  models: Record<string, string>;
  error: string | null;
  approved_at: string | null;
  review_status: ReviewStatus;
  /** Phase 2: post-hoc score filter the inspector applies by default.
   *  PATCH `/projects/{pid}/runs/{rid}` to persist a new value. */
  display_threshold: number;
}

export interface PerClassStats {
  n_detections: number;
  frames_present: number;
  max_in_frame: number;
  avg_per_frame: number;
  avg_per_present_frame: number;
  score_avg: number;
  score_p50: number;
}

export interface RunStats {
  frames_processed: number;
  frames_with_detections: number;
  total_ms: number;
  avg_ms_per_frame: number;
  p50_ms_per_frame: number;
  p95_ms_per_frame: number;
  n_detections_total: number;
  detections_per_class: Record<string, PerClassStats>;
  per_frame_count_min: number;
  per_frame_count_p50: number;
  per_frame_count_p95: number;
  per_frame_count_max: number;
  per_frame_count_avg: number;
  per_frame_count_histogram: Record<string, number>;
}

export interface RunProgress {
  stage: string;
  message: string;
  current_frame: number;
  total_frames: number;
  frames_with_detections: number;
  started_at: string;
  updated_at: string;
}

export interface RunDetail {
  manifest: RunManifest;
  stats: RunStats | null;
  progress: RunProgress | null;
}

export interface RunsResponse {
  runs: RunManifest[];
}

export interface PerFrameLabels {
  frame_idx: number;
  detections: {
    bbox_xyxy: [number, number, number, number];
    score: number;
    class_id: number;
    class_name: string;
  }[];
  masks?: unknown[];
}

// ---- Students / Optimize ------------------------------------------------

export interface StudentManifest {
  id: string;
  project_id: string;
  train_teacher_ids: string[];
  eval_teacher_ids: string[];
  task: Task;
  prompt: string;
  started_at: string;
  ended_at: string | null;
  status: "queued" | "running" | "completed" | "failed";
  models: Record<string, string>;
  error: string | null;
  export_threshold?: number;
  t_low?: number;
  treat_empty_as_negative?: boolean;
  architecture?: string;
}

export interface PerEvalTeacherStat {
  teacher_id: string;
  n_images: number;
  n_annotations: number;
  map50: number;
  map50_95: number;
  error?: string;
}

export interface PerTrainTeacherBucket {
  teacher_id: string;
  positive: number;
  uncertain: number;
  true_negative: number;
  /** Phase 3 review-source counters per teacher. All defaulted to 0 by the
   *  server so old stats.json files keep loading. */
  n_frames_curated?: number;
  n_frames_confirmed_empty?: number;
  n_frames_marked_missed?: number;
  n_frames_unreviewed_used?: number;
}

export interface StudentStats {
  train_images: number;
  train_annotations: number;
  train_seconds: number;
  epochs: number;
  map50: number;
  map50_95: number;
  avg_inference_ms: number;
  p50_inference_ms: number;
  p95_inference_ms: number;
  model_size_mb: number;
  per_eval_teacher: PerEvalTeacherStat[];
  n_positive_frames: number;
  n_uncertain_dropped: number;
  n_true_negative_frames: number;
  per_teacher_buckets: PerTrainTeacherBucket[];
  export_threshold: number;
  t_low: number;
  treat_empty_as_negative: boolean;
  imgsz: number;
  device: string;
  inference_warmup_discarded: boolean;
  /** Phase 3 review-source counters: same training-frame total as
   *  positive+uncertain+true_negative, sliced by what drove each frame. */
  n_frames_curated?: number;
  n_frames_confirmed_empty?: number;
  n_frames_marked_missed?: number;
  n_frames_unreviewed_used?: number;
}

export interface StudentDetail {
  manifest: StudentManifest;
  stats: StudentStats | null;
  progress: RunProgress | null;
}

export interface StudentsResponse {
  students: StudentManifest[];
}

export interface OptimizeRequest {
  train_teacher_ids: string[];
  eval_teacher_ids: string[];
  detect_impl?: string;
  segment_impl?: string;
  track_impl?: string;
  epochs?: number;
  export_threshold?: number;
  t_low?: number;
  treat_empty_as_negative?: boolean;
  architecture?: string;
}

export interface ArchitecturesResponse {
  architectures: string[];
}

export interface PreviewBucketsRequest {
  teacher_ids: string[];
  export_threshold?: number;
  t_low?: number;
  treat_empty_as_negative?: boolean;
}

export interface PreviewBucketsAggregate {
  positive: number;
  uncertain: number;
  true_negative: number;
  n_classes: number;
  class_names: string[];
}

export interface PreviewBucketsPerTeacher {
  teacher_id: string;
  positive: number;
  uncertain: number;
  true_negative: number;
}

export interface PreviewBucketsResponse {
  aggregate: PreviewBucketsAggregate;
  per_teacher: PreviewBucketsPerTeacher[];
}
