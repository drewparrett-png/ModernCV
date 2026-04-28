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

// ---- Modes / Learn / Runs ------------------------------------------------

export type Mode = "learn" | "optimize";

export type Task = "detection" | "segmentation";

export interface LearnRequest {
  task: Task;
  /** One chip per class. Each chip is a phrase the user typed and expects
   *  to see verbatim on detection labels. Joined with " . " server-side
   *  for the model and snapped back per-detection. */
  prompts: string[];
  /** Legacy single-string entry — still accepted by the backend, but new
   *  callers should use `prompts`. */
  prompt?: string;
  video_path: string;
  detect_impl?: string;
  segment_impl?: string;
  reid_impl?: string;
  track_impl?: string;
  max_frames?: number;
  /** Detector confidence floor for box scores. Lower = more permissive
   *  (catches small / faint objects but more false positives). */
  box_threshold?: number;
  /** Confidence floor for the text-token grounding pass. */
  text_threshold?: number;
  /** When true the GroundingDINO adapter skips the HF processor's resize
   *  step (default 1333 longest edge → small objects get blurred). */
  full_resolution?: boolean;
}

export interface RunManifest {
  id: string;
  task: Task;
  prompt: string;
  video_path: string;
  started_at: string;
  ended_at: string | null;
  /** "queued" → waiting in the FIFO; "running" → currently executing on
   *  the worker; "completed" / "failed" are terminal states. */
  status: "queued" | "running" | "completed" | "failed";
  models: Record<string, string>;
  error: string | null;
}

export interface PerClassStats {
  /** Total detections of this class across all frames. */
  n_detections: number;
  /** Frames that contained at least one detection of this class. */
  frames_present: number;
  /** Most detections of this class in any single frame. */
  max_in_frame: number;
  /** n_detections / frames_processed. Useful as a baseline rate. */
  avg_per_frame: number;
  /** n_detections / frames_present. Useful for "when this class shows up,
   *  how many do we see?". */
  avg_per_present_frame: number;
  /** Mean detector confidence across all detections of this class. */
  score_avg: number;
  /** Median detector confidence across all detections of this class. */
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

  // Detection breakdown — added phase 1. Older runs (pre-breakdown) come
  // back with `{}` for the dict fields and `0` for the scalars; the GUI
  // gracefully degrades.
  detections_per_class: Record<string, PerClassStats>;
  per_frame_count_min: number;
  per_frame_count_p50: number;
  per_frame_count_p95: number;
  per_frame_count_max: number;
  per_frame_count_avg: number;
  /** Histogram: count (string-keyed for JSON) → number of frames. */
  per_frame_count_histogram: Record<string, number>;
}

export interface RunProgress {
  stage: string; // "starting" | "loading_models" | "running" | "finalizing"
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

/** One line of labels/per_frame.jsonl, parsed. */
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
  /** Teachers whose curated COCO labels were merged into the training set. */
  train_teacher_ids: string[];
  /** Teachers held out for transferability evaluation (Student is scored
   *  against them but never sees them at train time). May be empty. */
  eval_teacher_ids: string[];
  task: Task;
  prompt: string;
  started_at: string;
  ended_at: string | null;
  status: "running" | "completed" | "failed";
  models: Record<string, string>;
  error: string | null;
  // Confidence-band thresholds carried from the OptimizeRequest. Defaulted
  // on the backend so old manifests without these keys still load — the
  // GUI treats absent fields as the spec defaults too.
  t_high?: number;
  t_low?: number;
  treat_empty_as_negative?: boolean;
  /** Trainer architecture (Phase 1.3). Defaulted to "yolov8n" on the
   *  backend so old manifests without this key load as the only
   *  architecture that existed before the dispatcher. */
  architecture?: string;
}

/** One row of the Student's per-eval-teacher transferability table. */
export interface PerEvalTeacherStat {
  teacher_id: string;
  n_images: number;
  n_annotations: number;
  map50: number;
  map50_95: number;
  /** Set when the trainer skipped this teacher (missing source video, etc.). */
  error?: string;
}

/** One row of the Student's per-train-teacher frame-bucket breakdown. */
export interface PerTrainTeacherBucket {
  teacher_id: string;
  positive: number;
  uncertain: number;
  true_negative: number;
}

export interface StudentStats {
  train_images: number;
  train_annotations: number;
  train_seconds: number;
  epochs: number;
  /** Mean across per_eval_teacher; 0 when no eval teachers were configured. */
  map50: number;
  map50_95: number;
  avg_inference_ms: number;
  p50_inference_ms: number;
  p95_inference_ms: number;
  model_size_mb: number;
  per_eval_teacher: PerEvalTeacherStat[];
  // ---- frame-bucket breakdown (Phase 0.5/0.6) --------------------------
  // Defaulted to safe values on the backend so old stats.json files keep
  // loading; the GUI renders an em-dash when `per_teacher_buckets` is
  // empty (legacy run, no breakdown was captured).
  n_positive_frames: number;
  n_uncertain_dropped: number;
  n_true_negative_frames: number;
  per_teacher_buckets: PerTrainTeacherBucket[];
  t_high: number;
  t_low: number;
  treat_empty_as_negative: boolean;
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
  /** Required, ≥1. Teachers whose curated COCO becomes the training set. */
  train_teacher_ids: string[];
  /** Optional. Teachers held out for transferability scoring. */
  eval_teacher_ids: string[];
  detect_impl?: string;
  segment_impl?: string;
  track_impl?: string;
  epochs?: number;
  /** Confidence floor for "this annotation becomes a YOLO label". Detections
   *  with score ≥ t_high promote their frame to *positive* and survive the
   *  per-annotation min_score filter at YOLO-label-write time. Default 0.35. */
  t_high?: number;
  /** Frames whose only detections sit in [t_low, t_high) are *uncertain* —
   *  the teacher saw something but wasn't sure. Dropped entirely from
   *  training unless treat_empty_as_negative is on. Default 0.15. */
  t_low?: number;
  /** Reproduces pre-Phase-0 behaviour: every zero/below-t_low-only frame
   *  becomes a true negative. Off by default — flip on only when you fully
   *  trust the teacher's "no detection" signal. */
  treat_empty_as_negative?: boolean;
  /** Trainer architecture (Phase 1.4). Defaulted to "yolov8n" on the
   *  backend; absent on the wire is treated as the same default. */
  architecture?: string;
}

// ---- Architectures list (Phase 1.4) -------------------------------------

/** Response for `GET /students/architectures` — names of every trainer
 *  registered in `pipeline.students.TRAINERS`. The GUI fetches this once
 *  on mount to populate the architecture dropdown. */
export interface ArchitecturesResponse {
  architectures: string[];
}

// ---- Bucket preview (Phase 0.4) ------------------------------------------

export interface PreviewBucketsRequest {
  teacher_ids: string[];
  t_high?: number;
  t_low?: number;
  treat_empty_as_negative?: boolean;
}

export interface PreviewBucketsAggregate {
  positive: number;
  uncertain: number;
  true_negative: number;
  n_classes: number;
  /** Class-name union across the selected teachers, in first-seen order. */
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
