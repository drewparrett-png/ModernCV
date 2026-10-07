/**
 * Types for the YOLO26 Studio API (server/studio.py). All geometry is in
 * original-image pixels.
 */

export type Split = "auto" | "train" | "val" | "test";
export type Pt = [number, number];
export type BBox = [number, number, number, number];

export interface StudioClass {
  id: number;
  name: string;
  color: string;
}

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface SyntheticMeta {
  seed: number;
  tilt_deg: number;
  hfov_deg: number;
  camera_height_m: number;
  n_visible: number;
  stack_top_m: number;
}

export interface StudioImage {
  id: string;
  file: string;
  filename: string;
  width: number;
  height: number;
  split: Split;
  source: string;
  added_at: string;
  has_depth: boolean;
  intrinsics: Intrinsics | null;
  negative: boolean;
  n_annotations: number;
  n_suggestions: number;
  synthetic?: SyntheticMeta;
}

export interface Annotation {
  id: string;
  class_id: number;
  bbox: BBox;
  polygon: Pt[] | null;
  source: string;
  score: number | null;
}

export interface Suggestion extends Annotation {
  class_name: string;
}

export interface SuggestionSet {
  source: string | null;
  created_at: string | null;
  items: Suggestion[];
}

export interface DatasetStats {
  n_images: number;
  n_labeled: number;
  n_negative: number;
  n_unlabeled: number;
  splits: { train: number; val: number; test: number };
  instances_per_class: Record<string, number>;
  n_instances: number;
  n_box_only: number;
}

export interface StudioState {
  classes: StudioClass[];
  images: StudioImage[];
  stats: DatasetStats;
}

// ---- Prompts ---------------------------------------------------------------

export interface BoxPrompt {
  type: "box";
  bbox: BBox;
  class_id: number;
  polarity: 1 | -1;
}

export interface StrokePrompt {
  type: "stroke";
  points: Pt[];
  radius: number;
  class_id: number;
  polarity: 1 | -1;
}

export type Prompt = BoxPrompt | StrokePrompt;

export interface SamResult {
  class_id: number | null;
  polygon: Pt[];
  bbox: BBox;
  score: number;
  area: number;
  source: string;
}

export type Scope = "image" | "unlabeled" | "all" | "ids";
export type YoloeFamily = "26" | "11";
/** Closed-vocabulary YOLO families (YOLO26 default; YOLO11 for comparison). */
export type YoloFamily = "26" | "11";
export type YoloeSize = "s" | "m" | "l";
export type YoloTask = "detect" | "segment" | "classify" | "pose" | "obb";
export type YoloSize = "n" | "s" | "m" | "l" | "x";
export type TrainTask = "detect" | "segment" | "obb";

export type ModelSpec =
  | { kind: "yolo"; family: YoloFamily; task: YoloTask; size: YoloSize }
  | { kind: "trained"; model_id: string; artifact?: string }
  | { kind: "yoloe-text"; family: YoloeFamily; size: YoloeSize; classes: string[] }
  | { kind: "yoloe-pf"; family: YoloeFamily; size: YoloeSize };

export interface PredictParams {
  conf?: number;
  iou?: number;
  imgsz?: number;
  max_det?: number;
  agnostic_nms?: boolean;
  augment?: boolean;
  classes?: number[];
  retina_masks?: boolean;
  half?: boolean;
  end2end?: boolean | null;
}

export interface Detection {
  class_id: number;
  class_name: string;
  score: number;
  bbox: BBox;
  polygon?: Pt[];
  keypoints?: [number, number, number][];
  obb?: { cx: number; cy: number; w: number; h: number; angle_deg: number; points: Pt[] };
  track_id?: number;
}

export interface Prediction {
  task: YoloTask | null;
  model_label: string;
  names: Record<string, string>;
  image: { width: number; height: number };
  speed: { preprocess?: number; inference?: number; postprocess?: number };
  detections: Detection[];
  classification: { top: { class_id: number; class_name: string; score: number }[] } | null;
}

// ---- Catalog ---------------------------------------------------------------

export interface Catalog {
  device: string;
  yolo: Record<YoloFamily, Record<YoloTask, { size: YoloSize; weights: string; cached: boolean }[]>>;
  yoloe: { family: YoloeFamily; size: YoloeSize; weights: string; cached: boolean; pf_cached: boolean }[];
  sam: { id: string; label: string; cached: boolean }[];
  depth: { id: string; label: string }[];
  export_formats: { id: string; label: string; available: boolean; note: string }[];
  trackers: string[];
  train_tasks: TrainTask[];
  augment_keys: string[];
  default_train_config: TrainConfig;
  /** Optional package → install hint when missing (null when installed). */
  missing_deps?: Record<string, string | null>;
}

// ---- Training --------------------------------------------------------------

export interface TrainConfig {
  epochs: number;
  imgsz: number;
  batch: number;
  patience: number;
  optimizer: string;
  lr0: number | null;
  cos_lr: boolean;
  freeze: number | null;
  val_pct: number;
  seed: number;
  workers: number;
  cache: boolean;
  augment: Record<string, number>;
}

export interface TrainProgress {
  stage: string;
  epoch: number;
  epochs: number;
  message?: string;
  metrics?: Record<string, number>;
  loss?: Record<string, number>;
  elapsed_s?: number;
  eta_s?: number;
  updated_at: string;
}

export interface PerClassMetric {
  class_id: number;
  class_name: string;
  box_map50?: number;
  box_map50_95?: number;
  box_precision?: number;
  box_recall?: number;
  mask_map50?: number;
  mask_map50_95?: number;
  mask_precision?: number;
  mask_recall?: number;
}

export interface ExportEntry {
  format: string;
  file: string;
  size_bytes: number;
  seconds: number;
  created_at: string;
  imgsz: number;
  half: boolean;
  int8: boolean;
  dynamic: boolean;
  nms: boolean;
}

export interface TrainedModel {
  id: string;
  name: string;
  task: TrainTask;
  size: YoloSize;
  /** Absent on models trained before YOLO11 support — those are YOLO26. */
  family?: YoloFamily;
  base: string;
  base_weights: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  config: TrainConfig;
  dataset: {
    class_ids: number[];
    names: string[];
    counts: { train: number; val: number; test: number };
    instances: number;
    warnings: string[];
  };
  metrics: Record<string, number> | null;
  per_class: PerClassMetric[] | null;
  speed?: Record<string, number>;
  train_seconds?: number;
  weights_bytes?: number;
  exports: ExportEntry[];
  /** Absent on the bare manifests returned by start / cancel. */
  progress?: TrainProgress | null;
  curve?: Record<string, number>[];
  plots?: string[];
}

export interface ValResult {
  summary: Record<string, number>;
  per_class: PerClassMetric[];
  speed: Record<string, number>;
  confusion: { labels: string[]; matrix: number[][] } | null;
}

export interface BenchRow {
  runtime: string;
  artifact: string;
  device: string;
  p50_ms?: number;
  p95_ms?: number;
  mean_ms?: number;
  fps?: number;
  n?: number;
  error?: string;
}

// ---- Pallet ----------------------------------------------------------------

export interface PalletBox {
  id: string;
  class_id: number;
  height_m: number | null;
  height_is_lower_bound?: boolean;
  dims_m?: [number, number] | null;
  yaw_deg?: number | null;
  tilt_deg?: number | null;
  centroid_px?: Pt;
  plane_xy_m?: Pt;
  footprint_m?: Pt[] | null;
  top_area_frac?: number;
  n_points?: number;
  flags: string[];
  layer?: number;
  pick_order?: number;
  blocked?: boolean;
  polygon?: Pt[];
  gt_height_m?: number;
}

export interface PalletResult {
  image_id: string;
  plane: {
    mode: string;
    note: string;
    normal: number[];
    d_m: number;
    inlier_ratio: number;
    rms_m: number;
    n_points: number;
    camera_tilt_deg: number;
    camera_height_m: number;
    /** Auto plane never reaches the bottom of the frame — likely a wall. */
    suspect_wall?: boolean;
  };
  boxes: PalletBox[];
  layers: { layer: number; n: number; mean_height_m: number; min_height_m: number; max_height_m: number }[];
  summary: {
    n_boxes: number;
    n_measured: number;
    n_layers: number;
    max_height_m: number;
    n_blocked: number;
    n_top_hidden: number;
    /** Most cartons show only side faces. */
    side_view?: boolean;
    top_band_m: number;
  };
  calibration?: { known_camera_height_m: number; depth_scale: number };
  depth_source: string;
  intrinsics: Intrinsics & { source: string };
  instance_source: string;
  height_range_m: [number, number];
  depth_range_m: [number, number];
  height_vis: string;
  depth_vis: string;
  warnings: string[];
  ground_truth?: { n_matched: number; mean_abs_err_m: number | null; max_abs_err_m: number | null };
}

export interface PalletRequest {
  instances: { kind: "annotations" | "suggestions" | "predict"; model?: ModelSpec; params?: PredictParams; min_score?: number };
  depth: { source: "auto" | "sensor" | "mono"; model?: string };
  use_image_intrinsics: boolean;
  hfov_deg: number;
  known_camera_height_m?: number | null;
  plane: { mode: "auto" | "painted" | "boxes"; strokes?: { points: Pt[]; radius: number }[] };
  layer_tol_m: number;
}

// ---- Tracking --------------------------------------------------------------

export interface TrackJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  error: string | null;
  video_path: string;
  tracker: string;
  model: ModelSpec;
  stride: number;
  max_frames: number;
  count_line: { axis: "x" | "y"; pos: number } | null;
  stats: {
    frames: number;
    seconds: number;
    fps: number | null;
    mean_infer_ms: number | null;
    unique_tracks: number;
    unique_by_class: Record<string, number>;
    line_counts: { forward: number; backward: number } | null;
  } | null;
  progress: { done: number; total: number } | null;
  has_video: boolean;
}
