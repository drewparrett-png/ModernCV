/** Fetch wrappers for the Studio API (server/studio.py). */

import { apiBase } from "../api";
import type {
  Annotation,
  BenchRow,
  Catalog,
  ExportEntry,
  ModelSpec,
  PalletRequest,
  PalletResult,
  Prediction,
  PredictParams,
  Prompt,
  SamResult,
  Scope,
  StudioClass,
  StudioImage,
  StudioState,
  SuggestionSet,
  TrackJob,
  TrainConfig,
  TrainedModel,
  ValResult,
  YoloeFamily,
  YoloeSize,
} from "./types";

const enc = encodeURIComponent;

async function json<T>(res: Response, label: string): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `${label}: HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

function post<T>(path: string, body: unknown, label = path): Promise<T> {
  return fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r, label));
}

function put<T>(path: string, body: unknown, label = path): Promise<T> {
  return fetch(`${apiBase}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r, label));
}

function get<T>(path: string, label = path): Promise<T> {
  return fetch(`${apiBase}${path}`).then((r) => json<T>(r, label));
}

function del<T>(path: string, label = path): Promise<T> {
  return fetch(`${apiBase}${path}`, { method: "DELETE" }).then((r) => json<T>(r, label));
}

const S = (pid: string) => `/projects/${enc(pid)}/studio`;

// ---- URLs ------------------------------------------------------------------

export const imageUrl = (pid: string, iid: string) => `${apiBase}${S(pid)}/images/${enc(iid)}/file`;
export const thumbUrl = (pid: string, iid: string, size = 160) =>
  `${apiBase}${S(pid)}/images/${enc(iid)}/thumb?size=${size}`;
export const depthPreviewUrl = (pid: string, iid: string, bust = "") =>
  `${apiBase}${S(pid)}/images/${enc(iid)}/depth.png${bust ? `?v=${bust}` : ""}`;
export const plotUrl = (pid: string, mid: string, name: string) =>
  `${apiBase}${S(pid)}/models/${enc(mid)}/plots/${enc(name)}`;
export const exportDownloadUrl = (pid: string, mid: string, file: string) =>
  `${apiBase}${S(pid)}/models/${enc(mid)}/exports/${enc(file)}`;
export const weightsUrl = (pid: string, mid: string) => `${apiBase}${S(pid)}/models/${enc(mid)}/weights`;
export const trackVideoUrl = (pid: string, tid: string) => `${apiBase}${S(pid)}/tracks/${enc(tid)}/overlay.mp4`;
export const datasetZipUrl = (pid: string, task: string) => `${apiBase}${S(pid)}/dataset.zip?task=${enc(task)}`;

// ---- Catalog / dataset -----------------------------------------------------

export const fetchCatalog = () => get<Catalog>("/studio/catalog");
export const fetchStudio = (pid: string) => get<StudioState>(S(pid));
export const putClasses = (pid: string, classes: Partial<StudioClass>[]) =>
  put<{ classes: StudioClass[]; images: StudioImage[] }>(`${S(pid)}/classes`, { classes });

export async function uploadImages(pid: string, files: File[]): Promise<{ images: StudioImage[]; errors: string[] }> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  const res = await fetch(`${apiBase}${S(pid)}/images`, { method: "POST", body: fd });
  return json(res, "upload images");
}

export const imagesFromVideo = (pid: string, body: { video_path: string; stride: number; max_frames: number; start_frame?: number }) =>
  post<{ images: StudioImage[] }>(`${S(pid)}/images/from_video`, body);
export const addSynthetic = (pid: string, body: { count: number; with_labels: boolean; seed?: number }) =>
  post<{ images: StudioImage[]; classes: StudioClass[] }>(`${S(pid)}/images/synthetic`, body);
export const patchImage = (pid: string, iid: string, body: Partial<Pick<StudioImage, "split" | "negative" | "intrinsics">>) =>
  fetch(`${apiBase}${S(pid)}/images/${enc(iid)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<StudioImage>(r, "patch image"));
export const deleteImage = (pid: string, iid: string) => del<{ deleted: string }>(`${S(pid)}/images/${enc(iid)}`);

export async function uploadDepth(
  pid: string,
  iid: string,
  file: File,
  opts: { scale_to_mm?: number; fx?: number; fy?: number; cx?: number; cy?: number } = {},
): Promise<StudioImage> {
  const fd = new FormData();
  fd.append("file", file, file.name);
  for (const [k, v] of Object.entries(opts)) if (v !== undefined && v !== null && !Number.isNaN(v)) fd.append(k, String(v));
  const res = await fetch(`${apiBase}${S(pid)}/images/${enc(iid)}/depth`, { method: "PUT", body: fd });
  return json(res, "upload depth");
}
export const deleteDepth = (pid: string, iid: string) => del<StudioImage>(`${S(pid)}/images/${enc(iid)}/depth`);

export const fetchAnnotations = (pid: string, iid: string) =>
  get<{ annotations: Annotation[] }>(`${S(pid)}/images/${enc(iid)}/annotations`);
export const putAnnotations = (pid: string, iid: string, annotations: Partial<Annotation>[]) =>
  put<{ annotations: Annotation[]; image: StudioImage }>(`${S(pid)}/images/${enc(iid)}/annotations`, { annotations });
export const fetchSuggestions = (pid: string, iid: string) => get<SuggestionSet>(`${S(pid)}/images/${enc(iid)}/suggestions`);
export const acceptSuggestions = (pid: string, iid: string, body: { ids?: string[]; min_score?: number; class_id?: number | null }) =>
  post<{ annotations: Annotation[]; suggestions: SuggestionSet; image: StudioImage; classes: StudioClass[]; accepted: number }>(
    `${S(pid)}/images/${enc(iid)}/suggestions/accept`,
    body,
  );
export const rejectSuggestions = (pid: string, iid: string, ids?: string[]) =>
  post<{ suggestions: SuggestionSet; image: StudioImage }>(`${S(pid)}/images/${enc(iid)}/suggestions/reject`, { ids });

// ---- Prompting -------------------------------------------------------------

export type Granularity = "auto" | "fine" | "medium" | "coarse";

export const samSegment = (pid: string, image_id: string, prompts: Prompt[], sam_model: string, granularity: Granularity = "auto") =>
  post<{ result: SamResult | null }>(`${S(pid)}/prompt/sam`, { image_id, prompts, sam_model, granularity });
export const paintLabels = (pid: string, image_id: string, prompts: Prompt[]) =>
  post<{ annotations: Annotation[]; added: number; image: StudioImage }>(`${S(pid)}/prompt/paint`, { image_id, prompts });

export interface VisualRef {
  image_id: string;
  prompts?: Prompt[];
  use_annotations?: boolean;
  annotation_ids?: string[];
}

export interface DetectOutcome {
  engine: string;
  counts: Record<string, number>;
  images: StudioImage[];
  n_examples?: number;
  n_total?: number;
}

export const visualPrompt = (
  pid: string,
  body: {
    refs: VisualRef[];
    scope: Scope;
    image_id?: string;
    image_ids?: string[];
    family: YoloeFamily;
    size: YoloeSize;
    conf: number;
    iou?: number;
    refine_with_sam: boolean;
    sam_model: string;
  },
) => post<DetectOutcome>(`${S(pid)}/prompt/visual`, body);

export const detectToSuggestions = (
  pid: string,
  body: {
    model: ModelSpec;
    scope: Scope;
    image_id?: string;
    params?: PredictParams;
    refine_with_sam?: boolean;
    sam_model?: string;
    assign_class_id?: number | null;
  },
) => post<DetectOutcome>(`${S(pid)}/prompt/detect`, body);

export const predict = (pid: string, body: { model: ModelSpec; image_id: string; params: PredictParams }) =>
  post<Prediction>(`${S(pid)}/predict`, body);

// ---- Models ----------------------------------------------------------------

export const fetchModels = (pid: string) => get<{ models: TrainedModel[] }>(`${S(pid)}/models`);
export const fetchModel = (pid: string, mid: string) => get<TrainedModel>(`${S(pid)}/models/${enc(mid)}`);
export const startTraining = (
  pid: string,
  body: { task: string; size: string; base: string; name?: string; config: Partial<TrainConfig> },
) => post<TrainedModel>(`${S(pid)}/models`, body);
export const cancelModel = (pid: string, mid: string) => post<TrainedModel>(`${S(pid)}/models/${enc(mid)}/cancel`, {});
export const deleteModel = (pid: string, mid: string) => del<{ deleted: string }>(`${S(pid)}/models/${enc(mid)}`);
export const validateModel = (pid: string, mid: string, body: { split: string; conf?: number | null; iou?: number }) =>
  post<ValResult>(`${S(pid)}/models/${enc(mid)}/val`, body);
export const exportModel = (
  pid: string,
  mid: string,
  body: { format: string; imgsz?: number; half?: boolean; int8?: boolean; dynamic?: boolean; nms?: boolean },
) => post<ExportEntry>(`${S(pid)}/models/${enc(mid)}/export`, body);
export const benchmarkModel = (pid: string, mid: string, n_images = 20) =>
  post<{ rows: BenchRow[] }>(`${S(pid)}/models/${enc(mid)}/benchmark`, { n_images });

// ---- Pallet / tracking -----------------------------------------------------

export const analyzePallet = (pid: string, iid: string, body: PalletRequest) =>
  post<PalletResult>(`${S(pid)}/pallet/${enc(iid)}`, body);

export const fetchTracks = (pid: string) => get<{ tracks: TrackJob[] }>(`${S(pid)}/tracks`);
export const fetchTrack = (pid: string, tid: string) => get<TrackJob>(`${S(pid)}/tracks/${enc(tid)}`);
export const startTrack = (
  pid: string,
  body: {
    video_path: string;
    model: ModelSpec;
    tracker: string;
    params: PredictParams;
    stride: number;
    max_frames: number;
    count_line: { axis: "x" | "y"; pos: number } | null;
  },
) => post<TrackJob>(`${S(pid)}/tracks`, body);
export const deleteTrack = (pid: string, tid: string) => del<{ deleted: string }>(`${S(pid)}/tracks/${enc(tid)}`);
