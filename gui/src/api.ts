import type {
  ArchitecturesResponse,
  BlocksResponse,
  GraphSpec,
  LearnRequest,
  OptimizeRequest,
  PerFrameLabels,
  PreviewBucketsRequest,
  PreviewBucketsResponse,
  RunDetail,
  RunResponse,
  RunsResponse,
  StudentDetail,
  StudentsResponse,
  VideosResponse,
} from "./types";

const API_BASE = "http://localhost:8000";

export const apiBase = API_BASE;

export async function fetchBlocks(): Promise<BlocksResponse> {
  const res = await fetch(`${API_BASE}/blocks`);
  if (!res.ok) throw new Error(`GET /blocks: ${res.status}`);
  return res.json();
}

export async function fetchVideos(): Promise<VideosResponse> {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error(`GET /videos: ${res.status}`);
  return res.json();
}

export async function runGraph(graph: GraphSpec): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, video_path: null }),
  });
  if (!res.ok) throw new Error(`POST /run: ${res.status}`);
  return res.json();
}

// ---- Learn / Runs --------------------------------------------------------

export async function runLearn(req: LearnRequest): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/learn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /learn: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchRuns(): Promise<RunsResponse> {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error(`GET /runs: ${res.status}`);
  return res.json();
}

export async function fetchRunDetail(id: string): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`GET /runs/${id}: ${res.status}`);
  return res.json();
}

/** Fetch the per_frame.jsonl, parsed into one PerFrameLabels per line. */
export async function fetchRunLabels(id: string): Promise<PerFrameLabels[]> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(id)}/labels`);
  if (!res.ok) throw new Error(`GET /runs/${id}/labels: ${res.status}`);
  const text = await res.text();
  return text
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as PerFrameLabels);
}

/** URL for a single frame — pass directly to <img src=…>, no parse needed. */
export function runFrameUrl(
  id: string,
  idx: number,
  source: "raw" | "overlay" = "overlay",
): string {
  return `${API_BASE}/runs/${encodeURIComponent(id)}/frame/${idx}?source=${source}`;
}

export function runOverlayUrl(id: string): string {
  return `${API_BASE}/runs/${encodeURIComponent(id)}/overlay.mp4`;
}

// ---- Rejections / Delete ------------------------------------------------

/** Per-run rejection map: frame_idx → list of detection indices to drop. */
export type RejectionMap = Record<string, number[]>;

export async function fetchRejections(id: string): Promise<RejectionMap> {
  const res = await fetch(
    `${API_BASE}/runs/${encodeURIComponent(id)}/rejections`,
  );
  if (!res.ok) throw new Error(`GET /runs/${id}/rejections: ${res.status}`);
  const body = (await res.json()) as { rejections: RejectionMap };
  return body.rejections;
}

export async function toggleRejection(
  id: string,
  frame_idx: number,
  det_idx: number,
): Promise<RejectionMap> {
  const res = await fetch(
    `${API_BASE}/runs/${encodeURIComponent(id)}/rejections/toggle`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_idx, det_idx }),
    },
  );
  if (!res.ok) throw new Error(`POST .../toggle: ${res.status}`);
  const body = (await res.json()) as { rejections: RejectionMap };
  return body.rejections;
}

export async function deleteRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE /runs/${id}: ${res.status}`);
  }
}

// ---- Model cache status -------------------------------------------------

export interface CacheStatus {
  impl: string;
  known: boolean;
  cached: boolean;
  estimated_bytes: number;
  model_id: string | null;
}

export async function fetchCacheStatus(impls: string[]): Promise<CacheStatus[]> {
  const qs = encodeURIComponent(impls.join(","));
  const res = await fetch(`${API_BASE}/models/cache_status?impls=${qs}`);
  if (!res.ok) throw new Error(`GET /models/cache_status: ${res.status}`);
  const body = (await res.json()) as { impls: CacheStatus[] };
  return body.impls;
}

// ---- Students / Optimize ------------------------------------------------

export async function runOptimize(
  req: OptimizeRequest,
): Promise<StudentDetail> {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /optimize: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchStudents(): Promise<StudentsResponse> {
  const res = await fetch(`${API_BASE}/students`);
  if (!res.ok) throw new Error(`GET /students: ${res.status}`);
  return res.json();
}

export async function fetchStudentDetail(id: string): Promise<StudentDetail> {
  const res = await fetch(`${API_BASE}/students/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`GET /students/${id}: ${res.status}`);
  return res.json();
}

export async function deleteStudent(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/students/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE /students/${id}: ${res.status}`);
  }
}

/** Names of every registered Student-trainer architecture (Phase 1.4).
 *  Cheap — reads `pipeline.students.list_trainers()`. Cached in the
 *  Zustand store after the initial mount fetch. */
export async function fetchArchitectures(): Promise<ArchitecturesResponse> {
  const res = await fetch(`${API_BASE}/students/architectures`);
  if (!res.ok) {
    throw new Error(`GET /students/architectures: ${res.status}`);
  }
  return res.json();
}

/** Live frame-bucket preview for the New Student form (Phase 0.4).
 *  Cheap on the backend — reads each teacher's coco.json and runs
 *  `classify_frames`, no frame extraction. Caller debounces. */
export async function previewBuckets(
  req: PreviewBucketsRequest,
): Promise<PreviewBucketsResponse> {
  const res = await fetch(`${API_BASE}/students/preview-buckets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /students/preview-buckets: ${res.status} ${text}`);
  }
  return res.json();
}
