import type {
  ArchitecturesResponse,
  BlocksResponse,
  DetectionsResponse,
  FrameState,
  FrameStateEntry,
  FrameStatesMap,
  GraphSpec,
  LearnRequest,
  OptimizeRequest,
  PerFrameLabels,
  PreviewBucketsRequest,
  PreviewBucketsResponse,
  Project,
  ProjectCreateRequest,
  ProjectsResponse,
  RunDetail,
  RunResponse,
  RunsResponse,
  StudentDetail,
  StudentRunDetail,
  StudentRunRequest,
  StudentRunsResponse,
  StudentsResponse,
  VideosResponse,
} from "./types";

const API_BASE = "http://localhost:8000";

export const apiBase = API_BASE;

// ---- Discovery ----------------------------------------------------------

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

// ---- Projects -----------------------------------------------------------

export async function fetchProjects(): Promise<ProjectsResponse> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) throw new Error(`GET /projects: ${res.status}`);
  return res.json();
}

export async function fetchProject(id: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`GET /projects/${id}: ${res.status}`);
  return res.json();
}

export async function createProject(req: ProjectCreateRequest): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /projects: ${res.status} ${text}`);
  }
  return res.json();
}

export async function renameProject(id: string, name: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PATCH /projects/${id}: ${res.status} ${text}`);
  }
  return res.json();
}

export async function deleteProject(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE /projects/${id}: ${res.status}`);
  }
}

// ---- Per-project: Learn / Runs -----------------------------------------

function p(projectId: string): string {
  return `${API_BASE}/projects/${encodeURIComponent(projectId)}`;
}

export async function runLearn(
  projectId: string,
  req: LearnRequest,
): Promise<RunDetail> {
  const res = await fetch(`${p(projectId)}/learn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST .../learn: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchRuns(projectId: string): Promise<RunsResponse> {
  const res = await fetch(`${p(projectId)}/runs`);
  if (!res.ok) throw new Error(`GET .../runs: ${res.status}`);
  return res.json();
}

export async function fetchRunDetail(
  projectId: string,
  id: string,
  threshold?: number,
): Promise<RunDetail> {
  const url = new URL(`${p(projectId)}/runs/${encodeURIComponent(id)}`);
  if (threshold !== undefined) {
    url.searchParams.set("threshold", String(threshold));
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`GET .../runs/${id}: ${res.status}`);
  return res.json();
}

export async function patchRunDisplayThreshold(
  projectId: string,
  id: string,
  display_threshold: number,
): Promise<RunDetail> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_threshold }),
    },
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PATCH .../runs/${id}: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchRunLabels(
  projectId: string,
  id: string,
): Promise<PerFrameLabels[]> {
  const res = await fetch(`${p(projectId)}/runs/${encodeURIComponent(id)}/labels`);
  if (!res.ok) throw new Error(`GET .../labels: ${res.status}`);
  const text = await res.text();
  return text
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as PerFrameLabels);
}

export function runFrameUrl(
  projectId: string,
  id: string,
  idx: number,
  source: "raw" | "overlay" = "overlay",
): string {
  return `${p(projectId)}/runs/${encodeURIComponent(id)}/frame/${idx}?source=${source}`;
}

export function runOverlayUrl(projectId: string, id: string): string {
  return `${p(projectId)}/runs/${encodeURIComponent(id)}/overlay.mp4`;
}

// ---- Per-frame review state (Phase 3) -----------------------------------

export async function fetchFrameStates(
  projectId: string,
  id: string,
): Promise<FrameStatesMap> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/frame_states`,
  );
  if (!res.ok) throw new Error(`GET .../frame_states: ${res.status}`);
  const body = (await res.json()) as { frame_states: FrameStatesMap };
  return body.frame_states;
}

export async function putFrameState(
  projectId: string,
  id: string,
  frame_idx: number,
  body: { state: FrameState; rejected_dets?: number[] },
): Promise<FrameStateEntry> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/frame_states/${frame_idx}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) {
    let detail = "";
    try {
      const err = await res.json();
      detail = err.detail ?? "";
    } catch {
      // ignore — fall through to plain status code
    }
    throw new Error(
      `PUT .../frame_states/${frame_idx}: ${res.status}${detail ? ` (${detail})` : ""}`,
    );
  }
  return (await res.json()) as FrameStateEntry;
}

export async function deleteFrameState(
  projectId: string,
  id: string,
  frame_idx: number,
): Promise<void> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/frame_states/${frame_idx}`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE .../frame_states/${frame_idx}: ${res.status}`);
  }
}

// ---- Detections / crops (Phase 4) --------------------------------------

export async function fetchDetections(
  projectId: string,
  id: string,
  opts: { sort?: "score_asc"; limit?: number; offset?: number } = {},
): Promise<DetectionsResponse> {
  const params = new URLSearchParams();
  params.set("sort", opts.sort ?? "score_asc");
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/detections?${params.toString()}`,
  );
  if (!res.ok) throw new Error(`GET .../detections: ${res.status}`);
  return (await res.json()) as DetectionsResponse;
}

export function detectionCropUrl(
  projectId: string,
  id: string,
  frame_idx: number,
  det_idx: number,
  pad = 24,
): string {
  return (
    `${p(projectId)}/runs/${encodeURIComponent(id)}` +
    `/detection_crop/${frame_idx}/${det_idx}.jpg?pad=${pad}`
  );
}

// ---- Delete a run -------------------------------------------------------

export async function deleteRun(projectId: string, id: string): Promise<void> {
  const res = await fetch(`${p(projectId)}/runs/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE .../runs/${id}: ${res.status}`);
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

// ---- Per-project: Students / Optimize ----------------------------------

export async function runOptimize(
  projectId: string,
  req: OptimizeRequest,
): Promise<StudentDetail> {
  const res = await fetch(`${p(projectId)}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST .../optimize: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchStudents(projectId: string): Promise<StudentsResponse> {
  const res = await fetch(`${p(projectId)}/students`);
  if (!res.ok) throw new Error(`GET .../students: ${res.status}`);
  return res.json();
}

export async function fetchStudentDetail(
  projectId: string,
  id: string,
): Promise<StudentDetail> {
  const res = await fetch(`${p(projectId)}/students/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`GET .../students/${id}: ${res.status}`);
  return res.json();
}

export async function deleteStudent(
  projectId: string,
  id: string,
): Promise<void> {
  const res = await fetch(`${p(projectId)}/students/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE .../students/${id}: ${res.status}`);
  }
}

export async function fetchArchitectures(
  projectId: string,
): Promise<ArchitecturesResponse> {
  const res = await fetch(`${p(projectId)}/students/architectures`);
  if (!res.ok) {
    throw new Error(`GET .../architectures: ${res.status}`);
  }
  return res.json();
}

export async function previewBuckets(
  projectId: string,
  req: PreviewBucketsRequest,
): Promise<PreviewBucketsResponse> {
  const res = await fetch(`${p(projectId)}/students/preview-buckets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST .../preview-buckets: ${res.status} ${text}`);
  }
  return res.json();
}

// ---- Student-run (Phase 5) ----------------------------------------------

export async function startStudentRun(
  projectId: string,
  studentId: string,
  req: StudentRunRequest,
): Promise<StudentRunDetail> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    },
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST .../run: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchStudentRuns(
  projectId: string,
  studentId: string,
): Promise<StudentRunsResponse> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}/runs`,
  );
  if (!res.ok) throw new Error(`GET .../runs: ${res.status}`);
  return res.json();
}

export async function fetchStudentRunDetail(
  projectId: string,
  studentId: string,
  runId: string,
): Promise<StudentRunDetail> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}` +
      `/runs/${encodeURIComponent(runId)}`,
  );
  if (!res.ok) throw new Error(`GET .../runs/${runId}: ${res.status}`);
  return res.json();
}

export async function deleteStudentRun(
  projectId: string,
  studentId: string,
  runId: string,
): Promise<void> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}` +
      `/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404) {
    throw new Error(`DELETE .../runs/${runId}: ${res.status}`);
  }
}

export function studentRunOverlayUrl(
  projectId: string,
  studentId: string,
  runId: string,
): string {
  return (
    `${p(projectId)}/students/${encodeURIComponent(studentId)}` +
    `/runs/${encodeURIComponent(runId)}/overlay.mp4`
  );
}

export function studentRunFrameUrl(
  projectId: string,
  studentId: string,
  runId: string,
  idx: number,
  source: "raw" | "overlay" = "overlay",
): string {
  return (
    `${p(projectId)}/students/${encodeURIComponent(studentId)}` +
    `/runs/${encodeURIComponent(runId)}/frame/${idx}?source=${source}`
  );
}

export interface TrainingCurve {
  epochs: number[];
  train_loss: number[];
  val_map50: number[];
  val_map50_95: number[];
}

export async function fetchTrainingCurve(
  projectId: string,
  studentId: string,
): Promise<TrainingCurve | null> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}/training_curve`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`GET training_curve: ${res.status}`);
  }
  return res.json();
}

export async function fetchStudentSamples(
  projectId: string,
  studentId: string,
): Promise<Record<string, string[]>> {
  const res = await fetch(
    `${p(projectId)}/students/${encodeURIComponent(studentId)}/samples`,
  );
  if (!res.ok) {
    if (res.status === 404) return {};
    throw new Error(`GET samples: ${res.status}`);
  }
  const body = await res.json();
  return body.samples ?? {};
}

export function studentSampleUrl(
  projectId: string,
  studentId: string,
  teacherId: string,
  name: string,
): string {
  return (
    `${p(projectId)}/students/${encodeURIComponent(studentId)}` +
    `/samples/${encodeURIComponent(teacherId)}/${encodeURIComponent(name)}`
  );
}
