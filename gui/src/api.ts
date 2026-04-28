import type {
  ArchitecturesResponse,
  BlocksResponse,
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
): Promise<RunDetail> {
  const res = await fetch(`${p(projectId)}/runs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`GET .../runs/${id}: ${res.status}`);
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

// ---- Rejections / Approve / Delete -------------------------------------

export type RejectionMap = Record<string, number[]>;

export async function fetchRejections(
  projectId: string,
  id: string,
): Promise<RejectionMap> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/rejections`,
  );
  if (!res.ok) throw new Error(`GET .../rejections: ${res.status}`);
  const body = (await res.json()) as { rejections: RejectionMap };
  return body.rejections;
}

export async function toggleRejection(
  projectId: string,
  id: string,
  frame_idx: number,
  det_idx: number,
): Promise<RejectionMap> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/rejections/toggle`,
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

export async function approveRun(
  projectId: string,
  id: string,
): Promise<RunDetail["manifest"]> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/approve`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(`POST .../approve: ${res.status}`);
  const body = await res.json();
  return body.manifest;
}

export async function unapproveRun(
  projectId: string,
  id: string,
): Promise<RunDetail["manifest"]> {
  const res = await fetch(
    `${p(projectId)}/runs/${encodeURIComponent(id)}/unapprove`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(`POST .../unapprove: ${res.status}`);
  const body = await res.json();
  return body.manifest;
}

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
