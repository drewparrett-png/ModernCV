import type { BlocksResponse, GraphSpec, RunResponse } from "./types";

const API_BASE = "http://localhost:8000";

export async function fetchBlocks(): Promise<BlocksResponse> {
  const res = await fetch(`${API_BASE}/blocks`);
  if (!res.ok) throw new Error(`GET /blocks: ${res.status}`);
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
