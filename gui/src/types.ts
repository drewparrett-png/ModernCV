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
