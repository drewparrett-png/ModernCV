import type { Edge, Node } from "reactflow";
import type { BlockKind, BlockNodeData } from "./types";

interface SeedEntry {
  kind: BlockKind;
  impl: string;
  params: Record<string, unknown>;
}

const SEED: SeedEntry[] = [
  { kind: "input", impl: "opencv", params: { path: "data/clip.mp4" } },
  { kind: "detect", impl: "yolov8n", params: {} },
  { kind: "segment", impl: "sam2-tiny", params: {} },
  { kind: "reid", impl: "dinov3-vits16", params: {} },
  { kind: "track", impl: "bytetrack", params: {} },
  { kind: "stats", impl: "basic", params: {} },
  { kind: "output", impl: "overlay-mp4", params: { path: "runs/out.mp4" } },
];

export function seedNodes(): Node<BlockNodeData>[] {
  return SEED.map((entry, i) => ({
    id: `n${i + 1}`,
    type: "block",
    position: { x: 80 + i * 220, y: 160 },
    data: { kind: entry.kind, impl: entry.impl, params: entry.params },
  }));
}

export function seedEdges(): Edge[] {
  return SEED.slice(0, -1).map((_, i) => ({
    id: `e${i + 1}`,
    source: `n${i + 1}`,
    target: `n${i + 2}`,
  }));
}
