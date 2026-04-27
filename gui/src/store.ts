import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "reactflow";

import { fetchBlocks, fetchVideos, runGraph } from "./api";
import { seedEdges, seedNodes } from "./seed";
import type {
  BlockKind,
  BlockNodeData,
  GraphSpec,
  RunResponse,
} from "./types";

interface State {
  blocks: Record<string, string[]>;
  videos: string[];
  dataDir: string;
  nodes: Node<BlockNodeData>[];
  edges: Edge[];
  runResult: RunResponse | null;
  runError: string | null;
  running: boolean;
  loadBlocks: () => Promise<void>;
  loadVideos: () => Promise<void>;
  setImpl: (nodeId: string, impl: string) => void;
  setParam: (nodeId: string, key: string, value: unknown) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  run: () => Promise<void>;
}

function patchNodeData(
  nodes: Node<BlockNodeData>[],
  nodeId: string,
  patch: (d: BlockNodeData) => BlockNodeData,
): Node<BlockNodeData>[] {
  return nodes.map((n) =>
    n.id === nodeId ? { ...n, data: patch(n.data) } : n,
  );
}

export const useStore = create<State>((set, get) => ({
  blocks: {},
  videos: [],
  dataDir: "",
  nodes: seedNodes(),
  edges: seedEdges(),
  runResult: null,
  runError: null,
  running: false,

  async loadBlocks() {
    const resp = await fetchBlocks();
    const blocks: Record<string, string[]> = {};
    for (const b of resp.blocks) blocks[b.kind] = b.impls;
    set({ blocks });
  },

  async loadVideos() {
    const resp = await fetchVideos();
    set({ videos: resp.videos, dataDir: resp.data_dir });
  },

  setImpl(nodeId, impl) {
    set({
      nodes: patchNodeData(get().nodes, nodeId, (d) => ({ ...d, impl })),
    });
  },

  setParam(nodeId, key, value) {
    set({
      nodes: patchNodeData(get().nodes, nodeId, (d) => ({
        ...d,
        params: { ...d.params, [key]: value },
      })),
    });
  },

  onNodesChange(changes) {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },

  onEdgesChange(changes) {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect(connection) {
    set({ edges: addEdge(connection, get().edges) });
  },

  async run() {
    const { nodes, edges } = get();
    const graph: GraphSpec = {
      nodes: nodes.map((n) => ({
        id: n.id,
        kind: n.data.kind as BlockKind,
        impl: n.data.impl,
        params: n.data.params,
      })),
      edges: edges
        .filter((e) => e.source && e.target)
        .map((e) => [e.source, e.target] as [string, string]),
    };
    set({ running: true, runError: null });
    try {
      const result = await runGraph(graph);
      set({ runResult: result, running: false });
    } catch (err) {
      set({
        runError: err instanceof Error ? err.message : String(err),
        running: false,
      });
    }
  },
}));
