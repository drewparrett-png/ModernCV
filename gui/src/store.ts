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

import {
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  deleteRun as apiDeleteRun,
  deleteStudent as apiDeleteStudent,
  fetchArchitectures,
  fetchBlocks,
  fetchCacheStatus,
  fetchProjects,
  fetchRunDetail,
  fetchRuns,
  fetchStudentDetail,
  fetchStudents,
  fetchVideos,
  renameProject as apiRenameProject,
  runGraph,
  runLearn,
  runOptimize,
} from "./api";
import { seedEdges, seedNodes } from "./seed";
import type {
  BlockKind,
  BlockNodeData,
  GraphSpec,
  LearnRequest,
  Mode,
  OptimizeRequest,
  Project,
  ProjectCreateRequest,
  ProjectSummary,
  RunDetail,
  RunResponse,
  StudentDetail,
} from "./types";

interface LearnFormState {
  videoPath: string;
  maxFrames: number | null;
  frameStride: number;
}

interface State {
  // Top-level mode (within a project)
  mode: Mode;
  setMode: (m: Mode) => void;

  // Catalog
  blocks: Record<string, string[]>;
  videos: string[];
  dataDir: string;
  architectures: string[];
  loadBlocks: () => Promise<void>;
  loadVideos: () => Promise<void>;
  loadArchitectures: () => Promise<void>;

  // ---- Projects (Phase 1) ------------------------------------------------
  projects: ProjectSummary[];
  currentProjectId: string | null;
  projectError: string | null;
  loadProjects: () => Promise<void>;
  setCurrentProject: (id: string | null) => void;
  createProject: (req: ProjectCreateRequest) => Promise<Project | null>;
  renameProject: (id: string, name: string) => Promise<void>;
  deleteProject: (id: string) => Promise<void>;
  /** Convenience selector — current project record from the summary list,
   *  or null if no project is selected / not yet loaded. */
  getCurrentProject: () => ProjectSummary | null;

  // Graph editor (existing)
  nodes: Node<BlockNodeData>[];
  edges: Edge[];
  runResult: RunResponse | null;
  runError: string | null;
  running: boolean;
  setImpl: (nodeId: string, impl: string) => void;
  setParam: (nodeId: string, key: string, value: unknown) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  run: () => Promise<void>;

  // Learn (Teachers) — multi-run, current-project-scoped
  learnForm: LearnFormState;
  setLearnField: <K extends keyof LearnFormState>(
    key: K,
    value: LearnFormState[K],
  ) => void;
  learnError: string | null;
  teacherDetails: Record<string, RunDetail>;
  teacherPolls: Record<string, ReturnType<typeof setInterval>>;
  selectedTeacherId: string | null;
  selectTeacher: (id: string | null) => void;
  loadTeachers: () => Promise<void>;
  startLearn: () => Promise<string | null>;
  deleteTeacher: (id: string) => Promise<void>;

  // Optimize (Students) — multi-run, current-project-scoped
  optimizeError: string | null;
  studentDetails: Record<string, StudentDetail>;
  studentPolls: Record<string, ReturnType<typeof setInterval>>;
  selectedStudentId: string | null;
  selectStudent: (id: string | null) => void;
  loadStudents: () => Promise<void>;
  startOptimize: (req: OptimizeRequest) => Promise<string | null>;
  deleteStudent: (id: string) => Promise<void>;

  // Optimize-mode tab routing (Phase 3)
  optimizeTab: "new" | "compare";
  setOptimizeTab: (t: "new" | "compare") => void;
  compareStudentIds: string[];
  toggleCompareStudent: (id: string) => void;
  clearCompareStudents: () => void;

  // Inspector
  inspectingRunId: string | null;
  openInspector: (id: string) => void;
  closeInspector: () => void;
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

const POLL_MS = 1500;

export const useStore = create<State>((set, get) => ({
  mode: "learn",
  setMode: (m) => set({ mode: m }),

  blocks: {},
  videos: [],
  dataDir: "",
  architectures: [],

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

  async loadArchitectures() {
    const pid = get().currentProjectId;
    if (!pid) {
      // architectures list is project-scoped via the URL; if no project is
      // selected we can't fetch yet — components fall back to "yolov8n".
      return;
    }
    const resp = await fetchArchitectures(pid);
    set({ architectures: resp.architectures });
  },

  // ---- Projects ----
  projects: [],
  currentProjectId: null,
  projectError: null,

  async loadProjects() {
    try {
      const resp = await fetchProjects();
      set({ projects: resp.projects, projectError: null });
    } catch (e) {
      set({ projectError: e instanceof Error ? e.message : String(e) });
    }
  },

  setCurrentProject(id) {
    if (get().currentProjectId === id) return;
    // Switching project invalidates teacher/student state — clear it so
    // stale rows don't bleed across projects. Polls are also cancelled.
    const polls = { ...get().teacherPolls, ...get().studentPolls };
    for (const handle of Object.values(polls)) {
      clearInterval(handle);
    }
    set({
      currentProjectId: id,
      teacherDetails: {},
      teacherPolls: {},
      selectedTeacherId: null,
      studentDetails: {},
      studentPolls: {},
      selectedStudentId: null,
      compareStudentIds: [],
      mode: "learn",
      learnError: null,
      optimizeError: null,
    });
    if (id) {
      // Fire and forget — the UI shows skeletons during these.
      void get().loadTeachers();
      void get().loadStudents();
      void get().loadArchitectures();
    }
  },

  async createProject(req) {
    try {
      const project = await apiCreateProject(req);
      await get().loadProjects();
      return project;
    } catch (e) {
      set({ projectError: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  async renameProject(id, name) {
    try {
      await apiRenameProject(id, name);
      await get().loadProjects();
    } catch (e) {
      set({ projectError: e instanceof Error ? e.message : String(e) });
    }
  },

  async deleteProject(id) {
    try {
      await apiDeleteProject(id);
      if (get().currentProjectId === id) {
        get().setCurrentProject(null);
      }
      await get().loadProjects();
    } catch (e) {
      set({ projectError: e instanceof Error ? e.message : String(e) });
    }
  },

  getCurrentProject() {
    const pid = get().currentProjectId;
    if (!pid) return null;
    return get().projects.find((p) => p.id === pid) ?? null;
  },

  // ---- Graph editor (unchanged) ----
  nodes: seedNodes(),
  edges: seedEdges(),
  runResult: null,
  runError: null,
  running: false,

  setImpl(nodeId, impl) {
    set({ nodes: patchNodeData(get().nodes, nodeId, (d) => ({ ...d, impl })) });
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

  // ---- Learn / Teachers ----

  learnForm: {
    videoPath: "",
    maxFrames: 60,
    frameStride: 1,
  },
  setLearnField(key, value) {
    set({ learnForm: { ...get().learnForm, [key]: value } });
  },

  learnError: null,
  teacherDetails: {},
  teacherPolls: {},
  selectedTeacherId: null,
  selectTeacher: (id) => set({ selectedTeacherId: id }),

  async loadTeachers() {
    const pid = get().currentProjectId;
    if (!pid) return;
    const resp = await fetchRuns(pid);
    const next: Record<string, RunDetail> = { ...get().teacherDetails };
    const seen = new Set<string>();
    for (const m of resp.runs) {
      seen.add(m.id);
      next[m.id] = {
        manifest: m,
        stats: next[m.id]?.stats ?? null,
        progress: next[m.id]?.progress ?? null,
      };
    }
    for (const id of Object.keys(next)) {
      if (!seen.has(id)) delete next[id];
    }
    set({ teacherDetails: next });
  },

  async startLearn() {
    const pid = get().currentProjectId;
    if (!pid) {
      set({ learnError: "No project selected." });
      return null;
    }
    const project = get().getCurrentProject();
    if (!project) {
      set({ learnError: "Project not loaded yet — try again in a moment." });
      return null;
    }
    const { learnForm } = get();
    if (!learnForm.videoPath) {
      set({ learnError: "Pick a video first." });
      return null;
    }
    set({ learnError: null });

    // Pre-flight cache check (unchanged from pre-Phase-1 behaviour).
    const implsToCheck =
      project.task === "detection"
        ? ["groundingdino", "bytetrack"]
        : ["groundingdino", "sam2-tiny", "dinov3-vits16", "bytetrack"];
    try {
      const statuses = await fetchCacheStatus(implsToCheck);
      const needsDownload = statuses.filter(
        (s) => s.known && !s.cached && s.estimated_bytes > 0,
      );
      if (needsDownload.length > 0) {
        const totalMB = needsDownload.reduce(
          (acc, s) => acc + s.estimated_bytes,
          0,
        ) / 1_000_000;
        const list = needsDownload
          .map((s) => `  • ${s.impl} (~${Math.round(s.estimated_bytes / 1_000_000)} MB)`)
          .join("\n");
        const ok = window.confirm(
          `This run needs to download model weights:\n\n${list}\n\nTotal: ~${Math.round(totalMB)} MB. Continue?`,
        );
        if (!ok) {
          set({ learnError: "Run cancelled — weights download declined." });
          return null;
        }
      }
    } catch (e) {
      console.warn("cache_status check failed; proceeding without prompt:", e);
    }

    const req: LearnRequest = {
      video_path: learnForm.videoPath,
      max_frames: learnForm.maxFrames ?? undefined,
      frame_stride: learnForm.frameStride > 1 ? learnForm.frameStride : undefined,
    };
    let initial: RunDetail;
    try {
      initial = await runLearn(pid, req);
    } catch (err) {
      set({ learnError: err instanceof Error ? err.message : String(err) });
      return null;
    }

    set({
      teacherDetails: { ...get().teacherDetails, [initial.manifest.id]: initial },
      selectedTeacherId: initial.manifest.id,
    });

    if (
      initial.manifest.status === "running" ||
      initial.manifest.status === "queued"
    ) {
      _startTeacherPoll(pid, initial.manifest.id);
    }
    return initial.manifest.id;
  },

  async deleteTeacher(id) {
    const pid = get().currentProjectId;
    if (!pid) return;
    const polls = get().teacherPolls;
    if (polls[id]) clearInterval(polls[id]);
    const newPolls = { ...polls };
    delete newPolls[id];

    const details = { ...get().teacherDetails };
    delete details[id];

    set({
      teacherPolls: newPolls,
      teacherDetails: details,
      selectedTeacherId: get().selectedTeacherId === id ? null : get().selectedTeacherId,
    });

    try {
      await apiDeleteRun(pid, id);
    } catch (e) {
      console.error("delete teacher failed", e);
    }
    await get().loadTeachers();
  },

  // ---- Optimize / Students ----

  optimizeError: null,
  studentDetails: {},
  studentPolls: {},
  selectedStudentId: null,
  selectStudent: (id) => set({ selectedStudentId: id }),

  async loadStudents() {
    const pid = get().currentProjectId;
    if (!pid) return;
    const resp = await fetchStudents(pid);
    const next: Record<string, StudentDetail> = { ...get().studentDetails };
    const seen = new Set<string>();
    for (const m of resp.students) {
      seen.add(m.id);
      next[m.id] = {
        manifest: m,
        stats: next[m.id]?.stats ?? null,
        progress: next[m.id]?.progress ?? null,
      };
    }
    for (const id of Object.keys(next)) {
      if (!seen.has(id)) delete next[id];
    }
    set({ studentDetails: next });
  },

  async startOptimize(req) {
    const pid = get().currentProjectId;
    if (!pid) {
      set({ optimizeError: "No project selected." });
      return null;
    }
    set({ optimizeError: null });
    let initial: StudentDetail;
    try {
      initial = await runOptimize(pid, req);
    } catch (err) {
      set({ optimizeError: err instanceof Error ? err.message : String(err) });
      return null;
    }
    set({
      studentDetails: { ...get().studentDetails, [initial.manifest.id]: initial },
      selectedStudentId: initial.manifest.id,
    });
    if (initial.manifest.status === "running") {
      _startStudentPoll(pid, initial.manifest.id);
    }
    return initial.manifest.id;
  },

  async deleteStudent(id) {
    const pid = get().currentProjectId;
    if (!pid) return;
    const polls = get().studentPolls;
    if (polls[id]) clearInterval(polls[id]);
    const newPolls = { ...polls };
    delete newPolls[id];

    const details = { ...get().studentDetails };
    delete details[id];

    set({
      studentPolls: newPolls,
      studentDetails: details,
      selectedStudentId: get().selectedStudentId === id ? null : get().selectedStudentId,
    });

    try {
      await apiDeleteStudent(pid, id);
    } catch (e) {
      console.error("delete student failed", e);
    }
    await get().loadStudents();
  },

  // ---- Optimize tabs (Phase 3) ----
  optimizeTab: "new",
  setOptimizeTab: (t) => set({ optimizeTab: t }),
  compareStudentIds: [],
  toggleCompareStudent: (id) => {
    const ids = get().compareStudentIds;
    if (ids.includes(id)) {
      set({ compareStudentIds: ids.filter((x) => x !== id) });
    } else {
      set({ compareStudentIds: [...ids, id] });
    }
  },
  clearCompareStudents: () => set({ compareStudentIds: [] }),

  // ---- Inspector ----
  inspectingRunId: null,
  openInspector: (id) => set({ inspectingRunId: id }),
  closeInspector: () => set({ inspectingRunId: null }),
}));

// ---- Poll helpers ----------------------------------------------------------

function _startTeacherPoll(projectId: string, id: string): void {
  const existing = useStore.getState().teacherPolls[id];
  if (existing) return;

  const handle = setInterval(async () => {
    try {
      const latest = await fetchRunDetail(projectId, id);
      useStore.setState((s) => ({
        teacherDetails: { ...s.teacherDetails, [id]: latest },
      }));
      const status = latest.manifest.status;
      if (status !== "queued" && status !== "running") {
        const polls = useStore.getState().teacherPolls;
        if (polls[id]) clearInterval(polls[id]);
        const next = { ...polls };
        delete next[id];
        useStore.setState({ teacherPolls: next });
      }
    } catch (e) {
      console.error("teacher poll failed", e);
    }
  }, POLL_MS);
  useStore.setState((s) => ({
    teacherPolls: { ...s.teacherPolls, [id]: handle },
  }));
}

function _startStudentPoll(projectId: string, id: string): void {
  const existing = useStore.getState().studentPolls[id];
  if (existing) return;

  const handle = setInterval(async () => {
    try {
      const latest = await fetchStudentDetail(projectId, id);
      useStore.setState((s) => ({
        studentDetails: { ...s.studentDetails, [id]: latest },
      }));
      if (latest.manifest.status !== "running") {
        const polls = useStore.getState().studentPolls;
        if (polls[id]) clearInterval(polls[id]);
        const next = { ...polls };
        delete next[id];
        useStore.setState({ studentPolls: next });
      }
    } catch (e) {
      console.error("student poll failed", e);
    }
  }, POLL_MS);
  useStore.setState((s) => ({
    studentPolls: { ...s.studentPolls, [id]: handle },
  }));
}

export const startTeacherPoll = _startTeacherPoll;
export const startStudentPoll = _startStudentPoll;
