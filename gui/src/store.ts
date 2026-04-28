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
  approveRun as apiApproveRun,
  deleteRun as apiDeleteRun,
  deleteStudent as apiDeleteStudent,
  fetchArchitectures,
  fetchBlocks,
  fetchCacheStatus,
  fetchRunDetail,
  fetchRuns,
  fetchStudentDetail,
  fetchStudents,
  fetchVideos,
  runGraph,
  runLearn,
  runOptimize,
  unapproveRun as apiUnapproveRun,
} from "./api";
import { seedEdges, seedNodes } from "./seed";
import type {
  BlockKind,
  BlockNodeData,
  GraphSpec,
  LearnRequest,
  Mode,
  OptimizeRequest,
  RunDetail,
  RunResponse,
  StudentDetail,
  Task,
} from "./types";

interface LearnFormState {
  task: Task;
  /** One chip per class — each is a complete phrase ("soccer ball",
   *  "player"). The detector treats each chip atomically and labels
   *  detections with the matching chip text. */
  prompts: string[];
  videoPath: string;
  maxFrames: number | null;
  /** Confidence threshold for box scores. Defaults match the GroundingDINO
   *  adapter's defaults (0.30 / 0.25); lower box_threshold to ~0.15-0.20
   *  for small-object prompts like "soccer ball" that score lower than
   *  player-sized prompts. */
  boxThreshold: number;
  textThreshold: number;
  /** Skip the GroundingDINO HF processor's resize step. The default
   *  shrinks 1080p footage to ~1333×750, which blurs small targets like
   *  the soccer ball. Tradeoff: ~2× per-frame latency. */
  fullResolution: boolean;
}

interface State {
  // Top-level mode
  mode: Mode;
  setMode: (m: Mode) => void;

  // Catalog
  blocks: Record<string, string[]>;
  videos: string[];
  dataDir: string;
  /** Names of every registered Student-trainer architecture (Phase 1.4).
   *  Fetched once on mount via `loadArchitectures`; drives the GUI's
   *  architecture <select>. Empty until the fetch resolves — components
   *  fall back to the spec default ("yolov8n") in that window. */
  architectures: string[];
  loadBlocks: () => Promise<void>;
  loadVideos: () => Promise<void>;
  loadArchitectures: () => Promise<void>;

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

  // Learn (Teachers) — multi-run
  learnForm: LearnFormState;
  setLearnField: <K extends keyof LearnFormState>(
    key: K,
    value: LearnFormState[K],
  ) => void;
  learnError: string | null;
  /** All runs visible to the UI: either persisted (loaded from /runs) or
   *  freshly kicked off this session. Keyed by id. */
  teacherDetails: Record<string, RunDetail>;
  /** Active polling handles per teacher run id. */
  teacherPolls: Record<string, ReturnType<typeof setInterval>>;
  /** Currently selected teacher in the Learn sidebar. */
  selectedTeacherId: string | null;
  selectTeacher: (id: string | null) => void;
  loadTeachers: () => Promise<void>;
  startLearn: () => Promise<string | null>;
  deleteTeacher: (id: string) => Promise<void>;
  /** Approve / un-approve the dataset behind a Teacher run.
   *  Both actions are optimistic: the local manifest flips immediately so
   *  the pill reacts without waiting for the server, and the previous
   *  manifest is restored if the server call fails. The single
   *  `teacherDetails` mutation is what makes Phase 1's pill update across
   *  Learn sidebar, Optimize selectors, and Run Inspector with one
   *  store write. */
  approveTeacher: (id: string) => Promise<void>;
  unapproveTeacher: (id: string) => Promise<void>;

  // Optimize (Students) — multi-run
  optimizeError: string | null;
  studentDetails: Record<string, StudentDetail>;
  studentPolls: Record<string, ReturnType<typeof setInterval>>;
  selectedStudentId: string | null;
  selectStudent: (id: string | null) => void;
  loadStudents: () => Promise<void>;
  startOptimize: (req: OptimizeRequest) => Promise<string | null>;
  deleteStudent: (id: string) => Promise<void>;

  // Optimize-mode tab routing (Phase 3). Compare is a sub-tab inside
  // Optimize, *not* a top-level mode — keeps the existing Learn/Optimize
  // shell untouched.
  optimizeTab: "new" | "compare";
  setOptimizeTab: (t: "new" | "compare") => void;
  /** Set of Student ids selected in the Compare tab. Lives in the store
   *  (not local state) so opening the Compare tab on a fresh mount
   *  doesn't blow away the user's selection while toggling between New
   *  and Compare sub-tabs. Spec: "persist nothing" — this resets on
   *  reload because Zustand state is in-memory only. */
  compareStudentIds: string[];
  toggleCompareStudent: (id: string) => void;
  clearCompareStudents: () => void;

  // Inspector — which run is currently being inspected (from Learn or Optimize)
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
    // Phase 1.4: pulls `pipeline.students.list_trainers()` from the server
    // once on mount. The architecture <select> in the New Student form
    // reads this; when it's empty (fetch failed or hasn't returned yet)
    // the form falls back to the spec default ("yolov8n") so the user
    // can still kick off a run.
    const resp = await fetchArchitectures();
    set({ architectures: resp.architectures });
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
    // Start with no chips — the placeholder ("Type and hit Enter to add a
    // search term") tells the user how to fill them in. Seeding a default
    // chip makes the form look pre-configured and the user has to remember
    // to delete it before typing what they actually want.
    task: "detection",
    prompts: [],
    videoPath: "",
    maxFrames: 60,
    boxThreshold: 0.3,
    textThreshold: 0.25,
    fullResolution: false,
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
    const resp = await fetchRuns();
    const next: Record<string, RunDetail> = { ...get().teacherDetails };
    // Merge: keep existing details (which may have stats/progress already
    // loaded) and just refresh the manifest field. Newly seen ids get a
    // skeleton detail; the next poll/click will fill in the rest.
    const seen = new Set<string>();
    for (const m of resp.runs) {
      seen.add(m.id);
      next[m.id] = {
        manifest: m,
        stats: next[m.id]?.stats ?? null,
        progress: next[m.id]?.progress ?? null,
      };
    }
    // Drop ids the backend no longer reports (deletes elsewhere).
    for (const id of Object.keys(next)) {
      if (!seen.has(id)) delete next[id];
    }
    set({ teacherDetails: next });
  },

  async startLearn() {
    const { learnForm } = get();
    if (!learnForm.videoPath) {
      set({ learnError: "Pick a video first." });
      return null;
    }
    const cleanPrompts = learnForm.prompts
      .map((p) => p.trim())
      .filter((p) => p.length > 0);
    if (cleanPrompts.length === 0) {
      set({
        learnError:
          "Enter at least one thing to look for (press Enter after typing).",
      });
      return null;
    }
    set({ learnError: null });

    // Pre-flight: check whether the impls we're about to use need to
    // download weights. Default detection chain is groundingdino + bytetrack;
    // segmentation adds sam2-tiny + dinov3-vits16. We only ask the user
    // about the ones the backend reports as a known download.
    const implsToCheck =
      learnForm.task === "detection"
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
      // Cache-status check failed — fall through and let /learn run; the
      // download will still happen, just without the consent gate. Better
      // than blocking on a transient error.
      console.warn("cache_status check failed; proceeding without prompt:", e);
    }

    const req: LearnRequest = {
      task: learnForm.task,
      prompts: cleanPrompts,
      video_path: learnForm.videoPath,
      max_frames: learnForm.maxFrames ?? undefined,
      box_threshold: learnForm.boxThreshold,
      text_threshold: learnForm.textThreshold,
      full_resolution: learnForm.fullResolution || undefined,
    };
    let initial: RunDetail;
    try {
      initial = await runLearn(req);
    } catch (err) {
      set({ learnError: err instanceof Error ? err.message : String(err) });
      return null;
    }

    // Insert into the multi-run table; auto-select.
    set({
      teacherDetails: { ...get().teacherDetails, [initial.manifest.id]: initial },
      selectedTeacherId: initial.manifest.id,
    });

    if (
      initial.manifest.status === "running" ||
      initial.manifest.status === "queued"
    ) {
      _startTeacherPoll(initial.manifest.id);
    }
    return initial.manifest.id;
  },

  async approveTeacher(id) {
    const prev = get().teacherDetails[id];
    if (!prev) return;
    // Optimistic flip: stamp `approved_at` locally so the pill updates
    // before the network call returns. We mirror the server's "approved"
    // derivation rule (approved_at != null → review_status="approved")
    // so the UI's intermediate state matches what the server will send
    // back. If the call fails, we restore the previous detail wholesale.
    const optimistic = {
      ...prev,
      manifest: {
        ...prev.manifest,
        approved_at: new Date().toISOString(),
        review_status: "approved" as const,
      },
    };
    set({
      teacherDetails: { ...get().teacherDetails, [id]: optimistic },
    });
    try {
      const updated = await apiApproveRun(id);
      set({
        teacherDetails: {
          ...get().teacherDetails,
          [id]: { ...optimistic, manifest: updated },
        },
      });
    } catch (e) {
      console.error("approve failed", e);
      set({
        teacherDetails: { ...get().teacherDetails, [id]: prev },
      });
      throw e;
    }
  },

  async unapproveTeacher(id) {
    const prev = get().teacherDetails[id];
    if (!prev) return;
    // Optimistic clear. We can't compute the exact "reviewed" vs
    // "unreviewed" fallback locally without inspecting the rejection
    // file, but the server is going to return the right status — assume
    // "unreviewed" optimistically and let the server response correct it
    // if the dataset has rejections (cheap correction; the real source
    // of truth is one HTTP RTT away).
    const optimistic = {
      ...prev,
      manifest: {
        ...prev.manifest,
        approved_at: null,
        review_status: "unreviewed" as const,
      },
    };
    set({
      teacherDetails: { ...get().teacherDetails, [id]: optimistic },
    });
    try {
      const updated = await apiUnapproveRun(id);
      set({
        teacherDetails: {
          ...get().teacherDetails,
          [id]: { ...optimistic, manifest: updated },
        },
      });
    } catch (e) {
      console.error("unapprove failed", e);
      set({
        teacherDetails: { ...get().teacherDetails, [id]: prev },
      });
      throw e;
    }
  },

  async deleteTeacher(id) {
    // Stop polling if active.
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
      await apiDeleteRun(id);
    } catch (e) {
      console.error("delete teacher failed", e);
    }
    // Keep the local state authoritative — refresh from server.
    await get().loadTeachers();
  },

  // ---- Optimize / Students ----

  optimizeError: null,
  studentDetails: {},
  studentPolls: {},
  selectedStudentId: null,
  selectStudent: (id) => set({ selectedStudentId: id }),

  async loadStudents() {
    const resp = await fetchStudents();
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
    set({ optimizeError: null });
    let initial: StudentDetail;
    try {
      initial = await runOptimize(req);
    } catch (err) {
      set({ optimizeError: err instanceof Error ? err.message : String(err) });
      return null;
    }
    set({
      studentDetails: { ...get().studentDetails, [initial.manifest.id]: initial },
      selectedStudentId: initial.manifest.id,
    });
    if (initial.manifest.status === "running") {
      _startStudentPoll(initial.manifest.id);
    }
    return initial.manifest.id;
  },

  async deleteStudent(id) {
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
      await apiDeleteStudent(id);
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

// ---- Poll helpers (live outside the store init so each can re-enter the
// store via useStore.getState()/.setState()) ------------------------------

function _startTeacherPoll(id: string): void {
  const existing = useStore.getState().teacherPolls[id];
  if (existing) return; // already polling

  const handle = setInterval(async () => {
    try {
      const latest = await fetchRunDetail(id);
      useStore.setState((s) => ({
        teacherDetails: { ...s.teacherDetails, [id]: latest },
      }));
      // Keep polling while the run is queued OR running — both are
      // non-terminal. Stop once the manifest hits a terminal state.
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

function _startStudentPoll(id: string): void {
  const existing = useStore.getState().studentPolls[id];
  if (existing) return;

  const handle = setInterval(async () => {
    try {
      const latest = await fetchStudentDetail(id);
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

// Keep these exported for components that want to manually seed a poll
// (e.g. when the page first loads and discovers a still-running run).
export const startTeacherPoll = _startTeacherPoll;
export const startStudentPoll = _startStudentPoll;
