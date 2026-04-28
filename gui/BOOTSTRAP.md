# GUI Bootstrap Brief — for Claude Code

This document is a self-contained brief for bootstrapping the `gui/` directory.
The rest of the repo (`pipeline/`, `server/`) is already scaffolded and is the
contract this GUI must talk to.

## What you're building

A React Flow node-graph editor that lets a user assemble a video-processing
pipeline by dragging nodes (Input → Detect → Segment → ReID → Track → Stats →
Output), picking an implementation per node from a dropdown, and clicking
"Run" to execute it on the FastAPI backend at `http://localhost:8000`.

This is v1: a single linear chain, no branching, no zoom-to-frame preview. We
will add those once the shell is solid.

## Backend contract (already implemented)

Read `server/main.py` and `server/schemas.py` for the source of truth. Quick
reference:

- `GET /health` → `{"status": "ok"}`
- `GET /blocks` →
  ```json
  {"blocks": [
    {"kind": "input",   "impls": ["opencv", "ffmpeg"]},
    {"kind": "detect",  "impls": ["yolov8n", "yolov8s", "yolov11n", "rtdetr", "groundingdino"]},
    {"kind": "segment", "impls": ["sam2-tiny", "sam2-small", "mobilesam", "fastsam"]},
    {"kind": "reid",    "impls": ["dinov3-vits16", "dinov3-vitb16", "dinov2-vits14", "osnet"]},
    {"kind": "track",   "impls": ["bytetrack", "botsort", "ocsort", "sam2-mask"]},
    {"kind": "stats",   "impls": ["basic", "calibrated", "team-cluster"]},
    {"kind": "output",  "impls": ["overlay-mp4", "json-tracks", "preview"]}
  ]}
  ```
- `POST /run` body:
  ```json
  {
    "graph": {
      "nodes": [
        {"id": "n1", "kind": "input",   "impl": "opencv",        "params": {"path": "data/clip30.mp4"}},
        {"id": "n2", "kind": "detect",  "impl": "yolov8n",       "params": {}},
        {"id": "n3", "kind": "segment", "impl": "sam2-tiny",     "params": {}},
        {"id": "n4", "kind": "reid",    "impl": "dinov3-vits16", "params": {}},
        {"id": "n5", "kind": "track",   "impl": "bytetrack",     "params": {}},
        {"id": "n6", "kind": "stats",   "impl": "basic",         "params": {}},
        {"id": "n7", "kind": "output",  "impl": "overlay-mp4",   "params": {"path": "runs/out.mp4"}}
      ],
      "edges": [["n1","n2"],["n2","n3"],["n3","n4"],["n4","n5"],["n5","n6"],["n6","n7"]]
    }
  }
  ```
  Returns `{"frames_processed": int, "error": str|null, "graph_node_count": int}`.

CORS is already wide-open on the backend, so the GUI can run on Vite's
`localhost:5173` and hit `localhost:8000` directly.

Note: every block currently raises `NotImplementedError` on first use, so a
`/run` call will return an error string. That's expected — the goal of v1 is
to prove the full round-trip works.

## Bootstrap steps

```bash
cd gui
npm create vite@latest . -- --template react-ts
npm install
npm install reactflow zustand
npm run dev
```

After scaffolding, replace boilerplate with the deliverables below. Keep the
CSS minimal — we want clarity over polish at this stage.

## Deliverables

### 1. `src/api.ts`

Tiny client for the backend:

- `getBlocks(): Promise<BlocksResponse>` (GET /blocks)
- `runGraph(graph: GraphSpec): Promise<RunResponse>` (POST /run)

Type the response shapes to match `server/schemas.py`. Base URL from
`import.meta.env.VITE_API_URL` with fallback `http://localhost:8000`.

### 2. `src/store.ts`

zustand store holding:

- `nodes: Node[]` (React Flow nodes — each carries `data: { kind, impl, params }`)
- `edges: Edge[]`
- `availableImpls: Record<BlockKind, string[]>` — fetched once from `/blocks`
- `runResult: RunResponse | null`
- actions: `addNode(kind)`, `updateNodeData(id, patch)`, `setEdges`,
  `runPipeline()` which serializes to GraphSpec and calls the API.

### 3. `src/nodes/`

One custom React Flow node per kind:

- `InputNode` — label, impl `<select>`, text input bound to `params.path`.
- `DetectNode` — label, impl `<select>`, optional confidence slider bound to
  `params.conf` (0.0–1.0, default 0.25).
- `SegmentNode` — label, impl `<select>`, "propagate masks" checkbox bound to
  `params.propagate` (only meaningful for `sam2-*`; show but don't gate).
- `ReIDNode` — label, impl `<select>`, no params yet.
- `TrackNode` — label, impl `<select>`, two number inputs bound to
  `params.high_thresh` (0.5) and `params.low_thresh` (0.1).
- `StatsNode` — label, impl `<select>`, no params yet.
- `OutputNode` — label, impl `<select>`, text input bound to `params.path`.

Every node renders a left target handle (except Input) and a right source
handle (except Output). All nodes pull their impl options from
`store.availableImpls[kind]`.

### 4. `src/App.tsx`

Layout:

```
┌─────────────────────────────────────────────────────┐
│  ModernCV   [Run]                                   │  ← top bar
├──────────┬──────────────────────────────────────────┤
│ Palette  │                                          │
│ + Input  │           React Flow canvas              │
│ + Detect │                                          │
│ + Segment│                                          │
│ + ReID   │                                          │
│ + Track  │                                          │
│ + Stats  │                                          │
│ + Output │                                          │
├──────────┴──────────────────────────────────────────┤
│  Result: { frames_processed: 0, error: "..." }      │  ← results
└─────────────────────────────────────────────────────┘
```

- Palette buttons call `store.addNode(kind)`. New nodes spawn at a sensible
  default position; user wires them up.
- Canvas uses `<ReactFlow nodes={...} edges={...} nodeTypes={...} />` with
  `onNodesChange`, `onEdgesChange`, `onConnect` plumbed through the store.
- Run button calls `store.runPipeline()`; results render in the bottom panel
  as pretty-printed JSON.
- On mount, fetch `/blocks` and populate `availableImpls`.

### 5. Seed graph

On first load (if no nodes exist), seed the canvas with the seven default
nodes wired in a chain so the user immediately sees the intended pipeline.
Defaults match the example POST body above.

## Out of scope for this pass

- Authentication, error toasts (use `console.error`), persistence across
  reloads, branching graphs, video preview pane, mini-map, theming.
  Add later, after the round-trip works.

## Definition of done

1. `npm run dev` starts Vite without errors.
2. With `uvicorn server.main:app --reload --port 8000` running, the canvas
   shows the seeded 7-node pipeline with each node's impl dropdown populated
   from the backend.
3. Editing a dropdown updates store state.
4. Clicking "Run" produces an HTTP POST visible in the backend log; the
   results panel shows the response (which will be a `NotImplementedError`
   from the first stub block — that's the expected end-state for v1).
5. Commit with a clear message; do not commit `node_modules/`.

## Files to read before starting

- `README.md` (project plan, decisions)
- `gui/README.md` (UI map and node-type table)
- `pipeline/blocks/__init__.py` and the individual block files (block kinds
  and AVAILABLE_IMPLS)
- `pipeline/graph.py` (GraphSpec shape)
- `server/schemas.py` (Pydantic models — match these exactly on the wire)
- `server/main.py` (endpoint behavior)
