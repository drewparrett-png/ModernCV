# GUI — React Flow + Vite

The pipeline editor. Each node on the canvas is a Block; each Block has a
dropdown of implementations (the "tool chain" feel). The graph serializes to
JSON and is sent to the FastAPI backend at `POST /run`.

## Why React Flow

- Mature node-graph library (selection, panning, edges, mini-map).
- Custom node types — we define one per BlockKind so each can have its own
  controls (impl dropdown, params).
- Easy export to JSON, which is exactly what the runner wants.

## Bootstrap

We didn't commit boilerplate to keep the repo clean. Scaffold with:

```bash
cd gui
npm create vite@latest . -- --template react-ts
npm install
npm install reactflow zustand
npm run dev
```

Then replace `src/App.tsx` with the editor shell that:

1. Calls `GET http://localhost:8000/blocks` on mount → populates a left-side
   palette of available BlockKinds and the impls for each.
2. Renders a React Flow canvas with the seven node types
   (Input, Detect, Segment, ReID, Track, Stats, Output).
3. Each custom node renders its label, an impl `<select>` populated from the
   `/blocks` response, and any params the impl declares.
4. A "Run" button serializes the graph to the same shape as
   `pipeline.graph.GraphSpec` and POSTs it to `/run`.
5. Result viewer panel at the bottom — frames-processed count for v1; later,
   a video preview canvas reading from a websocket.

## Planned node types

| Node      | Default impl       | Special UI                                |
|-----------|--------------------|-------------------------------------------|
| Input     | opencv             | File picker (or path text field)          |
| Detect    | yolov8n            | impl select; confidence slider            |
| Segment   | sam2-tiny          | impl select; "propagate" toggle for SAM2  |
| ReID      | dinov3-vits16      | impl select; embedding-dim readout        |
| Track     | bytetrack          | impl select; high/low score thresholds    |
| Stats     | basic              | impl select; metric checkboxes            |
| Output    | overlay-mp4        | output path; "draw masks" toggle          |

## Why not ComfyUI / Gradio / NiceGUI

We considered all three. ComfyUI gives us a node UI for free but binds us to
its abstractions; Gradio doesn't really do node graphs; NiceGUI is fine but
React Flow gives the best node-graph UX and the round-trip through JSON is
the most instructive for understanding pipeline plumbing.
