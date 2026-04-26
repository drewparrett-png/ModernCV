# ModernCV

A learning project: build a modern computer vision pipeline that segments and tracks
players and the ball in soccer footage, using **DINOv3** as a feature backbone for
re-identification, alongside detection (YOLO), segmentation (SAM2), and tracking
(ByteTrack). Wrapped in a node-graph GUI so each stage is inspectable and tunable.

Informed by an NVIDIA interview on modern CV pipelines.

## Goal

Process a tactical-camera soccer clip end to end:

```
video → detect → segment → re-id (DINOv3) → track → stats → overlay
```

Each stage is a "block" in a visual pipeline. Each block has one or more
implementations (e.g. `Segment` can be SAM2, MobileSAM, or FastSAM) that you swap
from a dropdown. The graph compiles to JSON and a Python runner executes it.

## Where DINOv3 plugs in

DINOv3 is a self-supervised vision **backbone** — it produces strong, generic
image features but is not itself a detector or segmenter. In this pipeline it
sits in the **Re-Identification** block: for each detected player crop we run
DINOv3 to get an embedding, and the tracker uses that embedding as the
appearance feature when matching detections across frames. This is where DINOv3
materially helps over a classical Re-ID head — occlusions, players leaving and
re-entering frame, similar-jersey teammates.

We can also use DINOv3 features for unsupervised team clustering (KMeans on
player embeddings) as a stretch goal.

## Decisions locked

- **Hardware**: Apple Silicon (M-series Mac). PyTorch MPS backend.
- **GUI**: React Flow (frontend) + FastAPI (backend). Graph serializes to JSON;
  Python runner executes it.
- **V1 scope**: GUI shell with stubbed blocks first, then a working tracking
  demo on a short SoccerNet clip — both tracks interleave.

### Apple-Silicon-aware model selections

| Block      | Primary                | Fallback           | Notes                              |
|------------|------------------------|--------------------|------------------------------------|
| Detect     | YOLOv8n / v8s          | YOLOv11n           | Ultralytics, MPS supported         |
| Segment    | SAM2.1-tiny            | MobileSAM, FastSAM | SAM2 has native video propagation  |
| Re-ID      | DINOv3 ViT-S/16        | DINOv3 ViT-B/16    | Skip ViT-L on M-series             |
| Track      | ByteTrack              | BoT-SORT, OC-SORT  | CPU-fine                            |
| Calibrate  | stub for v1            | TVCalib, PnLCalib  | Required for real stats later      |

## Data

We start with one ~30-second clip from **SoccerNet-Tracking**
(<https://soccernet.org>). Free for research after agreeing to terms. Annotations
let us validate every block against ground truth.

Other candidates: SoccerTrack (top-down/tactical), Roboflow Universe (small
detection sets), SkillCorner.

Sample clips go in `data/` which is gitignored.

## Repo layout

```
ModernCV/
├── pipeline/              # Python: blocks and model adapters
│   ├── blocks/            # input, detect, segment, reid, track, stats, output
│   ├── models/            # thin wrappers: yolo.py, sam2.py, dinov3.py, ...
│   ├── graph.py           # graph data structure
│   └── runner.py          # executes a graph spec
├── server/                # FastAPI app: POST /run with {graph, video}
├── gui/                   # React + React Flow
├── data/                  # gitignored — clips, weights
└── notebooks/             # exploration
```

## Roadmap

1. **Scaffold** (this commit): repo skeleton, stubs, pyproject, .gitignore.
2. **GUI shell**: Vite + React Flow with placeholder block nodes, palette,
   dropdown for impl per block, Run button calling FastAPI.
3. **Pipeline skeleton**: Block base class, graph runner that walks nodes and
   passes frames forward.
4. **First model**: YOLO detect block on a short clip, results overlaid in GUI.
5. **SAM2 segment block**: masks rendered as overlays.
6. **DINOv3 re-id block + ByteTrack**: stable IDs across occlusions.
7. **Stats block**: per-player position trace, simple distance-covered metric.
8. **Stretch**: field calibration, team clustering, action spotting.

## Running

(Stubs only at the moment.)

```bash
# Python deps (installs into a uv-managed venv if you have uv)
uv sync          # or: pip install -e .

# Backend
uvicorn server.main:app --reload --port 8000

# Frontend (after gui/ is bootstrapped)
cd gui && npm install && npm run dev
```

## Status

Scaffold only — every block currently raises `NotImplementedError`. See the
roadmap above for what lands next.
