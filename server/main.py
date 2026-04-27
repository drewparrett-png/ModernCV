"""FastAPI app — bridges the React Flow GUI to the Python runner.

Endpoints:
    GET  /health           — liveness
    GET  /blocks           — what blocks exist and what impls each supports
    POST /run              — execute a graph; returns summary JSON (v1)
                             (Will become a streaming/websocket endpoint when
                             we want live preview frames.)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.graph import GraphSpec
from pipeline.models.registry import REGISTRY
from pipeline.runner import run as run_graph
from server.schemas import (
    BlockKindInfo,
    BlocksResponse,
    RunRequest,
)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DATA_DIR = Path("data")

app = FastAPI(title="ModernCV", version="0.1.0")

# Permissive in dev — tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/videos")
def videos() -> dict:
    """List video files in data/ — used by the InputNode dropdown.

    Paths are returned relative to the project root so they can be passed
    straight to the OpenCV reader as-is. Sorted alphabetically; nested
    directories (e.g. data/dfl/clip_001.mp4) are walked recursively.
    """
    if not DATA_DIR.exists():
        return {"videos": [], "data_dir": str(DATA_DIR.resolve()), "count": 0}
    found = sorted(
        str(p)
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return {"videos": found, "data_dir": str(DATA_DIR.resolve()), "count": len(found)}


@app.get("/blocks", response_model=BlocksResponse)
def blocks() -> BlocksResponse:
    return BlocksResponse(
        blocks=[
            BlockKindInfo(kind=kind.value, impls=impls)
            for kind, impls in REGISTRY.items()
        ]
    )


@app.post("/run")
def run_endpoint(req: RunRequest) -> dict:
    graph = GraphSpec.from_dict(req.graph.model_dump())
    n_frames = 0
    last_error: str | None = None
    try:
        for _batch in run_graph(graph):
            n_frames += 1
    except NotImplementedError as e:
        # Expected during scaffold — surface which block stub bit us.
        last_error = str(e)
    return {
        "frames_processed": n_frames,
        "error": last_error,
        "graph_node_count": len(graph.nodes),
    }
