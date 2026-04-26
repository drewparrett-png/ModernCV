"""FastAPI app — bridges the React Flow GUI to the Python runner.

Endpoints:
    GET  /health           — liveness
    GET  /blocks           — what blocks exist and what impls each supports
    POST /run              — execute a graph; returns summary JSON (v1)
                             (Will become a streaming/websocket endpoint when
                             we want live preview frames.)
"""

from __future__ import annotations

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
