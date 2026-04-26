"""Pydantic schemas for the FastAPI surface.

These describe the JSON contract between the GUI and the runner. Keep them
narrow — the graph spec is simple, and adding fields piecemeal is fine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeSpecModel(BaseModel):
    id: str
    kind: str  # "input" | "detect" | "segment" | "reid" | "track" | "stats" | "output"
    impl: str
    params: dict[str, Any] = Field(default_factory=dict)


class GraphSpecModel(BaseModel):
    nodes: list[NodeSpecModel]
    edges: list[tuple[str, str]]


class RunRequest(BaseModel):
    graph: GraphSpecModel
    # Input video path is carried in the Input node's params for now; this
    # field exists for future overrides (e.g. uploaded blob path).
    video_path: str | None = None


class BlockKindInfo(BaseModel):
    kind: str
    impls: list[str]


class BlocksResponse(BaseModel):
    blocks: list[BlockKindInfo]
