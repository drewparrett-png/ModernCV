"""Graph data structure.

The GUI emits a JSON spec like:

    {
      "nodes": [
        {"id": "n1", "kind": "input",   "impl": "opencv",       "params": {"path": "data/clip.mp4"}},
        {"id": "n2", "kind": "detect",  "impl": "yolov8n",      "params": {}},
        {"id": "n3", "kind": "segment", "impl": "sam2-tiny",    "params": {}},
        {"id": "n4", "kind": "reid",    "impl": "dinov3-vits16","params": {}},
        {"id": "n5", "kind": "track",   "impl": "bytetrack",    "params": {}},
        {"id": "n6", "kind": "stats",   "impl": "basic",        "params": {}},
        {"id": "n7", "kind": "output",  "impl": "overlay-mp4",  "params": {"path": "runs/out.mp4"}}
      ],
      "edges": [["n1","n2"],["n2","n3"],["n3","n4"],["n4","n5"],["n5","n6"],["n6","n7"]]
    }

For v1 we assume a linear chain (one input, one output, single path). Branching
and merging is a later concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.blocks.base import BlockKind


@dataclass
class NodeSpec:
    id: str
    kind: BlockKind
    impl: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSpec:
    nodes: list[NodeSpec]
    edges: list[tuple[str, str]]

    @classmethod
    def from_dict(cls, data: dict) -> "GraphSpec":
        nodes = [
            NodeSpec(
                id=n["id"],
                kind=BlockKind(n["kind"]),
                impl=n["impl"],
                params=n.get("params", {}),
            )
            for n in data["nodes"]
        ]
        edges = [tuple(e) for e in data["edges"]]
        return cls(nodes=nodes, edges=edges)

    def linearize(self) -> list[NodeSpec]:
        """Topo-sort assuming linear chain. Raises if branching is detected."""
        by_id = {n.id: n for n in self.nodes}
        next_of: dict[str, str] = {}
        prev_of: dict[str, str] = {}
        for src, dst in self.edges:
            if src in next_of:
                raise ValueError(f"node {src!r} has multiple outgoing edges; v1 is linear-only")
            if dst in prev_of:
                raise ValueError(f"node {dst!r} has multiple incoming edges; v1 is linear-only")
            next_of[src] = dst
            prev_of[dst] = src

        roots = [n for n in self.nodes if n.id not in prev_of]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one source node, got {len(roots)}")

        order: list[NodeSpec] = []
        cur: str | None = roots[0].id
        while cur is not None:
            order.append(by_id[cur])
            cur = next_of.get(cur)
        if len(order) != len(self.nodes):
            raise ValueError("graph is disconnected")
        return order
