"""Studio project model — serialisable node graph (no secrets)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1

#: Canonical port type names used by the canvas and executor.
PORT_TYPES = frozenset({"text", "image", "video", "json", "number", "any"})


class StudioModelError(ValueError):
    """Raised when a project document is structurally invalid."""


@dataclass
class NodeSpec:
    """One node instance on the canvas."""

    id: str
    type: str
    pos: tuple[float, float] = (0.0, 0.0)
    params: dict[str, Any] = field(default_factory=dict)
    muted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "params": dict(self.params),
            "muted": bool(self.muted),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeSpec:
        if not isinstance(data, dict):
            raise StudioModelError("node must be an object")
        node_id = data.get("id")
        node_type = data.get("type")
        if not node_id or not isinstance(node_id, str):
            raise StudioModelError("node.id must be a non-empty string")
        if not node_type or not isinstance(node_type, str):
            raise StudioModelError(f"node {node_id!r}: type must be a non-empty string")
        pos = data.get("pos") or [0.0, 0.0]
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise StudioModelError(f"node {node_id!r}: pos must be [x, y]")
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise StudioModelError(f"node {node_id!r}: params must be an object")
        return cls(
            id=node_id,
            type=node_type,
            pos=(float(pos[0]), float(pos[1])),
            params=dict(params),
            muted=bool(data.get("muted", False)),
        )


@dataclass
class EdgeSpec:
    """A typed link: from_node.from_port → to_node.to_port."""

    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": [self.from_node, self.from_port],
            "to": [self.to_node, self.to_port],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeSpec:
        if not isinstance(data, dict):
            raise StudioModelError("edge must be an object")
        edge_id = data.get("id")
        if not edge_id or not isinstance(edge_id, str):
            raise StudioModelError("edge.id must be a non-empty string")
        src = data.get("from")
        dst = data.get("to")
        if not isinstance(src, (list, tuple)) or len(src) != 2:
            raise StudioModelError(f"edge {edge_id!r}: from must be [nodeId, port]")
        if not isinstance(dst, (list, tuple)) or len(dst) != 2:
            raise StudioModelError(f"edge {edge_id!r}: to must be [nodeId, port]")
        return cls(
            id=edge_id,
            from_node=str(src[0]),
            from_port=str(src[1]),
            to_node=str(dst[0]),
            to_port=str(dst[1]),
        )


@dataclass
class Project:
    """A full studio project document."""

    name: str = "untitled"
    version: int = PROJECT_VERSION
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, NodeSpec]:
        return {n.id: n for n in self.nodes}

    def validate(self) -> None:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise StudioModelError("duplicate node ids")
        id_set = set(ids)
        edge_ids = [e.id for e in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise StudioModelError("duplicate edge ids")
        for edge in self.edges:
            if edge.from_node not in id_set:
                raise StudioModelError(f"edge {edge.id!r}: unknown from node {edge.from_node!r}")
            if edge.to_node not in id_set:
                raise StudioModelError(f"edge {edge.id!r}: unknown to node {edge.to_node!r}")
            if edge.from_node == edge.to_node:
                raise StudioModelError(f"edge {edge.id!r}: self-loop is not allowed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "settings": dict(self.settings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        if not isinstance(data, dict):
            raise StudioModelError("project must be an object")
        version = int(data.get("version", PROJECT_VERSION))
        if version > PROJECT_VERSION:
            raise StudioModelError(f"unsupported project version {version}")
        nodes_raw = data.get("nodes") or []
        edges_raw = data.get("edges") or []
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            raise StudioModelError("nodes/edges must be arrays")
        project = cls(
            name=str(data.get("name") or "untitled"),
            version=version,
            nodes=[NodeSpec.from_dict(n) for n in nodes_raw],
            edges=[EdgeSpec.from_dict(e) for e in edges_raw],
            settings=dict(data.get("settings") or {}),
        )
        project.validate()
        return project

    @classmethod
    def from_json(cls, text: str) -> Project:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StudioModelError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> Project:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json() + "\n", encoding="utf-8")
        return out


def empty_demo_project() -> Project:
    """A tiny default graph: storyboard → mock video → export."""
    return Project(
        name="demo-mock-export",
        nodes=[
            NodeSpec(
                id="n_script",
                type="script.storyboard",
                pos=(40, 80),
                params={
                    "title": "demo walkthrough",
                    "prompt": "slow forward dolly through a calm harbour at dawn",
                    "duration": 2,
                    "aspect_ratio": "1:1",
                },
            ),
            NodeSpec(
                id="n_mock",
                type="video.mock",
                pos=(320, 80),
                params={"duration": 2, "aspect_ratio": "1:1", "fps": 12},
            ),
            NodeSpec(
                id="n_export",
                type="export.bundle",
                pos=(600, 80),
                params={"filename": "studio_demo.mp4"},
            ),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="n_script", from_port="prompt", to_node="n_mock", to_port="prompt"),
            EdgeSpec(id="e2", from_node="n_script", from_port="duration", to_node="n_mock", to_port="duration"),
            EdgeSpec(id="e3", from_node="n_mock", from_port="video", to_node="n_export", to_port="video"),
        ],
        settings={"default_video_provider": "mock", "export": {"vr180": False, "dome": False}},
    )
