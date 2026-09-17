"""Graph executor — topological run with per-node status for the canvas."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from studio.models import Project, StudioModelError
from studio.nodes.registry import NODE_REGISTRY, get_node_class

log = logging.getLogger(__name__)


class GraphError(StudioModelError):
    """Raised when the graph cannot be executed."""


@dataclass
class NodeRunResult:
    node_id: str
    status: str  # idle | running | ok | error | skipped
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cache_hit: bool = False


@dataclass
class RunReport:
    order: list[str]
    results: dict[str, NodeRunResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "results": {
                nid: {
                    "status": r.status,
                    "outputs": {k: _jsonable(v) for k, v in r.outputs.items()},
                    "error": r.error,
                    "cache_hit": r.cache_hit,
                }
                for nid, r in self.results.items()
            },
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def topological_order(project: Project) -> list[str]:
    """Return node ids in dependency order. Raises GraphError on cycles."""
    project.validate()
    indegree = {n.id: 0 for n in project.nodes}
    adjacency: dict[str, list[str]] = {n.id: [] for n in project.nodes}
    for edge in project.edges:
        indegree[edge.to_node] += 1
        adjacency[edge.from_node].append(edge.to_node)

    ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for nxt in adjacency[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if len(order) != len(project.nodes):
        raise GraphError("graph contains a cycle")
    return order


def incoming_map(project: Project) -> dict[str, dict[str, tuple[str, str]]]:
    """node_id → {to_port: (from_node, from_port)}"""
    incoming: dict[str, dict[str, tuple[str, str]]] = {n.id: {} for n in project.nodes}
    for edge in project.edges:
        incoming[edge.to_node][edge.to_port] = (edge.from_node, edge.from_port)
    return incoming


def _cache_key(node_type: str, params: dict[str, Any], upstream: dict[str, Any]) -> str:
    payload = json.dumps(
        {"type": node_type, "params": params, "upstream": {k: _jsonable(v) for k, v in upstream.items()}},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_graph(
    project: Project,
    *,
    work_dir: str,
    only_downstream_of: str | None = None,
    on_status: Callable[[str, str], None] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
) -> RunReport:
    """Execute the project graph.

    Parameters
    ----------
    project:
        Validated project document.
    work_dir:
        Absolute directory where nodes may write artefacts (tests use tmp_path).
    only_downstream_of:
        Optional node id — when set, only that node and its ancestors run;
        others are marked skipped. M0 implements full-graph runs; this
        argument is reserved for dirty-subgraph runs.
    on_status:
        Optional callback ``(node_id, status)`` for live UI updates.
    cache:
        Optional shared dict used as a content-addressed output cache. Pass
        the same dict across runs to skip re-executing unchanged nodes.
    """
    order = topological_order(project)
    nodes = project.node_map()
    incoming = incoming_map(project)
    report = RunReport(order=order)
    outputs: dict[str, dict[str, Any]] = {}
    if cache is None:
        cache = {}

    def set_status(nid: str, status: str) -> None:
        if on_status:
            on_status(nid, status)

    for nid in order:
        node = nodes[nid]
        if node.muted:
            report.results[nid] = NodeRunResult(node_id=nid, status="skipped", outputs={})
            outputs[nid] = {}
            set_status(nid, "skipped")
            continue

        upstream: dict[str, Any] = {}
        for port, (src_node, src_port) in incoming[nid].items():
            src_out = outputs.get(src_node, {})
            if src_port not in src_out:
                raise GraphError(f"node {nid!r}: upstream {src_node!r} has no output {src_port!r}")
            upstream[port] = src_out[src_port]

        key = _cache_key(node.type, node.params, upstream)
        if key in cache:
            report.results[nid] = NodeRunResult(
                node_id=nid,
                status="ok",
                outputs=dict(cache[key]),
                cache_hit=True,
            )
            outputs[nid] = dict(cache[key])
            set_status(nid, "ok")
            continue

        set_status(nid, "running")
        try:
            cls = get_node_class(node.type)
            instance = cls()
            result = instance.run(params=node.params, inputs=upstream, work_dir=work_dir, node_id=nid)
        except Exception as exc:
            log.exception("node %s failed", nid)
            report.results[nid] = NodeRunResult(node_id=nid, status="error", error=str(exc))
            set_status(nid, "error")
            raise GraphError(f"node {nid!r} ({node.type}) failed: {exc}") from exc

        outputs[nid] = result
        cache[key] = dict(result)
        report.results[nid] = NodeRunResult(node_id=nid, status="ok", outputs=dict(result))
        set_status(nid, "ok")

    return report


def list_node_types() -> list[dict[str, Any]]:
    """Catalogue for the frontend node palette."""
    catalog: list[dict[str, Any]] = []
    for type_name in sorted(NODE_REGISTRY):
        meta = NODE_REGISTRY[type_name].describe()
        catalog.append(meta)
    return catalog
