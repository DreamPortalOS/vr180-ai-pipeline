"""Graph executor — topological run with per-node status for the canvas."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from studio.models import Project, StudioModelError
from studio.nodes.registry import NODE_REGISTRY, get_node_class

log = logging.getLogger(__name__)


class GraphError(StudioModelError):
    """Raised when the graph cannot be executed."""


#: Run-scope selectors for ``run_graph`` (issue #419):
#:   "all"        — execute every non-locked node
#:   "node"       — execute only ``run_node``; its upstream comes from the
#:                  last run's outputs (``last_outputs``)
#:   "downstream" — execute ``run_node`` and all of its descendants
RUN_MODES = frozenset({"all", "node", "downstream"})


@dataclass
class NodeRunResult:
    node_id: str
    status: str  # idle | running | ok | error | skipped | cancelled
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


def ancestors_and_self(project: Project, node_id: str) -> set[str]:
    """Return {node_id} ∪ all upstream ancestors following edges backwards."""
    nodes = {n.id for n in project.nodes}
    if node_id not in nodes:
        raise GraphError(f"unknown node id {node_id!r}")
    parents: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in project.edges:
        parents[edge.to_node].append(edge.from_node)
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parents.get(cur, ()))
    return seen


def descendants_and_self(project: Project, node_id: str) -> set[str]:
    """Return {node_id} ∪ all downstream descendants."""
    nodes = {n.id for n in project.nodes}
    if node_id not in nodes:
        raise GraphError(f"unknown node id {node_id!r}")
    children: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in project.edges:
        children[edge.from_node].append(edge.to_node)
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(children.get(cur, ()))
    return seen


def run_graph(
    project: Project,
    *,
    work_dir: str,
    run_mode: str = "all",
    run_node: str | None = None,
    only_downstream_of: str | None = None,
    dirty_from: str | None = None,
    on_status: Callable[[str, str], None] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    last_outputs: dict[str, dict[str, Any]] | None = None,
    cancel: threading.Event | None = None,
) -> RunReport:
    """Execute the project graph.

    Parameters
    ----------
    project:
        Validated project document.
    work_dir:
        Absolute directory where nodes may write artefacts (tests use tmp_path).
    run_mode:
        One of :data:`RUN_MODES`. ``"all"`` executes every non-locked node;
        ``"node"`` executes only ``run_node`` (its upstream is taken from the
        last run's outputs); ``"downstream"`` executes ``run_node`` and all of
        its descendants. Nodes outside a scoped run are not executed — their
        last output is reused so scoped nodes can still resolve inputs.
    run_node:
        Target node id for ``run_mode`` ``"node"`` / ``"downstream"``.
    only_downstream_of:
        Legacy alias (issue #359): run that node and its **ancestors**.
        Prefer ``run_mode`` for new callers; kept so older clients keep working.
    dirty_from:
        Optional node id — force recompute of this node and **descendants**
        by dropping their cache keys for this run; other nodes still use cache.
    on_status:
        Optional callback ``(node_id, status)`` for live UI updates.
    cache:
        Optional shared dict used as a content-addressed output cache. Pass
        the same dict across runs to skip re-executing unchanged nodes.
    last_outputs:
        Optional shared dict ``node_id → last outputs`` kept across runs. A
        locked node reuses its entry here instead of executing; a scoped run
        resolves upstream from outside the scope via this dict. The server
        keeps one alongside ``cache``.
    cancel:
        Optional :class:`threading.Event`. When set, the next node about to
        execute is marked ``cancelled`` and remaining nodes are not started;
        already-completed outputs are preserved in the report.
    """
    if run_mode not in RUN_MODES:
        raise GraphError(f"unknown run_mode {run_mode!r}; expected one of {sorted(RUN_MODES)}")
    order = topological_order(project)
    nodes = project.node_map()
    incoming = incoming_map(project)
    report = RunReport(order=order)
    outputs: dict[str, dict[str, Any]] = {}
    if cache is None:
        cache = {}
    if last_outputs is None:
        last_outputs = {}

    # Resolve the active run set from the requested mode (issue #419). The
    # legacy only_downstream_of path is preserved as node+ancestors.
    run_set: set[str] | None = None
    if run_mode in ("node", "downstream"):
        if not run_node:
            raise GraphError(f"run_mode={run_mode!r} requires run_node")
        run_set = {run_node} if run_mode == "node" else descendants_and_self(project, run_node)
    elif only_downstream_of:
        run_set = ancestors_and_self(project, only_downstream_of)

    dirty_set: set[str] = set()
    if dirty_from:
        dirty_set = descendants_and_self(project, dirty_from)

    def set_status(nid: str, status: str) -> None:
        if on_status:
            on_status(nid, status)

    for nid in order:
        node = nodes[nid]

        # Locked nodes never execute in any mode; they reuse their last output
        # (issue #419). A locked node feeding a running one supplies that output.
        if node.locked:
            retained = dict(last_outputs.get(nid, {}))
            outputs[nid] = retained
            report.results[nid] = NodeRunResult(node_id=nid, status="locked", outputs=dict(retained))
            set_status(nid, "locked")
            continue

        if node.muted:
            report.results[nid] = NodeRunResult(node_id=nid, status="skipped", outputs={})
            outputs[nid] = {}
            set_status(nid, "skipped")
            continue

        if run_set is not None and nid not in run_set:
            # Outside the requested scope: do not execute, but reuse the last
            # output so a scoped node can still resolve its upstream inputs.
            retained = dict(last_outputs.get(nid, {}))
            outputs[nid] = retained
            report.results[nid] = NodeRunResult(node_id=nid, status="skipped", outputs=dict(retained))
            set_status(nid, "skipped")
            continue

        # Cancellation point (issue #419): a stop requested before this node
        # takes effect here, at the node boundary. The node is marked
        # cancelled and remaining nodes are not started; completed outputs
        # already in the report survive.
        if cancel is not None and cancel.is_set():
            report.results[nid] = NodeRunResult(node_id=nid, status="cancelled", outputs={})
            set_status(nid, "cancelled")
            continue

        upstream: dict[str, Any] = {}
        for port, (src_node, src_port) in incoming[nid].items():
            src_out = outputs.get(src_node, {})
            if src_port not in src_out:
                if run_set is not None and src_node not in run_set:
                    raise GraphError(
                        f"node {nid!r}: upstream {src_node!r} is outside the "
                        f"run scope and has no last output for port {port!r}"
                    )
                raise GraphError(f"node {nid!r}: upstream {src_node!r} has no output {src_port!r}")
            upstream[port] = src_out[src_port]

        key = _cache_key(node.type, node.params, upstream)
        if key in cache and nid not in dirty_set:
            outs = dict(cache[key])
            report.results[nid] = NodeRunResult(
                node_id=nid,
                status="ok",
                outputs=outs,
                cache_hit=True,
            )
            outputs[nid] = outs
            last_outputs[nid] = dict(outs)
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
        last_outputs[nid] = dict(result)
        report.results[nid] = NodeRunResult(node_id=nid, status="ok", outputs=dict(result))
        set_status(nid, "ok")

    return report


def extract_gallery(report: RunReport | dict[str, Any]) -> dict[str, Any] | None:
    """Pull stills/contact-sheet gallery info out of a run report, if any."""
    results: dict[str, Any]
    if isinstance(report, RunReport):
        results = {
            nid: {"status": r.status, "outputs": r.outputs, "error": r.error, "cache_hit": r.cache_hit}
            for nid, r in report.results.items()
        }
    else:
        results = report.get("results") or {}

    shots: list[dict[str, Any]] = []
    sheet = None
    for nid, res in results.items():
        outs = (res or {}).get("outputs") or {}
        if outs.get("sheet"):
            sheet = outs["sheet"]
        stills = outs.get("stills")
        if isinstance(stills, dict) and isinstance(stills.get("shots"), list):
            for s in stills["shots"]:
                shots.append(
                    {
                        "id": s.get("id"),
                        "description": s.get("description"),
                        "image": s.get("image"),
                        "duration": s.get("duration"),
                        "source_node": nid,
                    }
                )
        # also accept top-level list outputs
        if isinstance(outs.get("shots"), list):
            for s in outs["shots"]:
                if isinstance(s, dict) and s.get("image"):
                    shots.append(
                        {
                            "id": s.get("id"),
                            "description": s.get("description"),
                            "image": s.get("image"),
                            "duration": s.get("duration"),
                            "source_node": nid,
                        }
                    )

    if not shots and not sheet:
        return None
    return {"sheet": sheet, "shots": shots, "count": len(shots)}


def list_node_types() -> list[dict[str, Any]]:
    """Catalogue for the frontend node palette."""
    catalog: list[dict[str, Any]] = []
    for type_name in sorted(NODE_REGISTRY):
        meta = NODE_REGISTRY[type_name].describe()
        catalog.append(meta)
    return catalog
