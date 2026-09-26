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


#: How many historical results per node are kept for the ◀▶ history switcher.
HISTORY_DEPTH = 5


@dataclass
class RunHistory:
    """Keeps the last ``HISTORY_DEPTH`` results for one node id, newest last.

    Used by the canvas inline-preview arrows (issue #418). Each entry mirrors
    the shape of ``NodeRunResult.to_dict`` minus ``node_id``.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, result: NodeRunResult) -> None:
        self.entries.append(
            {
                "status": result.status,
                "outputs": {k: _jsonable(v) for k, v in result.outputs.items()},
                "error": result.error,
                "cache_hit": result.cache_hit,
            }
        )
        del self.entries[:-HISTORY_DEPTH]

    def to_dict(self) -> dict[str, Any]:
        return list(self.entries)

    def recent(self, n: int = HISTORY_DEPTH) -> list[dict[str, Any]]:
        return self.entries[-n:]


@dataclass
class RunReport:
    order: list[str]
    results: dict[str, NodeRunResult] = field(default_factory=dict)
    history: dict[str, RunHistory] = field(default_factory=dict)

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
            "history": {nid: hist.to_dict() for nid, hist in self.history.items() if hist.entries},
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
    only_downstream_of: str | None = None,
    dirty_from: str | None = None,
    on_status: Callable[[str, str], None] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    history: dict[str, RunHistory] | None = None,
) -> RunReport:
    """Execute the project graph.

    Parameters
    ----------
    project:
        Validated project document.
    work_dir:
        Absolute directory where nodes may write artefacts (tests use tmp_path).
    only_downstream_of:
        Optional node id — run only that node and its **ancestors** (inputs
        needed to produce it). Unrelated/sibling nodes are marked skipped.
    dirty_from:
        Optional node id — force recompute of this node and **descendants**
        by dropping their cache keys for this run; other nodes still use cache.
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
    if history is None:
        history = {}

    run_set: set[str] | None = None
    if only_downstream_of:
        run_set = ancestors_and_self(project, only_downstream_of)

    dirty_set: set[str] = set()
    if dirty_from:
        dirty_set = descendants_and_self(project, dirty_from)

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
        if run_set is not None and nid not in run_set:
            report.results[nid] = NodeRunResult(node_id=nid, status="skipped", outputs={})
            outputs[nid] = {}
            set_status(nid, "skipped")
            continue
        if nid not in history:
            history[nid] = RunHistory()

        upstream: dict[str, Any] = {}
        for port, (src_node, src_port) in incoming[nid].items():
            src_out = outputs.get(src_node, {})
            if src_port not in src_out:
                if run_set is not None and src_node not in run_set:
                    raise GraphError(
                        f"node {nid!r}: upstream {src_node!r} excluded by only_downstream_of={only_downstream_of!r}"
                    )
                raise GraphError(f"node {nid!r}: upstream {src_node!r} has no output {src_port!r}")
            upstream[port] = src_out[src_port]

        key = _cache_key(node.type, node.params, upstream)
        if key in cache and nid not in dirty_set:
            cached = NodeRunResult(
                node_id=nid,
                status="ok",
                outputs=dict(cache[key]),
                cache_hit=True,
            )
            report.results[nid] = cached
            outputs[nid] = dict(cache[key])
            history[nid].append(cached)
            set_status(nid, "ok")
            continue

        set_status(nid, "running")
        try:
            cls = get_node_class(node.type)
            instance = cls()
            result = instance.run(params=node.params, inputs=upstream, work_dir=work_dir, node_id=nid)
        except Exception as exc:
            log.exception("node %s failed", nid)
            failed = NodeRunResult(node_id=nid, status="error", error=str(exc))
            report.results[nid] = failed
            history[nid].append(failed)
            set_status(nid, "error")
            raise GraphError(f"node {nid!r} ({node.type}) failed: {exc}") from exc

        outputs[nid] = result
        cache[key] = dict(result)
        ok = NodeRunResult(node_id=nid, status="ok", outputs=dict(result))
        report.results[nid] = ok
        history[nid].append(ok)
        set_status(nid, "ok")

    report.history = history
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
                        "motion": s.get("motion"),
                        "index": s.get("index"),
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
                            "motion": s.get("motion"),
                            "index": s.get("index"),
                            "source_node": nid,
                        }
                    )

    # Pass-through nodes (e.g. checkpoint.review) re-emit the same stills, so
    # every shot showed up twice in the drawer (lead browser QA, #418). Keep the
    # first occurrence per shot id — the producing node runs first.
    seen: set[Any] = set()
    unique: list[dict[str, Any]] = []
    for s in shots:
        key = s.get("id")
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        unique.append(s)
    shots = unique

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


#: Script/storyboard node types whose canvas cards drive the bottom drawer
#: (issue #418): selecting one lists its shots as cards.
STORYBOARD_NODE_TYPES = frozenset({"script.storyboard", "script.shot_list", "text.polish_shots"})


def reorder_shots(shots: list[dict[str, Any]], order: list[str]) -> list[dict[str, Any]]:
    """Return ``shots`` re-ordered by ``order`` (a permutation of shot ids).

    Ids present in ``order`` come first, in that order; ids not listed keep
    their original relative order at the end. Unknown ids are ignored so a
    stale ``shot_order`` stored in an older project JSON cannot drop shots.
    """
    if not order:
        return list(shots)
    rank = {str(sid): i for i, sid in enumerate(order)}
    missing = [s for s in shots if str(s.get("id")) not in rank]
    known = [s for s in shots if str(s.get("id")) in rank]
    known.sort(key=lambda s: rank[str(s.get("id"))])
    return known + missing
