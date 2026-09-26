"""Run-mode / locked / cancellation behaviour for ``run_graph`` (issue #419).

These pin the three editor run scopes (全部 / 仅本节点 / 本节点及下游),
node locking (零调用 + 沿用上次输出) and cooperative cancellation at the node
boundary. They use a counting node so the assertions are about *which nodes were
executed*, not what ffmpeg produced — fast and independent of the real backends.
"""

from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from studio.graph import GraphError, run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes.base import PortSpec, StudioNode
from studio.nodes.registry import register_node


@register_node
class _CountingNode(StudioNode):
    """A pure node that records each execution by node id.

    One node type is registered once for the whole session; tests reset the
    shared call log via the autouse fixture. It has an optional ``value`` input
    so it can be both a graph root and a downstream node.
    """

    type_name = "test.counter"
    category = "tool"
    label = "Counter"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="value", type="any", required=False),)
    outputs: tuple[PortSpec, ...] = (PortSpec(name="value", type="any"),)
    calls: ClassVar[list[str]] = []

    def run(self, *, params, inputs, work_dir, node_id) -> dict:
        _CountingNode.calls.append(node_id)
        return {"value": f"out-{node_id}"}


def _chain() -> Project:
    """a -> b -> c, all counting nodes."""
    return Project(
        name="chain",
        nodes=[
            NodeSpec(id="a", type="test.counter", params={}),
            NodeSpec(id="b", type="test.counter", params={}),
            NodeSpec(id="c", type="test.counter", params={}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="a", from_port="value", to_node="b", to_port="value"),
            EdgeSpec(id="e2", from_node="b", from_port="value", to_node="c", to_port="value"),
        ],
    )


@pytest.fixture(autouse=True)
def _reset_calls():
    _CountingNode.calls = []
    yield
    _CountingNode.calls = []


def test_run_mode_all_calls_every_node(tmp_path) -> None:
    last: dict = {}
    report = run_graph(_chain(), work_dir=str(tmp_path), last_outputs=last)
    assert _CountingNode.calls == ["a", "b", "c"]
    assert report.results["a"].status == "ok"
    # last_outputs is populated so a later scoped run can resolve upstream.
    assert last["a"] == {"value": "out-a"}
    assert last["c"] == {"value": "out-c"}


def test_run_mode_node_calls_only_the_target(tmp_path) -> None:
    last: dict = {}
    run_graph(_chain(), work_dir=str(tmp_path), last_outputs=last)  # populate
    _CountingNode.calls = []

    report = run_graph(_chain(), work_dir=str(tmp_path), run_mode="node", run_node="b", last_outputs=last)

    # Only b was executed; a and c were outside the scope and skipped.
    assert _CountingNode.calls == ["b"]
    assert report.results["a"].status == "skipped"
    assert report.results["c"].status == "skipped"
    assert report.results["b"].status == "ok"
    assert report.results["b"].outputs["value"] == "out-b"


def test_run_mode_downstream_calls_target_and_descendants(tmp_path) -> None:
    last: dict = {}
    run_graph(_chain(), work_dir=str(tmp_path), last_outputs=last)  # populate
    _CountingNode.calls = []

    report = run_graph(_chain(), work_dir=str(tmp_path), run_mode="downstream", run_node="b", last_outputs=last)

    # b and its descendant c ran; the upstream a was skipped (its last output
    # fed b without re-executing).
    assert _CountingNode.calls == ["b", "c"]
    assert report.results["a"].status == "skipped"
    assert report.results["b"].status == "ok"
    assert report.results["c"].status == "ok"


def test_run_mode_node_requires_run_node(tmp_path) -> None:
    with pytest.raises(GraphError, match="requires run_node"):
        run_graph(_chain(), work_dir=str(tmp_path), run_mode="node")


def test_unknown_run_mode_rejected(tmp_path) -> None:
    with pytest.raises(GraphError, match="unknown run_mode"):
        run_graph(_chain(), work_dir=str(tmp_path), run_mode="bogus")


def test_locked_node_zero_calls_and_output_retained(tmp_path) -> None:
    last: dict = {}
    run_graph(_chain(), work_dir=str(tmp_path), last_outputs=last)  # populate last outputs
    saved_c = dict(last["c"])
    _CountingNode.calls = []

    project = _chain()
    project.nodes[2].locked = True  # lock c

    report = run_graph(project, work_dir=str(tmp_path), last_outputs=last)

    # Locked => zero executions, but the last output is reused verbatim.
    assert "c" not in _CountingNode.calls
    assert report.results["c"].status == "locked"
    assert report.results["c"].outputs == saved_c
    # a and b still ran normally.
    assert _CountingNode.calls == ["a", "b"]


def test_cancel_marks_pending_node_cancelled_and_preserves_done(tmp_path) -> None:
    """A stop requested after the first node lands on the next pending node as
    `cancelled`; the node that already finished keeps its output in the report.
    """
    cancel = threading.Event()

    def on_status(nid: str, status: str) -> None:
        if nid == "a" and status == "ok":
            cancel.set()  # stop before b runs

    report = run_graph(_chain(), work_dir=str(tmp_path), on_status=on_status, cancel=cancel)

    assert report.results["a"].status == "ok"
    assert report.results["a"].outputs == {"value": "out-a"}  # preserved
    assert report.results["b"].status == "cancelled"
    assert report.results["c"].status == "cancelled"
    # Only a was actually executed; b and c never ran.
    assert _CountingNode.calls == ["a"]


def test_legacy_only_downstream_of_still_works(tmp_path) -> None:
    """The old only_downstream_of path (node + ancestors) is preserved."""
    report = run_graph(_chain(), work_dir=str(tmp_path), only_downstream_of="b")
    assert report.results["a"].status == "ok"
    assert report.results["b"].status == "ok"
    assert report.results["c"].status == "skipped"
