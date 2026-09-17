"""Node implementations for Immersive Node Studio."""

from __future__ import annotations

from studio.nodes.base import StudioNode
from studio.nodes.convert import DomeConvertNode, DomeCoverageNode, Vr180ConvertNode
from studio.nodes.export import ExportBundleNode
from studio.nodes.llm import LlmPolishNode
from studio.nodes.mock_video import MockVideoNode
from studio.nodes.preview import PreviewNode
from studio.nodes.quality import QualityCheckNode
from studio.nodes.registry import NODE_REGISTRY, get_node_class, register_node
from studio.nodes.script import StoryboardNode
from studio.nodes.seedance_video import SeedanceVideoNode

__all__ = [
    "NODE_REGISTRY",
    "DomeConvertNode",
    "DomeCoverageNode",
    "ExportBundleNode",
    "LlmPolishNode",
    "MockVideoNode",
    "PreviewNode",
    "QualityCheckNode",
    "SeedanceVideoNode",
    "StoryboardNode",
    "StudioNode",
    "Vr180ConvertNode",
    "get_node_class",
    "register_node",
]


def _bootstrap() -> None:
    for cls in (
        StoryboardNode,
        LlmPolishNode,
        MockVideoNode,
        SeedanceVideoNode,
        QualityCheckNode,
        DomeConvertNode,
        DomeCoverageNode,
        Vr180ConvertNode,
        PreviewNode,
        ExportBundleNode,
    ):
        register_node(cls)


_bootstrap()
