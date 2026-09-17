"""Node implementations for Immersive Node Studio."""

from __future__ import annotations

from studio.nodes.base import StudioNode
from studio.nodes.export import ExportBundleNode
from studio.nodes.mock_video import MockVideoNode
from studio.nodes.preview import PreviewNode
from studio.nodes.registry import NODE_REGISTRY, get_node_class, register_node
from studio.nodes.script import StoryboardNode

# Import for side-effect registration
__all__ = [
    "NODE_REGISTRY",
    "ExportBundleNode",
    "MockVideoNode",
    "PreviewNode",
    "StoryboardNode",
    "StudioNode",
    "get_node_class",
    "register_node",
]


def _bootstrap() -> None:
    for cls in (
        StoryboardNode,
        MockVideoNode,
        PreviewNode,
        ExportBundleNode,
    ):
        register_node(cls)


_bootstrap()
