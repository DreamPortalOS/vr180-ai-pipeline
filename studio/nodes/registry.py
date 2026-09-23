"""Node type registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from studio.nodes.base import StudioNode

NODE_REGISTRY: dict[str, type[StudioNode]] = {}


class UnknownNodeTypeError(KeyError):
    """Raised when a project references an unregistered node type."""


def register_node(cls: type[StudioNode]) -> type[StudioNode]:
    if not cls.type_name:
        raise ValueError(f"{cls.__name__} must set type_name")
    NODE_REGISTRY[cls.type_name] = cls
    return cls


def get_node_class(type_name: str) -> type[StudioNode]:
    try:
        return NODE_REGISTRY[type_name]
    except KeyError as exc:
        raise UnknownNodeTypeError(f"unknown node type {type_name!r}; known: {sorted(NODE_REGISTRY)}") from exc
