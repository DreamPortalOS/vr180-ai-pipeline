"""Base class and port schema for studio nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PortSpec:
    name: str
    type: str
    required: bool = False


class StudioNode(ABC):
    """A pure-ish unit of work in the studio graph.

    Implementations must not touch the network in tests; work_dir is always
    an absolute path supplied by the caller (tmp_path in tests).
    """

    type_name: str = ""
    category: str = "tool"
    label: str = ""
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "type": cls.type_name,
            "category": cls.category,
            "label": cls.label or cls.type_name,
            "inputs": [{"name": p.name, "type": p.type, "required": p.required} for p in cls.inputs],
            "outputs": [{"name": p.name, "type": p.type} for p in cls.outputs],
            "param_schema": cls.param_schema(),
        }

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        """Lightweight form schema for the inspector (name/type/default/label)."""
        return []

    @abstractmethod
    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        """Execute and return a mapping of output-port → value."""
