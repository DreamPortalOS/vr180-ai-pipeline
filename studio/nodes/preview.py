"""Preview node — passes values through and echoes a short summary for the canvas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode


class PreviewNode(StudioNode):
    type_name = "tool.preview"
    category = "tool"
    label = "预览"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="value", type="any"),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="value", type="any"),
        PortSpec(name="summary", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "label", "type": "string", "default": "preview", "label": "标签"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = work_dir, node_id
        label = str(params.get("label") or "preview")
        value = inputs.get("value")
        if value is None:
            raise ValueError("preview node requires an input on port 'value'")

        kind = type(value).__name__
        detail: Any = value
        if isinstance(value, str) and Path(value).exists():
            kind = "path"
            detail = {
                "path": value,
                "exists": True,
                "suffix": Path(value).suffix.lower(),
                "size_bytes": Path(value).stat().st_size,
            }
        elif isinstance(value, str):
            detail = value if len(value) <= 200 else value[:200] + "…"

        return {
            "value": value,
            "summary": {"label": label, "kind": kind, "detail": detail},
        }
