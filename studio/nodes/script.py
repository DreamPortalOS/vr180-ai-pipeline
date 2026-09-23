"""Storyboard / prompt source node."""

from __future__ import annotations

from typing import Any

from studio.nodes.base import PortSpec, StudioNode


class StoryboardNode(StudioNode):
    type_name = "script.storyboard"
    category = "script"
    label = "分镜脚本"
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="prompt", type="text"),
        PortSpec(name="duration", type="number"),
        PortSpec(name="storyboard", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "title", "type": "string", "default": "untitled shot", "label": "标题"},
            {"name": "prompt", "type": "string", "default": "", "label": "画面描述"},
            {"name": "duration", "type": "number", "default": 5, "label": "时长(秒)"},
            {"name": "aspect_ratio", "type": "string", "default": "1:1", "label": "画幅"},
            {"name": "negative", "type": "string", "default": "", "label": "否定词"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = inputs, work_dir, node_id
        title = str(params.get("title") or "untitled shot")
        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("storyboard prompt must not be empty")
        duration = float(params.get("duration") or 5)
        if duration <= 0:
            raise ValueError("duration must be > 0")
        aspect = str(params.get("aspect_ratio") or "1:1")
        negative = str(params.get("negative") or "")
        storyboard = {
            "title": title,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect,
            "negative": negative,
        }
        return {
            "prompt": prompt,
            "duration": duration,
            "storyboard": storyboard,
        }
