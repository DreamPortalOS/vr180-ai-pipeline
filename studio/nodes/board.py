"""``storyboard.board`` — the manual storyboard board node (分镜板)."""

from __future__ import annotations

from typing import Any

from studio.board import board_to_stills
from studio.nodes.base import PortSpec, StudioNode


class StoryboardBoardNode(StudioNode):
    """Emit the shots the operator drew in the drawer, in board order.

    Nothing is generated here: text → image and shot → video happen from the
    drawer (``/api/board/*``) so the operator can iterate per shot.  The node
    only hands the picked stills to downstream nodes (``video.from_stills``,
    ``checkpoint.review`` …) in the same ``stills`` shape as
    ``image.batch_stills``.
    """

    type_name = "storyboard.board"
    category = "script"
    label = "分镜板"
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="stills", type="json"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "aspect",
                "type": "string",
                "default": "dome",
                "label": "画幅",
                "options": [["dome", "1:1 穹顶母版"], ["16:9", "16:9 平面"], ["2:1", "2:1 VR180"]],
            },
            {"name": "style", "type": "string", "default": "", "label": "统一风格（附加到每镜 prompt）"},
            {"name": "shots", "type": "json", "default": [], "label": "镜头（在下方分镜板编辑）"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        shots = params.get("shots") or []
        if not isinstance(shots, list):
            raise ValueError("storyboard.board: shots must be a list")
        stills = board_to_stills(shots, str(params.get("aspect") or "dome"))
        if not stills["shots"]:
            raise ValueError("分镜板还没有选定图片的镜头：先在下方分镜板为镜头出图并选图")
        return {
            "stills": stills,
            "meta": {
                "count": len(stills["shots"]),
                "pending": stills["pending"],
                "aspect": stills["summary"]["board_aspect"],
            },
        }
