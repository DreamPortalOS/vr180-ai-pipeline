"""Source quality check node — wraps scripts.check_source_quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode


class QualityCheckNode(StudioNode):
    type_name = "qa.source_quality"
    category = "qa"
    label = "源片质检"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="image", type="image"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="report", type="json"),
        PortSpec(name="passed", type="number"),
        PortSpec(name="video", type="video"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "mode", "type": "string", "default": "auto", "label": "mode (auto|mock)"},
            {"name": "pairs", "type": "number", "default": 6, "label": "抽帧对数"},
            {
                "name": "skip",
                "type": "string",
                "default": "",
                "label": "跳过检查，逗号分隔",
            },
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
        media = inputs.get("video") or inputs.get("image") or params.get("path")
        if not media:
            raise ValueError("quality node requires video or image input")
        path = Path(str(media))
        mode = str(params.get("mode") or "auto").lower()

        if mode == "mock":
            report = {
                "source": str(path),
                "summary": {"pass": 1, "warn": 0, "fail": 0, "skip": 0, "overall": "pass"},
                "exit_code": 0,
                "checks": [
                    {
                        "name": "exists",
                        "status": "pass",
                        "detail": f"mock check on {path.name}",
                        "measured": {"exists": path.is_file()},
                        "advice": "",
                    }
                ],
                "mode": "mock",
            }
            return {"report": report, "passed": 1, "video": str(media)}

        if not path.is_file():
            report = {
                "source": str(path),
                "summary": {"pass": 0, "warn": 0, "fail": 1, "skip": 0, "overall": "fail"},
                "exit_code": 1,
                "checks": [
                    {
                        "name": "exists",
                        "status": "fail",
                        "detail": f"file not found: {path}",
                        "measured": {"exists": False},
                        "advice": "检查上游节点输出路径",
                    }
                ],
            }
            return {"report": report, "passed": 0, "video": str(media)}

        from scripts.check_source_quality import run_checks

        skip_raw = str(params.get("skip") or "").strip()
        skip = tuple(s.strip() for s in skip_raw.split(",") if s.strip())
        pairs = int(params.get("pairs") or 6)
        report_obj = run_checks(path, pairs=pairs, skip=skip)
        report = report_obj.to_dict()
        report["mode"] = "full"
        return {"report": report, "passed": 0 if report_obj.failed else 1, "video": str(media)}
