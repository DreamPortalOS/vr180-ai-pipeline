"""Export node — copy artefacts and write a run summary under work_dir."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode


class ExportBundleNode(StudioNode):
    type_name = "export.bundle"
    category = "export"
    label = "导出"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="prompt", type="text"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="path", type="text"),
        PortSpec(name="manifest", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "filename", "type": "string", "default": "export.mp4", "label": "文件名"},
            {"name": "subdir", "type": "string", "default": "studio_export", "label": "子目录"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        video = inputs.get("video")
        if not video:
            raise ValueError("export node requires a video input")
        video_path = Path(str(video))
        if not video_path.is_file():
            raise ValueError(f"export video not found: {video_path}")

        filename = str(params.get("filename") or f"{node_id}.mp4")
        # Prevent path escape from the export subdirectory.
        safe_name = Path(filename).name
        subdir = str(params.get("subdir") or "studio_export")
        safe_subdir = Path(subdir).name

        out_dir = Path(work_dir) / safe_subdir / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / safe_name
        shutil.copy2(video_path, dest)

        prompt = inputs.get("prompt")
        manifest = {
            "export_path": str(dest),
            "source_video": str(video_path),
            "prompt": prompt,
            "size_bytes": dest.stat().st_size,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return {"path": str(dest), "manifest": manifest}
