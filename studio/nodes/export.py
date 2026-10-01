"""Export node — copy artefacts and write a run summary under work_dir."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_paths(params: dict[str, Any], work_dir: str, node_id: str) -> tuple[Path, Path]:
    # Keep the established node directory and basename-only filename behavior.
    filename = Path(str(params.get("filename") or f"{node_id}.mp4")).name
    subdir = Path(str(params.get("subdir") or "studio_export")).name
    out_dir = Path(work_dir) / subdir / node_id
    return out_dir / filename, out_dir / "manifest.json"


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

    @classmethod
    def cache_is_valid(
        cls,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> bool:
        """Reuse only an export whose current files still match the receipt."""
        try:
            dest, manifest_path = _export_paths(params, work_dir, node_id)
            source = Path(str(inputs["video"]))
            manifest = outputs["manifest"]
            digest = manifest["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                return False
            expected = {
                "export_path": str(dest),
                "source_video": str(source),
                "prompt": inputs.get("prompt"),
                "size_bytes": dest.stat().st_size,
                "sha256": digest,
            }
            if outputs["path"] != str(dest) or manifest != {**expected, "manifest_path": str(manifest_path)}:
                return False
            disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            return disk == expected and _sha256(source) == digest and _sha256(dest) == digest
        except (OSError, ValueError, TypeError, KeyError):
            # Missing, old or malformed cache records must rematerialize safely.
            return False

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

        dest, manifest_path = _export_paths(params, work_dir, node_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, dest)

        prompt = inputs.get("prompt")
        manifest = {
            "export_path": str(dest),
            "source_video": str(video_path),
            "prompt": prompt,
            "size_bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return {"path": str(dest), "manifest": manifest}
