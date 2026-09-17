"""Mock video node — local placeholder clip without paid APIs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class MockVideoNode(StudioNode):
    type_name = "video.mock"
    category = "generate"
    label = "Mock 视频"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="prompt", type="text"),
        PortSpec(name="duration", type="number"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "duration", "type": "number", "default": 2, "label": "时长(秒)"},
            {"name": "aspect_ratio", "type": "string", "default": "1:1", "label": "画幅"},
            {"name": "fps", "type": "number", "default": 12, "label": "帧率"},
            {"name": "size", "type": "number", "default": 256, "label": "短边像素"},
        ]

    @staticmethod
    def _geometry(aspect_ratio: str, short_side: int) -> tuple[int, int]:
        short_side = max(16, int(short_side))
        if short_side % 2:
            short_side += 1
        try:
            w_s, h_s = (int(x) for x in aspect_ratio.split(":", 1))
        except ValueError as exc:
            raise ValueError(f"invalid aspect_ratio {aspect_ratio!r}") from exc
        if w_s <= 0 or h_s <= 0:
            raise ValueError(f"invalid aspect_ratio {aspect_ratio!r}")
        if w_s >= h_s:
            height = short_side
            width = round(short_side * w_s / h_s)
        else:
            width = short_side
            height = round(short_side * h_s / w_s)
        if width % 2:
            width += 1
        if height % 2:
            height += 1
        return width, height

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        prompt = str(inputs.get("prompt") or params.get("prompt") or "mock shot")
        duration = float(inputs.get("duration") or params.get("duration") or 2)
        if duration <= 0:
            raise ValueError("duration must be > 0")
        # Keep mock runs tiny — Studio M0 is about the canvas, not quality.
        duration = min(duration, 5.0)
        aspect = str(params.get("aspect_ratio") or "1:1")
        fps = int(params.get("fps") or 12)
        short_side = int(params.get("size") or 256)
        width, height = self._geometry(aspect, short_side)

        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mock.mp4"

        src = f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"
        cmd = [
            _FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            src,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-t",
            str(duration),
            str(out_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found on PATH; install ffmpeg for mock video") from exc
        if proc.returncode != 0 or not out_path.is_file():
            stderr = (proc.stderr or "").strip()[-500:]
            raise RuntimeError(f"ffmpeg mock render failed: {stderr}")

        return {
            "video": str(out_path),
            "meta": {
                "path": str(out_path),
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "prompt": prompt,
                "provider": "mock",
            },
        }
