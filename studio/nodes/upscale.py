"""SeedVR2 upscale node — mock path for canvas; real CUDA backend optional."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class Seedvr2UpscaleNode(StudioNode):
    type_name = "video.seedvr2"
    category = "convert"
    label = "SeedVR2 超分"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="video", type="video", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "mode",
                "type": "string",
                "default": "mock",
                "label": "mode (mock|seedvr2)",
            },
            {"name": "scale", "type": "number", "default": 2, "label": "放大倍数"},
            {"name": "batch_size", "type": "number", "default": 5, "label": "batch (4n+1)"},
            {"name": "filename", "type": "string", "default": "upscaled.mp4", "label": "输出文件名"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        src = inputs.get("video")
        if not src:
            raise ValueError("seedvr2 node requires video input")
        src_path = Path(str(src))
        if not src_path.is_file():
            raise FileNotFoundError(f"video not found: {src_path}")

        mode = str(params.get("mode") or "mock").lower()
        scale = max(1, int(params.get("scale") or 2))
        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(params.get("filename") or "upscaled.mp4")

        if mode == "mock":
            # Geometry-only stand-in: lanczos scale via ffmpeg (no model weights).
            cmd = [
                _FFMPEG,
                "-y",
                "-i",
                str(src_path),
                "-vf",
                f"scale=iw*{scale}:ih*{scale}:flags=lanczos",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
            if proc.returncode != 0 or not out_path.is_file():
                raise RuntimeError(f"mock upscale failed: {(proc.stderr or '')[-400:]}")
            return {
                "video": str(out_path),
                "meta": {
                    "mode": "mock",
                    "scale": scale,
                    "path": str(out_path),
                    "note": "lanczos geometry mock — not SeedVR2 reconstruction",
                },
            }

        if mode != "seedvr2":
            raise ValueError(f"unknown seedvr2 mode {mode!r}; use mock|seedvr2")

        from pipeline.video_upscaler import SeedVR2Upscaler

        batch = int(params.get("batch_size") or 5)
        try:
            upscaler = SeedVR2Upscaler(batch_size=batch)
        except Exception as exc:  # missing CUDA/deps → clear operator message
            raise RuntimeError(
                f"SeedVR2 backend unavailable: {exc}\n"
                "Use mode=mock on this machine, or run scripts/setup_seedvr2.py on a CUDA host."
            ) from exc

        factor = scale if scale in (2, 3, 4) else 2
        result_path = upscaler.upscale(str(src_path), str(out_path), factor=factor)
        final = Path(result_path) if result_path else out_path
        if not final.is_file():
            raise RuntimeError(f"SeedVR2 produced no file at {final}")
        return {
            "video": str(final),
            "meta": {
                "mode": "seedvr2",
                "scale": factor,
                "batch_size": batch,
                "path": str(final),
            },
        }
