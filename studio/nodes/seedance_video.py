"""Seedance video node — paid Ark API with mock fallback and cost estimate."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode
from studio.settings import settings_from_params

# Rough operator-facing estimates (元). Not a bill — used only for pre-submit UI.
# Seedance 4k/10s measured ~50 元 (OWNER_BRIEF); MiniMax 2K API ~9.5 元/10s (DECISION_MINIMAX).
_COST_TABLE_YUAN = {
    ("480p", 5): 1.0,
    ("480p", 10): 2.0,
    ("720p", 5): 3.0,
    ("720p", 10): 6.0,
    ("1080p", 5): 12.0,
    ("1080p", 10): 24.0,
    ("4k", 5): 25.0,
    ("4k", 10): 50.0,
    ("minimax-2k", 5): 4.8,
    ("minimax-2k", 10): 9.5,
    ("minimax-768p", 5): 3.0,
    ("minimax-768p", 10): 6.0,
}


def estimate_cost_yuan(resolution: str, duration: int | float) -> int | float | None:
    """Return a rough cost estimate, or None if the tier is unknown."""
    duration_i = int(duration)
    if duration_i <= 5:
        duration_i = 5
    elif duration_i <= 10:
        duration_i = 10
    else:
        duration_i = 10
    return _COST_TABLE_YUAN.get((resolution, duration_i))


class SeedanceVideoNode(StudioNode):
    type_name = "video.seedance"
    category = "generate"
    label = "Seedance 视频"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="prompt", type="text"),
        PortSpec(name="image", type="image"),
        PortSpec(name="duration", type="number"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "provider",
                "type": "string",
                "default": "mock",
                "label": "provider (mock|seedance|minimax)",
            },
            {"name": "model", "type": "string", "default": "doubao-seedance-2-0-fast-260128", "label": "模型"},
            {"name": "duration", "type": "number", "default": 5, "label": "时长(秒)"},
            {"name": "resolution", "type": "string", "default": "480p", "label": "分辨率档"},
            {"name": "ratio", "type": "string", "default": "1:1", "label": "画幅"},
            {"name": "confirm_paid", "type": "boolean", "default": False, "label": "确认付费提交"},
            {"name": "size", "type": "number", "default": 128, "label": "mock 短边像素"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        prompt = str(inputs.get("prompt") or params.get("prompt") or "cinematic immersive shot").strip()
        image = inputs.get("image")
        duration = float(inputs.get("duration") or params.get("duration") or 5)
        provider = str(params.get("provider") or "mock").lower()
        settings = settings_from_params({"seedance_provider": provider})

        if provider == "mock" or (not provider and settings.seedance_provider == "mock"):
            return self._mock_render(prompt, duration, params, work_dir, node_id)

        if provider not in {"seedance", "minimax"}:
            raise ValueError(f"unknown video provider {provider!r}; use mock|seedance|minimax")

        resolution = str(params.get("resolution") or "480p")
        if provider == "minimax":
            # Map studio resolution labels onto MiniMax cost tiers.
            res_key = resolution if resolution in {"minimax-2k", "minimax-768p"} else "minimax-2k"
        else:
            res_key = resolution
        estimate = estimate_cost_yuan(res_key, duration)
        confirm = bool(params.get("confirm_paid", False))
        if not confirm:
            raise ValueError(
                f"paid {provider} submit blocked: set confirm_paid=true after reviewing cost estimate "
                f"(~{estimate} 元 for {res_key}/{duration:.0f}s). Use provider=mock for free canvas runs."
            )

        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ratio = str(params.get("ratio") or "1:1")

        if provider == "minimax":
            if not os.environ.get("MINIMAX_API_KEY"):
                raise ValueError("MINIMAX_API_KEY is not set; configure MiniMax open-platform credentials")
            from integrations.minimax import MiniMaxProvider

            provider_obj = MiniMaxProvider()
            out_path = out_dir / "minimax.mp4"
            model = str(params.get("model") or "")
            kwargs: dict[str, Any] = {"duration": int(duration), "aspect_ratio": ratio}
            if model:
                kwargs["model"] = model
            if res_key.endswith("2k"):
                kwargs.setdefault("resolution", "2k")
            if image:
                result = provider_obj.generate_from_image(str(image), prompt=prompt, **kwargs)
            else:
                result = provider_obj.generate(prompt, **kwargs)
        else:
            if not os.environ.get("ARK_API_KEY") and not settings.ark_api_key:
                raise ValueError("ARK_API_KEY is not set; configure Volcengine Ark credentials")
            from integrations.seedance import SeedanceProvider

            provider_obj = SeedanceProvider(api_key=settings.ark_api_key or None)
            out_path = out_dir / "seedance.mp4"
            model = str(params.get("model") or "doubao-seedance-2-0-fast-260128")
            if image:
                result = provider_obj.generate_from_image(
                    str(image),
                    prompt=prompt,
                    duration=int(duration),
                    aspect_ratio=ratio,
                    resolution=resolution,
                    model=model,
                )
            else:
                result = provider_obj.generate(
                    prompt,
                    duration=int(duration),
                    aspect_ratio=ratio,
                    resolution=resolution,
                    model=model,
                )

        self._download(result.video_url, out_path)
        meta = {
            "provider": provider,
            "job_id": result.job_id,
            "resolution": res_key if provider == "minimax" else resolution,
            "duration": duration,
            "cost_estimate_yuan": estimate,
            "path": str(out_path),
            "metadata": result.metadata,
        }
        return {"video": str(out_path), "meta": meta}

    def _mock_render(
        self,
        prompt: str,
        duration: float,
        params: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        duration = min(max(duration, 0.5), 5.0)
        short_side = int(params.get("size") or 128)
        if short_side % 2:
            short_side += 1
        ratio = str(params.get("ratio") or "1:1")
        try:
            w_s, h_s = (int(x) for x in ratio.split(":", 1))
        except ValueError as exc:
            raise ValueError(f"invalid ratio {ratio!r}") from exc
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

        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "seedance_mock.mp4"
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        src = f"testsrc2=size={width}x{height}:rate=12:duration={duration}"
        cmd = [
            ffmpeg,
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"mock seedance render failed: {(proc.stderr or '')[-400:]}")
        return {
            "video": str(out_path),
            "meta": {
                "provider": "mock",
                "path": str(out_path),
                "width": width,
                "height": height,
                "duration": duration,
                "prompt": prompt,
                "cost_estimate_yuan": 0,
            },
        }

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        import httpx

        with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as res:
            if res.status_code >= 400:
                raise RuntimeError(f"download failed HTTP {res.status_code}")
            with dest.open("wb") as fh:
                for chunk in res.iter_bytes():
                    fh.write(chunk)
