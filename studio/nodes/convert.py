"""Dome conversion + coverage + VR180 CLI nodes for Studio M2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from studio.coverage import analyze_media
from studio.nodes.base import PortSpec, StudioNode


class DomeConvertNode(StudioNode):
    type_name = "convert.dome"
    category = "convert"
    label = "球幕转换"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="video", type="video", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "size", "type": "number", "default": 256, "label": "输出边长 (生产 4096)"},
            {"name": "dome_fov", "type": "number", "default": 180, "label": "鱼眼 FOV°"},
            {"name": "coverage_h", "type": "number", "default": 120, "label": "源水平覆盖°"},
            {"name": "input_projection", "type": "string", "default": "rectilinear", "label": "输入投影"},
            {"name": "pitch", "type": "number", "default": 0, "label": "pitch°"},
            {"name": "yaw", "type": "number", "default": 0, "label": "yaw°"},
            {"name": "roll", "type": "number", "default": 0, "label": "roll°"},
            {"name": "crf", "type": "number", "default": 23, "label": "CRF"},
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
            raise ValueError("dome convert requires a video input")
        src_path = Path(str(src))
        if not src_path.is_file():
            raise FileNotFoundError(f"source not found: {src_path}")

        from pipeline.fulldome_mapper import FulldomeMapper

        size = int(params.get("size") or 256)
        mapper = FulldomeMapper(
            dome_fov=float(params.get("dome_fov") or 180),
            coverage_h_fov=float(params.get("coverage_h") or 120),
            output_size=size,
            crf=int(params.get("crf") or 23),
            input_projection=str(params.get("input_projection") or "rectilinear"),
            pitch=float(params.get("pitch") or 0),
            yaw=float(params.get("yaw") or 0),
            roll=float(params.get("roll") or 0),
        )
        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dome_master.mp4"
        mapper.convert(str(src_path), str(out_path))
        meta = {
            "path": str(out_path),
            "size": size,
            "dome_fov": float(params.get("dome_fov") or 180),
            "coverage_h": float(params.get("coverage_h") or 120),
            "input_projection": str(params.get("input_projection") or "rectilinear"),
            "delivery": "4096² circular domemaster, no warp/split (DECISION_DOME)",
        }
        return {"video": str(out_path), "meta": meta}


class DomeCoverageNode(StudioNode):
    type_name = "qa.dome_coverage"
    category = "qa"
    label = "球幕覆盖度"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="video", type="video", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="report", type="json"),
        PortSpec(name="passed", type="number"),
        PortSpec(name="video", type="video"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "frame_index", "type": "number", "default": 0, "label": "抽帧序号"},
            {"name": "min_deg", "type": "number", "default": 85, "label": "合格天顶角下限"},
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
        media = inputs.get("video")
        if not media:
            raise ValueError("coverage node requires a video/image input")
        stats = analyze_media(str(media), frame_index=int(params.get("frame_index") or 0))
        # Issue #401: ``passed`` follows the shared ``level`` (bad ⇒ 0, warn ⇒ 1
        # with level passed through so the UI can mark it yellow), not a second
        # ``coverage_deg >= min_deg`` threshold that disagreed with ``level`` and
        # let a ``level="bad"`` master read ``passed=True``.
        min_deg = float(params.get("min_deg") or 85)
        passed = 0 if stats.level == "bad" else 1
        report = stats.to_dict()
        report["min_deg"] = min_deg
        report["passed"] = bool(passed)
        return {"report": report, "passed": passed, "video": str(media)}


class Vr180ConvertNode(StudioNode):
    type_name = "convert.vr180"
    category = "convert"
    label = "VR180 转换"
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
                "default": "cli",
                "label": "mode (cli|mock)",
            },
            {"name": "src_hfov", "type": "number", "default": 120, "label": "源水平 FOV°"},
            {"name": "max_disparity", "type": "number", "default": 0.02, "label": "最大视差"},
            {"name": "eye_size", "type": "number", "default": 256, "label": "每眼边长 (生产 2880)"},
            {
                "name": "backend",
                "type": "string",
                "default": "apache",
                "label": "backend (apache|full|auto)",
            },
            {"name": "extra_args", "type": "string", "default": "", "label": "额外 CLI 参数（空格分隔）"},
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
            raise ValueError("vr180 convert requires a video input")
        src_path = Path(str(src))
        if not src_path.is_file():
            raise FileNotFoundError(f"source not found: {src_path}")

        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vr180_sbs.mp4"
        mode = str(params.get("mode") or "cli").lower()

        if mode == "mock":
            # Geometric stand-in: copy source so the canvas can wire export.
            import shutil

            shutil.copy2(src_path, out_path)
            return {
                "video": str(out_path),
                "meta": {
                    "mode": "mock",
                    "path": str(out_path),
                    "note": "mock passthrough — not stereo; use mode=cli for real VR180",
                },
            }

        if mode != "cli":
            raise ValueError(f"unknown vr180 mode {mode!r}; use cli|mock")

        eye = int(params.get("eye_size") or 256)
        if eye % 2:
            eye += 1
        backend = str(params.get("backend") or "apache").lower()
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            "--input",
            str(src_path),
            "--output",
            str(out_path),
            "--src-hfov",
            str(params.get("src_hfov") or 120),
            "--max-disparity",
            str(params.get("max_disparity") or 0.02),
            "--output-width",
            str(eye),
            "--output-height",
            str(eye),
            "--upscale",
            "0",
            "--device",
            "cpu",
            "--max-frames",
            str(params.get("max_frames") or 8),
        ]
        if backend == "apache":
            # Commercial-safe profile (#368): no DepthCrafter / StereoCrafter.
            cmd.extend(
                [
                    "--depth-model",
                    "depth-anything",
                    "--stereo-model",
                    "default",
                    "--no-temporal",
                ]
            )
        elif backend == "full":
            # Advanced backends if deployed; pipeline falls back with a warning.
            cmd.extend(["--depth-model", "depthcrafter", "--stereo-model", "stereocrafter"])
        elif backend != "auto":
            raise ValueError(f"unknown vr180 backend {backend!r}; use apache|full|auto")
        extra = str(params.get("extra_args") or "").strip()
        if extra:
            cmd.extend(extra.split())
        # Never shell=True; list form only (CLAUDE.md boundary).
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(Path.cwd()), timeout=1800)
        if proc.returncode != 0 or not out_path.is_file():
            stderr = (proc.stderr or "")[-800:]
            stdout = (proc.stdout or "")[-400:]
            raise RuntimeError(f"run_pipeline VR180 failed (exit {proc.returncode}): {stderr or stdout}")
        return {
            "video": str(out_path),
            "meta": {
                "mode": "cli",
                "backend": backend,
                "path": str(out_path),
                "eye_size": eye,
                "src_hfov": float(params.get("src_hfov") or 120),
            },
        }
