"""Production workflow nodes: storyboard shots → stills → video → concat → audio."""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


# ---------------------------------------------------------------------------
# Script / storyboard
# ---------------------------------------------------------------------------


class ProjectBriefNode(StudioNode):
    type_name = "script.project"
    category = "script"
    label = "项目脚本"
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="brief", type="json"),
        PortSpec(name="style", type="text"),
        PortSpec(name="target", type="text"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "title", "type": "string", "default": "未命名沉浸式短片", "label": "片名"},
            {"name": "theme", "type": "string", "default": "峡谷航拍穿越", "label": "主题"},
            {
                "name": "style",
                "type": "string",
                "default": "cinematic golden hour, photoreal, continuous forward motion",
                "label": "统一风格",
            },
            {"name": "target", "type": "string", "default": "dual", "label": "目标 (dual|dome|vr180)"},
            {"name": "total_seconds", "type": "number", "default": 12, "label": "成片时长(秒)"},
            {"name": "aspect_ratio", "type": "string", "default": "1:1", "label": "画幅"},
            {
                "name": "negative",
                "type": "string",
                "default": "cuts, shake, shallow DOF, text, watermark",
                "label": "全局否定",
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
        _ = inputs, work_dir, node_id
        brief = {
            "title": str(params.get("title") or "untitled"),
            "theme": str(params.get("theme") or ""),
            "style": str(params.get("style") or ""),
            "target": str(params.get("target") or "dual"),
            "total_seconds": float(params.get("total_seconds") or 12),
            "aspect_ratio": str(params.get("aspect_ratio") or "1:1"),
            "negative": str(params.get("negative") or ""),
        }
        return {"brief": brief, "style": brief["style"], "target": brief["target"]}


class ShotListNode(StudioNode):
    """Expand a project brief into a structured multi-shot storyboard."""

    type_name = "script.shot_list"
    category = "script"
    label = "分镜列表"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="brief", type="json", required=True),
        PortSpec(name="style", type="text"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="shots", type="json"),
        PortSpec(name="summary", type="json"),
        PortSpec(name="total_seconds", type="number"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "shot_texts",
                "type": "string",
                "default": (
                    "wide establishing over canyon rim\n"
                    "slow push into the gorge between red walls\n"
                    "emerge into sunlit river bend"
                ),
                "label": "分镜描述（一行一镜）",
            },
            {
                "name": "shot_durations",
                "type": "string",
                "default": "4,4,4",
                "label": "各镜时长(秒)，逗号分隔",
            },
            {
                "name": "motions",
                "type": "string",
                "default": "dolly_in,dolly_in,dolly_in",
                "label": "运动标签，逗号分隔",
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
        brief = inputs.get("brief") or {}
        if not isinstance(brief, dict):
            raise ValueError("shot_list requires brief json input")
        style = str(inputs.get("style") or brief.get("style") or "").strip()
        theme = str(brief.get("theme") or "").strip()
        negative = str(brief.get("negative") or "").strip()
        aspect = str(brief.get("aspect_ratio") or "1:1")

        lines = [ln.strip() for ln in str(params.get("shot_texts") or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("shot_texts must contain at least one shot line")
        if len(lines) > 12:
            raise ValueError("shot_texts max 12 shots for studio canvas")

        raw_durs = [s.strip() for s in str(params.get("shot_durations") or "").split(",") if s.strip()]
        raw_motions = [s.strip() for s in str(params.get("motions") or "").split(",") if s.strip()]

        shots: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            dur = float(raw_durs[i]) if i < len(raw_durs) and raw_durs[i] else 4.0
            dur = max(1.0, min(dur, 10.0))
            motion = raw_motions[i] if i < len(raw_motions) else "dolly_in"
            prompt_parts = [line]
            if theme:
                prompt_parts.append(theme)
            if style:
                prompt_parts.append(style)
            prompt = ", ".join(prompt_parts)
            if negative:
                prompt = f"{prompt}. avoid: {negative}"
            shots.append(
                {
                    "index": i,
                    "id": f"shot_{i + 1:02d}",
                    "description": line,
                    "motion": motion,
                    "duration": dur,
                    "aspect_ratio": aspect,
                    "prompt": prompt,
                }
            )
        total = sum(s["duration"] for s in shots)
        summary = {
            "count": len(shots),
            "total_seconds": total,
            "theme": theme,
            "title": brief.get("title"),
        }
        return {"shots": {"shots": shots, "summary": summary}, "summary": summary, "total_seconds": total}


class PolishShotsNode(StudioNode):
    type_name = "text.polish_shots"
    category = "script"
    label = "批量润色分镜"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="shots", type="json", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="shots", type="json"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "provider", "type": "string", "default": "mock", "label": "provider (mock|litellm)"},
            {"name": "target", "type": "string", "default": "vr180", "label": "目标"},
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
        payload = inputs.get("shots")
        if not isinstance(payload, dict) or "shots" not in payload:
            raise ValueError("polish_shots requires shots json")
        provider = str(params.get("provider") or "mock").lower()
        target = str(params.get("target") or "vr180")
        from studio.nodes.llm import LlmPolishNode

        polished: list[dict[str, Any]] = []
        for shot in payload["shots"]:
            out = LlmPolishNode().run(
                params={"provider": provider, "target": target},
                inputs={"prompt": shot.get("prompt") or shot.get("description") or ""},
                work_dir=work_dir,
                node_id=f"{node_id}_{shot.get('index')}",
            )
            item = dict(shot)
            item["prompt_raw"] = shot.get("prompt")
            item["prompt"] = out["prompt"]
            polished.append(item)
        result = dict(payload)
        result["shots"] = polished
        return {"shots": result, "meta": {"provider": provider, "count": len(polished)}}


# ---------------------------------------------------------------------------
# Storyboard stills (debug before video)
# ---------------------------------------------------------------------------


def _write_placeholder_png(path: Path, *, w: int, h: int, seed: int, label: str) -> None:
    """Write a tiny valid PNG without external deps beyond stdlib+zlib."""
    import zlib

    # Simple gradient + noiseless pattern derived from seed/label so shots differ.
    rows = bytearray()
    hsh = sum(ord(c) for c in label) + seed * 17
    for y in range(h):
        rows.append(0)  # filter none
        for x in range(w):
            r = (x * 255 // max(1, w - 1) + hsh) % 256
            g = (y * 255 // max(1, h - 1) + hsh // 3) % 256
            b = ((x + y + hsh) * 3) % 256
            rows.extend((r & 255, g & 255, b & 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


class BatchStillNode(StudioNode):
    """Generate one still per shot + a contact sheet path for canvas review."""

    type_name = "image.batch_stills"
    category = "generate"
    label = "分镜图批出"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="shots", type="json", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="stills", type="json"),
        PortSpec(name="sheet", type="image"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "width", "type": "number", "default": 320, "label": "图宽"},
            {"name": "height", "type": "number", "default": 320, "label": "图高"},
            {
                "name": "provider",
                "type": "string",
                "default": "mock",
                "label": "provider (mock|file)",
            },
            {
                "name": "source_dir",
                "type": "string",
                "default": "",
                "label": "file 模式：已有分镜图目录",
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
        payload = inputs.get("shots")
        if not isinstance(payload, dict) or "shots" not in payload:
            raise ValueError("batch_stills requires shots json")
        shots = payload["shots"]
        w = max(64, int(params.get("width") or 320))
        h = max(64, int(params.get("height") or 320))
        if w % 2:
            w += 1
        if h % 2:
            h += 1
        provider = str(params.get("provider") or "mock").lower()
        out_dir = Path(work_dir) / "studio_out" / node_id / "stills"
        out_dir.mkdir(parents=True, exist_ok=True)

        stills: list[dict[str, Any]] = []
        if provider == "file":
            src_dir = Path(str(params.get("source_dir") or ""))
            if not src_dir.is_dir():
                raise ValueError(f"source_dir not found: {src_dir}")
            for shot in shots:
                sid = shot.get("id") or f"shot_{shot.get('index')}"
                matches = sorted(src_dir.glob(f"{sid}.*")) + sorted(src_dir.glob(f"*{sid}.*"))
                if not matches:
                    raise FileNotFoundError(f"no still for {sid} in {src_dir}")
                stills.append({**shot, "image": str(matches[0])})
        else:
            for shot in shots:
                sid = shot.get("id") or f"shot_{shot.get('index')}"
                path = out_dir / f"{sid}.png"
                _write_placeholder_png(
                    path, w=w, h=h, seed=int(shot.get("index") or 0), label=str(shot.get("description") or sid)
                )
                stills.append({**shot, "image": str(path)})

        # Contact sheet: horizontal strip via ffmpeg hstack if possible, else first still.
        sheet_path = out_dir / "contact_sheet.png"
        images = [s["image"] for s in stills]
        if len(images) >= 2:
            inputs_args: list[str] = []
            for img in images:
                inputs_args.extend(["-i", img])
            n = len(images)
            # scale each then hstack
            filt = "".join(f"[{i}:v]scale=160:160[s{i}];" for i in range(n))
            filt += "".join(f"[s{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"
            cmd = [
                _FFMPEG,
                "-y",
                *inputs_args,
                "-filter_complex",
                filt,
                "-map",
                "[out]",
                str(sheet_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
            if proc.returncode != 0 or not sheet_path.is_file():
                shutil.copy2(images[0], sheet_path)
        else:
            shutil.copy2(images[0], sheet_path)

        stills_payload = {"shots": stills, "summary": payload.get("summary") or {}}
        meta = {
            "provider": provider,
            "count": len(stills),
            "sheet": str(sheet_path),
            "out_dir": str(out_dir),
            "review": "人工确认分镜图后再进入视频节点",
        }
        return {"stills": stills_payload, "sheet": str(sheet_path), "meta": meta}


class ReviewGateNode(StudioNode):
    """Pass-through checkpoint that surfaces stills for human review on canvas."""

    type_name = "checkpoint.review"
    category = "tool"
    label = "分镜审核门"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="stills", type="json", required=True),
        PortSpec(name="sheet", type="image"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="stills", type="json"),
        PortSpec(name="sheet", type="image"),
        PortSpec(name="report", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "require_ack", "type": "boolean", "default": False, "label": "强制人工确认(否则报错)"},
            {"name": "ack", "type": "boolean", "default": True, "label": "已确认分镜图"},
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
        stills = inputs.get("stills")
        if not isinstance(stills, dict):
            raise ValueError("review gate requires stills json")
        sheet = inputs.get("sheet") or stills.get("sheet")
        shots = stills.get("shots") or []
        require = bool(params.get("require_ack", False))
        ack = bool(params.get("ack", True))
        if require and not ack:
            raise ValueError("分镜审核门：请先确认分镜图（ack=true）再进入视频流程")
        report = {
            "count": len(shots),
            "sheet": sheet,
            "ack": ack,
            "shots": [
                {
                    "id": s.get("id"),
                    "duration": s.get("duration"),
                    "image": s.get("image"),
                    "description": s.get("description"),
                }
                for s in shots
            ],
        }
        return {"stills": stills, "sheet": sheet, "report": report}


# ---------------------------------------------------------------------------
# Video from stills + concat
# ---------------------------------------------------------------------------


class VideosFromStillsNode(StudioNode):
    """Per-shot short clip from each storyboard still (mock I2V; seedance later)."""

    type_name = "video.from_stills"
    category = "generate"
    label = "分镜图→视频"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="stills", type="json", required=True),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="videos", type="json"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "provider", "type": "string", "default": "mock", "label": "provider (mock)"},
            {"name": "fps", "type": "number", "default": 12, "label": "帧率"},
            {"name": "size", "type": "number", "default": 256, "label": "短边像素"},
            {"name": "max_clips", "type": "number", "default": 8, "label": "最多镜头数"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        payload = inputs.get("stills")
        if not isinstance(payload, dict) or "shots" not in payload:
            raise ValueError("from_stills requires stills json")
        shots = payload["shots"]
        max_clips = int(params.get("max_clips") or 8)
        shots = shots[:max_clips]
        fps = int(params.get("fps") or 12)
        size = max(64, int(params.get("size") or 256))
        if size % 2:
            size += 1
        provider = str(params.get("provider") or "mock").lower()
        if provider != "mock":
            raise ValueError(
                "video.from_stills currently supports provider=mock only; use seedance node per-shot for paid I2V"
            )

        out_dir = Path(work_dir) / "studio_out" / node_id / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        clips: list[dict[str, Any]] = []
        for shot in shots:
            still = shot.get("image")
            if not still or not Path(still).is_file():
                raise FileNotFoundError(f"missing still for {shot.get('id')}: {still}")
            dur = float(shot.get("duration") or 4)
            dur = max(1.0, min(dur, 10.0))
            out_path = out_dir / f"{shot.get('id') or 'clip'}.mp4"
            cmd = [
                _FFMPEG,
                "-y",
                "-loop",
                "1",
                "-i",
                str(still),
                "-t",
                str(dur),
                "-vf",
                f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
                "-r",
                str(fps),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
            if proc.returncode != 0 or not out_path.is_file():
                raise RuntimeError(f"clip render failed for {shot.get('id')}: {(proc.stderr or '')[-300:]}")
            clips.append({**shot, "video": str(out_path), "duration": dur})

        videos = {
            "clips": clips,
            "summary": {
                "count": len(clips),
                "total_seconds": sum(c["duration"] for c in clips),
                "fps": fps,
                "size": size,
            },
        }
        return {
            "videos": videos,
            "meta": {"provider": "mock", "count": len(clips), "out_dir": str(out_dir)},
        }


class ConcatVideosNode(StudioNode):
    type_name = "video.concat"
    category = "convert"
    label = "镜头拼合"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="videos", type="json", required=True),)
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
                "default": "concat_demuxer",
                "label": "mode (concat_demuxer)",
            },
            {"name": "filename", "type": "string", "default": "assembled.mp4", "label": "输出文件名"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        payload = inputs.get("videos")
        if not isinstance(payload, dict) or "clips" not in payload:
            raise ValueError("concat requires videos json with clips[]")
        clips = payload["clips"]
        paths = [c["video"] for c in clips if c.get("video")]
        if len(paths) < 1:
            raise ValueError("concat: no clip paths")
        if len(paths) == 1:
            # still copy for stable export path
            out_dir = Path(work_dir) / "studio_out" / node_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / str(params.get("filename") or "assembled.mp4")
            shutil.copy2(paths[0], out_path)
            return {"video": str(out_path), "meta": {"mode": "copy", "inputs": 1}}

        from pipeline.segment_concat import ConcatSegment, concat_segments

        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(params.get("filename") or "assembled.mp4")
        segments = [ConcatSegment(path=Path(p)) for p in paths]
        concat_segments(segments, out_path, mode="demux")
        return {
            "video": str(out_path),
            "meta": {
                "mode": str(params.get("mode") or "concat_demuxer"),
                "inputs": len(paths),
                "path": str(out_path),
                "clips": [{"id": c.get("id"), "duration": c.get("duration")} for c in clips],
            },
        }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class BgmToneNode(StudioNode):
    """Generate a soft sine BGM placeholder (wav) for mux testing."""

    type_name = "audio.bgm_tone"
    category = "audio"
    label = "配乐(占位音)"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="duration", type="number"),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="audio", type="text"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "duration", "type": "number", "default": 12, "label": "时长(秒)"},
            {"name": "freq_hz", "type": "number", "default": 220, "label": "频率 Hz"},
            {"name": "amplitude", "type": "number", "default": 0.15, "label": "音量 0-1"},
            {"name": "filename", "type": "string", "default": "bgm.wav", "label": "文件名"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        duration = float(inputs.get("duration") or params.get("duration") or 12)
        duration = max(1.0, min(duration, 60.0))
        freq = float(params.get("freq_hz") or 220)
        amp = float(params.get("amplitude") or 0.15)
        rate = 22050
        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / str(params.get("filename") or "bgm.wav")
        n = int(duration * rate)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            frames = bytearray()
            for i in range(n):
                # gentle fade in/out to avoid clicks
                t = i / rate
                env = 1.0
                if t < 0.2:
                    env = t / 0.2
                elif t > duration - 0.2:
                    env = max(0.0, (duration - t) / 0.2)
                sample = int(max(-1.0, min(1.0, amp * env * math.sin(2 * math.pi * freq * t))) * 32767)
                frames.extend(struct.pack("<h", sample))
            wf.writeframes(bytes(frames))
        return {
            "audio": str(path),
            "meta": {"path": str(path), "duration": duration, "freq_hz": freq, "kind": "placeholder_bgm"},
        }


class AudioMuxNode(StudioNode):
    type_name = "audio.mux"
    category = "audio"
    label = "音轨合成"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video", required=True),
        PortSpec(name="audio", type="text"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "filename", "type": "string", "default": "with_audio.mp4", "label": "输出文件名"},
            {"name": "volume", "type": "number", "default": 0.8, "label": "配乐音量"},
            {"name": "mode", "type": "string", "default": "replace", "label": "mode (replace|mix)"},
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
        audio = inputs.get("audio")
        if not video:
            raise ValueError("mux requires video input")
        if not audio:
            # passthrough if no audio wired
            return {"video": str(video), "meta": {"mode": "passthrough", "note": "no audio input"}}
        vpath = Path(str(video))
        apath = Path(str(audio))
        if not vpath.is_file():
            raise FileNotFoundError(f"video not found: {vpath}")
        if not apath.is_file():
            raise FileNotFoundError(f"audio not found: {apath}")
        out_dir = Path(work_dir) / "studio_out" / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(params.get("filename") or "with_audio.mp4")
        vol = float(params.get("volume") or 0.8)
        mode = str(params.get("mode") or "replace")
        if mode == "mix":
            # keep original audio if present and mix — for silent mock clips this is same as replace
            cmd = [
                _FFMPEG,
                "-y",
                "-i",
                str(vpath),
                "-i",
                str(apath),
                "-filter_complex",
                f"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0,volume={vol}[a]",
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-shortest",
                str(out_path),
            ]
        else:
            cmd = [
                _FFMPEG,
                "-y",
                "-i",
                str(vpath),
                "-i",
                str(apath),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-af",
                f"volume={vol}",
                "-shortest",
                str(out_path),
            ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"audio mux failed: {(proc.stderr or '')[-400:]}")
        return {
            "video": str(out_path),
            "meta": {"mode": mode, "path": str(out_path), "volume": vol, "audio": str(apath)},
        }
