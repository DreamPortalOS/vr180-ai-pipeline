#!/usr/bin/env python3
"""K-5 (#149): end-to-end smoke — one command = lead's whole manual acceptance loop.

Every acceptance round tonight the lead hand-repeats the same ritual: run a
real conversion → byte-scan for sv3d/st3d → ffprobe the audio/resolution → run
vr180_qa → diff the log for which backends actually took effect.  This script
freezes that ritual into a single command — and doubles as the tripwire for
"real-run-only" defects: tonight's dozen-plus bugs (lost metadata, lost audio,
silently-ignored backend flags, empty depth dir, wrong weight path) were ALL
green-in-CI, broken-on-real-run, and ALL catchable by one e2e smoke.

Checks (each prints ✅/❌ plus the MEASURED value, not just pass/fail):
  1. pipeline exit code == 0
  2. product exists and ffprobe can read it; resolution matches the quality tier
  3. byte-scan: sv3d/st3d boxes REALLY present in the mp4 (a QA report saying
     so is not enough — tonight we had QA say yes while the bytes said no)
  4. --copy-audio-from: an audio stream is present in the output
  5. log assertion: the backend names actually in effect match the requested
     ``--depth-model`` / ``--stereo-model`` (captured from the
     "🎚️  Streaming backends: depth=..., stereo=..." line) — the two times
     tonight those flags were silently ignored, this was the only signal
  6. sidecar JSON exists and carries the required immersive fields

Any failure prints the measured value + a localisation hint and exits non-zero;
all-pass prints a one-line summary.  ``--json`` emits a machine-readable report
for a future CI hook.

Profiles:
  fast : --quality preview --max-frames 8, pure Depth-Anything, NO heavy models
         (seconds; for CI/daily).  NOTE: preview is the non-streaming legacy
         path, so the streaming-backends log assertion is skipped for it.
         Requires a Depth-Anything weights download from HF on first run —
         on a cache-cold CI runner use ``ci`` instead.
  ci   : NO model at all — ``--force-sbs`` treats the (wide, synthesized) input as
         already-stereo, so depth AND stereo stages are skipped entirely and the
         pipeline runs the real SBS-split → equirect → ffmpeg encode → sv3d/st3d
         injection → sidecar chain.  This exercises the wiring tonight's
         green-in-CI defects lived in (projection, metadata injection, sidecar)
         with zero HF download — stable on a cache-cold CI runner.  The backend
         log assertion is N/A (no depth/stereo backend runs on an SBS input).
  full : --depth-model depthcrafter --stereo-model stereocrafter --comfort safe
         --quality high  (lead's on-machine heavy-model acceptance; NOT for CI)

Usage:
    python scripts/e2e_smoke.py --input video.mp4 --profile fast
    python scripts/e2e_smoke.py --input ci_src.mp4 --profile ci
    python scripts/e2e_smoke.py -i video.mp4 --profile full --copy-audio-from video.mp4
    python scripts/e2e_smoke.py -i video.mp4 --profile fast --json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline.spherical_injector import _STEREO_LEFT_RIGHT, _find_box_recursive
from pipeline.streaming_pipeline import QUALITY_PRESETS

log = logging.getLogger("e2e-smoke")

# ---------------------------------------------------------------------------
# Profile table (module constant — edit here to add/remove profiles)
# ---------------------------------------------------------------------------

#: Shared quality-tier resolution source of truth is pipeline.streaming_pipeline
#: QUALITY_PRESETS; this table only adds the CLI flags + expected backends.
PROFILES: dict[str, dict] = {
    "fast": {
        "description": "preview 档 + 8 帧 + 纯 Depth-Anything（不碰重模型，秒级）",
        "args": ("--quality", "preview", "--max-frames", "8"),
        "expected_depth": "depth-anything",
        "expected_stereo": "default",
        "quality": "preview",
        # fast 走 preview 非 streaming 旧路径 → 无 streaming-backends 日志行，跳过断言
        "skip_backend_log": True,
    },
    "ci": {
        "description": "CI 档：零模型下载 — --force-sbs 走真实 SBS 拆分+投影+编码+注入+sidecar",
        # --force-sbs：把（宽幅合成的）输入当作已立体 SBS，depth/stereo 阶段整体跳过，
        # 真实跑 SBS 拆分 → equirect 投影 → ffmpeg 编码 → sv3d/st3d 注入 → sidecar 全链路，
        # 零 HF 下载，缓存冷的 CI runner 上也稳。小 256²/眼让编码远低于 5 分钟预算。
        "args": (
            "--force-sbs",
            "--quality",
            "preview",
            "--max-frames",
            "8",
            "--output-width",
            "256",
            "--output-height",
            "256",
        ),
        "expected_depth": None,
        "expected_stereo": None,
        "quality": "preview",
        # SBS 输入不跑 depth/stereo 后端 → 无 streaming-backends 日志行，跳过断言
        "skip_backend_log": True,
        # 分辨率断言用显式的 256²/眼覆盖 preview 的 1920²（小片快编，CI <5 分钟）
        "eye_size": 256,
    },
    "full": {
        "description": "DepthCrafter + StereoCrafter + comfort safe + quality high（重模型本机验收）",
        "args": (
            "--depth-model",
            "depthcrafter",
            "--stereo-model",
            "stereocrafter",
            "--comfort",
            "safe",
            "--quality",
            "high",
        ),
        "expected_depth": "depthcrafter",
        "expected_stereo": "stereocrafter",
        "quality": "high",
        "skip_backend_log": False,
    },
}

#: Regex that lifts the effective backend names out of the pipeline log.
#: Emitted by StreamingPipeline.process_stream (I-5, #120).
_BACKENDS_RE = re.compile(r"Streaming backends:\s*depth=([^\s,]+),\s*stereo=([^\s]+)")

#: Default in-repo sample when --input is omitted (git-ignored local asset;
#: the smoke is an on-machine acceptance tool, not a CI job).
DEFAULT_INPUT = Path("video") / "e2e_smoke_sample.mp4"

#: Required immersive-block fields in the sidecar (D-3 contract).
SIDECAR_REQUIRED_IMMERSIVE = ("projection", "fov_deg", "stereo_layout", "eye_resolution")

_SCAN_BOXES = (b"sv3d", b"st3d")


# ---------------------------------------------------------------------------
# Check result + report
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One smoke assertion: name, pass/fail, MEASURED value, localisation hint."""

    name: str
    ok: bool
    measured: str = ""
    hint: str = ""


@dataclass
class SmokeReport:
    """Aggregate result for one e2e_smoke run."""

    input: str
    output: str
    profile: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


# ---------------------------------------------------------------------------
# Individual assertion helpers (pure / injectable — tests drive these directly)
# ---------------------------------------------------------------------------


def build_pipeline_command(
    input_path: str,
    output_path: str,
    profile: str,
    copy_audio_from: str | None = None,
) -> list[str]:
    """Assemble the exact run_pipeline argv for one smoke run (list form)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (choose from {sorted(PROFILES)})")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_pipeline.py")),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        *PROFILES[profile]["args"],
    ]
    if copy_audio_from:
        cmd += ["--copy-audio-from", str(copy_audio_from)]
    return cmd


def check_exit_code(returncode: int) -> Check:
    """1. pipeline exit code == 0."""
    ok = returncode == 0
    return Check(
        "pipeline exit code",
        ok,
        measured=f"exit={returncode}",
        hint="" if ok else "管线本身没跑完 — 先看上面的 run_pipeline 日志栈，别先怀疑断言。",
    )


def _ffprobe_streams(path: str, ffprobe: str = "ffprobe") -> dict:
    """Return ffprobe's parsed JSON for *path* (raises on failure)."""
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # list argv, no shell
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def check_output_probe(
    output_path: str,
    quality: str,
    probe: dict | None = None,
    eye_size: int | None = None,
) -> Check:
    """2. product exists + ffprobe-readable + resolution matches the quality tier.

    ``probe`` is injectable so tests never shell out to ffprobe.  Expected
    frame = (2×eye) × eye for the SBS equirect output.  ``eye_size`` overrides
    the quality-tier default when the profile passed an explicit
    ``--output-width/--output-height`` (the ``ci`` profile uses a small
    256²/eye so the CI encode stays well under the time budget).
    """
    p = Path(output_path)
    if not p.is_file():
        return Check(
            "output exists + ffprobe",
            False,
            measured=f"missing: {output_path}",
            hint="产物根本没落盘 — 管线 exit=0 却没写文件，查 run_pipeline 的输出路径/权限。",
        )
    try:
        info = probe if probe is not None else _ffprobe_streams(output_path)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return Check(
            "output exists + ffprobe",
            False,
            measured=f"ffprobe unreadable: {exc}",
            hint="文件在但 ffprobe 读不出 — 编码器可能中途死掉，看管线 stderr 尾部。",
        )

    width = height = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
            break
    eye = eye_size if eye_size is not None else QUALITY_PRESETS[quality]
    want_w, want_h = eye * 2, eye
    ok = (width, height) == (want_w, want_h)
    return Check(
        "output exists + ffprobe",
        ok,
        measured=f"{width}×{height} (eye={eye}² → want {want_w}×{want_h})",
        hint=(
            ""
            if ok
            else f"分辨率对不上（期望 {eye}²/眼）。"
            "查 output_width/output_height 是否被 --output-width 覆盖或 preset 没生效。"
        ),
    )


def _scan_boxes(path: str) -> dict[str, dict]:
    """Byte-scan the mp4 for sv3d/st3d boxes (mirrors vr180_qa's scanner)."""
    found: dict[str, dict] = {}
    data = bytearray(Path(path).read_bytes())
    for box_type in _SCAN_BOXES:
        offset = _find_box_recursive(data, box_type, 0, len(data))
        if offset == -1:
            continue
        entry: dict = {"offset": offset, "stereo_mode": None}
        if box_type == b"st3d":
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            if size >= 13 and offset + 13 <= len(data):
                entry["stereo_mode"] = data[offset + 12]
        found[box_type.decode("ascii")] = entry
    return found


def check_metadata_bytes(output_path: str, boxes: dict | None = None) -> Check:
    """3. BYTE-SCAN sv3d/st3d really present + st3d mode == left-right.

    This is deliberately a raw byte scan, NOT the QA report — tonight's
    metadata-loss bug had QA saying "present" while the bytes had nothing.
    """
    if boxes is None:
        if not Path(output_path).is_file():
            return Check(
                "sv3d/st3d byte-scan",
                False,
                measured=f"missing: {output_path}",
                hint="产物不存在，无从扫字节。",
            )
        boxes = _scan_boxes(output_path)
    has_sv3d = "sv3d" in boxes
    has_st3d = "st3d" in boxes
    st3d_mode = boxes.get("st3d", {}).get("stereo_mode")
    ok = has_sv3d and has_st3d and st3d_mode == _STEREO_LEFT_RIGHT
    present = ",".join(sorted(boxes)) or "none"
    return Check(
        "sv3d/st3d byte-scan",
        ok,
        measured=f"boxes={present}, st3d_mode={st3d_mode}",
        hint=(
            ""
            if ok
            else "字节里没有 sv3d/st3d（或 st3d 不是 left-right）— 查 spherical_injector "
            "是否真的跑了 / 后续 audio remux 是否把 sample-entry boxes 冲掉（issue #91）。"
        ),
    )


def check_audio_stream(
    output_path: str,
    copy_audio_from: str | None,
    probe: dict | None = None,
) -> Check:
    """4. When --copy-audio-from was passed, an audio stream must be present.

    Not requested → the check passes trivially (recorded as N/A so the report
    shows it was intentionally skipped, not silently dropped).
    """
    if not copy_audio_from:
        return Check("audio stream", True, measured="N/A (no --copy-audio-from)")
    try:
        info = probe if probe is not None else _ffprobe_streams(output_path)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return Check(
            "audio stream",
            False,
            measured=f"ffprobe unreadable: {exc}",
            hint="读不出流信息 — 先看产物是否可 ffprobe。",
        )
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    ok = audio is not None
    codec = audio.get("codec_name", "?") if audio else "none"
    return Check(
        "audio stream",
        ok,
        measured=f"audio={codec} (requested --copy-audio-from={copy_audio_from})",
        hint=(
            ""
            if ok
            else "要求了 --copy-audio-from 但输出没有音轨 — 查 audio_mux 的 remux 分支"
            " / 源文件本身是否真有音轨（ffprobe 源文件确认）。"
        ),
    )


def check_backend_log(
    log_text: str,
    expected_depth: str,
    expected_stereo: str,
) -> Check:
    """5. The backend names actually in effect match the requested ones.

    Scrapes the "🎚️  Streaming backends: depth=..., stereo=..." line.  This
    is the single most valuable assertion tonight: ``--depth-model`` /
    ``--stereo-model`` were silently ignored twice, and this line was the
    only place the truth showed up.
    """
    m = _BACKENDS_RE.search(log_text or "")
    if not m:
        return Check(
            "backend log assertion",
            False,
            measured="no 'Streaming backends' line found in log",
            hint=(
                "日志里抓不到生效后端行 — 要么走了非 streaming 路径（quality=preview 不streaming），"
                "要么 I-5 的 backend 日志被删了。full profile 必须能抓到。"
            ),
        )
    depth_name, stereo_name = m.group(1), m.group(2)
    ok = depth_name == expected_depth and stereo_name == expected_stereo
    return Check(
        "backend log assertion",
        ok,
        measured=f"depth={depth_name}, stereo={stereo_name} (want depth={expected_depth}, stereo={expected_stereo})",
        hint=(
            ""
            if ok
            else "生效后端 ≠ 请求后端 — --depth-model/--stereo-model 被静默忽略或 fallback 了。"
            "查 build_depth_backend / build_stereo_backend 的 fallback WARNING。"
        ),
    )


def check_sidecar(
    output_path: str,
    sidecar: dict | None = None,
) -> Check:
    """6. sidecar JSON exists and carries the D-3 required immersive fields.

    ``sidecar`` is injectable so tests never read the filesystem JSON.
    """
    sidecar_path = Path(output_path).parent / (Path(output_path).stem + ".json")
    if sidecar is None:
        if not sidecar_path.is_file():
            return Check(
                "sidecar JSON",
                False,
                measured=f"missing: {sidecar_path}",
                hint="sidecar 没落盘 — 查 _write_sidecar_from_args 是否被调用 / 是否吞了异常。",
            )
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return Check(
                "sidecar JSON",
                False,
                measured=f"unreadable: {exc}",
                hint="sidecar 在但解析不出 JSON。",
            )
    immersive = sidecar.get("immersive", {}) if isinstance(sidecar, dict) else {}
    missing = [f for f in SIDECAR_REQUIRED_IMMERSIVE if f not in immersive]
    ok = not missing
    return Check(
        "sidecar JSON",
        ok,
        measured=(
            f"immersive fields present ({len(SIDECAR_REQUIRED_IMMERSIVE)}/{len(SIDECAR_REQUIRED_IMMERSIVE)})"
            if ok
            else f"missing immersive fields: {missing}"
        ),
        hint=("" if ok else "sidecar 缺 D-3 必需字段 — 查 normalize_immersive / write_sidecar 的字段装配。"),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_report(report: SmokeReport) -> None:
    """Human-readable per-check ✅/❌ with measured values + hints."""
    icon = {True: "✅", False: "❌"}
    print(f"\nE2E SMOKE — {report.profile} profile")
    print(f"  input : {report.input}")
    print(f"  output: {report.output}")
    print("-" * 64)
    for c in report.checks:
        print(f"{icon[c.ok]} {c.name}: {c.measured}")
        if not c.ok and c.hint:
            print(f"   💡 {c.hint}")
    print("-" * 64)
    if report.ok:
        print(f"✅ ALL {len(report.checks)} CHECKS PASSED ({report.profile})")
    else:
        print(f"❌ {len(report.failed)}/{len(report.checks)} CHECKS FAILED ({report.profile})")


def run_smoke(
    input_path: str,
    output_path: str,
    profile: str,
    copy_audio_from: str | None = None,
    runner=subprocess.run,
) -> SmokeReport:
    """Run one full smoke: pipeline subprocess + all six assertions.

    ``runner`` is injectable so tests drive the assertion logic with a fake
    subprocess result (no real conversion, no ffprobe shell-out for the
    probe-dependent checks — those inject their probe/box/sidecar fixtures
    directly via the check helpers in the test-suite's own unit tests).
    """
    prof = PROFILES[profile]
    report = SmokeReport(input=str(input_path), output=str(output_path), profile=profile)

    cmd = build_pipeline_command(input_path, output_path, profile, copy_audio_from)
    log.info("🚀 %s", " ".join(cmd))
    proc = runner(cmd, capture_output=True, text=True)  # list argv, no shell
    combined_log = (proc.stdout or "") + "\n" + (proc.stderr or "")

    report.checks.append(check_exit_code(proc.returncode))
    report.checks.append(check_output_probe(output_path, prof["quality"], eye_size=prof.get("eye_size")))
    report.checks.append(check_metadata_bytes(output_path))
    report.checks.append(check_audio_stream(output_path, copy_audio_from))
    # The streaming-backends log line is only emitted by StreamingPipeline.
    # fast (preview) is the non-streaming legacy path and ci is a --force-sbs
    # run (no depth/stereo backend executes), so only the streaming (full)
    # profile asserts it.  Non-streaming profiles get an explicit N/A pass so
    # the report shows the check was intentionally skipped, not silently dropped.
    if prof.get("skip_backend_log"):
        report.checks.append(
            Check(
                "backend log assertion",
                True,
                measured=f"N/A ({profile} profile = non-streaming path, no streaming-backends log line)",
            )
        )
    else:
        report.checks.append(check_backend_log(combined_log, prof["expected_depth"], prof["expected_stereo"]))
    report.checks.append(check_sidecar(output_path))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  Accept optional *argv* for testing."""
    parser = argparse.ArgumentParser(
        description="端到端冒烟：一条命令 = 真实转换 + 字节扫描 + ffprobe + 后端日志断言 + sidecar 校验",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_INPUT),
        help=f"输入视频（默认仓内小样 {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="fast",
        help=(
            "fast = preview+8帧+纯Depth-Anything（秒级，日常）；"
            "ci = 零模型下载（--force-sbs 真实 SBS 拆分+投影+编码+注入+sidecar，CI 用）；"
            "full = DepthCrafter+StereoCrafter 重模型（本机验收）"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出产物路径（默认 <input_stem>_e2e_<profile>.mp4）",
    )
    parser.add_argument(
        "--copy-audio-from",
        default=None,
        metavar="PATH",
        help="传入则断言输出含音轨（默认从该文件 remux 音轨进产物）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON 报告")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns 0 = all checks passed, 1 = any failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    input_path = args.input
    if not Path(input_path).is_file():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or str(Path(input_path).with_name(f"{Path(input_path).stem}_e2e_{args.profile}.mp4"))

    report = run_smoke(
        input_path=input_path,
        output_path=output_path,
        profile=args.profile,
        copy_audio_from=args.copy_audio_from,
    )

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
