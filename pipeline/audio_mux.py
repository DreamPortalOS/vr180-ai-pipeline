#!/usr/bin/env python3
"""Lossless audio passthrough utilities (issue #73, H-1).

Generated AI video (e.g. Seedance 2.0) often ships with a synced AAC track
that the VR180 conversion pipeline (rawvideo frames) currently discards. This
module remuxes the source's audio stream into the finished VR180 file with
``-c copy`` — no re-encoding, no dependency additions.

All heavy lifting goes through ffmpeg/ffprobe invoked as a subprocess list
(no ``shell=True``), so the functions are fully unit-testable with mocked
``subprocess.run`` on CI (CPU-only, no models, no network).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("audio-mux")

# FFmpeg/ffprobe binary defaults (overridable via env for CI/test wiring).
_FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"


def _probe_streams(path: str, ffprobe: str = _FFPROBE_BIN) -> dict:
    """Run ffprobe on *path* and return the parsed JSON (capture_output, list cmd)."""
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned non-JSON: {exc}") from None


def has_audio_stream(path: str, ffprobe: str = _FFPROBE_BIN) -> bool:
    """Return ``True`` if *path* contains at least one audio stream.

    Uses ffprobe; does not open the file with a media framework.
    """
    try:
        info = _probe_streams(path, ffprobe=ffprobe)
    except (RuntimeError, ValueError):
        return False
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def audio_stream_info(path: str, ffprobe: str = _FFPROBE_BIN) -> dict | None:
    """Return codec / bit_rate info of the first audio stream, or ``None``.

    Mirrors ``has_audio_stream`` but returns the parsed metadata so callers
    (and QA) can report codec and bitrate.
    """
    try:
        info = _probe_streams(path, ffprobe=ffprobe)
    except (RuntimeError, ValueError):
        return None
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            return {
                "codec_name": s.get("codec_name", "unknown"),
                "bit_rate": s.get("bit_rate"),
                "sample_rate": s.get("sample_rate"),
            }
    return None


def _atomic_replace(src: str, dst: str) -> None:
    """Replace *dst* with *src* atomically (rename). Falls back to copy+unlink
    if rename crosses a device boundary."""
    src_p, dst_p = Path(src), Path(dst)
    try:
        src_p.replace(dst_p)
        return
    except OSError:
        # Cross-device fallback: copy then unlink.
        shutil.copy2(src, dst)
        if src_p.exists():
            src_p.unlink()


def copy_audio_to(
    vr180_path: str,
    audio_source: str,
    ffmpeg: str = _FFMPEG_BIN,
    *,
    shortest: bool = True,
) -> str:
    """Lossless remux: take video from *vr180_path* + audio from *audio_source*.

    Produces a new file (``<vr180_path>.aout.mp4``), then atomically replaces
    *vr180_path*. The remux is ``-c copy`` only — no re-encoding. The sv3d/st3d
    boxes embedded during conversion are preserved because ffmpeg -c copy does
    not touch user-data boxes.

    Args:
        vr180_path:   The VR180 video file (video-only) to attach audio to.
        audio_source: Source file whose audio stream is copied in.
        ffmpeg:       Path to the ffmpeg binary.
        shortest:     If True (default), add ``-shortest`` so the output is
                      clipped to the shorter of the two streams (prevents the
                      VR video from being extended by a long audio tail).

    Returns:
        The final output path (== *vr180_path* after the atomic replace).

    Raises:
        RuntimeError: if ffmpeg exits non-zero.
    """
    if not Path(vr180_path).is_file():
        raise FileNotFoundError(f"VR180 video not found: {vr180_path}")
    if not Path(audio_source).is_file():
        raise FileNotFoundError(f"Audio source not found: {audio_source}")

    tmp = str(Path(vr180_path).with_suffix(".aout.mp4"))
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        vr180_path,
        "-i",
        audio_source,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c",
        "copy",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(tmp)

    log.info("🔊 Audio remux: video=%s audio=%s → %s", vr180_path, audio_source, tmp)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio remux failed (exit {result.returncode}): {result.stderr.strip()[-500:]}")
    _atomic_replace(tmp, vr180_path)
    log.info("✅ Audio remux done → %s", vr180_path)
    return vr180_path
