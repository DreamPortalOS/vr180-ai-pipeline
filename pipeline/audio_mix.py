#!/usr/bin/env python3
"""External ambience audio mix-in (issue #396, S-4).

AI-generated sources are silent apart from ``--copy-audio-from`` passthrough.
This module mixes one external ambience track into the finished artefact
(dome + VR180 routes) with pure ffmpeg command construction + execution.

All ffmpeg invocations are subprocess **list** form (no ``shell=True``), so the
command builders are pure functions and unit-testable with mocked
``subprocess.run`` on CI (CPU-only, no models, no network).

Encode contract: video stream is ``-c:v copy`` (never re-encoded),
audio is AAC 192k stereo.

Looping implementation note: ``-stream_loop -1`` on the ambience input never
sends EOF, so a downstream ``areverse`` (used for the duration-independent
fade-out) buffers forever and the run hangs even with ``-shortest`` (measured:
TIMEOUT on ffmpeg N-125258).  The loop therefore lives **in the filtergraph**:
``aloop=loop=-1`` + ``atrim=0:<video-duration>`` + ``asetpts`` — the trim
re-introduces a finite EOF, so ``areverse`` and ``-shortest`` both terminate
(measured: exit 0).  Without a probed duration the fallback is a finite
``aloop=loop=99`` repeat, still clamped by ``-shortest``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("audio-mix")

_FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_CHANNELS = 2
DEFAULT_AUDIO_GAIN_DB = 0.0
DEFAULT_AUDIO_FADE_S = 1.0
#: Fallback repeat count when the video duration cannot be probed.
LOOP_FALLBACK_COUNT = 99


def _probe_duration_s(path: str, ffprobe: str = _FFPROBE_BIN) -> float | None:
    """Return the container duration in seconds, or ``None`` on any failure."""
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return float(json.loads(result.stdout).get("format", {}).get("duration", 0) or 0) or None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def _mix_chain(gain_db: float, loop: bool, fade_s: float, duration_s: float | None = None) -> str:
    """One ambience-track filter chain (no labels, no input index)."""
    parts = [f"volume={gain_db:g}dB"]
    if loop:
        if duration_s and duration_s > 0:
            parts.append("aloop=loop=-1:size=2e9")
            parts.append(f"atrim=0:{duration_s:g}")
            parts.append("asetpts=N/SR/TB")
        else:
            parts.append(f"aloop=loop={LOOP_FALLBACK_COUNT}:size=2e9")
    if fade_s > 0:
        # Duration-independent fade-out: reverse / fade-in / reverse back.
        parts.append(f"afade=t=in:st=0:d={fade_s:g}")
        parts.append("areverse")
        parts.append(f"afade=t=in:st=0:d={fade_s:g}")
        parts.append("areverse")
    return ",".join(parts)


def build_audio_mix_filter(
    *,
    gain_db: float = DEFAULT_AUDIO_GAIN_DB,
    loop: bool = False,
    fade_s: float = DEFAULT_AUDIO_FADE_S,
    dual: bool = False,
    duration_s: float | None = None,
) -> str:
    """Build the ``-filter_complex`` string for the ambience mix.

    Args:
        gain_db: Gain applied to the ambience track (dB, ``volume`` filter).
        loop: Loop the ambience track when it is shorter than the video
            (infinite ``aloop`` trimmed to *duration_s* by ``atrim``, cut to
            video length by ``-shortest``; finite fallback without duration).
        fade_s: Fade in/out length in seconds (``<= 0`` disables both fades).
        dual: ``True`` when a ``--copy-audio-from`` track is also present —
            the copy track passes through (``anull``) and the ambience chain
            is combined with it via ``amix``.
        duration_s: Probed video duration in seconds (loop trim target).

    Returns:
        The filter_complex string, ending in the ``[aout]`` label.
    """
    chain = _mix_chain(gain_db, loop, fade_s, duration_s)
    if not dual:
        return f"[1:a]{chain}[aout]"
    return f"[1:a]anull[a1];[2:a]{chain}[a2];[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[aout]"


def build_audio_mix_command(
    video_path: str,
    mix_path: str,
    out_path: str,
    *,
    copy_audio_from: str | None = None,
    gain_db: float = DEFAULT_AUDIO_GAIN_DB,
    loop: bool = False,
    fade_s: float = DEFAULT_AUDIO_FADE_S,
    duration_s: float | None = None,
    ffmpeg: str = _FFMPEG_BIN,
    ffprobe: str = _FFPROBE_BIN,
) -> list[str]:
    """Build the full ffmpeg argv for the ambience mix (pure function).

    Video is copied (``-c:v copy``); audio is AAC 192k stereo. ``-shortest``
    clamps the output to the video length when the ambience track loops.
    When *loop* is set and *duration_s* is None, the video duration is
    probed via ffprobe (mockable ``_probe_duration_s``) so the in-graph
    ``aloop`` is trimmed back to finite; pass an explicit *duration_s* (or a
    falsy one) to skip the probe.
    """
    if loop and not duration_s:
        probed = _probe_duration_s(video_path, ffprobe=ffprobe)
        duration_s = probed
    dual = copy_audio_from is not None
    filter_complex = build_audio_mix_filter(gain_db=gain_db, loop=loop, fade_s=fade_s, dual=dual, duration_s=duration_s)
    cmd = [ffmpeg, "-y", "-i", video_path]
    if dual:
        cmd += ["-i", copy_audio_from]
    cmd += ["-i", mix_path]
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        AUDIO_CODEC,
        "-b:a",
        AUDIO_BITRATE,
        "-ac",
        str(AUDIO_CHANNELS),
        "-shortest",
        out_path,
    ]
    return cmd


def mix_external_audio(
    video_path: str,
    mix_path: str,
    out_path: str,
    *,
    copy_audio_from: str | None = None,
    gain_db: float = DEFAULT_AUDIO_GAIN_DB,
    loop: bool = False,
    fade_s: float = DEFAULT_AUDIO_FADE_S,
    duration_s: float | None = None,
    ffmpeg: str = _FFMPEG_BIN,
    ffprobe: str = _FFPROBE_BIN,
) -> str:
    """Mix the external ambience track into *video_path* (ffmpeg, list form).

    Writes *out_path* (``video -c:v copy`` + mixed AAC stereo) and returns it.
    Callers mix onto the finished artefact of either route (dome or VR180);
    no route-specific post-processing happens here. When *loop* is set, the
    video duration is probed (one ffprobe call) so the in-graph ``aloop`` is
    trimmed back to finite; pass *duration_s* to skip the probe.

    Raises:
        FileNotFoundError: when the video or an audio input is missing.
        RuntimeError: when ffmpeg exits non-zero.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(mix_path).is_file():
        raise FileNotFoundError(f"Ambience audio not found: {mix_path}")
    if copy_audio_from is not None and not Path(copy_audio_from).is_file():
        raise FileNotFoundError(f"Audio source not found: {copy_audio_from}")

    duration_s = duration_s if not loop else (duration_s or _probe_duration_s(video_path, ffprobe=ffprobe))
    cmd = build_audio_mix_command(
        video_path,
        mix_path,
        out_path,
        copy_audio_from=copy_audio_from,
        gain_db=gain_db,
        loop=loop,
        fade_s=fade_s,
        duration_s=duration_s,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    log.info("Audio mix: video=%s ambience=%s -> %s", video_path, mix_path, out_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed (exit {result.returncode}): {result.stderr.strip()[-500:]}")
    log.info("Audio mix done -> %s", out_path)
    return out_path
