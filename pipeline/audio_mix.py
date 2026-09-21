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
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("audio-mix")

_FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_CHANNELS = 2
DEFAULT_AUDIO_GAIN_DB = 0.0
DEFAULT_AUDIO_FADE_S = 1.0


def _mix_chain(gain_db: float, loop: bool, fade_s: float) -> str:
    """One ambience-track filter chain (no labels, no input index)."""
    parts = [f"volume={gain_db:g}dB"]
    if loop:
        parts.append("aloop=loop=-1:size=2e9")
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
) -> str:
    """Build the ``-filter_complex`` string for the ambience mix.

    Args:
        gain_db: Gain applied to the ambience track (dB, ``volume`` filter).
        loop: Loop the ambience track when it is shorter than the video
            (``aloop``, cut to video length by ``-shortest``).
        fade_s: Fade in/out length in seconds (``<= 0`` disables both fades).
        dual: ``True`` when a ``--copy-audio-from`` track is also present —
            both tracks are normalised and combined with ``amix``.

    Returns:
        The filter_complex string, ending in the ``[aout]`` label.
    """
    chain = _mix_chain(gain_db, loop, fade_s)
    if not dual:
        return f"[1:a]{chain}[aout]"
    return f"[1:a]{chain}[a1];[2:a]{chain}[a2];[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[aout]"


def build_audio_mix_command(
    video_path: str,
    mix_path: str,
    out_path: str,
    *,
    copy_audio_from: str | None = None,
    gain_db: float = DEFAULT_AUDIO_GAIN_DB,
    loop: bool = False,
    fade_s: float = DEFAULT_AUDIO_FADE_S,
    ffmpeg: str = _FFMPEG_BIN,
) -> list[str]:
    """Build the full ffmpeg argv for the ambience mix (pure function).

    Video is copied (``-c:v copy``); audio is AAC 192k stereo. ``-shortest``
    clamps the output to the video length when the ambience track loops.
    """
    dual = copy_audio_from is not None
    filter_complex = build_audio_mix_filter(gain_db=gain_db, loop=loop, fade_s=fade_s, dual=dual)
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
    *,
    copy_audio_from: str | None = None,
    gain_db: float = DEFAULT_AUDIO_GAIN_DB,
    loop: bool = False,
    fade_s: float = DEFAULT_AUDIO_FADE_S,
    ffmpeg: str = _FFMPEG_BIN,
    route: str = "vr180",
) -> str:
    """Mix the external ambience track into *video_path* (ffmpeg, list form).

    Produces a temp file then atomically replaces *video_path*. For the VR180
    route, sv3d/st3d is re-injected afterwards (``-c:v copy`` with a remapped
    audio track drops the sample-entry boxes, issue #91); the dome route
    carries no spherical boxes and skips that step.

    Returns:
        The final output path (== *video_path* after the atomic replace).
    """
    from pipeline.audio_mux import _atomic_replace

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(mix_path).is_file():
        raise FileNotFoundError(f"Ambience audio not found: {mix_path}")
    if copy_audio_from is not None and not Path(copy_audio_from).is_file():
        raise FileNotFoundError(f"Audio source not found: {copy_audio_from}")

    tmp = str(Path(video_path).with_suffix(".amix.mp4"))
    cmd = build_audio_mix_command(
        video_path,
        mix_path,
        tmp,
        copy_audio_from=copy_audio_from,
        gain_db=gain_db,
        loop=loop,
        fade_s=fade_s,
        ffmpeg=ffmpeg,
    )
    log.info("🎧 Audio mix: video=%s ambience=%s → %s", video_path, mix_path, tmp)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed (exit {result.returncode}): {result.stderr.strip()[-500:]}")
    _atomic_replace(tmp, video_path)
    if route == "vr180":
        from pipeline.spherical_injector import inject_spherical_metadata

        log.info("🎧 Re-injecting sv3d/st3d (ffmpeg -c:v copy drops sample-entry boxes)")
        import os

        inject_spherical_metadata(video_path, video_path + ".vr.mp4", stereo_mode="sbs")
        os.replace(video_path + ".vr.mp4", video_path)
    log.info("✅ Audio mix done → %s", video_path)
    return video_path
