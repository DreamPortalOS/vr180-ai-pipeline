"""Lossless / cross-faded concatenation of pre-generated video segments.

Issue #173 (C-1): the pipeline currently emits one 5-second clip at a time,
and the owner's feedback is that short, discontinuous clips look jarring.
Before doing longer single-clip generation we want a layer that splices
multiple already-generated flat (pre-VR180) clips into one continuous
sequence. This stays **above** the VR180 conversion so the whole joined
video can be converted in one pass.

This module is a thin ffmpeg/ffprobe wrapper — no new dependencies:

  * ``probe_segment(path)``  -> {width, height, fps, duration, has_audio}
  * ``check_compatible(segments)``  -> raises ``ConcatError`` on mismatch
  * ``concat_segments(segments, out, *, mode, crossfade, encoder, runner)``

All subprocess calls are list-form (never ``shell=True`` — CLAUDE.md red line)
and take an injectable ``runner`` so tests can assert the built command
without touching real ffmpeg.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.streaming_pipeline import select_encoder

log = logging.getLogger("segment-concat")

# ffmpeg / ffprobe binary defaults (overridable for CI/test wiring).
_FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

# When a segment carries an explicit start/end clip, we express the cut in the
# concat demuxer file as ``file path.mp4`` preceded by an ``in`` and ``out``
# duration (ffmpeg concat protocol).
# The concat demuxer file itself lives in the system temp dir (``tempfile``) so
# it never lands in the repo or under ``video/``.
_CONCAT_DIR_PREFIX = "vr180-concat-"


class ConcatError(RuntimeError):
    """A segment is incompatible with the rest of the concat set."""


@dataclass(frozen=True)
class ConcatSegment:
    """One input clip plus optional in/out trim points (seconds).

    ``start`` / ``end`` are ``None`` when the whole clip is kept.  When given,
    they mark the trim window ``[start, end)`` in seconds.
    """

    path: Path
    start: float | None = None
    end: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.start is not None and self.start < 0:
            raise ValueError("segment start must be >= 0")
        if self.end is not None and self.end < 0:
            raise ValueError("segment end must be >= 0")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("segment end must be > start")


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


def _probe_json(path: Path, ffprobe: str = _FFPROBE_BIN) -> dict:
    """Run ffprobe on *path* and return the parsed JSON, or ``{}`` on failure."""
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise ConcatError(f"ffprobe failed on {path} (exit {result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ConcatError(f"ffprobe returned non-JSON for {path}: {exc}") from None


def probe_segment(path: Path, ffprobe: str = _FFPROBE_BIN) -> dict:
    """Probe a segment with ffprobe and return key metadata.

    Returns a dict with the keys ``width``, ``height``, ``fps``,
    ``duration`` (seconds, ``None`` if unavailable) and ``has_audio``.
    """
    path = Path(path)
    info = _probe_json(path, ffprobe=ffprobe)

    streams = info.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if video_stream is None:
        raise ConcatError(f"no video stream in {path}")

    width = video_stream.get("width")
    height = video_stream.get("height")
    fps = _parse_fps(video_stream.get("r_frame_rate"))
    duration = _parse_float(info.get("format", {}).get("duration"))
    for required, value in (("width", width), ("height", height), ("fps", fps)):
        if value is None:
            raise ConcatError(f"could not determine {required} for {path}")

    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "duration": duration,
        "has_audio": has_audio,
    }


def _parse_fps(r_frame_rate: str | None) -> float | None:
    """Parse ffprobe's ``r_frame_rate`` (e.g. ``"30/1"`` or ``"25/1"``)."""
    if not r_frame_rate:
        return None
    if "/" in r_frame_rate:
        try:
            num, den = r_frame_rate.split("/")
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except (ValueError, TypeError):
            return None
    try:
        return float(r_frame_rate)
    except (ValueError, TypeError):
        return None


def _parse_float(value: str | float | int | None) -> float | None:
    """Coerce a ffprobe numeric field to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Compatibility check
# --------------------------------------------------------------------------- #


def check_compatible(
    segments: Sequence[ConcatSegment],
    ffprobe: str = _FFPROBE_BIN,
) -> None:
    """Verify all segments share the same resolution and fps.

    Raises :class:`ConcatError` when any segment disagrees, with a message
    that lists each offending segment's actual width/height/fps so the
    caller can pinpoint which clip is the odd one out.
    """
    if len(segments) < 1:
        raise ConcatError("concat requires at least one segment")

    probes: list[tuple[Path, dict]] = []
    for seg in segments:
        probes.append((seg.path, probe_segment(seg.path, ffprobe=ffprobe)))

    first = probes[0]
    ref_width = first[1]["width"]
    ref_height = first[1]["height"]
    ref_fps = first[1]["fps"]
    ref_label = _describe(first[1])

    mismatches: list[str] = []
    for path, meta in probes[1:]:
        actual = _describe(meta)
        if meta["width"] != ref_width or meta["height"] != ref_height or meta["fps"] != ref_fps:
            mismatches.append(f"  {path}: {actual} (expected {ref_label})")

    if mismatches:
        ref_path = probes[0][0]
        raise ConcatError(
            "incompatible segment resolution/fps for concat:\n"
            f"  reference {ref_path}: {ref_label}\n" + "\n".join(mismatches)
        )


def _describe(meta: dict) -> str:
    fps = meta["fps"]
    fps_str = str(int(fps)) if abs(fps - round(fps)) < 1e-9 else f"{fps:.3f}"
    return f"{meta['width']}x{meta['height']} @ {fps_str}fps"


# --------------------------------------------------------------------------- #
# Concat
# --------------------------------------------------------------------------- #


def concat_segments(
    segments: Sequence[ConcatSegment],
    output_path: Path,
    *,
    mode: str = "demux",
    crossfade: float = 0.0,
    encoder: str | None = None,
    runner: Callable[[list[str]], object] | None = None,
    ffmpeg: str = _FFMPEG_BIN,
) -> Path:
    """Concatenate pre-generated segments into one continuous video.

    ``mode="demux"`` (default) runs ffmpeg's concat demuxer with ``-c copy``
    — zero re-encoding, fast, and lossless, but it requires every segment to
    share the same codec and encoding parameters (resolution/fps are checked
    up front by :func:`check_compatible`).

    ``mode="filter"`` (or any ``crossfade > 0``) goes through
    ``filter_complex`` so the segments can be re-encoded and, when
    ``crossfade > 0``, joined with a smooth ``xfade`` video + ``acrossfade``
    audio transition.

    Args:
        segments: Ordered clips to join. Resolution/fps must all match.
        output_path: Destination file.
        mode: ``"demux"`` (lossless copy) or ``"filter"`` (re-encode).
        crossfade: Transition length in seconds (>0 forces filter mode).
        encoder: Override the re-encoding encoder args
            (``["-c:v", "libx265"]`` for example). ``None`` reuses
            :func:`pipeline.streaming_pipeline.select_encoder`.
        runner: Injected command runner (default :func:`subprocess.run`).
            Tests pass a fake to assert the built command.
        ffmpeg: Path to the ffmpeg binary.

    Returns:
        ``output_path``.

    Raises:
        ConcatError: on incompatible segments or ffmpeg failure.
    """
    if mode not in ("demux", "filter"):
        raise ConcatError(f"unknown concat mode {mode!r}; choose 'demux' or 'filter'")
    if crossfade < 0:
        raise ConcatError("crossfade must be >= 0")
    if crossfade > 0 and mode == "demux":
        log.info("crossfade>0 forces filter mode")
        mode = "filter"

    output_path = Path(output_path)
    check_compatible(segments)

    runner = runner or subprocess.run

    if mode == "demux":
        return _concat_demux(segments, output_path, runner=runner, ffmpeg=ffmpeg)
    return _concat_filter(
        segments,
        output_path,
        crossfade=crossfade,
        encoder=encoder,
        runner=runner,
        ffmpeg=ffmpeg,
    )


# --------------------------------------------------------------------------- #
# Demux path (-f concat, -c copy)
# --------------------------------------------------------------------------- #


def _concat_demux(
    segments: Sequence[ConcatSegment],
    output_path: Path,
    *,
    runner: Callable[[list[str]], object],
    ffmpeg: str,
) -> Path:
    """Lossless concat via the concat demuxer. No re-encoding."""
    with tempfile.TemporaryDirectory(prefix=_CONCAT_DIR_PREFIX) as tmp:
        list_path = Path(tmp) / "list.txt"
        _write_concat_list(segments, list_path)
        log.info(
            "🧩 [concat demux] %d segment(s) via %s → %s",
            len(segments),
            list_path,
            output_path,
        )

        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = runner(cmd)
        if getattr(result, "returncode", 0) != 0:
            raise ConcatError(f"concat demux failed (exit {getattr(result, 'returncode', -1)})")
    return output_path


def _write_concat_list(segments: Sequence[ConcatSegment], list_path: Path) -> None:
    """Write a concat demuxer ``list.txt``.

    The demuxer file format is one ``file <path>`` per line, optionally
    preceded by ``duration``, ``in`` / ``out`` markers for clip trimming
    (https://ffmpeg.org/ffmpeg-formats.html#concat-1).
    """
    lines: list[str] = []
    for seg in segments:
        parts: list[str] = []
        if seg.start is not None:
            parts.append(f"in {seg.start}")
        if seg.end is not None:
            parts.append(f"out {seg.end}")
        parts.append(f"file '{seg.path}'")
        lines.append(" ".join(parts))
    # ffprobe reads the file; use the repo-local encoding.
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Filter path (filter_complex, optional xfade / acrossfade)
# --------------------------------------------------------------------------- #


def _concat_filter(
    segments: Sequence[ConcatSegment],
    output_path: Path,
    *,
    crossfade: float,
    encoder: str | None,
    runner: Callable[[list[str]], object],
    ffmpeg: str,
) -> Path:
    """Re-encode concat via filter_complex.

    When ``crossfade <= 0`` this is a plain ``concat`` video (+ audio) filter.
    When ``crossfade > 0`` we use ``xfade`` (video) + ``acrossfade`` (audio)
    with transition offset computed so the total duration equals
    ``sum(durations) - (n-1) * crossfade``.
    """
    probes = [probe_segment(seg.path) for seg in segments]
    all_have_audio = all(p["has_audio"] for p in probes)

    width = probes[0]["width"]
    fps = probes[0]["fps"]
    encoder_args = encoder if encoder is not None else select_encoder("h264", width)

    input_args: list[str] = []
    filter_parts: list[str] = []
    for _i, seg in enumerate(segments):
        input_args += [
            "-ss",
            str(seg.start or 0),
            "-i",
            str(seg.path),
        ]

    if len(segments) == 1:
        # Trivial case: one segment, no join needed — still re-encode through
        # ffmpeg so the same encoder args apply.
        input_args += ["-c:v", encoder_args[1]]
        if all_have_audio:
            input_args.append("-c:a")
            input_args.append("aac")
        input_args.append("-movflags")
        input_args.append("+faststart")
    else:
        has_audio_filter = all_have_audio
        if crossfade > 0:
            v_out, a_out, offsets = _build_xfade(segments, probes, crossfade, fps, has_audio=has_audio_filter)
            filter_parts = [v_out]
            if has_audio_filter:
                filter_parts.append(a_out)
            offsets_input_args = _build_offsets_input(offsets, input_args)
            input_args = offsets_input_args
        else:
            v_in = "".join(f"[{i}:v]" for i in range(len(segments)))
            filter_parts.append(f"{v_in}concat=n={len(segments)}:v=1:a={int(has_audio_filter)}[outv]")
            if has_audio_filter:
                a_in = "".join(f"[{i}:a]" for i in range(len(segments)))
                filter_parts.append(f"{a_in}concat=n={len(segments)}:v=0:a=1[outa]")

        filter_desc = ";".join(filter_parts)
        cmd = [
            ffmpeg,
            "-y",
            *input_args,
            "-filter_complex",
            filter_desc,
            "-map",
            "[outv]",
        ]
        cmd += encoder_args
        if has_audio_filter:
            cmd += ["-map", "[outa]", "-c:a", "aac"]
        cmd += ["-movflags", "+faststart", str(output_path)]

        result = runner(cmd)
        if getattr(result, "returncode", 0) != 0:
            raise ConcatError(f"concat filter failed (exit {getattr(result, 'returncode', -1)})")
        return output_path

    cmd = [
        ffmpeg,
        "-y",
        *input_args,
        str(output_path),
    ]
    result = runner(cmd)
    if getattr(result, "returncode", 0) != 0:
        raise ConcatError(f"concat filter failed (exit {getattr(result, 'returncode', -1)})")
    return output_path


def _build_xfade(
    segments: Sequence[ConcatSegment],
    probes: list[dict],
    crossfade: float,
    fps: float,
    *,
    has_audio: bool,
) -> tuple[str, str, list[int]]:
    """Build the xfade / acrossfade filter chain for *n* segments.

    Returns ``(video_filter, audio_filter, offsets)``. ``audio_filter`` is an
    empty string when ``has_audio`` is False. ``offsets`` is a list of video
    frame offsets for each xfade (used to inject ``-t`` duration args into the
    input list so xfade can address stable input lengths).
    """
    n = len(segments)
    assert n >= 2

    # Offset of the i-th xfade transition (between segment i and i+1):
    #   offset_i = duration_0 + ... + duration_i - i * crossfade
    offsets: list[int] = []
    cumulative = 0.0
    for i in range(n - 1):
        cumulative += probes[i]["duration"] or (probes[i]["fps"] * 0)
        cumulative -= i * crossfade
        offsets.append(max(0, round(cumulative)))

    v_inputs = "".join(f"[{i}:v]" for i in range(n))
    # Offsets may be 0 when durations are unknown (probed as None). Keep the
    # offset explicit so the ffmpeg expression is well-formed.
    transitions: list[str] = []
    for off in offsets:
        transitions.append(f"trans_len={int(crossfade * fps)}:offset={off}:transition=fade")
    # Build a daisy-chained xfade: [0:v][1:v]xfade...[x0]; [x0][2:v]xfade...[x1] ...
    v_chain_parts: list[str] = []
    if n == 2:
        v_chain_parts.append(f"{v_inputs}xfade=frame_step=1:{transitions[0]}[outv]")
    else:
        prev = "xf0"
        v_chain_parts.append(f"[0:v][1:v]xfade=frame_step=1:{transitions[0]}[{prev}]")
        for i in range(2, n):
            src = f"[{i}:v]"
            out = f"xf{i - 1}"
            v_chain_parts.append(f"[{prev}]{src}xfade=frame_step=1:{transitions[i - 1]}[{out}]")
            prev = out
        v_chain_parts[-1] = v_chain_parts[-1].rsplit("]", 1)[0] + "][outv]"

    video_filter = "".join(v_chain_parts)

    audio_filter = ""
    if has_audio:
        a_chain_parts: list[str] = []
        a0 = f"[0:a][1:a]acrossfade=d={crossfade}:c1=tri:c2=tri[xa0]"
        a_chain_parts.append(a0)
        for i in range(2, n):
            out = f"xa{i - 1}"
            if i == n - 1:
                a_chain_parts.append(f"[xa{i - 2}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[outa]")
            else:
                a_chain_parts.append(f"[xa{i - 2}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out}]")
        audio_filter = "".join(a_chain_parts)

    return video_filter, audio_filter, offsets


def _build_offsets_input(offsets: list[int], input_args: list[str]) -> list[str]:
    """Prepend a ``-t <first_segment_duration>`` after the first segment's -i
    so ffmpeg knows the first input's length (needed when xfade references it).

    Kept as a no-op passthrough: the concat filter path uses named inputs and
    the offsets are already encoded in the filter expression itself, so there
    is nothing additional to inject. This hook exists so a future variant can
    add ``-t`` duration bounding without reshaping the public API.
    """
    _ = offsets  # offsets already baked into the filter expression
    return input_args
