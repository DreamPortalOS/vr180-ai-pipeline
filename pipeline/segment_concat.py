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
    warnings: list[str] | None = None,
) -> None:
    """Verify all segments share the same resolution and fps.

    Raises :class:`ConcatError` when any segment disagrees, with a message
    that lists each offending segment's actual width/height/fps so the
    caller can pinpoint which clip is the odd one out.

    Audio-track presence is *not* a hard error: a concat set that is
    uniformly audio-bearing or uniformly silent is fine. But a **mixed** set
    (some segments have an audio track, some don't) silently drops audio
    from the whole output — segment_concat deliberately emits no audio in
    that case (issue #173) rather than padding the silent clips. That is a
    "fails quietly" footgun: an owner splicing 10 clips where one happens to
    have no audio track gets a 20-minute silent film and only discovers it
    on the headset. So when the set is mixed we surface it: if *warnings* is
    a list, a human-readable note naming the audio-less segments is appended
    to it. The caller that doesn't pass *warnings* (the default) is
    unaffected — the check still returns without raising.
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

    # Audio-track consistency. A mixed set is not a hard error, but the caller
    # asked to be told (issue #196): when audio is dropped the output goes
    # silent, which is easy to miss until playback. Surface it via *warnings*.
    if warnings is not None:
        audio_mismatch = _audio_mismatch_warning(probes)
        if audio_mismatch is not None:
            warnings.append(audio_mismatch)


def _audio_mismatch_warning(probes: list[tuple[Path, dict]]) -> str | None:
    """Build a warning string for a mixed-audio concat set, or ``None``.

    Returns ``None`` when the set is uniform (all have audio, or none do) —
    those are silent-but-fine. When the set is mixed, names each audio-less
    segment by 1-based index and filename and explains the consequence.
    """
    has_audio_flags = [meta["has_audio"] for _, meta in probes]
    # Uniform set (all True or all False) → nothing to warn about.
    if all(has_audio_flags) or not any(has_audio_flags):
        return None

    silent_idx = [i for i, has in enumerate(has_audio_flags) if not has]
    parts = [f"segment {i + 1} ({probes[i][0].name})" for i in silent_idx]
    listed = ", ".join(parts)
    return (
        f"audio track mismatch: {listed} have no audio track — "
        "concat output will have NO audio track. Re-encode the silent "
        "segments with an audio track (or strip audio from all) to keep "
        "audio in the output."
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

    # The concat demuxer list file lives in the system temp dir, not the repo.
    # ffmpeg then resolves any relative paths inside that list against the list
    # file's directory — so relative caller paths would be resolved against
    # TMPDIR and silently fail (lead real-run failure; same root cause as
    # PR #75's seedvr2-cli cwd mismatch). Absolutize segment paths when writing
    # the list, and absolutize output_path for the same reason.
    output_path = Path(output_path).resolve()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    check_compatible(segments, warnings=warnings)
    for w in warnings:
        log.warning("⚠️ %s", w)

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
        # Absolute, forward-slash path. The concat demuxer resolves any
        # relative file entry against the list file's (system temp) directory,
        # so a caller-relative path would become TMPDIR/relative/path.mp4
        # and fail. ``.resolve()`` absolutizes; the demuxer accepts both
        # forward and back slashes, but forward slashes are the portable
        # cross-platform form recommended by the ffmpeg docs.
        seg_path = Path(seg.path).resolve().as_posix()
        parts.append(f"file '{seg_path}'")
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
    """Re-encode concat via ``filter_complex``.

    When ``crossfade <= 0`` this is a plain ``concat`` video (+ audio) filter.
    When ``crossfade > 0`` we daisy-chain ``xfade`` (video) + ``acrossfade``
    (audio). Each transition's ``offset`` is the timestamp (seconds from the
    start of the *output* so far) at which the crossfade into the next clip
    begins, so the total output duration is ``sum(durations) - (n-1)*crossfade``.
    """
    probes = [probe_segment(seg.path) for seg in segments]
    all_have_audio = all(p["has_audio"] for p in probes)

    width = probes[0]["width"]
    encoder_args = encoder if encoder is not None else select_encoder("h264", width)

    # Input list: one -ss (seek to optional start) + -i per segment.  -ss
    # before -i is fastest (input-seek, demuxer-level) and avoids decoding
    # frames before the trim point.
    input_args: list[str] = []
    for seg in segments:
        if seg.start is not None:
            input_args += ["-ss", str(seg.start)]
        input_args += ["-i", str(seg.path)]

    n = len(segments)
    if n == 1:
        # Trivial: one segment, no join. Re-encode so the encoder args apply.
        cmd = [ffmpeg, "-y", *input_args]
        cmd += encoder_args
        if all_have_audio:
            cmd += ["-c:a", "aac"]
        cmd += ["-movflags", "+faststart", str(output_path)]
    else:
        if crossfade > 0:
            v_filter, a_filter = _build_xfade(probes, crossfade, has_audio=all_have_audio)
            filter_desc = v_filter if not all_have_audio else ";".join([v_filter, a_filter])
        else:
            v_in = "".join(f"[{i}:v]" for i in range(n))
            filter_desc = f"{v_in}concat=n={n}:v=1:a={int(all_have_audio)}[outv]"
            if all_have_audio:
                a_in = "".join(f"[{i}:a]" for i in range(n))
                filter_desc += ";" + f"{a_in}concat=n={n}:v=0:a=1[outa]"

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
        if all_have_audio:
            cmd += ["-map", "[outa]", "-c:a", "aac"]
        cmd += ["-movflags", "+faststart", str(output_path)]

    result = runner(cmd)
    if getattr(result, "returncode", 0) != 0:
        raise ConcatError(f"concat filter failed (exit {getattr(result, 'returncode', -1)})")
    return output_path


def _build_xfade(
    probes: list[dict],
    crossfade: float,
    *,
    has_audio: bool,
) -> tuple[str, str]:
    """Build the daisy-chained xfade / acrossfade filter for *n >= 2* clips.

    ``xfade`` takes two video inputs and a crossfade ``duration`` plus an
    ``offset`` (seconds into the *first* input where the transition begins).
    For >2 clips we chain: ``[x0] = xfade(0,1)``; ``[x1] = xfade(x0,2)``; …
    Each chained xfade's offset is the end of the previous result minus the
    crossfade duration (so transitions overlap correctly). The same shape
    applies to ``acrossfade`` for audio.

    Returns ``(video_filter, audio_filter)``. ``audio_filter`` is ``""`` when
    ``has_audio`` is False.
    """
    n = len(probes)
    assert n >= 2
    durations = [p["duration"] for p in probes]
    # Fall back to a nominal 0s when ffprobe couldn't read duration: xfade
    # then starts at offset 0, which still produces a well-formed filter.
    dur = [d if d is not None and d > 0 else 0.0 for d in durations]

    v_parts: list[str] = []
    a_parts: list[str] = []
    prev_v = "0:v"
    prev_a = "0:a"
    # Running output length (seconds) of the xfade chain so far. The i-th
    # transition's offset is measured into the *current* chained output, so it
    # equals (length so far - crossfade), clamped to >= 0. After each
    # transition the chain's length grows by dur[i] - crossfade.
    chain_len = dur[0]
    for i in range(1, n):
        offset = max(0.0, chain_len - crossfade)
        out_label = "outv" if i == n - 1 else f"xv{i}"
        v_parts.append(f"[{prev_v}][{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset}[{out_label}]")
        prev_v = out_label
        chain_len = offset + dur[i]
        if has_audio:
            out_a = "outa" if i == n - 1 else f"xa{i}"
            a_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out_a}]")
            prev_a = out_a

    return "".join(v_parts), "".join(a_parts)
