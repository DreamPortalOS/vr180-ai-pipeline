#!/usr/bin/env python3
"""Synthesise a placeholder deep-sea ambience track with ffmpeg (issue #429).

The dome short needs *some* audio track, but the repo ships no audio assets
and third-party music must not be downloaded.  This script renders a
royalty-free underwater placeholder from ffmpeg lavfi sources only:

* low-passed pink noise ............ water-flow bed
* slow low-frequency sine (hum_hz, with a slow tremolo swell)
* sparse bubble chirps (seeded band-passed noise bursts at fixed times)

Output is a 48 kHz stereo WAV; ``alimiter`` + a fixed headroom trim keep the
peak at or below -1 dBFS.  All ffmpeg argv construction is pure (list form,
no ``shell=True``) so unit tests assert on strings without needing ffmpeg.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 48000
CHANNELS = 2
# Fixed trim so the limiter ceiling (-1 dBFS) can never be exceeded.
HEADROOM_DB = -1.0
FADE_S = 2.0
# Bubble voice: band-passed noise bursts, seeded placement.
BUBBLE_CENTER_HZ = 1200.0
BUBBLE_Q = 2.0
BUBBLE_DUR_S = 0.18
BUBBLE_GAIN = 3.0
BUBBLE_RATE_HZ = 0.06  # ~1 chirp per ~17 s of programme
_BUBBLE_MARGIN_S = 1.0

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Pure construction helpers (no subprocess, no I/O)
# ---------------------------------------------------------------------------


def bubble_times(duration: float, seed: int) -> list[float]:
    """Deterministic bubble onset times for ``seed`` (seconds, sorted)."""
    usable = duration - 2 * _BUBBLE_MARGIN_S
    if usable <= 0:
        return []
    count = max(2, round(duration * BUBBLE_RATE_HZ))
    rng = random.Random(seed)
    return sorted(rng.uniform(_BUBBLE_MARGIN_S, duration - _BUBBLE_MARGIN_S) for _ in range(count))


def build_filter_complex(duration: float, seed: int, hum_hz: float = 55.0) -> str:
    """Build the lavfi ``filter_complex`` string for the ambience mix."""
    bubbles = bubble_times(duration, seed)
    fade_out_start = max(duration - FADE_S, 0.0)

    parts = [
        # --- water bed: pink noise, low-passed, gentle level -----------------
        f"anoisesrc=color=pink:sample_rate={SAMPLE_RATE}:duration={duration:g}:seed={seed}"
        f"[bed_src];[bed_src]lowpass=f=400,volume=0.35[bed]",
        # --- deep hum: low sine with a slow amplitude swell ------------------
        f"sine=frequency={hum_hz:g}:sample_rate={SAMPLE_RATE}:duration={duration:g}"
        f"[hum_src];[hum_src]tremolo=f=0.12:d=0.8,volume=0.30[hum]",
    ]
    mix_inputs = ["[bed]", "[hum]"]
    for i, onset in enumerate(bubbles):
        parts.append(
            f"anoisesrc=color=white:sample_rate={SAMPLE_RATE}:duration={BUBBLE_DUR_S:g}:seed={seed + 1000 + i}"
            f"[bub{i}_src];[bub{i}_src]bandpass=f={BUBBLE_CENTER_HZ:g}:w={BUBBLE_Q}c,"
            f"volume={BUBBLE_GAIN:g}dB,adelay={round(onset * 1000)}|{round(onset * 1000)},"
            f"apad=whole_dur={duration:g}[bub{i}]"
        )
        mix_inputs.append(f"[bub{i}]")
    n_inputs = len(mix_inputs)
    parts.append(
        f"{''.join(mix_inputs)}amix=inputs={n_inputs}:duration=longest:dropout_transition=0,"
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo,"
        f"alimiter=limit=0.891251:attack=7:release=100,"
        f"volume={HEADROOM_DB:g}dB,"
        f"afade=t=in:st=0:d={FADE_S:g},afade=t=out:st={fade_out_start:g}:d={FADE_S:g}[aout]"
    )
    return ";".join(parts)


def build_ffmpeg_command(
    duration: float,
    out: str,
    seed: int = 7,
    hum_hz: float = 55.0,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg argv (list form, no shell) that renders the track."""
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-filter_complex",
        build_filter_complex(duration, seed, hum_hz),
        "-map",
        "[aout]",
        "-t",
        f"{duration:g}",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-c:a",
        "pcm_s16le",
        out,
    ]


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------


def render_ambience(
    duration: float,
    out: str,
    seed: int = 7,
    hum_hz: float = 55.0,
    ffmpeg: str | None = None,
) -> str:
    """Run ffmpeg to render the ambience WAV; returns ``out``."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not found on PATH")
    cmd = build_ffmpeg_command(duration, out, seed, hum_hz, ffmpeg=exe)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True, help="Output length in seconds")
    parser.add_argument("--out", required=True, help="Destination .wav path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--style", default="underwater", choices=["underwater"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration <= 0:
        print("error: --duration must be > 0", file=sys.stderr)
        return 2
    render_ambience(args.duration, args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
