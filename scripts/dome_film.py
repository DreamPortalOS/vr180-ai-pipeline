#!/usr/bin/env python3
"""Dome film composer — multi-scene fulldome short in one command (D-1, issue #427).

The owner's first dome short is cut from keyframe-anchored AI clips.  Each
scene arrives as a **16:9 ~1080p** clip (Gemini only emits 16:9): the 1:1
domemaster sat letterboxed on black.  This script chains the tools that
already exist one clip at a time —

    crop11        (16:9 → 1:1, inscribed circle blacked, audio copied)
    rimlift-video (brighten the dark outer annulus, one gain map per clip)
    upscale       (lanczos scale, or seedvr2 on the CUDA box)
    concat        (xfade between scenes)
    audio-mix     (optional external ambience)
    encode        (H.265 10-bit 4096² delivery master)
    dome_qa       (machine acceptance, JSON report next to the film)

— so a JSON plan produces a finished domemaster and its QA report in one run.

Plan file (``--plan``):

.. code-block:: json

    {
      "size": 4096,
      "upscale": "lanczos",
      "crossfade": 1.0,
      "audio": "path/or/null",
      "audio_gain_db": 0,
      "scenes": [
        {"clip": "a.mp4", "trim_start": 0.0, "trim_end": null,
         "rimlift": true, "is_169": true}
      ]
    }

Per scene the chain is: optional ``crop11`` (when ``is_169``), optional
``rimlift-video`` (when ``rimlift``), then an upscale to ``size``×``size``
(``lanczos`` default; ``seedvr2`` routes through the existing
``--video-upscale seedvr2`` CLI backend with the 12 GB defaults).  Everything
outside the inscribed circle stays pure black at every hop.

All ffmpeg/ffprobe invocations are subprocess **list** form (never
``shell=True`` — CLAUDE.md red line).  Intermediates live in a temp dir that
is removed on success (``--keep-intermediates`` keeps them).  ``--dry-run``
prints every step's command and the predicted output size/duration and writes
nothing.

Usage:
    python scripts/dome_film.py --plan plan.json --output dome_film.mp4
    python scripts/dome_film.py --plan plan.json --output dome_film.mp4 --dry-run
    python scripts/dome_film.py --plan plan.json --output dome_film.mp4 --keep-intermediates
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Bootstrap the repo root so ``python scripts/dome_film.py`` runs bare (same
# pattern as scripts/dome_qa.py / scripts/run_pipeline.py) — pipeline/ and
# scripts/ are then importable without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the geometry / rim-lift tools (issue #408) rather than re-implementing
# them: crop11 + rimlift-video + the pure gain-map functions.
from scripts import dome_frame_tools as dft  # noqa: E402

# Reuse the concat engine (C-1, issue #173) and the audio mix (S-4, #396).
from pipeline.segment_concat import ConcatError, ConcatSegment, concat_segments, probe_segment  # noqa: E402

log = logging.getLogger("dome_film")

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"

# --------------------------------------------------------------------------- #
# Defaults / encode contract
# --------------------------------------------------------------------------- #
#: Delivery canvas (the card's 4096² master).
DEFAULT_SIZE = 4096
#: Default upscale engine: lanczos runs anywhere (CPU CI box); seedvr2 needs CUDA.
DEFAULT_UPSCALE = "lanczos"
UPSCALE_CHOICES = ("lanczos", "seedvr2", "none")
#: H.265 10-bit delivery master (issue #427) — Quest HMD, 10-bit avoids banding
#: on the graded dome footage.  8-bit intermediates feed the concat so the
#: encoder's own yuv420p10le conversion is the only 10-bit hop.
FINAL_CRF = 20
FINAL_PIX_FMT = "yuv420p10le"
FINAL_CODEC = "libx265"
FINAL_PRESET = "medium"
#: Wall-clock ceiling for one ffmpeg hop (matches dome_frame_tools).
FFMPEG_TIMEOUT_SEC = dft.FFMPEG_TIMEOUT_SEC


# --------------------------------------------------------------------------- #
# Plan model
# --------------------------------------------------------------------------- #


@dataclass
class Scene:
    """One dome scene: a source clip plus its per-scene switches.

    ``clip`` is the 16:9 (or already-1:1) source.  ``trim_start`` / ``trim_end``
    are the optional cut window in seconds.  ``is_169`` routes the clip through
    ``crop11``; ``rimlift`` through ``rimlift-video``.
    """

    clip: Path
    trim_start: float | None = None
    trim_end: float | None = None
    rimlift: bool = True
    is_169: bool = True


@dataclass
class Plan:
    """The whole composition, parsed from the JSON plan file."""

    scenes: list[Scene]
    size: int = DEFAULT_SIZE
    upscale: str = DEFAULT_UPSCALE
    crossfade: float = 1.0
    audio: Path | None = None
    audio_gain_db: float = 0.0
    # Intermediates / QA report dir (filled in by main, under the output's
    # parent so --keep-intermediates leaves a reviewable folder next to it).
    work_dir: Path | None = field(default=None, repr=False)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _load_plan(plan_path: str | Path) -> Plan:
    """Parse the JSON plan into a :class:`Plan`, validating the fields we use.

    Relative scene/audio paths resolve against the plan file's directory so a
    plan checked in next to the material is portable across machines.
    """
    path = Path(plan_path)
    if not path.is_file():
        raise FileNotFoundError(f"plan not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"plan {path} must be a JSON object")

    base = path.resolve().parent
    scenes: list[Scene] = []
    for i, raw in enumerate(data.get("scenes") or []):
        _require(isinstance(raw, dict) and raw.get("clip"), f"scene #{i} needs a 'clip' path")
        clip = Path(raw["clip"])
        if not clip.is_absolute():
            clip = (base / clip).resolve()
        scenes.append(
            Scene(
                clip=clip,
                trim_start=_opt_float(raw.get("trim_start")),
                trim_end=_opt_float(raw.get("trim_end")),
                rimlift=bool(raw.get("rimlift", True)),
                is_169=bool(raw.get("is_169", True)),
            )
        )
    _require(scenes, "plan has no scenes")

    upscale = str(data.get("upscale", DEFAULT_UPSCALE)).lower()
    _require(upscale in UPSCALE_CHOICES, f"upscale must be one of {UPSCALE_CHOICES}, got {upscale!r}")
    crossfade = float(data.get("crossfade", 1.0))
    _require(crossfade >= 0.0, f"crossfade must be >= 0, got {crossfade}")

    audio = data.get("audio")
    audio_path = None
    if audio:
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = (base / audio_path).resolve()

    size = int(data.get("size", DEFAULT_SIZE))
    _require(size > 0 and size % 2 == 0, f"size must be a positive even number, got {size}")
    return Plan(
        scenes=scenes,
        size=size,
        upscale=upscale,
        crossfade=crossfade,
        audio=audio_path,
        audio_gain_db=float(data.get("audio_gain_db", 0.0)),
    )


def _opt_float(v) -> float | None:
    """``None`` stays ``None``; anything else coerces to float (or raises)."""
    if v is None:
        return None
    f = float(v)
    if f < 0:
        raise ValueError(f"trim points must be >= 0, got {f}")
    return f


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], *, label: str) -> None:
    """Run one ffmpeg/ffprobe argv (list form), raising on failure."""
    log.debug("%s: %s", label, " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{result.stderr[-2000:]}")


def _probe(path: Path) -> dict:
    """width/height/fps/duration/has_audio via the shared probe."""
    return probe_segment(Path(path))


def _black_circle_and_scale_filter(size: int) -> str:
    """Filter that masks to the inscribed circle *and* scales to ``size``².

    The circle mask runs in the frame's own coordinates (``W``/``H`` of the
    input frame) and the scale is lanczos to the square delivery canvas — one
    filtergraph so the crop11→upscale hops stay a single encode.  The mask
    predicate is the exact one :func:`dome_frame_tools.crop11_filter` uses, so
    a 1:1 input and a 16:9 input end on the same black surround.  The chain is
    labelled ``[dome]`` so ``-filter_complex``'s input pad is explicit (an
    unlabelled chain in ``-filter_complex`` cannot bind to the input stream).
    """
    circle = dft.geq_circle_expression()
    body = (
        "format=gbrp,"
        f"geq=r='if({circle},r(X,Y),0)':g='if({circle},g(X,Y),0)':b='if({circle},b(X,Y),0)',"
        f"scale={size}:{size}:flags=lanczos,format={dft.CROP11_PIX_FMT}"
    )
    return f"[0:v]{body}[dome]"


def build_upscale_command(input_path: Path, output_path: Path, size: int, *, ffmpeg: str = _FFMPEG) -> list[str]:
    """ffmpeg argv: mask to the circle + lanczos-scale to ``size``² (list form).

    Used for the ``lanczos`` upscale backend.  Audio is copied through; the
    video is re-encoded at the intermediate quality so concat sees clean 8-bit
    frames.
    """
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        _black_circle_and_scale_filter(size),
        "-map",
        "[dome]",
        "-map",
        "0:a:0?",
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-crf",
        str(dft.CROP11_CRF),
        "-pix_fmt",
        dft.CROP11_PIX_FMT,
        str(output_path),
    ]


def _upscale_seedvr2(input_path: Path, output_path: Path, size: int) -> str:
    """SeedVR2 upscale to ``size``² via the existing ``--video-upscale seedvr2`` backend.

    Routes through :class:`pipeline.video_upscaler.SeedVR2Upscaler` (the same
    class ``run_pipeline.py --video-upscale seedvr2`` drives) with the 12 GB
    defaults baked into :class:`~pipeline.video_upscaler.CLIBackend`.  The
    circle-mask runs *before* SeedVR2 (in the caller's crop11/rimlift hops) so
    the AI upscaler only ever sees clean 1:1 frames; the final mask re-asserts
    the black surround after upscaling.
    """
    from pipeline.video_upscaler import SeedVR2Upscaler  # deferred: CUDA-only import

    upscaler = SeedVR2Upscaler(batch_size=5)
    factor = max(2, round(size / max(1, _probe(input_path)["height"])))
    factor = min(factor, 4)  # SeedVR2 supports 2/3/4
    out = upscaler.upscale(input_path=str(input_path), output_path=str(output_path), factor=factor)
    return str(out)


def build_final_encode_command(
    concat_path: Path,
    output_path: Path,
    size: int,
    *,
    ffmpeg: str = _FFMPEG,
) -> list[str]:
    """ffmpeg argv for the H.265 10-bit ``size``² delivery master (list form).

    The concat intermediate is 8-bit 4:2:0; this hop is the only 10-bit
    conversion.  ``-c:a copy`` keeps whatever audio the concat/mix produced.
    """
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(concat_path),
        "-c:v",
        FINAL_CODEC,
        "-preset",
        FINAL_PRESET,
        "-crf",
        str(FINAL_CRF),
        "-pix_fmt",
        FINAL_PIX_FMT,
        "-tag:v",
        "hvc1",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


# --------------------------------------------------------------------------- #
# Per-scene chain
# --------------------------------------------------------------------------- #


def _scene_step(scene: Scene, idx: int, plan: Plan, work: Path, *, dry_run: bool) -> Path:
    """Run one scene through crop11 → rimlift → upscale.  Returns the 1:1 path.

    Each hop writes into *work* and returns the path the next hop reads.  In
    ``dry_run`` mode nothing runs and the would-be paths are printed instead.
    """
    current = scene.clip
    tag = f"scene{idx:02d}"

    # 1) crop11 — 16:9 → 1:1 + circle mask (audio copied).
    if scene.is_169:
        out = work / f"{tag}_crop11.mp4"
        if dry_run:
            print(f"[{tag}] crop11      {current} → {out}  (16:9 → 1:1, circle black)")
        else:
            dft.crop11_file(current, out)
        current = out

    # 2) rimlift-video — brighten the outer annulus (one gain map per clip).
    if scene.rimlift:
        out = work / f"{tag}_rimlift.mp4"
        if dry_run:
            print(f"[{tag}] rimlift     {current} → {out}  (rim → ≥40% of centre)")
        else:
            dft.rimlift_video(current, out)
        current = out

    # 3) upscale to the delivery canvas (lanczos / seedvr2 / none).
    if plan.upscale != "none":
        out = work / f"{tag}_{plan.size}.mp4"
        if dry_run:
            if plan.upscale == "seedvr2":
                print(f"[{tag}] upscale     {current} → {out}  (seedvr2 → {plan.size}², 12GB defaults)")
            else:
                cmd = build_upscale_command(current, out, plan.size)
                print(f"[{tag}] upscale     {current} → {out}  (lanczos → {plan.size}²)")
                print(f"        $ {' '.join(str(c) for c in cmd)}")
        else:
            if plan.upscale == "seedvr2":
                _upscale_seedvr2(current, out, plan.size)
            else:
                _run(build_upscale_command(current, out, plan.size), label=f"upscale {tag}")
        current = out

    return current


# --------------------------------------------------------------------------- #
# Composition driver
# --------------------------------------------------------------------------- #


def compose(plan: Plan, output_path: Path, *, keep_intermediates: bool, dry_run: bool) -> Path:
    """Run the full chain and return the finished master path.

    In ``dry_run`` mode every hop is printed (with its predicted size/duration
    where ffmpeg can know it) and nothing is written.
    """
    output_path = Path(output_path).resolve()
    n = len(plan.scenes)

    if dry_run:
        total = 0.0
        print(f"# dome_film dry-run — {n} scene(s), size {plan.size}², crossfade {plan.crossfade}s")
        for i, scene in enumerate(plan.scenes):
            try:
                dur = _probe(scene.clip)["duration"]
            except (ConcatError, OSError, RuntimeError):
                dur = None
            if dur is not None:
                start = scene.trim_start or 0.0
                end = scene.trim_end if scene.trim_end is not None else dur
                seg = max(0.0, end - start)
            else:
                seg = 0.0
            total += seg
            print(f"\n── scene {i}: {scene.clip.name} (~{seg:.2f}s) ──")
            _scene_step(scene, i, plan, Path("<work>"), dry_run=True)
        out_dur = total - max(0, n - 1) * plan.crossfade
        print(f"\n── concat ×{n} → {plan.size}², crossfade {plan.crossfade}s → ~{out_dur:.2f}s ──")
        if plan.audio:
            print(f"── audio-mix    {plan.audio} (gain {plan.audio_gain_db} dB) ──")
        print(f"── encode       → {output_path}  ({FINAL_CODEC} 10-bit {FINAL_PIX_FMT})")
        print(f"── dome_qa      → {output_path.with_suffix('.qa.json')}")
        return output_path

    # Real run.  Intermediates go in a temp dir (cleaned unless
    # --keep-intermediates).  When kept, they land next to the output for review.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_intermediates:
        work = output_path.parent / (output_path.stem + "_work")
        work.mkdir(parents=True, exist_ok=True)
        _cleanup = None
    else:
        _tmp = tempfile.TemporaryDirectory(prefix="dome-film-")
        _cleanup = _tmp
        work = Path(_tmp.name)
    plan.work_dir = work

    try:
        # ── per-scene chain ────────────────────────────────────────────────
        prepared: list[Path] = []
        for i, scene in enumerate(plan.scenes):
            if not scene.clip.is_file():
                raise FileNotFoundError(f"scene {i} clip not found: {scene.clip}")
            prepared.append(_scene_step(scene, i, plan, work, dry_run=False))

        # ── concat with crossfade ──────────────────────────────────────────
        segments = [
            ConcatSegment(p, start=s.trim_start, end=s.trim_end) for p, s in zip(prepared, plan.scenes, strict=True)
        ]
        concat_out = work / "_concat.mp4"
        log.info("🎬 Concatenating %d scene(s) → %s (crossfade %.2fs)", n, concat_out, plan.crossfade)
        try:
            concat_segments(
                segments,
                concat_out,
                mode="filter" if plan.crossfade > 0 else "demux",
                crossfade=plan.crossfade,
                encoder=["-c:v", "libx264", "-crf", str(dft.CROP11_CRF), "-pix_fmt", dft.CROP11_PIX_FMT],
            )
        except ConcatError as exc:
            raise RuntimeError(f"concat failed: {exc}") from exc
        current = concat_out

        # ── optional audio mix ─────────────────────────────────────────────
        if plan.audio is not None:
            if not plan.audio.is_file():
                raise FileNotFoundError(f"audio not found: {plan.audio}")
            from pipeline.audio_mix import mix_external_audio

            mixed = work / "_mixed.mp4"
            log.info("🎧 Mixing ambience %s (gain %.1f dB)", plan.audio, plan.audio_gain_db)
            mix_external_audio(str(current), str(plan.audio), str(mixed), gain_db=plan.audio_gain_db)
            current = mixed

        # ── final H.265 10-bit encode ──────────────────────────────────────
        log.info("📦 Encoding %s² H.265 10-bit master → %s", plan.size, output_path)
        _run(build_final_encode_command(current, output_path, plan.size), label="final encode")

        # ── dome QA → JSON next to the film ───────────────────────────────
        qa_path = _run_qa(output_path, plan.size)
        log.info("✅ dome film ready → %s (QA %s)", output_path, qa_path)
        return output_path
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()


def _run_qa(output_path: Path, size: int) -> Path:
    """Run ``scripts/dome_qa.py`` and write the JSON report beside the film."""
    from scripts import dome_qa

    report = dome_qa.run_qa(str(output_path), size=size)
    payload = json.loads(json.dumps(report.__dict__, default=str))
    payload["summary"] = report.summary
    qa_path = output_path.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "🧪 dome_qa %s (coverage r/R=%.2f, outside %.2f) → %s",
        report.verdict,
        report.coverage_r,
        report.outside_mean,
        qa_path,
    )
    return qa_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dome_film.py",
        description="Compose a multi-scene fulldome short from a JSON plan (issue #427).",
    )
    p.add_argument("--plan", required=True, help="JSON plan file (scenes, size, upscale, crossfade, audio)")
    p.add_argument("--output", "-o", required=True, help="Output dome film (H.265 10-bit, size²)")
    p.add_argument("--dry-run", action="store_true", help="Print every step's command/size/duration; write nothing")
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the per-scene intermediates in <output>_work/ (default: clean up)",
    )
    p.add_argument("--ffmpeg", default=_FFMPEG, help="ffmpeg binary")
    p.add_argument("--ffprobe", default=_FFPROBE, help="ffprobe binary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        plan = _load_plan(args.plan)
        compose(plan, Path(args.output), keep_intermediates=args.keep_intermediates, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        log.error("dome_film failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
