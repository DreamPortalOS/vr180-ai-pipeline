#!/usr/bin/env python3
"""One-command image → VR180 orchestrator (issue #56, G-3).

Turns a single image into a Quest-playable VR180 in one command:

    python scripts/image_to_vr180.py --image cat.png

Pipeline (each stage writes a job-manifest checkpoint so ``--resume-from``
skips already-done stages — generation is expensive and must never be
re-run because a downstream stage failed):

  1. **prepare**   — ``pipeline.image_prep.prepare_image`` (normalise I2V input)
  2. **generate**  — ``integrations.factory.get_provider(...).generate_from_image``
  3. **streamcheck** — ffprobe stream-info on the generated video; fail-fast
     if resolution or fps is anomalous before any expensive conversion
  4. **upscale**   — optional SeedVR2 video super-resolution
     (``pipeline.video_upscaler.SeedVR2Upscaler``)
  5. **convert**   — VR180 conversion via ``scripts.run_pipeline`` stage
     functions (depth → stereo → equirect → metadata). Applies the
     ``--quality`` preset the same way ``run_pipeline.main`` does before
     any stage runs (V-2 lesson).
  6. **qa**        — ``scripts.vr180_qa.run_qa``; a failing verdict exits 3.

Only *real* modules are imported at call time; the model/ffmpeg-heavy paths
are behind injectable callables so the orchestration logic is fully
unit-testable on CI (CPU-only, no keys, no models).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("image-to-vr180")

# Exit code when final QA fails (machine-detectable, non-zero).
EXIT_QA_FAILED = 3

# Manifest stage names, in the order they run.
STAGE_ORDER = ("prepare", "generate", "streamcheck", "upscale", "convert", "audio", "qa")


# H-1: audio passthrough. The generated video (``args.generated_video``) is the
# default audio source (Seedance 2.0 embeds a synced AAC track). A caller may
# override it via ``copy_audio_from`` to attach an arbitrary audio track.
DEFAULT_AUDIO_SOURCE_ATTR = "generated_video"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class JobArgs:
    """Resolved runtime inputs for one run. Plain-ish container used by both
    the CLI path and the injected test path so stage functions share one
    source of truth for paths / provider / quality / upscale."""

    image: str
    prompt: str = ""
    provider: str = "mock"
    duration: int = 5
    # Generation tier (H-2): passed through to the provider as kwargs so
    # raising the tier is a CLI switch, not a code change. Defaults keep
    # the quota discipline (480p / 5s / adaptive).
    gen_resolution: str = "480p"
    gen_ratio: str = "adaptive"
    upscale: str = "none"
    quality: str = "preview"

    # I-3 (#88): comfort preset passed through to the VR180 conversion
    # stage.  Mirrors --quality: it's a starting point, and explicit
    # max_disparity / convergence overrides the preset via run_pipeline's
    # resolve_comfort.  Stored as strings / None (the argparse types) so the
    # synthetic argv run_convert_default builds is lossless.
    comfort: str = "balanced"
    max_disparity: float | None = None
    convergence: float | None = None

    # H-1: explicit audio source for the audio remux stage. When ``None``
    # (default), the stage falls back to ``args.generated_video`` and detects
    # an audio stream there automatically.
    copy_audio_from: str | None = None

    workdir: str = ""
    manifest_path: str | None = None
    resume_from: str | None = None

    # Resolved artefact paths (filled in by :func:`resolve_paths`).
    prepared_image: str = ""
    generated_video: str = ""
    upscaled_video: str = ""
    vr180_output: str = ""

    # Quality preset resolution (mirrors run_pipeline.apply_quality_preset).
    output_width: int | None = None
    output_height: int | None = None
    bitrate: str | None = None


# ---------------------------------------------------------------------------
# Resolved paths & defaults
# ---------------------------------------------------------------------------


def _default_workdir(image: str) -> str:
    src = Path(image)
    return str(src.parent / f"{src.stem}_vr180")


def resolve_paths(args: JobArgs) -> None:
    """Fill in artefact paths on *args* so every stage reads the same values.

    Paths live under ``args.workdir`` (created lazily by :func:`ensure_workdir`).

    K-1.2 (issue #106): when a caller (e.g. :mod:`scripts.batch_runner`) has
    already set ``args.vr180_output`` to a *scene-named* target, we MUST NOT
    overwrite it with the image-stem default — that was the bug that let two
    jobs sharing an input image silently collide on the same final file.
    When the final output is caller-driven, the intermediate artefacts
    (``_prep.png`` / ``_generated.mp4`` / ``_upscaled.mp4``) are likewise
    keyed off the output filename stem so co-located jobs cannot clobber each
    other's intermediates. The standalone CLI path (no pre-set output) keeps
    its original image-stem behaviour, so its contract is unchanged.
    """
    if not args.workdir:
        args.workdir = _default_workdir(args.image)
    w = Path(args.workdir)

    # K-1.2: if the caller chose the final filename (scene naming), honor it
    # and derive the intermediate stems from it so co-located jobs are
    # isolated. Otherwise fall back to the image stem.
    if args.vr180_output:
        stem = Path(args.vr180_output).stem
    else:
        stem = Path(args.image).stem
        args.vr180_output = str(w / f"{stem}_vr180.mp4")

    args.prepared_image = str(w / f"{stem}_prep.png")
    args.generated_video = str(w / f"{stem}_generated.mp4")
    args.upscaled_video = str(w / f"{stem}_upscaled.mp4")


# ---------------------------------------------------------------------------
# Manifest helpers (reuses pipeline.job_manifest; does not duplicate it)
# ---------------------------------------------------------------------------


def _load_or_create_manifest(args: JobArgs) -> dict:
    """Load an existing manifest (resume) or create a fresh one (new run).

    A ``--manifest`` path creates/overwrites a job manifest whose source is
    the input image. ``--resume-from`` loads a previously-written manifest
    and validates the source hash + done-stage hashes (so a replaced image
    or a tampered generated video is caught before we skip it).
    """
    from pipeline.job_manifest import (
        ManifestError,
        completed_stages,
        load_manifest,
        new_manifest,
        validate_source,
        validate_stage_outputs,
    )

    image = Path(args.image)

    if args.resume_from:
        try:
            manifest = load_manifest(args.resume_from)
            validate_source(manifest, str(image))
            for name in completed_stages(manifest):
                validate_stage_outputs(manifest, name)
        except ManifestError as exc:
            raise RuntimeError(f"Cannot resume from {args.resume_from}: {exc}") from None
        log.info("📂 Resume: completed stages %s", completed_stages(manifest) or "[]")
        return manifest

    if args.manifest_path:
        # Ensure manifest parent dir exists before new_manifest hashes the image.
        Path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        src_hash = (
            __import__("pipeline.job_manifest", fromlist=["sha256_file"]).sha256_file(str(image))
            if image.is_file()
            else "nohash"
        )
        job_id = f"{image.stem}-{src_hash[:8]}-{uuid.uuid4().hex[:4]}"
        return new_manifest(job_id, str(image), stage_names=STAGE_ORDER)

    return {}


def _stage_names_done(manifest: dict) -> set[str]:
    return {s["name"] for s in manifest.get("stages", []) if s.get("status") == "done"}


def _record_stage(manifest: dict, name: str, *, inputs: list[str], outputs: list[str], params: dict) -> None:
    if not manifest:
        return
    from pipeline.job_manifest import mark_stage_done

    mark_stage_done(
        manifest,
        name,
        inputs=[str(p) for p in inputs],
        outputs=[str(p) for p in outputs],
        params=dict(params),
    )


def _persist_manifest(manifest: dict, args: JobArgs) -> None:
    if not manifest or not args.manifest_path:
        return
    from pipeline.job_manifest import save_manifest

    save_manifest(manifest, args.manifest_path)


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------


def stage_prepare(args: JobArgs) -> str:
    """Normalise the input image for image-to-video ingestion."""
    from pipeline.image_prep import prepare_image

    log.info("🛠️  Stage prepare: %s → %s", args.image, args.prepared_image)
    Path(args.prepared_image).parent.mkdir(parents=True, exist_ok=True)
    result = prepare_image(
        input_path=args.image,
        out_path=args.prepared_image,
        target_aspect="16:9",
        target_width=1280,
        mode="letterbox",
    )
    if result.warnings:
        for w in result.warnings:
            log.warning("⚠️  prepare: %s", w)
    log.info("✅ Stage prepare done → %s (%dx%d)", args.prepared_image, result.width, result.height)
    return args.prepared_image


# ---------------------------------------------------------------------------
# Stage 2: generate (image-to-video provider)
# ---------------------------------------------------------------------------


def stage_generate(args: JobArgs, prepared_image: str | None = None) -> str:
    """Call the provider's ``generate_from_image`` and return the video path."""
    from integrations import factory

    if prepared_image is None:
        prepared_image = args.prepared_image
    Path(args.generated_video).parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "🛠️  Stage generate: provider=%s duration=%ds resolution=%s ratio=%s prompt=%r → %s",
        args.provider,
        args.duration,
        args.gen_resolution,
        args.gen_ratio,
        args.prompt or "(none)",
        args.generated_video,
    )

    # Generation-tier kwargs (H-2): handed to the provider so it can put
    # resolution/ratio into the request body. Seedance reads these; the mock
    # provider ignores them (per the card's "mock provider 忽略即可").
    gen_kwargs: dict[str, str] = {
        "resolution": args.gen_resolution,
        "ratio": args.gen_ratio,
    }

    # Ensure the mock provider (and any other provider) writes into our workdir
    # so downstream stages and manifest hashing stay inside tmp_path/CI scratch.
    os.environ["MOCK_PROVIDER_OUTPUT_DIR"] = args.workdir

    provider = factory.get_provider(args.provider)
    result = provider.generate_from_image(
        image_path=prepared_image,
        prompt=args.prompt,
        duration=args.duration,
        aspect_ratio="16:9",
        fps=24,
        **gen_kwargs,
    )

    video_path = result.video_url
    if not video_path or not Path(video_path).is_file():
        raise RuntimeError(f"Provider {args.provider} returned no valid video file: {video_path!r}")

    # Rename to the canonical generated_video path (providers pick their own
    # filename; we normalise so the manifest + downstream stages are stable).
    if Path(video_path) != Path(args.generated_video):
        Path(args.generated_video).parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(video_path, args.generated_video)
        video_path = args.generated_video

    log.info(
        "✅ Stage generate done → %s (%s, job=%s)",
        video_path,
        result.provider,
        result.job_id or "-",
    )
    return video_path


# ---------------------------------------------------------------------------
# Stage 3: stream check (fail-fast on resolution / fps anomalies)
# ---------------------------------------------------------------------------


def stage_streamcheck(video_path: str) -> None:
    """Probe the generated video and fail-fast if it looks unusable.

    Catches a wasted conversion run when the provider returned a broken or
    absurdly small/large stream.  Reuses the ffprobe stream extraction that
    ``scripts.vr180_qa`` already ships (no duplicate box/ffmpeg logic here).
    """
    from scripts.vr180_qa import QAReport, _extract_stream_info, _probe

    log.info("🛠️  Stage streamcheck: %s", video_path)
    probe = _probe(video_path)
    report = QAReport(path=video_path)
    _extract_stream_info(probe, report)

    width, height, fps = report.width, report.height, report.fps
    log.info(
        "   stream info: %dx%d %s %.2g fps %.1fs",
        width,
        height,
        report.codec,
        fps,
        report.duration_s,
    )

    anomalies: list[str] = []
    if width <= 0 or height <= 0:
        anomalies.append(f"non-positive resolution {width}x{height}")
    if fps <= 0:
        anomalies.append(f"non-positive fps {fps}")
    # Absurdly tiny video (a stub) would waste the conversion. Guardrail:
    # at least a few frames at a modest resolution are needed to convert.
    if width < 160 or height < 160:
        anomalies.append(f"video too small ({width}x{height})")
    if report.duration_s < 0.5:
        anomalies.append(f"video too short ({report.duration_s:.2f}s)")

    if anomalies:
        raise RuntimeError("Generated video stream check failed: " + "; ".join(anomalies))

    log.info("✅ Stage streamcheck passed")


# ---------------------------------------------------------------------------
# Stage 4: optional upscale (SeedVR2)
# ---------------------------------------------------------------------------


def stage_upscale(args: JobArgs, input_video: str) -> str:
    """Run SeedVR2 video super-resolution when ``--upscale seedvr2`` is set."""
    if args.upscale != "seedvr2":
        log.info("🛠️  Stage upscale: skipped (--upscale %s)", args.upscale)
        return input_video

    Path(args.upscaled_video).parent.mkdir(parents=True, exist_ok=True)
    log.info("🛠️  Stage upscale: %s → %s", input_video, args.upscaled_video)

    upscaler = SeedVR2Upscaler(batch_size=5)
    out = upscaler.upscale(input_path=input_video, output_path=args.upscaled_video, factor=2)
    log.info("✅ Stage upscale done → %s", out)
    return out


class SeedVR2Upscaler:
    """Facade that delegates to ``pipeline.video_upscaler.SeedVR2Upscaler``.

    Kept as a thin, swappable object so tests can inject a fake upscaler and
    so the CUDA-requiring real backend is only imported at call time.
    """

    def __init__(self, batch_size: int = 5) -> None:
        self.batch_size = batch_size

    def upscale(self, input_path: str, output_path: str, factor: int = 2) -> str:
        from pipeline.video_upscaler import SeedVR2Upscaler as _Real

        up = _Real(batch_size=self.batch_size)
        return up.upscale(input_path=input_path, output_path=output_path, factor=factor)


# ---------------------------------------------------------------------------
# Stage 5: VR180 conversion
# ---------------------------------------------------------------------------


def _apply_quality_preset(args: JobArgs) -> None:
    """Resolve ``--quality`` into per-eye size / bitrate the same way
    ``scripts.run_pipeline.apply_quality_preset`` does, so the converter
    always sees concrete dimensions (V-2 lesson)."""
    from pipeline.streaming_pipeline import resolve_quality, scaled_bitrate_mbps

    eye_size, _streaming = resolve_quality(args.quality)
    args.output_width = eye_size
    args.output_height = eye_size
    mbps = scaled_bitrate_mbps(eye_size, max_mbps=200.0)
    args.bitrate = f"{mbps:g}M"
    log.info("🎚️  --quality %s → %d²/eye, bitrate %.1f Mbps", args.quality, eye_size, mbps)


def stage_convert(args: JobArgs, input_video: str, convert: Callable[..., str]) -> str:
    """Run VR180 conversion.

    *convert* is an injectable callable ``(args: JobArgs, input_path: str)
    -> output_path``. The default (``None``) is the real run that calls
    :func:`run_convert_default`.
    """
    if convert is None:
        convert = run_convert_default
    log.info("🛠️  Stage convert: %s → %s", input_video, args.vr180_output)

    _apply_quality_preset(args)
    out = convert(args, input_video)

    if not out or not Path(out).is_file():
        raise RuntimeError(f"Converter returned no valid output: {out!r}")
    log.info("✅ Stage convert done → %s", out)
    return out


def run_convert_default(args: JobArgs, input_path: str) -> str:
    """Real VR180 conversion: run the run_pipeline stage loop end-to-end.

    Delegates to ``scripts.run_pipeline`` by building a synthetic argv and
    running its main(). This reuses the exact stage functions (depth →
    stereo → equirect → metadata) and the same ``apply_quality_preset``
    resolution path — no conversion logic is duplicated here.
    """
    import scripts.run_pipeline as rp

    output = args.vr180_output
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--input",
        input_path,
        "--output",
        output,
        "--quality",
        args.quality,
        "--bitrate",
        args.bitrate,
        "--comfort",
        args.comfort,
        "--device",
        "cpu",
        "--max-frames",
        "60",
        "--no-ffmpeg-v360",
    ]
    # I-3: forward explicit comfort overrides so the converter resolves the
    # same effective values the operator asked for at the image-to-vr180 CLI.
    if args.max_disparity is not None:
        argv += ["--max-disparity", str(args.max_disparity)]
    if args.convergence is not None:
        argv += ["--convergence", str(args.convergence)]

    # run_pipeline.main() reads sys.argv; run_pipeline.apply_quality_preset
    # is already applied above (and again inside main — idempotent), keeping
    # the V-2 rule that quality must be resolved before any stage runs.
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_pipeline", *argv]
        rp.main()
    finally:
        sys.argv = old_argv
    return output


# ---------------------------------------------------------------------------
# Stage 6: audio passthrough (issue #73, H-1)
# ---------------------------------------------------------------------------


def _audio_source_path(args: JobArgs, generated_video: str) -> str:
    """Resolve the audio source path for the remux stage.

    Explicit ``copy_audio_from`` wins; otherwise fall back to the generated
    video (which for real providers like Seedance carries the synced AAC
    track).
    """
    if args.copy_audio_from:
        return args.copy_audio_from
    return generated_video


def stage_audio(args: JobArgs, vr180_path: str, generated_video: str) -> dict[str, Any]:
    """Optional lossless audio remux: attach a source audio track to the VR180.

    Order matters: this stage runs *after* convert (which injected sv3d/st3d)
    and *before* QA. ffmpeg ``-c copy`` never touches user-data boxes, so the
    VR180 metadata survives the remux; the following QA stage then verifies
    sv3d/st3d are still present — a natural gate.

    Three paths:
      1. **No audio**  — source has no audio stream → skip, log, record ``copied=False``.
      2. **Copied**    — remux succeeds → atomic replace, record ``copied=True`` + codec.
      3. **Failed**    — remux raises → propagate RuntimeError (pipeline stops).

    Returns:
        A small result dict ``{"copied": bool, "codec": str|None, "source": str}``
        used by the manifest recorder.
    """
    from pipeline.audio_mux import audio_stream_info, copy_audio_to, has_audio_stream

    source = _audio_source_path(args, generated_video)
    result: dict[str, Any] = {"copied": False, "codec": None, "source": source}

    if not Path(source).is_file():
        log.info("🔊 Stage audio: source not found (%s) — skipping", source)
        return result

    if not has_audio_stream(source):
        log.info("🔊 Stage audio: no audio stream in %s — skipping (video will be silent)", source)
        return result

    info = audio_stream_info(source)
    codec = (info or {}).get("codec_name", "unknown") if info else "unknown"
    log.info("🔊 Stage audio: copying %s track (%s) %s → %s", codec, source, codec, vr180_path)
    copy_audio_to(vr180_path, source)
    result["copied"] = True
    result["codec"] = codec
    log.info("✅ Stage audio done — copied %s track", codec)
    return result


# ---------------------------------------------------------------------------
# Stage 7: QA
# ---------------------------------------------------------------------------


def stage_qa(video_path: str) -> int:
    """Run VR180 QA; return exit code (0 pass, 3 fail)."""
    from scripts.vr180_qa import format_human, run_qa

    log.info("🛠️  Stage qa: %s", video_path)
    report = run_qa(video_path)
    print(format_human(report))
    if report.failed:
        log.error("❌ QA failed for %s — see report above", video_path)
        return EXIT_QA_FAILED
    log.info("✅ Stage qa passed — %s", report.verdict)
    return 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _stage_label(name: str) -> str:
    labels = {
        "prepare": "prepare_image",
        "generate": "generate_from_image",
        "streamcheck": "streamcheck",
        "upscale": "upscale",
        "convert": "vr180_convert",
        "audio": "audio_remux",
        "qa": "qa",
    }
    return labels.get(name, name)


def run_pipeline(
    args: JobArgs,
    *,
    prepare: Callable[[JobArgs], str] | None = None,
    generate: Callable[[JobArgs, str | None], str] | None = None,
    streamcheck: Callable[[str], None] | None = None,
    upscale: Callable[[JobArgs, str], str] | None = None,
    convert: Callable[[JobArgs, str, Callable[..., str] | None], str] | None = None,
) -> dict[str, Any]:
    """Execute the full image→VR180 pipeline with manifest checkpointing.

    All stage callables default to ``None`` and resolve at *call* time to
    the current module-level function, so ``patch.object(i2v, "stage_x",
    fake)`` in a test actually reaches the pipeline. This keeps the
    call-time default in sync with any injected fake and avoids the
    "default bound at definition" gotcha.

    Injected stage callables let tests drive the whole orchestration
    (call order, resume skip, manifest contents, QA-fail exit) without
    touching providers, ffmpeg, or models.

    Returns:
        ``{"output": <vr180_path>, "qa_exit": <int>, "manifest": <dict>|None}``.
    Raises:
        RuntimeError when final QA fails.
    """
    if prepare is None:
        prepare = stage_prepare
    if generate is None:
        generate = stage_generate
    if streamcheck is None:
        streamcheck = stage_streamcheck
    if upscale is None:
        upscale = stage_upscale
    if convert is None:
        convert = stage_convert

    resolve_paths(args)
    ensure_workdir(args)
    manifest = _load_or_create_manifest(args)
    done = _stage_names_done(manifest)

    # Staged execution: each stage is attempted in order; completed stages
    # are skipped; every finished stage is recorded before the next runs
    # (so a downstream failure never forces regeneration).
    prepared: str | None = None
    generated: str | None = None
    upscaled: str | None = None
    converted: str | None = None
    qa_exit = 0

    if "prepare" not in done:
        log.info("▶  Stage prepare starting")
        prepared = prepare(args)
        log.info("✔  Stage prepare complete → %s", prepared)
        _record_stage(
            manifest,
            "prepare",
            inputs=[args.image],
            outputs=[args.prepared_image],
            params={"target_aspect": "16:9", "target_width": 1280, "mode": "letterbox"},
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage prepare: skipped (already done in manifest)")
        prepared = args.prepared_image

    if "generate" not in done:
        log.info("▶  Stage generate starting")
        generated = generate(args, prepared)
        log.info("✔  Stage generate complete → %s", generated)
        _record_stage(
            manifest,
            "generate",
            inputs=[prepared],
            outputs=[generated],
            params={
                "provider": args.provider,
                "duration": args.duration,
                "resolution": args.gen_resolution,
                "ratio": args.gen_ratio,
            },
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage generate: skipped (already done in manifest)")
        generated = args.generated_video

    if "streamcheck" not in done:
        log.info("▶  Stage streamcheck starting")
        streamcheck(generated)
        log.info("✔  Stage streamcheck complete")
        _record_stage(
            manifest,
            "streamcheck",
            inputs=[generated],
            outputs=[],
            params={},
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage streamcheck: skipped (already done in manifest)")

    if "upscale" not in done:
        log.info("▶  Stage upscale starting")
        upscaled = upscale(args, generated)
        log.info("✔  Stage upscale complete → %s", upscaled)
        _record_stage(
            manifest,
            "upscale",
            inputs=[generated],
            outputs=[upscaled] if upscaled != generated else [],
            params={"upscale": args.upscale},
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage upscale: skipped (already done in manifest)")
        upscaled = generated

    if "convert" not in done:
        log.info("▶  Stage convert starting")
        converted = convert(args, upscaled, None)
        log.info("✔  Stage convert complete → %s", converted)
        _record_stage(
            manifest,
            "convert",
            inputs=[upscaled],
            outputs=[converted],
            params={"quality": args.quality, "output_width": args.output_width},
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage convert: skipped (already done in manifest)")
        converted = args.vr180_output

    # H-1: audio passthrough. Runs after sv3d/st3d injection, before QA.
    if "audio" not in done:
        log.info("▶  Stage audio starting")
        audio_result = stage_audio(args, converted, generated)
        log.info("✔  Stage audio complete (copied=%s)", audio_result.get("copied"))
        _record_stage(
            manifest,
            "audio",
            inputs=[converted, audio_result.get("source", "")],
            outputs=[converted],
            params={
                "copied": audio_result.get("copied"),
                "codec": audio_result.get("codec"),
                "source": audio_result.get("source"),
            },
        )
        _persist_manifest(manifest, args)
    else:
        log.info("⏭️  Stage audio: skipped (already done in manifest)")

    log.info("▶  Stage qa starting")
    qa_exit = stage_qa(converted)
    log.info("✔  Stage qa complete (exit=%d)", qa_exit)
    _record_stage(
        manifest,
        "qa",
        inputs=[converted],
        outputs=[converted],
        params={"qa_exit": qa_exit},
    )
    _persist_manifest(manifest, args)

    if qa_exit != 0:
        raise RuntimeError(f"QA failed for {converted} — refusing to deliver a non-VR180 artefact")

    # D-1: sidecar — one JSON per output artefact (see pipeline.sidecar).
    # Runs after QA so the qa.block reflects the actual verdict.
    _write_sidecar(converted, args)

    return {"output": converted, "qa_exit": qa_exit, "manifest": manifest or None}


def ensure_workdir(args: JobArgs) -> None:
    """Create the workdir so stage output paths are writable."""
    Path(args.workdir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# D-1: sidecar JSON metadata (issue #78)
# ---------------------------------------------------------------------------


def _write_sidecar(output_path: str, args: JobArgs) -> None:
    """Write the per-artefact sidecar JSON after QA succeeds.

    Best-effort: a sidecar failure never aborts a deliverable. The QA block is
    computed inside :func:`pipeline.sidecar.build_sidecar` by re-running
    :func:`scripts.vr180_qa.run_qa`; that is cheap (ffprobe + box scan only)
    and keeps this orchestrator decoupled from the QA report shape.
    """
    from pipeline.sidecar import write_sidecar

    immersive = {
        "projection": "equirect",
        "fov_deg": 180,
        "stereo_layout": "side_by_side",
        "spatial_metadata": ["sv3d", "st3d"],
    }
    w = int(args.output_width or 0)
    h = int(args.output_height or 0)
    if w and h:
        immersive["eye_resolution"] = [w, h]

    generation = {
        "route": "vr180",
        "i2v_backend": args.provider,
        "upscaler": args.upscale if args.upscale != "none" else None,
        "source_image": args.image,
        "prompt": args.prompt,
    }
    # Drop None values so the JSON stays clean for DreamPortal.
    generation = {k: v for k, v in generation.items() if v is not None}

    try:
        write_sidecar(output_path, immersive=immersive, generation=generation)
    except Exception as exc:
        log.warning("⚠️  Sidecar write failed for %s: %s", output_path, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command image → VR180 pipeline (G-3). "
        "Normalises the image, generates video via an I2V provider, "
        "optionally super-resolves it, converts to VR180, and runs QA.",
    )
    parser.add_argument("--image", "-i", required=True, help="Input image path (required)")
    parser.add_argument(
        "--prompt",
        default="",
        help="Motion-only prompt describing camera movement (default: empty)",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["kling", "seedance", "veo", "mock"],
        help="Video generation provider (default: mock)",
    )
    parser.add_argument("--duration", type=int, default=5, help="Generated video duration in seconds (default: 5)")
    parser.add_argument(
        "--gen-resolution",
        default="480p",
        choices=["480p", "720p", "1080p"],
        help="Generation resolution tier (default: 480p — quota discipline). "
        "Higher tiers consume more quota; a reminder is logged when >480p is selected.",
    )
    parser.add_argument(
        "--gen-ratio",
        default="adaptive",
        help="Generation aspect ratio passed through to the provider (default: adaptive). "
        "Seedance accepts e.g. adaptive/16:9/9:16/1:1.",
    )
    parser.add_argument(
        "--upscale",
        default="none",
        choices=["seedvr2", "none"],
        help="Optional video super-resolution (default: none)",
    )
    parser.add_argument(
        "--quality",
        default="preview",
        choices=["preview", "standard", "high"],
        help="VR180 quality preset passed through to the converter (default: preview)",
    )
    parser.add_argument(
        "--comfort",
        default="balanced",
        choices=["safe", "balanced", "strong"],
        help=(
            "I-3: stereo comfort preset passed through to the converter (default: balanced). "
            "safe = low disparity / far convergence (watch-long-no-nausea, weak 3D); "
            "balanced = middle ground; strong = max pop, needs solid depth. "
            "Explicit --max-disparity / --convergence override the preset."
        ),
    )
    parser.add_argument(
        "--max-disparity",
        type=float,
        default=None,
        help="Explicit max-disparity override for the converter (overrides --comfort preset).",
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=None,
        help="Explicit convergence-plane override for the converter (overrides --comfort preset).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        metavar="PATH",
        help="Write/update a job manifest JSON at this path after the run",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="MANIFEST",
        help="Resume from a job manifest: completed stages are hash-validated, then skipped",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for intermediate artefacts (default: <image_dir>/<image_stem>_vr180)",
    )
    parser.add_argument(
        "--copy-audio-from",
        default=None,
        metavar="PATH",
        help="H-1: attach an audio track from this source file to the VR180 output. "
        "If omitted, the audio stage detects an audio stream in the generated video "
        "(e.g. Seedance's synced AAC) and copies it in; if present there, no flag needed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        job = JobArgs(
            image=str(Path(args.image).resolve()),
            prompt=args.prompt,
            provider=args.provider,
            duration=args.duration,
            gen_resolution=args.gen_resolution,
            gen_ratio=args.gen_ratio,
            upscale=args.upscale,
            quality=args.quality,
            comfort=args.comfort,
            max_disparity=args.max_disparity,
            convergence=args.convergence,
            copy_audio_from=args.copy_audio_from,
            workdir=args.workdir or "",
            manifest_path=args.manifest,
            resume_from=args.resume_from,
        )
        # H-2: remind the operator that a high generation tier burns more
        # quota. Default stays 480p so the discipline is unchanged.
        if job.gen_resolution != "480p":
            log.warning("⚠️  高档位（%s）消耗更多额度，请确认后再继续。", job.gen_resolution)
        resolve_paths(job)
        log.info("Image-to-VR180 pipeline starting → input=%s output=%s", job.image, job.vr180_output)

        result = run_pipeline(job)
        log.info("✅ Pipeline complete → %s", result["output"])
        print(f"\n📦 VR180 output: {result['output']}")
        return 0
    except RuntimeError as exc:
        log.error("❌ %s", exc)
        # A QA failure surfaces as a RuntimeError from run_pipeline with the
        # EXIT_QA_FAILED context; propagate that code so callers/scripting
        # can detect "failed QA" vs "other error" (1).
        if "QA failed" in str(exc):
            return EXIT_QA_FAILED
        return 1
    except (ValueError, FileNotFoundError, KeyboardInterrupt) as exc:
        log.error("❌ %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
