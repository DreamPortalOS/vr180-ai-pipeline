#!/usr/bin/env python3
"""Batch runner for issue-to-VR180 jobs (issue #87, K-1).

Turns a *batch* of image→VR180 jobs into a reproducible, failure-tolerant
production run:

    python scripts/batch_runner.py --jobs jobs.json

A jobs file is a JSON array of per-job specs:

    [{"image": "a.png", "prompt": "slow dolly in", "scene_id": "s01", "duration": 5},
     {"image": "b.png", "prompt": "gentle orbit", "scene_id": "s02"}]

Per-job fields that are missing fall back to the *CLI-level* defaults
(``--provider``, ``--gen-resolution``, ``--quality``, ``--upscale``,
``--duration``, ``--gen-ratio``) — the exact same defaults and the same
meaning as ``scripts.image_to_vr180``, simply forwarded.

The orchestrator's function entry point
(:func:`scripts.image_to_vr180.run_pipeline`) is called for every job; no
flow logic is duplicated here.

Fault tolerance: a failing job is recorded with its error and the runner
continues with the next one. ``--fail-fast`` switches to abort-on-first-error.
A summary table (success / failure / artefact path / duration) is printed at
the end. ``--dry-run`` prints the resolved job list without running anything.

Batch-level checkpoint (``--state batch_state.json``) records each job's
status and artefact so a re-run can resume: already-succeeded jobs are
skipped. This is a *batch-level* state file (NOT a per-stage job manifest;
job_manifest is untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("batch-runner")


# Exit codes (machine-detectable).
EXIT_OK = 0
EXIT_PARTIAL = 2  # some jobs failed (without --fail-fast)
EXIT_FAILED = 1  # fatal / fail-fast / bad input


# ---------------------------------------------------------------------------
# Per-job resolved spec
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    """A single resolved job: per-job overrides merged with CLI defaults."""

    scene_id: str
    image: str
    prompt: str = ""
    provider: str = "mock"
    duration: int = 5
    gen_resolution: str = "480p"
    gen_ratio: str = "adaptive"
    upscale: str = "none"
    quality: str = "preview"
    workdir: str = ""
    copy_audio_from: str | None = None


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    scene_id: str
    status: str  # "success" | "failed" | "skipped" | "not_run"
    output_path: str = ""
    error: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Job spec resolution
# ---------------------------------------------------------------------------


def _cli_defaults(args: argparse.Namespace) -> dict[str, Any]:
    """Extract CLI-level default values to fall back to for each job."""
    return {
        "provider": args.provider,
        "duration": args.duration,
        "gen_resolution": args.gen_resolution,
        "gen_ratio": args.gen_ratio,
        "upscale": args.upscale,
        "quality": args.quality,
        "workdir": args.workdir or "",
        "copy_audio_from": args.copy_audio_from,
    }


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the jobs JSON file.

    Raises ``RuntimeError`` when the file is missing, malformed, or
    structurally invalid (non-array / wrong item type).
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Jobs file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Jobs file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Jobs file must be a JSON array, got {type(data).__name__}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"Job at index {i} is not an object")
    return data


def resolve_job(raw: dict[str, Any], defaults: dict[str, Any], idx: int) -> JobSpec:
    """Merge a raw job dict with CLI defaults into a resolved :class:`JobSpec`.

    ``image`` and ``scene_id`` are required per job; everything else falls
    back to the CLI-level defaults (or the JobSpec module defaults).
    Raises ``RuntimeError`` on missing required fields.
    """
    image = raw.get("image")
    if not image:
        raise RuntimeError(f"Job at index {idx} is missing required field 'image'")
    scene_id = str(raw.get("scene_id", f"job-{idx:03d}"))

    return JobSpec(
        scene_id=scene_id,
        image=str(image),
        prompt=str(raw.get("prompt", defaults.get("prompt", ""))),
        provider=str(raw.get("provider", defaults["provider"])),
        duration=int(raw.get("duration", defaults["duration"])),
        gen_resolution=str(raw.get("gen_resolution", defaults["gen_resolution"])),
        gen_ratio=str(raw.get("gen_ratio", defaults["gen_ratio"])),
        upscale=str(raw.get("upscale", defaults["upscale"])),
        quality=str(raw.get("quality", defaults["quality"])),
        workdir=str(raw.get("workdir", defaults["workdir"])),
        copy_audio_from=raw.get("copy_audio_from", defaults["copy_audio_from"]),
    )


def resolve_all(raw_jobs: list[dict[str, Any]], defaults: dict[str, Any]) -> list[JobSpec]:
    """Resolve every raw job into a :class:`JobSpec`."""
    return [resolve_job(raw, defaults, i) for i, raw in enumerate(raw_jobs)]


# ---------------------------------------------------------------------------
# State (batch-level checkpoint)
# ---------------------------------------------------------------------------


def _state_load(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the ``jobs`` map from the state file, or ``{}`` if absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        log.warning("State file %s is not valid JSON — starting fresh", path)
        return {}
    return data.get("jobs", {})


def _state_save(path: str | Path, jobs: list[JobResult], merge_with: dict[str, dict[str, Any]] | None = None) -> None:
    """Persist the batch state with one record per job.

    When resuming, ``merge_with`` carries the previously-persisted records
    so a re-visited *skipped* job keeps its durable ``success`` status
    instead of being rewritten as ``skipped`` by this pass.
    """
    state = {
        "version": 1,
        "jobs": {},
        "order": [],
    }
    if merge_with:
        state["jobs"] = dict(merge_with)
    for j in jobs:
        existing = state["jobs"].get(j.scene_id)
        # Keep a durable ``success`` record intact when this pass only re-
        # visited it as ``skipped`` (resume). A real re-run (success/failed)
        # overwrites the prior record as expected.
        if existing and existing.get("status") == "success" and j.status == "skipped":
            continue
        state["jobs"][j.scene_id] = asdict(j)
        state["order"].append(j.scene_id)
    # Deduplicate ``order`` while preserving first-seen sequence.
    seen: set[str] = set()
    state["order"] = [sid for sid in state["order"] if not (sid in seen or seen.add(sid))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _is_done(state: dict[str, dict[str, Any]], scene_id: str) -> bool:
    rec = state.get(scene_id)
    return isinstance(rec, dict) and rec.get("status") == "success"


def _artefact_of(state: dict[str, dict[str, Any]], scene_id: str) -> str:
    rec = state.get(scene_id, {})
    return str(rec.get("output_path", ""))


# ---------------------------------------------------------------------------
# Run one job (delegates to the orchestrator)
# ---------------------------------------------------------------------------


def _i2v() -> Any:
    """Return the canonical orchestrator module. Imported lazily so it can be
    patched (patch by name) by tests without a "bound-at-definition" gotcha."""
    import scripts.image_to_vr180 as i2v

    return i2v


def _make_job_args(spec: JobSpec, tmp_workdir: Path | None = None) -> Any:
    """Build the orchestrator's JobArgs from a resolved JobSpec.

    Returns an instance of the orchestrator's ``JobArgs`` dataclass.
    """
    import scripts.image_to_vr180 as i2v

    image = str(spec.image)
    workdir = spec.workdir
    if tmp_workdir is not None:
        # Allow tests to force a workdir into tmp_path.
        workdir = str(tmp_workdir)

    job = i2v.JobArgs(
        image=image,
        prompt=spec.prompt,
        provider=spec.provider,
        duration=spec.duration,
        gen_resolution=spec.gen_resolution,
        gen_ratio=spec.gen_ratio,
        upscale=spec.upscale,
        quality=spec.quality,
        copy_audio_from=spec.copy_audio_from,
        workdir=workdir or "",
    )
    i2v.resolve_paths(job)
    if workdir:
        i2v.ensure_workdir(job)
    return job


def run_one_job(
    spec: JobSpec,
    *,
    tmp_workdir: Path | None = None,
    run_pipeline=None,
) -> JobResult:
    """Run a single job through the orchestrator and return a :class:`JobResult`.

    ``run_pipeline`` may be injected (tests) so the real I2V pipeline is
    never invoked in tests. By default it resolves to
    :func:`scripts.image_to_vr180.run_pipeline`.
    """
    if run_pipeline is None:
        run_pipeline = _i2v().run_pipeline

    job = _make_job_args(spec, tmp_workdir=tmp_workdir)
    start = time.perf_counter()
    try:
        result = run_pipeline(job)
        elapsed = time.perf_counter() - start
        return JobResult(
            scene_id=spec.scene_id,
            status="success",
            output_path=str(result.get("output", job.vr180_output)),
            duration_s=round(elapsed, 3),
        )
    except BaseException as exc:
        elapsed = time.perf_counter() - start
        return JobResult(
            scene_id=spec.scene_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_s=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def format_summary(results: list[JobResult]) -> str:
    """Render a compact summary table of the batch run.

    Columns: scene_id / status / output_path / duration / error.
    """
    if not results:
        return "No jobs to report."

    def col_width(rows: list[str]) -> int:
        return max((len(r) for r in rows), default=0)

    sid_w = col_width([r.scene_id for r in results])
    out_w = col_width([r.output_path for r in results])
    err_w = col_width([r.error for r in results])

    hdr = f"{'scene_id':<{sid_w}}  {'status':<8}  {'output_path':<{out_w}}  {'duration_s':>10}  {'error':<{err_w}}"
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for r in results:
        lines.append(
            f"{r.scene_id:<{sid_w}}  {r.status:<8}  {r.output_path:<{out_w}}  {r.duration_s:>10.3f}  {r.error:<{err_w}}"
        )

    ok = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status == "failed")
    skip = sum(1 for r in results if r.status == "skipped")
    total = len(results)
    lines.append("")
    lines.append(f"Total: {total}  |  succeeded: {ok}  failed: {fail}  skipped: {skip}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------


def dry_run_print(specs: list[JobSpec]) -> str:
    """Return a human-readable listing of the jobs that *would* run."""
    lines = [f"Dry run — {len(specs)} job(s) would be executed:"]
    for i, s in enumerate(specs, start=1):
        lines.append(
            f"  [{i}] scene={s.scene_id} image={s.image} prompt={s.prompt!r} "
            f"provider={s.provider} dur={s.duration}s {s.gen_resolution}/{s.gen_ratio} "
            f"quality={s.quality} upscale={s.upscale}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch image → VR180 runner (K-1). "
        "Runs a list of jobs from --jobs, tolerating per-job failures, "
        "with checkpoint resume and a summary table.",
    )
    parser.add_argument("--jobs", required=True, help="Path to a JSON array of job specs (required)")
    parser.add_argument("--state", default=None, metavar="PATH", help="Batch state JSON path for checkpoint resume")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved job list without running anything")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Abort on the first failed job (default: continue and report all)"
    )

    # Defaults shared with scripts.image_to_vr180 — forwarded, not re-defined.
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
        help="Generation resolution tier (default: 480p)",
    )
    parser.add_argument("--gen-ratio", default="adaptive", help="Generation aspect ratio (default: adaptive)")
    parser.add_argument(
        "--upscale", default="none", choices=["seedvr2", "none"], help="Optional video super-resolution (default: none)"
    )
    parser.add_argument(
        "--quality",
        default="preview",
        choices=["preview", "standard", "high"],
        help="VR180 quality preset (default: preview)",
    )
    parser.add_argument(
        "--workdir", default=None, help="Working directory override for all jobs (default: per-job/auto)"
    )
    parser.add_argument(
        "--copy-audio-from",
        default=None,
        metavar="PATH",
        help="H-1: audio source forwarded to every job (default: per-job/auto)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        raw_jobs = load_jobs(args.jobs)
    except RuntimeError as exc:
        log.error("❌ %s", exc)
        return EXIT_FAILED

    defaults = _cli_defaults(args)
    try:
        specs = resolve_all(raw_jobs, defaults)
    except RuntimeError as exc:
        log.error("❌ %s", exc)
        return EXIT_FAILED

    if args.dry_run:
        print(dry_run_print(specs))
        return EXIT_OK

    state = _state_load(args.state) if args.state else {}

    results: list[JobResult] = []
    for spec in specs:
        if _is_done(state, spec.scene_id):
            log.info(
                "⏭️  %s: already succeeded (resuming) — output %s", spec.scene_id, _artefact_of(state, spec.scene_id)
            )
            results.append(
                JobResult(
                    scene_id=spec.scene_id,
                    status="skipped",
                    output_path=_artefact_of(state, spec.scene_id),
                )
            )
            if args.state:
                _state_save(args.state, results, merge_with=state)
            continue

        log.info("▶  Running %s (image=%s)", spec.scene_id, spec.image)
        result = run_one_job(spec)
        results.append(result)

        if args.state:
            _state_save(args.state, results, merge_with=state)

        if result.status == "success":
            log.info("✅ %s: %s (%.3fs)", spec.scene_id, result.output_path, result.duration_s)
        else:
            log.error("❌ %s: %s (%.3fs)", spec.scene_id, result.error, result.duration_s)

        if result.status == "failed" and args.fail_fast:
            log.error("⏹  --fail-fast: aborting after first failure")
            print(format_summary(results))
            return EXIT_FAILED

    print(format_summary(results))

    n_fail = sum(1 for r in results if r.status == "failed")
    return EXIT_PARTIAL if n_fail else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
