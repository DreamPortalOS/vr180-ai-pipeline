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

Output naming (issue #96 / K-1.1): the final artefact per job is named by
scene, not by input image, via :func:`pipeline.naming.compose_scene_name`
(D-4). ``scene_id`` falls back to a zero-padded job index (``job001``) when
omitted, so jobs are never silently homonymous. The summary table and the
state file record each job's *actual* scene-named output path.

Collision policy (auto-suffix, chosen over hard-fail): before writing, the
runner checks whether the target path already exists. If it does NOT match a
previously-succeeded artefact of the same scene run, the runner appends
``_c1``, ``_c2``, … to the filename until a free name is found. This is
robust to partial prior runs in a reused workdir while still guaranteeing no
silent overwrite.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("batch-runner")

# Collision-guard: when a scene-named target already exists and is NOT a
# previously-succeeded artefact of the same scene_id, we append an auto-
# suffix ``_c<N>`` (N = smallest integer >= 1 that yields a free filename).
# This is the chosen policy over hard-fail: it is robust to partial prior
# runs in the same workdir while still guaranteeing no silent overwrite.
_COLLISION_SUFFIX_RE = re.compile(r"_c(\d+)$")


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
# Scene-oriented naming (D-4 / issue #81)
# ---------------------------------------------------------------------------

# Regex mirror of pipeline.naming._SCENE_ID_RE — used to sanitize user-supplied
# scene_ids that would otherwise fail compose_scene_name. Characters outside
# the convention are stripped (rather than rejecting) so the runner stays
# tolerant of free-form identifiers while still emitting a valid filename.
_VALID_SCENE_ID_RE = re.compile(r"[^a-z0-9_-]+")


def _safe_scene_id(raw: str) -> str:
    """Sanitize a user-supplied scene_id into a valid naming token.

    Lowercases and strips any character not allowed by :func:`compose_scene_name`
    (``[a-z0-9_-]+``). Collapses empties to ``"scene"``.
    """
    token = _VALID_SCENE_ID_RE.sub("", raw.lower()).strip("_-")
    return token or "scene"


def _build_output_path(spec: JobSpec, workdir: Path) -> Path:
    """Compose the scene-named final output path for one job.

    Uses :func:`pipeline.naming.compose_scene_name` (D-4) so the final file
    is parseable by downstream assembly and — critically — is distinct per
    ``scene_id`` even when multiple jobs share the same input image.

    ``route`` is always ``vr180`` (this runner's route); ``preset`` mirrors the
    job's ``quality`` (preview/standard/high map to standalone/pcvr/...), with
    ``preview`` defaulting to the convention's standalone default.
    """
    from pipeline.naming import SceneAssetSpec, compose_scene_name

    # Map the runner's quality tiers onto the naming convention's presets.
    preset_by_quality = {"preview": "standalone", "standard": "standalone", "high": "pcvr"}
    preset = preset_by_quality.get(spec.quality, "standalone")

    spec_scene = SceneAssetSpec(
        scene_id=_safe_scene_id(spec.scene_id),
        scene_name=spec.scene_id,
        segment_index=1,
        route="vr180",
        preset=preset,
    )
    filename = compose_scene_name(spec_scene, extension="mp4")
    return workdir / filename


def _resolve_collision(target: Path, prior_successes: set[str]) -> Path:
    """Return a free output path, appending ``_c<N>`` if ``target`` already exists
    and is NOT a previously-succeeded artefact of this same scene run.

    A file that matches a known prior-success path is the orchestrator's own
    durable output from an earlier run of this very job — that is safe and
    expected (resume). Any other existing file is a foreign artefact and we
    refuse to silently overwrite it; instead we emit ``_c1``, ``_c2``, … until
    a free name is found.
    """
    target_str = str(target)
    if not target.is_file():
        return target
    # Same path as a previously-succeeded artefact → the orchestrator would
    # just re-write it anyway (idempotent). Safe to keep.
    if target_str in prior_successes:
        return target
    # Auto-suffix policy: append _c1 / _c2 / … until free.
    stem = target.stem
    suffix = target.suffix
    n = 1
    while True:
        candidate = target.parent / f"{stem}_c{n}{suffix}"
        if not candidate.is_file():
            return candidate
        n += 1


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


def _prior_success_outputs(state: dict[str, dict[str, Any]]) -> set[str]:
    """Return the set of output_path values of previously-succeeded jobs."""
    return {
        rec.get("output_path", "")
        for rec in state.values()
        if isinstance(rec, dict) and rec.get("status") == "success" and rec.get("output_path")
    }


def run_one_job(
    spec: JobSpec,
    *,
    tmp_workdir: Path | None = None,
    run_pipeline=None,
    state: dict[str, dict[str, Any]] | None = None,
) -> JobResult:
    """Run a single job through the orchestrator and return a :class:`JobResult`.

    The final ``vr180_output`` is recomputed from the job's ``scene_id`` via
    :func:`_build_output_path` (D-4 scene naming). This guarantees that two
    jobs sharing the same input image still produce distinct, parseable files.

    ``run_pipeline`` may be injected (tests) so the real I2V pipeline is
    never invoked in tests. By default it resolves to
    :func:`scripts.image_to_vr180.run_pipeline`.

    ``state`` carries previously-succeeded records so collision detection can
    distinguish "this file is my own durable output from a prior run" from
    "a foreign file happens to occupy this name" (see :func:`_resolve_collision`).
    """
    if run_pipeline is None:
        run_pipeline = _i2v().run_pipeline

    job = _make_job_args(spec, tmp_workdir=tmp_workdir)
    state = state or {}
    prior = _prior_success_outputs(state)

    # Re-derive the final output path per scene so co-located jobs sharing an
    # input image do not silently overwrite each other. The orchestrator then
    # writes into this path regardless of its internal (image-stem-based)
    # ``vr180_output`` default.
    workdir = Path(job.workdir)
    target = _resolve_collision(_build_output_path(spec, workdir), prior)
    job.vr180_output = str(target)
    # Guarantee the output directory is writable. When the runner omits
    # --workdir the orchestrator sets a default workdir but does not always
    # create it; and the scene-named target may live in a sub-location the
    # orchestrator never makedirs's. Either way, the runner owns the final
    # path now, so it must ensure its parent exists.
    target.parent.mkdir(parents=True, exist_ok=True)

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
        result = run_one_job(spec, state=state)
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
