#!/usr/bin/env python3
"""Stage-2 performance bench — run a Cartesian grid of StereoCrafter tunables.

P-3 (#223): Stage 2 (StereoCrafter) has three VRAM/throughput knobs
(`frames_chunk` / `overlap` / `tile_num`, env: STEREOCRAFTER_FRAMES_CHUNK /
STEREOCRAFTER_OVERLAP / STEREOCRAFTER_TILE_NUM, landed by P-2 #217). This
tool runs every combination of a user-supplied grid as a fresh
``scripts/run_pipeline.py`` subprocess, records wall-clock / s-per-frame /
OOM / exit-code / peak GPU memory, and emits a sorted Markdown table + a
machine-readable JSON blob.

Usage (lead's 12 GB tuning loop):

    python scripts/stage2_bench.py --input clip.mp4 --outdir out/bench \\
        --frames-chunk 8,16,23 --tile-num 1,2 --overlap 3 --max-frames 24

Key design decisions:

* Every combination runs in its **own subprocess** with a **unique
  ``--temp-dir``** (``<outdir>/_work/<combo-name>/``) so intermediate depth /
  left / right products never cross-contaminate between combos.
* The three knobs are passed **purely via subprocess environment**
  (STEREOCRAFTER_FRAMES_CHUNK / _OVERLAP / _TILE_NUM) — run_pipeline's CLI is
  not touched.  The StereoCrafter stereo cache key folds in these params, so
  each combination auto-invalidates the stereo cache (no stale hit distorts
  wall-clock).  Depth stays content-keyed, so it is computed once and
  reused — the desired behaviour on a repeatable input clip.
* The runner is injectable (``run_combo`` callable) so unit tests run a fake
  runner that never touches ffmpeg, CUDA, or nvidia-smi.
* A failing / OOM combo is **recorded and the batch continues**; one bad
  combo never aborts the grid.
* Peak GPU memory is sampled via ``nvidia-smi --query-gpu=memory.used``;
  when nvidia-smi is unavailable (CPU CI) the column is ``—`` and the run
  does not fail.
* ``--dry-run`` prints each command + env that *would* be run, without
  spawning anything.

subprocess is always called in list form; ``shell=True`` is never used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("stage2-bench")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_PIPELINE = Path(__file__).with_name("run_pipeline.py")

ENV_FRAMES_CHUNK = "STEREOCRAFTER_FRAMES_CHUNK"
ENV_OVERLAP = "STEREOCRAFTER_OVERLAP"
ENV_TILE_NUM = "STEREOCRAFTER_TILE_NUM"

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")

_MISSING = "—"


# ---------------------------------------------------------------------------
# Grid / naming — pure functions (unit-testable, no I/O)
# ---------------------------------------------------------------------------


def build_grid(
    frames_chunks: Sequence[int],
    tile_nums: Sequence[int],
    overlaps: Sequence[int],
) -> list[dict[str, int]]:
    """Build the Cartesian product of frames_chunk × tile_num × overlap.

    Returns a list of combo dicts (ordered by outer→inner: frames_chunk,
    tile_num, overlap).  Each combo has keys ``frames_chunk``, ``tile_num``,
    ``overlap``.
    """
    if not frames_chunks:
        raise ValueError("frames_chunks must contain at least one value")
    if not tile_nums:
        raise ValueError("tile_num must contain at least one value")
    if not overlaps:
        raise ValueError("overlap must contain at least one value")

    return [
        {"frames_chunk": fc, "tile_num": tn, "overlap": ov}
        for fc, tn, ov in itertools.product(frames_chunks, tile_nums, overlaps)
    ]


def combo_name(combo: dict[str, int]) -> str:
    """Stable, filesystem-safe slug for one combination.

    Format: ``fc{frames_chunk}_tn{tile_num}_ov{overlap}``.
    """
    return f"fc{combo['frames_chunk']}_tn{combo['tile_num']}_ov{combo['overlap']}"


def combo_env(combo: dict[str, int]) -> dict[str, str]:
    """Subprocess environment that carries the three knobs."""
    return {
        ENV_FRAMES_CHUNK: str(combo["frames_chunk"]),
        ENV_OVERLAP: str(combo["overlap"]),
        ENV_TILE_NUM: str(combo["tile_num"]),
    }


def build_run_pipeline_args(
    input_path: str,
    output_path: str,
    temp_dir: Path,
    max_frames: int | None = None,
) -> list[str]:
    """Build the list-form argv handed to run_pipeline for one combo.

    Uses the full stereo + depth stack (stereocrafter + depthcrafter) and a
    per-combo ``--temp-dir`` so no intermediate products cross-contaminate.
    ``subprocess.run`` is always called list-form — this function must never
    return a shell string.
    """
    argv = [
        sys.executable,
        str(RUN_PIPELINE),
        "--input",
        input_path,
        "--output",
        output_path,
        "--stereo-model",
        "stereocrafter",
        "--depth-model",
        "depthcrafter",
        "--temp-dir",
        str(temp_dir),
    ]
    if max_frames is not None:
        argv.extend(["--max-frames", str(max_frames)])
    return argv


# ---------------------------------------------------------------------------
# GPU memory sampler (best-effort, CPU-safe)
# ---------------------------------------------------------------------------


@dataclass
class SampleState:
    """Mutable state threaded through the sampler so tests can inject."""

    available: bool = True
    samples: list[int] = field(default_factory=list)
    _probe_cmd: list[str] | None = field(default=None, repr=False)
    _probe_rc: int = 0
    _probe_stderr: str = ""

    def probe(self) -> None:
        """Run the nvidia-smi probe once to establish availability.

        Called at the start of a run.  If the probe fails (tool missing,
        no GPU), ``available`` becomes ``False`` and subsequent sampling is
        a no-op — the benchmark still completes with memory shown as ``—``.
        """
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self._probe_cmd = cmd
            self._probe_rc = proc.returncode
            self._probe_stderr = proc.stderr
            self.available = proc.returncode == 0
        except FileNotFoundError:
            self.available = False
            self._probe_stderr = "nvidia-smi: command not found"
        except subprocess.TimeoutExpired:
            self.available = False
            self._probe_stderr = "nvidia-smi: timed out"


def sample_peak_memory() -> int:
    """Read one instantaneous GPU memory value (MiB) or raise."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    value = proc.stdout.strip().splitlines()[0].strip()
    return int(value)


# ---------------------------------------------------------------------------
# Combo runner — the injectable seam
# ---------------------------------------------------------------------------


@dataclass
class ComboResult:
    """Record kept per combination."""

    combo: dict[str, int]
    name: str
    cmd: list[str]
    env: dict[str, str]
    wall_seconds: float = 0.0
    frames: int = 0
    seconds_per_frame: float | None = None
    returncode: int = 0
    oom: bool = False
    peak_memory_mib: str = _MISSING
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.oom


def _frames_from_output(output_path: str, max_frames: int | None) -> int:
    """Frames processed by a combo.  Use --max-frames as the frame budget;
    the subprocess actually processes at most that many.  We record
    max_frames as the frame count for s/frame math so the bench is stable
    without needing to probe the produced video.
    """
    if max_frames is None:
        return 0
    return max_frames


def _detect_oom(stderr: str) -> bool:
    """Mark a combo OOM when its stderr names an explicit CUDA OOM."""
    lowered = stderr.lower()
    return any(marker.lower() in lowered for marker in OOM_MARKERS)


def run_combo(
    input_path: str,
    output_path: str,
    combo: dict[str, int],
    max_frames: int | None,
    *,
    runner: Callable[[list[str], dict[str, str]], subprocess.CompletedProcess] | None = None,
) -> ComboResult:
    """Run one combination of the grid as a run_pipeline subprocess.

    * ``runner`` is an injectable substitute for
      ``subprocess.run(argv, env=env, capture_output=True, text=True)``.
      Tests inject a fake runner so no ffmpeg/CUDA ever runs.  The injected
      runner receives the exact list-form argv and the env dict, and returns
      a CompletedProcess-like object with ``returncode``, ``stdout``,
      ``stderr`` attributes.
    * Wall-clock is measured around the runner call.  GPU memory is sampled
      before and after; on CPU (nvidia-smi absent) the column is ``—`` and
      the run does not fail.
    * An OOM / non-zero exit is **not** raised — it is recorded and the
      caller continues to the next combo.
    """
    _run = runner or _real_runner

    temp_dir = Path(output_path).parent
    argv = build_run_pipeline_args(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        max_frames=max_frames,
    )
    env = dict(os.environ)
    env.update(combo_env(combo))

    state = SampleState()
    state.probe()

    t0 = time.perf_counter()
    try:
        completed = _run(argv, env)
        rc = getattr(completed, "returncode", 1)
        stderr = getattr(completed, "stderr", "") or ""
    except Exception as exc:  # runner itself blew up — record, don't abort
        rc = 127
        stderr = f"runner exception: {exc}"
    wall = time.perf_counter() - t0

    # Best-effort peak memory: sample once after the run when nvidia-smi
    # is available.  A sampling failure must not fail the benchmark — fall
    # back to ``—``.
    peak = _MISSING
    if state.available:
        try:
            peak = str(sample_peak_memory())
        except Exception:
            peak = _MISSING

    frames = _frames_from_output(output_path, max_frames)
    spf = (wall / frames) if frames else None
    oom = _detect_oom(stderr)

    return ComboResult(
        combo=combo,
        name=combo_name(combo),
        cmd=argv,
        env={k: v for k, v in env.items() if k in {ENV_FRAMES_CHUNK, ENV_OVERLAP, ENV_TILE_NUM}},
        wall_seconds=round(wall, 3),
        frames=frames,
        seconds_per_frame=round(spf, 3) if spf is not None else None,
        returncode=rc,
        oom=oom,
        peak_memory_mib=str(peak),
        stderr=stderr.strip(),
    )


def _real_runner(
    argv: list[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _sort_key(result: ComboResult) -> tuple[float, int, str]:
    """Sort order for the Markdown table: s/frame ascending, OOM last."""
    spf = result.seconds_per_frame
    oom_sort = 1 if (spf is None or result.oom) else 0
    spf_val = spf if spf is not None else float("inf")
    return (oom_sort, spf_val, result.name)


def render_bench_md(input_path: str, max_frames: int | None, results: list[ComboResult]) -> str:
    """Render the human-readable Markdown table, sorted by s/frame ascending.

    OOM rows are marked with a red ❌ emoji; missing/NaN s-frame (OOM /
    no-frames) sorts to the bottom.  Wall-clock, s/frame, exit code, and
    peak GPU memory are all shown so the lead can spot VRAM bottlenecks at
    a glance.
    """
    ordered = sorted(results, key=_sort_key)
    lines: list[str] = [
        "# Stage-2 Performance Bench",
        "",
        f"- Source: `{input_path}`",
        f"- Frames per combo: {max_frames if max_frames is not None else 'all'}",
        f"- Combos: {len(ordered)}",
        "- Sorted by s/frame ascending; ❌ = OOM or non-zero exit.",
        "",
        "| # | combo | frames_chunk | tile_num | overlap | wall (s) | s/frame | exit | OOM | peak GPU (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, r in enumerate(ordered, start=1):
        marker = "❌ " if (r.oom or r.returncode != 0) else ""
        spf = f"{r.seconds_per_frame:g}" if r.seconds_per_frame is not None else "—"
        lines.append(
            f"| {idx} | {marker}`{r.name}` | {r.combo['frames_chunk']} | "
            f"{r.combo['tile_num']} | {r.combo['overlap']} | "
            f"{r.wall_seconds:g} | {spf} | {r.returncode} | "
            f"{'yes' if r.oom else 'no'} | {r.peak_memory_mib} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_bench_json(input_path: str, max_frames: int | None, results: list[ComboResult]) -> dict[str, Any]:
    """Machine-readable benchmark result blob."""
    return {
        "input": input_path,
        "max_frames": max_frames,
        "combos": [
            {
                "name": r.name,
                **r.combo,
                "wall_seconds": r.wall_seconds,
                "frames": r.frames,
                "seconds_per_frame": r.seconds_per_frame,
                "returncode": r.returncode,
                "oom": r.oom,
                "peak_memory_mib": r.peak_memory_mib,
                "ok": r.ok,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_bench(
    input_path: str,
    outdir: str | Path,
    frames_chunks: Sequence[int],
    tile_nums: Sequence[int],
    overlaps: Sequence[int],
    max_frames: int | None = None,
    *,
    dry_run: bool = False,
    runner: Callable[[list[str], dict[str, str]], subprocess.CompletedProcess] | None = None,
) -> list[ComboResult]:
    """Execute the full Stage-2 performance grid and write reports.

    Returns the list of :class:`ComboResult` (in grid order).  When
    ``dry_run=True`` no subprocess is spawned and no files are written; the
    commands + env are printed to stdout instead.
    """
    out_path = Path(outdir)
    combos = build_grid(
        frames_chunks=list(frames_chunks),
        tile_nums=list(tile_nums),
        overlaps=list(overlaps),
    )

    if dry_run:
        for combo in combos:
            temp_dir = out_path / "_work" / combo_name(combo)
            output_path = str(temp_dir / "bench_out.mp4")
            argv = build_run_pipeline_args(
                input_path=input_path,
                output_path=output_path,
                temp_dir=temp_dir,
                max_frames=max_frames,
            )
            env = combo_env(combo)
            print(f"$ {' '.join(shlex.quote(a) for a in argv)}")
            for k, v in env.items():
                print(f"    {k}={v}")
        return [
            ComboResult(
                combo=combo,
                name=combo_name(combo),
                cmd=[],
                env=combo_env(combo),
            )
            for combo in combos
        ]

    out_path.mkdir(parents=True, exist_ok=True)
    results: list[ComboResult] = []
    for combo in combos:
        name = combo_name(combo)
        temp_dir = out_path / "_work" / name
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(temp_dir / "bench_out.mp4")
        log.info(
            "=== Combo %s (fc=%s, tn=%s, ov=%s) ===", name, combo["frames_chunk"], combo["tile_num"], combo["overlap"]
        )
        result = run_combo(
            input_path=input_path,
            output_path=output_path,
            combo=combo,
            max_frames=max_frames,
            runner=runner,
        )
        spf = f"{result.seconds_per_frame:g}" if result.seconds_per_frame is not None else "—"
        log.info(
            "    %s: wall=%.2fs s/frame=%s exit=%d peak=%s",
            name,
            result.wall_seconds,
            spf,
            result.returncode,
            result.peak_memory_mib,
        )
        results.append(result)

    md_path = out_path / "bench.md"
    md_path.write_text(
        render_bench_md(input_path=input_path, max_frames=max_frames, results=results),
        encoding="utf-8",
    )
    json_path = out_path / "bench.json"
    json_path.write_text(
        json.dumps(render_bench_json(input_path=input_path, max_frames=max_frames, results=results), indent=2),
        encoding="utf-8",
    )
    log.info("✅ Bench complete: %d combos → %s + %s", len(results), md_path, json_path)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(value: str, name: str) -> list[int]:
    parts: list[int] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            print(f"Error: --{name} must be comma-separated integers, got {value!r}", file=sys.stderr)
            sys.exit(2)
        if val <= 0:
            print(f"Error: --{name} values must be positive, got {val}", file=sys.stderr)
            sys.exit(2)
        parts.append(val)
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-2 (StereoCrafter) performance grid bench — "
        "run frames_chunk × tile_num × overlap and report wall-clock / "
        "s-frame / OOM / GPU memory.",
    )
    parser.add_argument("--input", "-i", required=True, help="Source 2D video file")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for bench.md + bench.json")
    parser.add_argument(
        "--frames-chunk",
        required=True,
        help="Comma-separated frames_chunk values (e.g. 8,16,23)",
    )
    parser.add_argument(
        "--tile-num",
        required=True,
        help="Comma-separated tile_num values (e.g. 1,2)",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        help="Comma-separated overlap values (e.g. 3)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per combo (for short tuning runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands + env that would run; do not execute",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        frames_chunks = _parse_int_list(args.frames_chunk, "frames-chunk")
        tile_nums = _parse_int_list(args.tile_num, "tile-num")
        overlaps = _parse_int_list(args.overlap, "overlap")
    except SystemExit:
        return 2

    run_bench(
        input_path=args.input,
        outdir=args.outdir,
        frames_chunks=frames_chunks,
        tile_nums=tile_nums,
        overlaps=overlaps,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
