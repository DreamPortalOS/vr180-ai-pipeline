#!/usr/bin/env python3
"""K-4 (#138): one-shot comparison-sample builder — same source, N recipes.

Every time the lead prepares samples for the owner they hand-type several
``run_pipeline`` commands, QA each output one by one, and ``adb push`` them
one by one — and the owner's feedback quality depends on being able to A/B
(recipes that differ by *exactly one* parameter axis).  This script turns that
routine into a single command:

1. **Recipes** — a module-level table (:data:`DEFAULT_RECIPES`).  Each recipe
   is a short name plus the *extra* ``run_pipeline`` CLI flags it adds on top
   of the shared base flags, e.g.::

       baseline      = --comfort strong                       (per-frame depth)
       temporal      = --depth-model depthcrafter --comfort safe
       temporal_occl = temporal + --stereo-model stereocrafter

   Add/remove recipes by editing the constant — nothing else changes.
2. **Render** — each recipe is rendered through a callable runner
   (:func:`default_render_runner`), which drives the existing
   ``scripts/run_pipeline.py`` CLI as a subprocess.  **No conversion logic is
   duplicated here.**  Tests inject a fake runner, so CI never renders.
3. **Naming** — output filenames are composed via
   :func:`pipeline.naming.compose_scene_name` (D-4, merged): the source stem
   becomes the ``scene_name`` and the recipe short name becomes the ``preset``
   field, so every artefact carries its recipe in its name.
4. **QA + summary** — every product is validated with
   :func:`scripts.vr180_qa.run_qa`; results land in one table
   (recipe / resolution / audio / QA verdict / elapsed seconds).
5. **Depth-stability metrics** (K-16, #206) — after QA each rendered recipe is
   annotated with three objective "is it dizzying?" cells
   (temporal_jitter / flicker_ratio / edge_consistency, each with an OK/WARN/FAIL
   mark) by calling :mod:`scripts.depth_stability`.  Missing depth products or a
   depth_stability failure leaves the cell as ``—`` and never breaks the A/B set;
   ``--no-metrics`` skips the call entirely.  A lead-measured single-frame depth
   baseline is printed under the table so improvement is visible at a glance.
6. **comparison.md** — the table above plus *empty* scoring columns
   (清晰度/重影/立体感/舒适度) for the owner to fill in inside the headset.
7. **--push-to-quest** — optional ``adb -s <serial> push`` of every product
   followed by a media-scan broadcast.  Serial comes from ``--quest-serial``
   or the ``QUEST_SERIAL`` env var; when unset the step is skipped gracefully
   with a hint (never an error).
8. **--dry-run** — prints the resolved recipes, output names and the exact
   command lines without rendering anything.

Usage:
    python scripts/make_comparison.py --input video.mp4 --outdir out/cmp
    python scripts/make_comparison.py -i video.mp4 -o out/cmp --dry-run
    python scripts/make_comparison.py -i video.mp4 -o out/cmp --push-to-quest
    python scripts/make_comparison.py -i video.mp4 -o out/cmp \\
        --extra-arg=--quality --extra-arg=high
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# K-15 (#205): let this script run directly (``python scripts/make_comparison.py``)
# without the caller having to set PYTHONPATH — put the repo root on sys.path
# before importing the ``pipeline`` / ``scripts`` packages.  Idempotent (no
# duplicate entries) and a no-op when PYTHONPATH already points here.  With the
# repo root on sys.path the ``scripts.vr180_qa`` package import below succeeds
# for direct invocation too, so the flat-import fallback is no longer reached
# (kept for safety / other invocation styles — ``# pragma: no cover``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.naming import SceneAssetSpec, compose_scene_name  # noqa: E402

try:
    # Package import (tests, ``python -m scripts.make_comparison``).
    from scripts.vr180_qa import run_qa
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    # Direct invocation (``python scripts/make_comparison.py``) puts scripts/
    # itself on sys.path, shadowing the ``scripts`` package — import the QA
    # module flat instead.
    from vr180_qa import run_qa  # type: ignore[no-redef]

# depth_stability is only imported for the optional metrics step; keep it
# lazy-tolerant so a broken import never blocks the core comparison flow.
try:
    from scripts.depth_stability import (
        StabilityReport,
        compute_report,
        load_depth_npy_dir,
    )
except ModuleNotFoundError:  # pragma: no cover - depth_stability lives next door
    from depth_stability import (  # type: ignore[no-redef]
        StabilityReport,
        compute_report,
        load_depth_npy_dir,
    )

log = logging.getLogger("make-comparison")

# ---------------------------------------------------------------------------
# Recipe table (module constant — edit here to add/remove recipes)
# ---------------------------------------------------------------------------

#: Shared base flags every recipe renders with.  ``standard`` is the V-1
#: baseline tier (2880²/eye, streaming) — cheap enough that an N-recipe
#: comparison stays feasible, sharp enough for headset A/B.
DEFAULT_BASE_ARGS: tuple[str, ...] = ("--quality", "standard")

#: Default recipe set.  Each recipe is one axis of difference against
#: ``baseline`` so the owner's A/B verdicts map to a single parameter change.
#: ``baseline`` mirrors the old hand-graded samples (per-frame depth, strong
#: comfort); ``temporal`` isolates DepthCrafter's temporal consistency;
#: ``temporal_occl`` adds StereoCrafter's disocclusion handling on top.
DEFAULT_RECIPES: tuple[dict, ...] = (
    {
        "name": "baseline",
        "description": "单帧深度 + comfort strong（旧样片基线）",
        "args": ("--comfort", "strong"),
    },
    {
        "name": "temporal",
        "description": "DepthCrafter 时序一致深度 + comfort safe",
        "args": ("--depth-model", "depthcrafter", "--comfort", "safe"),
    },
    {
        "name": "temporal_occl",
        "description": "temporal + StereoCrafter 遮挡修复",
        "args": (
            "--depth-model",
            "depthcrafter",
            "--stereo-model",
            "stereocrafter",
            "--comfort",
            "safe",
        ),
    },
)

#: Owner-facing scoring columns (left empty in comparison.md).
SCORE_COLUMNS: tuple[str, ...] = ("清晰度", "重影", "立体感", "舒适度")

#: Filename extension for every rendered artefact.
OUTPUT_EXTENSION = "mp4"

#: adb destination on the Quest for pushed artefacts.
QUEST_PUSH_DIR = "/sdcard/Movies/vr180-comparison"

#: Env var read when --quest-serial is not given.
QUEST_SERIAL_ENV = "QUEST_SERIAL"

#: The three depth-stability metric columns added to the summary table (K-16,
#: #206).  Each cell shows the value + an OK/WARN/FAIL mark derived from the
#: thresholds in :mod:`scripts.depth_stability` (reused, never re-defined here).
DEPTH_METRIC_COLUMNS: tuple[str, ...] = (
    "temporal_jitter",
    "flicker_ratio",
    "edge_consistency",
)

#: Lead's measured single-frame depth baseline (the "dizzy" axis): per-frame
#: Depth-Anything depth maps flip ~52 % of pixels frame-to-frame and keep only
#: ~17 % edge overlap.  Shown under the table so the owner can see at a glance
#: whether a recipe improved on it.  Source: lead headset + metric run.
#:
#: ``flicker_ratio`` baseline first (lower is better), then ``edge_consistency``
#: (higher is better) — matches the order they appear in the note line.
DEPTH_BASELINE_FLICKER: float = 0.5221
DEPTH_BASELINE_EDGE: float = 0.1726

#: Placeholder shown in a metric cell when no value could be produced (no depth
#: products, or the depth_stability call raised).  Never fails the comparison.
DEPTH_METRIC_NA: str = "—"

#: depth-model names a recipe may select, in resolution-search order.  Used by
#: the default depth-dir resolver to enumerate the model-scoped checkpoint dirs
#: written by run_pipeline's :func:`get_depth_dir` (I-6, #121).
DEPTH_MODEL_NAMES: tuple[str, ...] = ("depth-anything", "depthcrafter")


# ---------------------------------------------------------------------------
# Recipe expansion / naming / rendering — injectable seams for tests
# ---------------------------------------------------------------------------


@dataclass
class Recipe:
    """One comparison recipe: short name + extra run_pipeline CLI args."""

    name: str
    args: list[str]
    description: str = ""


def load_recipes(table: Sequence[dict] | None = None) -> list[Recipe]:
    """Expand a recipe *table* (defaults to :data:`DEFAULT_RECIPES`) into
    :class:`Recipe` objects, validating as we go.

    Raises ``ValueError`` on duplicate or empty names, or a non-list ``args``.
    """
    raw = list(table) if table is not None else list(DEFAULT_RECIPES)
    if not raw:
        raise ValueError("recipe table is empty — nothing to compare")

    recipes: list[Recipe] = []
    seen: set[str] = set()
    for entry in raw:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"recipe is missing a non-empty 'name': {entry!r}")
        if name in seen:
            raise ValueError(f"duplicate recipe name: {name!r}")
        seen.add(name)
        args = entry.get("args", ())
        if not isinstance(args, (list, tuple)) or not all(isinstance(a, str) for a in args):
            raise ValueError(f"recipe {name!r}: 'args' must be a list/tuple of CLI strings, got {args!r}")
        recipes.append(
            Recipe(name=name, args=list(args), description=str(entry.get("description", ""))),
        )
    return recipes


def recipe_output_name(input_path: str, recipe_name: str) -> str:
    """Compose the artefact filename for one recipe via D-4 naming.

    ``cmp_<stem>_<recipe>_seg01_vr180_sbs_standalone.mp4`` — the recipe short
    name rides in the slugified ``scene_name`` field so it round-trips through
    :func:`parse_scene_name` (recipe names with underscores would corrupt the
    strict ``preset`` field, whose parser treats an underscore as the D-3
    projection-mark separator).  The source stem is concatenated into
    ``scene_name`` — slugified on compose — rather than ``scene_id`` (strict
    lowercase regex), so arbitrary source filenames compose without error.
    Naming rule lives in :mod:`pipeline.naming`; nothing is re-implemented here.
    """
    stem = Path(input_path).stem
    return compose_scene_name(
        SceneAssetSpec(
            scene_id="cmp",
            scene_name=f"{stem} {recipe_name}",
            segment_index=1,
            route="vr180",
            projection_mark="sbs",
            preset="standalone",
        ),
        extension=OUTPUT_EXTENSION,
    )


def build_render_command(
    input_path: str,
    output_path: str,
    recipe: Recipe,
    base_args: Sequence[str] = (),
) -> list[str]:
    """Assemble the exact ``run_pipeline`` command line for one recipe.

    Kept separate from the runner so ``--dry-run`` can print *precisely* what
    would execute, and tests can assert on the argv without spawning anything.
    """
    return [
        sys.executable,
        str(Path(__file__).with_name("run_pipeline.py")),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        *base_args,
        *recipe.args,
    ]


def default_render_runner(cmd: Sequence[str]) -> str:
    """Run one recipe by invoking the existing run_pipeline CLI as a subprocess.

    List-form argv (no ``shell=True``) per repo discipline; the child's output
    is inherited so the operator watches the render live.  Returns the
    ``--output`` path from the command line.  Raises ``RuntimeError`` on a
    non-zero exit (with the exit code) so the orchestrator can mark the recipe
    failed and continue with the rest — one bad recipe must not kill the A/B.
    """
    log.info("🎬 Render: %s", " ".join(cmd))
    # argv is list-form (no shell) per repo discipline.
    proc = subprocess.run(list(cmd))
    if proc.returncode != 0:
        raise RuntimeError(f"run_pipeline exited with code {proc.returncode}: {' '.join(cmd)}")
    # The output path is the token after --output.
    idx = list(cmd).index("--output")
    return str(cmd[idx + 1])


# ---------------------------------------------------------------------------
# Result table + comparison.md
# ---------------------------------------------------------------------------


@dataclass
class RecipeResult:
    """One row of the comparison summary."""

    recipe: str
    description: str = ""
    output: str = ""
    ok: bool = False
    elapsed_s: float = 0.0
    error: str = ""
    resolution: str = "-"
    audio: str = "-"
    verdict: str = "未渲染"
    qa_failed: bool = False
    # K-16 (#206): depth-stability cells.  Each is "<value> <verdict>" (e.g.
    # "0.5221 FAIL") or DEPTH_METRIC_NA when the metric could not be computed.
    temporal_jitter: str = DEPTH_METRIC_NA
    flicker_ratio: str = DEPTH_METRIC_NA
    edge_consistency: str = DEPTH_METRIC_NA
    extra: dict = field(default_factory=dict)


def qa_recipe_output(output_path: str, result: RecipeResult) -> RecipeResult:
    """Run vr180_qa against one rendered artefact and fill in the row.

    Resolution, audio and verdict are pulled from the QA report; a report with
    failing checks is flagged (``qa_failed``) but the row is still produced —
    the summary table must show *all* recipes, including broken ones.
    """
    report = run_qa(output_path)
    if report.width and report.height:
        result.resolution = f"{report.width}×{report.height}"
    result.audio = report.audio_codec or "无"
    result.verdict = report.verdict
    result.qa_failed = report.failed
    result.extra["qa"] = {c.name: f"{c.status}: {c.detail}" for c in report.checks}
    return result


# ---------------------------------------------------------------------------
# Depth-stability metrics (K-16, #206) — turns "is it dizzying?" into a number
# ---------------------------------------------------------------------------


def default_depth_dir_resolver(
    input_path: str,
    recipe: Recipe,
    result: RecipeResult,
) -> str | None:
    """Locate the depth-product directory one recipe wrote, if any.

    run_pipeline writes ``depth_*.npy`` into a model-scoped checkpoint dir
    (I-6, #121): ``<input_dir>/<input_stem>_vr180_temp/depth/<depth_model>/``
    (or ``--temp-dir/.../depth/<model>``).  The recipe flags tell us which model
    it *asked* for, but the resolver also checks the other model dir so a depth
    stage that silently fell back (e.g. depthcrafter→depth-anything) is still
    found.  Returns the first dir containing ``depth_*.npy``, else ``None``.

    Tests inject a fake resolver; the default never loads a model.
    """
    stem = Path(input_path).stem
    base_temp = Path(input_path).parent / f"{stem}_vr180_temp"
    # Prefer the model the recipe selected (depthcrafter if it asked, else
    # depth-anything), then fall back to the other model dir.
    requested = "depthcrafter" if any(a == "depthcrafter" for a in recipe.args) else "depth-anything"
    ordered = [requested] + [m for m in DEPTH_MODEL_NAMES if m != requested]
    for model in ordered:
        cand = base_temp / "depth" / model
        if cand.is_dir() and any(cand.glob("depth_*.npy")):
            return str(cand)
    return None


def run_depth_metrics(depth_dir: str | Path) -> StabilityReport:
    """Load a depth dir and compute the three stability metrics.

    Thin wrapper over :mod:`scripts.depth_stability` — this exists as a seam so
    tests can inject a fake (or assert it was never called) without depending on
    numpy/real depth files.  Thresholds come entirely from ``compute_report``
    (the ``JITTER_*`` / ``FLICKER_*`` / ``EDGE_*`` constants in depth_stability),
    so the OK/WARN/FAIL marks are the *same* thresholds everywhere.
    """
    depths = load_depth_npy_dir(depth_dir)
    return compute_report(depths)


def _metric_cell(metric) -> str:
    """Format one :class:`MetricResult`-like as ``"<value> <verdict>"``."""
    return f"{float(metric.value):.4f} {metric.ok}"


def apply_depth_metrics(
    input_path: str,
    recipe: Recipe,
    result: RecipeResult,
    depth_dir_resolver: Callable[..., str | None] = default_depth_dir_resolver,
    metrics_runner: Callable[[str | Path], StabilityReport] = run_depth_metrics,
) -> RecipeResult:
    """Fill one row's three depth-metric cells, degrading to ``—`` on any miss.

    This is a best-effort *post-hoc* statistic: a missing depth dir, an empty
    dir, or an exception from depth_stability all leave the cells as ``—`` and
    log a warning — **never** raise, never fail the comparison.  The render/QA
    verdict above is what gates shipping; this just annotates.
    """
    try:
        depth_dir = depth_dir_resolver(input_path, recipe, result)
    except Exception as exc:  # resolver must not be able to kill the A/B
        log.warning("⚠️  depth-dir resolver for %s raised: %s", recipe.name, exc)
        depth_dir = None
    if not depth_dir:
        # No depth products for this recipe — not an error (e.g. --force-sbs).
        log.info("depth metrics: no depth dir for %s — cells stay —", recipe.name)
        return result
    try:
        report = metrics_runner(depth_dir)
    except Exception as exc:  # the headline K-16 requirement: never fail here
        log.warning("⚠️  depth_stability failed for %s (%s): %s", recipe.name, depth_dir, exc)
        return result
    result.temporal_jitter = _metric_cell(report.temporal_jitter)
    result.flicker_ratio = _metric_cell(report.flicker_ratio)
    result.edge_consistency = _metric_cell(report.edge_consistency)
    return result


def _status_cell(result: RecipeResult) -> str:
    if not result.ok:
        return f"❌ {result.error or 'render failed'}"
    return "⚠️ QA fail" if result.qa_failed else "✅"


def render_summary_table(results: Sequence[RecipeResult]) -> str:
    """Render the recipe / 分辨率 / 音频 / QA / 耗时 / 深度稳定性 summary as Markdown.

    The last three columns (temporal_jitter / flicker_ratio / edge_consistency)
    are the K-16 (#206) objective "dizzy" axis: each cell is ``"<value> <OK/
    WARN/FAIL>"`` or ``—`` when depth_stability could not run for that recipe.
    """
    lines = [
        "| recipe | 说明 | 分辨率 | 音频 | QA 判定 | 耗时 | 状态 | " + " | ".join(DEPTH_METRIC_COLUMNS) + " |",
        "| --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in DEPTH_METRIC_COLUMNS) + " |",
    ]
    for r in results:
        lines.append(
            f"| {r.recipe} | {r.description or '-'} | {r.resolution} | {r.audio} "
            f"| {r.verdict} | {r.elapsed_s:.1f}s | {_status_cell(r)} | "
            f"{r.temporal_jitter} | {r.flicker_ratio} | {r.edge_consistency} |"
        )
    return "\n".join(lines)


def render_comparison_md(
    results: Sequence[RecipeResult],
    source: str,
    outdir: str | Path,
    pushed: bool = False,
) -> str:
    """Render ``comparison.md``: summary table + empty owner scoring columns."""
    lines = [
        "# VR180 对比样片（comparison）",
        "",
        f"- 源视频: `{source}`",
        f"- 输出目录: `{outdir}`",
        f"- 配方数: {len(results)}",
        f"- 已推送 Quest: {'是' if pushed else '否'}",
        "",
        "## 汇总",
        "",
        render_summary_table(results),
        "",
        (
            "> 深度稳定性（客观「晕」指标）：flicker_ratio 越低越好；"
            "edge_consistency 越高越好；temporal_jitter 越低越好。"
            f" 参照 lead 实测单帧深度基线 flicker_ratio={DEPTH_BASELINE_FLICKER:.4f}"
            f" / edge_consistency={DEPTH_BASELINE_EDGE:.4f}"
            "（=「非常晕」的量化解释，越远离这组数字越好）。"
            " `—` 表示该配方无深度产物或统计失败，不影响出片。"
        ),
        "",
        "## 头显打分（A/B）",
        "",
        "> 打分 1–5；同一源、逐配方对比，重点看差异轴（见「说明」列）。",
        "",
        "| 文件 | recipe | " + " | ".join(SCORE_COLUMNS) + " | 备注 |",
        "| --- | --- | " + " | ".join("---" for _ in SCORE_COLUMNS) + " | --- |",
    ]
    empty_scores = " | ".join("" for _ in SCORE_COLUMNS)
    for r in results:
        filename = Path(r.output).name if r.output else "-"
        lines.append(f"| {filename} | {r.recipe} | {empty_scores} |  |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quest push (adb)
# ---------------------------------------------------------------------------


def resolve_quest_serial(serial: str | None = None) -> str | None:
    """Resolve the Quest adb serial: explicit flag > ``QUEST_SERIAL`` env > None."""
    if serial:
        return serial
    env = os.environ.get(QUEST_SERIAL_ENV, "").strip()
    return env or None


def push_to_quest(
    paths: Sequence[str | Path],
    serial: str,
    remote_dir: str = QUEST_PUSH_DIR,
    adb: str = "adb",
) -> list[list[str]]:
    """Push every artefact to the Quest and trigger a media scan.

    Returns the list of adb commands executed.  ``adb push`` failures raise
    ``RuntimeError`` (a half-pushed comparison set is worse than a loud error);
    the media-scan broadcast is best-effort (it only affects gallery indexing).
    """
    commands: list[list[str]] = []
    for path in paths:
        cmd = [adb, "-s", serial, "push", str(path), remote_dir + "/"]
        log.info("📲 %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)  # list argv, no shell
        if proc.returncode != 0:
            raise RuntimeError(f"adb push failed for {path}: {proc.stderr.strip()}")
        commands.append(cmd)

    # Force the media scanner to index the new files so they show up in the
    # headset's gallery / file browsers without a reboot.  Best-effort.
    scan = [
        adb,
        "-s",
        serial,
        "shell",
        "am",
        "broadcast",
        "-a",
        "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
        "-d",
        f"file://{remote_dir}",
    ]
    log.info("📲 %s", " ".join(scan))
    subprocess.run(scan, capture_output=True, text=True)  # list argv, no shell
    commands.append(scan)
    return commands


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_dry_run(
    input_path: str,
    outdir: Path,
    recipes: Sequence[Recipe],
    base_args: Sequence[str],
) -> None:
    """Print the resolved recipes + exact commands without rendering."""
    print("DRY RUN — 将执行以下配方（不渲染）:")
    print(f"  源视频: {input_path}")
    print(f"  输出目录: {outdir}")
    for recipe in recipes:
        out_name = recipe_output_name(input_path, recipe.name)
        cmd = build_render_command(input_path, outdir / out_name, recipe, base_args)
        desc = f" — {recipe.description}" if recipe.description else ""
        print(f"\n[{recipe.name}]{desc}")
        print(f"  输出: {out_name}")
        print(f"  命令: {' '.join(cmd)}")


def run_comparison(
    input_path: str,
    outdir: str | Path,
    recipes: Sequence[Recipe] | None = None,
    base_args: Sequence[str] = DEFAULT_BASE_ARGS,
    render_runner: Callable[[Sequence[str]], str] | None = None,
    qa_runner: Callable[[str, RecipeResult], RecipeResult] = qa_recipe_output,
    push: bool = False,
    quest_serial: str | None = None,
    dry_run: bool = False,
    push_runner: Callable[..., list] = push_to_quest,
    metrics: bool = True,
    depth_dir_resolver: Callable[..., str | None] = default_depth_dir_resolver,
    metrics_runner: Callable[[str | Path], StabilityReport] = run_depth_metrics,
) -> list[RecipeResult]:
    """Render every recipe, QA each product, write comparison.md, optionally push.

    Args:
        render_runner: injectable render callable ``(cmd) -> output_path``.
            Defaults to :func:`default_render_runner` (subprocess to the real
            run_pipeline CLI).  Tests inject a fake — no real conversion runs.
        qa_runner: injectable QA callable ``(output_path, result) -> result``.
        push_runner: injectable adb push callable (tests assert the calls).
        metrics: when True (default), after QA each rendered recipe is annotated
            with the three depth-stability cells via
            :func:`apply_depth_metrics` (K-16, #206).  ``--no-metrics`` flips
            this off so depth_stability is never called.  The metrics step is
            best-effort: it never raises into the orchestration — a miss leaves
            the cells as ``—`` and the comparison proceeds.

    Returns:
        The per-recipe result rows (also written into ``comparison.md``).
    """
    recipe_list = list(recipes) if recipes is not None else load_recipes()
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _print_dry_run(input_path, out_path, recipe_list, base_args)
        return [
            RecipeResult(
                recipe=r.name,
                description=r.description,
                output=str(out_path / recipe_output_name(input_path, r.name)),
            )
            for r in recipe_list
        ]

    runner = render_runner or default_render_runner
    results: list[RecipeResult] = []
    for recipe in recipe_list:
        out_name = recipe_output_name(input_path, recipe.name)
        dest = out_path / out_name
        row = RecipeResult(recipe=recipe.name, description=recipe.description, output=str(dest))
        cmd = build_render_command(input_path, dest, recipe, base_args)
        log.info("=== Recipe %s/%s: %s ===", len(results) + 1, len(recipe_list), recipe.name)
        started = time.monotonic()
        try:
            runner(cmd)
            row.ok = True
        except Exception as exc:  # one failed recipe must not kill the A/B set
            row.ok = False
            row.error = str(exc)
            log.error("❌ Recipe %s failed: %s", recipe.name, exc)
        row.elapsed_s = time.monotonic() - started

        if row.ok:
            try:
                qa_runner(str(dest), row)
            except Exception as exc:
                log.warning("⚠️  QA failed for %s: %s", dest, exc)
                row.verdict = f"QA 错误: {exc}"
                row.qa_failed = True

            if metrics:
                # K-16 (#206): annotate the row with depth-stability cells.
                # apply_depth_metrics swallows every error itself (cells stay —),
                # so this block can never break the render/QA flow above.
                apply_depth_metrics(
                    input_path,
                    recipe,
                    row,
                    depth_dir_resolver=depth_dir_resolver,
                    metrics_runner=metrics_runner,
                )
        results.append(row)

    # Push before writing comparison.md so the md records whether it happened.
    pushed = False
    if push:
        serial = resolve_quest_serial(quest_serial)
        artefacts = [r.output for r in results if r.ok and r.output]
        if not serial:
            log.warning(
                "⚠️  --push-to-quest 但未配置 serial（--quest-serial 或环境变量 %s）— 跳过推送。",
                QUEST_SERIAL_ENV,
            )
        elif not artefacts:
            log.warning("⚠️  没有渲染成功的产物可推送 — 跳过。")
        else:
            push_runner(artefacts, serial)
            pushed = True
            log.info("✅ 已推送 %d 个产物到 Quest (%s)", len(artefacts), serial)

    md_path = out_path / "comparison.md"
    md_path.write_text(
        render_comparison_md(results, source=str(input_path), outdir=out_path, pushed=pushed),
        encoding="utf-8",
    )
    log.info("✅ comparison.md → %s", md_path)

    print("\n" + render_summary_table(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  Accept optional *argv* for testing."""
    parser = argparse.ArgumentParser(
        description="一键出对比样片：同一源视频 × 多配方渲染 + QA 汇总 + comparison.md（含头显打分空列）",
    )
    parser.add_argument("--input", "-i", required=True, help="源视频文件 (MP4, MOV, etc.)")
    parser.add_argument("--outdir", "-o", required=True, help="产物 + comparison.md 输出目录")
    parser.add_argument(
        "--recipe",
        action="append",
        default=None,
        help="只跑指定配方（可重复）。默认跑全部内置配方: " + ", ".join(r["name"] for r in DEFAULT_RECIPES),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="追加传给每条 run_pipeline 的参数（可重复），如 --extra-arg=--max-frames --extra-arg=60",
    )
    parser.add_argument(
        "--push-to-quest",
        action="store_true",
        help=f"渲染后用 adb push 全部产物到 Quest 并触发媒体扫描（serial 取 --quest-serial 或 env {QUEST_SERIAL_ENV}；未配置则跳过）",
    )
    parser.add_argument("--quest-serial", default=None, help=f"adb serial（默认读 env {QUEST_SERIAL_ENV}）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的配方与命令，不渲染")
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="跳过深度稳定性统计（temporal_jitter/flicker_ratio/edge_consistency）。"
        "默认开启统计；本开关只跳过调用 depth_stability，不影响渲染/QA。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0 = all rendered, 1 = any failure)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        recipes = load_recipes()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.recipe:
        wanted = set(args.recipe)
        unknown = wanted - {r.name for r in recipes}
        if unknown:
            valid = ", ".join(r.name for r in recipes)
            print(f"Error: unknown recipe(s): {sorted(unknown)} — choose from {valid}", file=sys.stderr)
            return 1
        recipes = [r for r in recipes if r.name in wanted]

    base_args = [*DEFAULT_BASE_ARGS, *args.extra_arg]

    results = run_comparison(
        input_path=args.input,
        outdir=args.outdir,
        recipes=recipes,
        base_args=base_args,
        push=args.push_to_quest,
        quest_serial=args.quest_serial,
        dry_run=args.dry_run,
        metrics=not args.no_metrics,
    )

    if args.dry_run:
        return 0
    failed = [r.recipe for r in results if not r.ok]
    if failed:
        print(f"❌ 渲染失败的配方: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
