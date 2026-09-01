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
5. **comparison.md** — the table above plus *empty* scoring columns
   (清晰度/重影/立体感/舒适度) for the owner to fill in inside the headset.
6. **--push-to-quest** — optional ``adb -s <serial> push`` of every product
   followed by a media-scan broadcast.  Serial comes from ``--quest-serial``
   or the ``QUEST_SERIAL`` env var; when unset the step is skipped gracefully
   with a hint (never an error).
7. **--dry-run** — prints the resolved recipes, output names and the exact
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

from pipeline.naming import SceneAssetSpec, compose_scene_name

try:
    # Package import (tests, ``python -m scripts.make_comparison``).
    from scripts.vr180_qa import run_qa
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    # Direct invocation (``python scripts/make_comparison.py``) puts scripts/
    # itself on sys.path, shadowing the ``scripts`` package — import the QA
    # module flat instead.
    from vr180_qa import run_qa  # type: ignore[no-redef]

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


def _status_cell(result: RecipeResult) -> str:
    if not result.ok:
        return f"❌ {result.error or 'render failed'}"
    return "⚠️ QA fail" if result.qa_failed else "✅"


def render_summary_table(results: Sequence[RecipeResult]) -> str:
    """Render the recipe / 分辨率 / 音频 / QA / 耗时 summary as Markdown."""
    lines = [
        "| recipe | 说明 | 分辨率 | 音频 | QA 判定 | 耗时 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.recipe} | {r.description or '-'} | {r.resolution} | {r.audio} "
            f"| {r.verdict} | {r.elapsed_s:.1f}s | {_status_cell(r)} |"
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
) -> list[RecipeResult]:
    """Render every recipe, QA each product, write comparison.md, optionally push.

    Args:
        render_runner: injectable render callable ``(cmd) -> output_path``.
            Defaults to :func:`default_render_runner` (subprocess to the real
            run_pipeline CLI).  Tests inject a fake — no real conversion runs.
        qa_runner: injectable QA callable ``(output_path, result) -> result``.
        push_runner: injectable adb push callable (tests assert the calls).

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
