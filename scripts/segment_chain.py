"""Chain multi-scene video generation: scene N's last frame seeds scene N+1.

A multi-scene fulldome / VR180 short needs continuity between shots.  Seedance
(Ark) already supports ``--return-last-frame`` (see ``scripts/generate.py`` and
``integrations.seedance.PASSTHROUGH_FIELDS``); this script drives that feature
across a *plan* of scenes, feeding scene N's final frame into scene N+1's
image-to-video request so the whole cut reads as one continuous camera move.

Usage::

    python -m scripts.segment_chain plan.json --out-dir video/chain
    python -m scripts.segment_chain plan.json --out-dir video/chain --resume
    python -m scripts.segment_chain plan.json --out-dir video/chain --dry-run
    python -m scripts.segment_chain plan.json --out-dir video/chain --concat

The plan is a JSON array, one object per scene::

    [
      {"prompt": "fly over mountains at dawn", "duration": 10, "seed_image": "first.png"},
      {"prompt": "descend into the valley", "duration": 10},
      {"prompt": "sweep across the lake", "duration": 8}
    ]

``seed_image`` is optional and only honoured for the **first** scene.  Every
later scene is seeded by ``scene_{N-1}_last.png`` written by the previous one.
Per scene the script writes ``scene_N.mp4`` and ``scene_N_last.png`` into
``--out-dir``.

The provider is called *in-process* (``integrations.factory.get_provider``),
never as a subprocess — the generation surface is the same
``generate`` / ``generate_from_image`` pair ``scripts.generate.py`` uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# K-15 (#205): runnable directly (``python scripts/segment_chain.py``) without
# the caller setting PYTHONPATH — put the repo root on sys.path before
# importing the repo's top-level packages.  Same shim as scripts/generate.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
from integrations.base import GenerationResult, VideoGenProvider  # noqa: E402
from integrations.factory import get_provider, list_providers  # noqa: E402
from integrations.seedance import MODEL_FAST, VALID_RESOLUTIONS  # noqa: E402
from integrations.usage_ledger import (  # noqa: E402
    BudgetExceededError,
    enforce_budget,
    estimate_cost,
    estimate_next_tokens,
    price_per_mtoken,
    read_records,
)

log = logging.getLogger(__name__)

#: Default duration (seconds) when a plan entry omits it.
DEFAULT_DURATION = 5

#: Defaults shared with scripts/generate.py (quota discipline: 480p / fast).
DEFAULT_MODEL = MODEL_FAST
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_FPS = 24

#: File-name template helpers (kept as functions so callers never hand-roll the
#: name and the two sides — video and last-frame — can't drift apart).
_SCENE_VIDEO_TPL = "scene_{idx}.mp4"
_SCENE_FRAME_TPL = "scene_{idx}_last.png"
_CONCAT_LIST_NAME = "concat_list.txt"
_CONCAT_OUTPUT_NAME = "chain.mp4"


def scene_video_path(out_dir: str | os.PathLike[str], idx: int) -> Path:
    """Return the output path for scene *idx*'s video."""
    return Path(out_dir) / _SCENE_VIDEO_TPL.format(idx=idx)


def scene_frame_path(out_dir: str | os.PathLike[str], idx: int) -> Path:
    """Return the path where scene *idx*'s last frame is saved."""
    return Path(out_dir) / _SCENE_FRAME_TPL.format(idx=idx)


def load_plan(path: str | os.PathLike[str]) -> list[dict]:
    """Load and validate a scene plan JSON array.

    Raises
    ------
    OSError
        If *path* cannot be read.
    ValueError
        If the file is not a JSON array of objects each carrying a non-empty
        ``prompt`` (or the array is empty).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"计划必须是 JSON 数组，得到 {type(data).__name__}")
    if not data:
        raise ValueError("计划为空：至少需要一场")

    plan: list[dict] = []
    for idx, scene in enumerate(data):
        if not isinstance(scene, dict):
            raise ValueError(f"第 {idx} 场必须是对象，得到 {type(scene).__name__}")
        prompt = scene.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"第 {idx} 场缺少非空 prompt")
        plan.append(scene)
    return plan


def _save_asset(url: str, out_path: str | os.PathLike[str]) -> Path:
    """Copy the asset at *url* (local path or http(s) URL) to *out_path*.

    Mirrors ``scripts.generate._download_video``: the mock provider (and the
    local SVD provider) return a *local* file path as ``video_url``, so a
    local-path branch keeps the whole chain runnable on CI with no network.
    """
    target = Path(out_path)
    if url.startswith("http://") or url.startswith("https://"):
        log.info("下载 %s -> %s", url, target)
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        return target

    if not os.path.exists(url):
        raise RuntimeError(f"文件不存在：{url}")
    with open(url, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def extract_last_frame(result: GenerationResult) -> str | None:
    """Locate the last-frame image URL in a :class:`GenerationResult`.

    Seedance stores the whole Ark poll response in ``result.metadata`` (see
    ``SeedanceProvider._poll_task``); when ``return_last_frame`` was set the
    ``content`` object carries the image URL alongside ``video_url``.  The
    lookup is defensive — a few known shapes are probed and the first hit wins,
    so a provider that nests it differently still chains.
    """
    meta = result.metadata or {}
    candidates: list[object] = [
        meta.get("last_frame"),
        meta.get("last_frame_url"),
        meta.get("image_url"),
    ]
    content = meta.get("content")
    if isinstance(content, dict):
        candidates.extend([content.get("last_frame"), content.get("last_frame_url"), content.get("image_url")])
        outputs = content.get("outputs")
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict):
                    candidates.extend([item.get("last_frame"), item.get("image_url")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _estimate(
    model: str,
    resolution: str,
    duration: int,
) -> tuple[float | None, float | None]:
    """Estimate (tokens, 元) for one scene from ledger history.

    There is no hard-coded price — ``estimate_next_tokens`` averages the cost of
    comparable past runs, and ``estimate_cost`` is ``None`` when
    ``VR180_ARK_PRICE_PER_MTOKEN`` is unset.  Both unknowns are surfaced as
    ``None`` rather than invented numbers.
    """
    records = read_records()
    tokens = estimate_next_tokens(records, model=model, resolution=resolution, duration=duration)
    cost = estimate_cost(tokens, price_per_mtoken())
    return tokens, cost


def _concat(out_dir: str | os.PathLike[str], count: int) -> Path:
    """Concatenate ``count`` scene videos with the ffmpeg concat demuxer.

    Uses a list file (``-f concat -safe 0``) and ``-c copy`` for a lossless,
    fast join.  *count* must match the number of ``scene_N.mp4`` files present;
    heterogeneous codecs will fail loudly rather than corrupt silently.
    """
    out_dir = Path(out_dir)
    list_path = out_dir / _CONCAT_LIST_NAME
    videos = [scene_video_path(out_dir, i) for i in range(count)]
    missing = [str(v) for v in videos if not v.exists()]
    if missing:
        raise RuntimeError(f"拼接失败：缺少场次视频 {missing}")

    with list_path.open("w", encoding="utf-8") as handle:
        for video in videos:
            handle.write(f"file '{video.as_posix()}'\n")

    out_path = out_dir / _CONCAT_OUTPUT_NAME
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    log.info("ffmpeg concat %d 场 -> %s", count, out_path)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat 失败（exit {result.returncode}）：{result.stderr[-2000:]}")
    return out_path


def run_chain(
    plan: list[dict],
    out_dir: str | os.PathLike[str],
    provider: VideoGenProvider,
    *,
    resume: bool = False,
    dry_run: bool = False,
    concat: bool = False,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    fps: int = DEFAULT_FPS,
    budget_cap: float | str | None = None,
    budget_cap_tokens: float | str | None = None,
) -> int:
    """Run every scene in *plan* sequentially, chaining last frames.

    Returns a process exit code (0 = all scenes done).  A budget gate from
    :mod:`integrations.usage_ledger` is enforced **before each scene's
    submission**; if it fires, :class:`BudgetExceededError` is caught and the
    chain stops immediately (exit 2 — the same convention as ``generate.py``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, scene in enumerate(plan):
        prompt = scene["prompt"]
        duration = int(scene.get("duration", DEFAULT_DURATION))
        video_path = scene_video_path(out_dir, idx)
        frame_path = scene_frame_path(out_dir, idx)

        # Image source: the plan's own seed_image for the first scene only;
        # every later scene reads the previous scene's saved last frame.
        image = scene.get("seed_image") if idx == 0 else str(scene_frame_path(out_dir, idx - 1))

        if resume and video_path.exists() and frame_path.exists():
            log.info("场景 %d 已存在，跳过（--resume）：%s", idx, video_path)
            continue

        if dry_run:
            tokens, cost = _estimate(model, resolution, duration)
            cost_text = "未知" if cost is None else f"≈{cost:.4f} 元"
            token_text = "未知" if tokens is None else f"≈{tokens:,.0f} tokens"
            print(f"场景 {idx}: prompt={prompt!r} duration={duration}s image={image or '-'}")
            print(
                f"          provider={provider.name} model={model} resolution={resolution} "
                f"ratio={aspect_ratio} fps={fps}"
            )
            print(f"          预估 {token_text}，费用 {cost_text}")
            continue

        # W-11 (#328): the local budget soft-gate, enforced before anything is
        # submitted.  The provider (seedance) also gates internally; this
        # chain-level check is what stops the *rest of the cut* on a block.
        try:
            enforce_budget(
                cap=budget_cap,
                cap_tokens=budget_cap_tokens,
                model=model,
                resolution=resolution,
                duration=duration,
            )
        except BudgetExceededError as exc:
            log.error("场景 %d 触发预算软闸，停止链式生成：%s", idx, exc)
            return 2

        kwargs: dict[str, object] = {
            "return_last_frame": True,
            "model": model,
            "resolution": resolution,
        }
        if budget_cap is not None:
            kwargs["budget_cap"] = budget_cap
        if budget_cap_tokens is not None:
            kwargs["budget_cap_tokens"] = budget_cap_tokens

        if image:
            log.info(
                "场景 %d：图生视频 provider=%s image=%s duration=%ds ratio=%s fps=%d",
                idx,
                provider.name,
                image,
                duration,
                aspect_ratio,
                fps,
            )
            try:
                result = provider.generate_from_image(
                    image_path=image,
                    prompt=prompt,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    fps=fps,
                    **kwargs,
                )
            except BudgetExceededError as exc:
                log.error("场景 %d 触发预算软闸，停止链式生成：%s", idx, exc)
                return 2
            except NotImplementedError as exc:
                log.error("provider %s 不支持图生视频（场景 %d 需要 image 输入）：%s", provider.name, idx, exc)
                return 1
            except (RuntimeError, ValueError) as exc:
                log.error("场景 %d 生成失败：%s", idx, exc)
                return 1
        else:
            log.info(
                "场景 %d：文生视频 provider=%s duration=%ds ratio=%s fps=%d",
                idx,
                provider.name,
                duration,
                aspect_ratio,
                fps,
            )
            try:
                result = provider.generate(
                    prompt=prompt,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    fps=fps,
                    **kwargs,
                )
            except BudgetExceededError as exc:
                log.error("场景 %d 触发预算软闸，停止链式生成：%s", idx, exc)
                return 2
            except (RuntimeError, ValueError) as exc:
                log.error("场景 %d 生成失败：%s", idx, exc)
                return 1

        _save_asset(result.video_url, video_path)
        log.info("场景 %d 完成：%s", idx, video_path)

        frame_url = extract_last_frame(result)
        if frame_url:
            _save_asset(frame_url, frame_path)
            log.info("场景 %d 末帧已保存：%s", idx, frame_path)
        elif idx < len(plan) - 1:
            log.error("场景 %d 未返回末帧（provider 未支持 return_last_frame），无法链到场景 %d。", idx, idx + 1)
            return 1

    if dry_run:
        return 0
    if concat and plan:
        _concat(out_dir, len(plan))
        log.info("拼接完成：%s", out_dir / _CONCAT_OUTPUT_NAME)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="链式生成多场景视频：场景 N 的末帧作为场景 N+1 的首帧。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available providers: " + ", ".join(list_providers()) + "\n\n"
            "计划文件是一个 JSON 数组，每场一个对象：\n"
            '  [{"prompt": "...", "duration": 10, "seed_image": "first.png"}, ...]\n'
            "seed_image 仅第一场可选；后续每场以 scene_{N-1}_last.png 作为 image 输入。\n\n"
            "用量账本 / 预算软闸（issue #328）：\n"
            "  VR180_LEDGER_PATH            账本路径（默认 ~/.vr180/usage_ledger.jsonl）\n"
            "  VR180_ARK_PRICE_PER_MTOKEN  单价，元/百万token（未配置则只记 token、不估金额）\n"
            "  VR180_BUDGET_CAP            累计预算上限（元），等价于 --budget-cap\n"
            "  VR180_BUDGET_CAP_TOKENS     累计预算上限（token），等价于 --budget-cap-tokens\n"
        ),
    )
    parser.add_argument("plan", help="JSON 计划文件路径（场景数组）。")
    parser.add_argument(
        "--out-dir",
        "-o",
        required=True,
        help="输出目录；每场写入 scene_N.mp4 与 scene_N_last.png。",
    )
    parser.add_argument(
        "--provider",
        "-p",
        default="seedance",
        choices=list_providers(),
        help="视频生成 provider（默认 seedance）。用 mock 可离线跑通整条链。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="已存在的 scene_N.mp4（且末帧 scene_N_last.png 在）跳过，不重新生成。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印每场请求参数与预估费用，不调用 provider（零 POST）。",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="结束后用 ffmpeg concat demuxer 拼接所有场次为 chain.mp4。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Seedance 模型 id（默认 {DEFAULT_MODEL}）。",
    )
    parser.add_argument(
        "--gen-resolution",
        default=DEFAULT_RESOLUTION,
        choices=list(VALID_RESOLUTIONS),
        help="生成分辨率档（默认 480p —— 额度纪律）。",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        default=DEFAULT_ASPECT_RATIO,
        help='画面比例（默认 "16:9"），映射到 Seedance 请求体 ratio。',
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="目标帧率（默认 24）。",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="累计预算软闸（元）；达到 100% 时在提交前拒绝并停止链式生成。",
    )
    parser.add_argument(
        "--budget-cap-tokens",
        type=float,
        default=None,
        metavar="TOKENS",
        help="累计预算软闸（token）；未配置单价时的闸门。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        plan = load_plan(args.plan)
    except (OSError, ValueError) as exc:
        log.error("读取计划失败：%s", exc)
        return 2

    try:
        provider = get_provider(args.provider)
    except ValueError as exc:
        log.error("Provider error: %s", exc)
        return 1

    return run_chain(
        plan,
        args.out_dir,
        provider,
        resume=args.resume,
        dry_run=args.dry_run,
        concat=args.concat,
        model=args.model,
        resolution=args.gen_resolution,
        aspect_ratio=args.aspect_ratio,
        fps=args.fps,
        budget_cap=args.budget_cap,
        budget_cap_tokens=args.budget_cap_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
