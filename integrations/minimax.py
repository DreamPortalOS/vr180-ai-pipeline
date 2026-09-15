"""MiniMax (Hailuo) video generation provider — the 2K lane (issue #353).

Why this provider exists: a Seedance 4k/10s run measured at ~50 元 a clip, and
at that price iteration is the bottleneck, not quality.  MiniMax's ``MiniMax-H3``
bills **$0.13 per second at 2K** (≈9.5 元 for a 10s clip at 7.3 元/$), i.e. about
a fifth of the Seedance 4k cost, and it is the only MiniMax tier that reaches
2K at all.

Verified contract (2026-09-15, platform.minimax.io — see the PR for links):

* submit  ``POST https://api.minimax.io/v2/video_generation``
* query   ``GET  https://api.minimax.io/v2/query/video_generation/{task_id}``
* auth    ``Authorization: Bearer $MINIMAX_API_KEY``
* body    ``{model, content[], duration, resolution, ratio}`` where ``content``
  is the same typed-part list Ark uses: a ``{"type": "text"}`` part plus, for
  image-to-video, a ``{"type": "image_url", "role": "first_frame"}`` part.
* async   submit returns ``task_id``; poll until ``status == "succeeded"`` and
  read the download URL off the task.  Same shape as Seedance, so the same
  discipline applies: **a client-side timeout is not a failure and the money is
  already spent** — recover with :meth:`MiniMaxProvider.resume`.

Answers to the three questions the card asked us to verify:

* **2K = 1440 px on the short edge.**  MiniMax documents the tier as "2K"; the
  1440 short-edge figure is the published definition, so 1:1 lands at roughly
  1440×1440.  MiniMax does **not** publish per-ratio pixel dimensions, so treat
  1440×1440 as expected-but-unconfirmed until the first real clip is probed.
* **1:1 is supported.**  ``ratio`` accepts
  ``adaptive / 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16``.  Our square requirement
  is met — with one operational caveat, see :data:`RATIO_IGNORED_NOTE`.
* **10s is supported.**  ``duration`` is an integer in [4, 15].

Credentials: ``MINIMAX_API_KEY`` (MiniMax 开放平台 console).  ``MINIMAX_API_BASE``
switches to the mainland China host (``https://api.minimaxi.com``), which serves
the same paths under a different domain.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path

import httpx

from integrations import usage_ledger
from integrations.base import GenerationResult, VideoGenProvider
from pipeline.image_prep import validate_image_for_i2v

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints / credentials
# ---------------------------------------------------------------------------

#: Global host. The mainland host serves identical paths, see ``ENV_API_BASE``.
BASE_URL_GLOBAL = "https://api.minimax.io"

#: Mainland China host (同一套路径，换域名).
BASE_URL_CN = "https://api.minimaxi.com"

ENV_API_KEY = "MINIMAX_API_KEY"
ENV_API_BASE = "MINIMAX_API_BASE"

_SUBMIT_PATH = "/v2/video_generation"
_QUERY_PATH = "/v2/query/video_generation/{task_id}"
_POLL_INTERVAL = 3.0

# ---------------------------------------------------------------------------
# Model / tier contract
# ---------------------------------------------------------------------------

#: The only MiniMax model with a 2K tier, and therefore the only one this
#: provider defaults to. The Hailuo-2.x family tops out at 1080p and bills in
#: prepaid "video points" instead of per second, which the budget gate below
#: cannot reason about — so it is deliberately not wired up here.
MODEL_H3 = "MiniMax-H3"

#: Model ids this provider knows how to price and validate.
KNOWN_MODELS = (MODEL_H3,)

RESOLUTION_768P = "768P"
RESOLUTION_2K = "2K"

#: Canonical tier order — cheapest first.
VALID_RESOLUTIONS = (RESOLUTION_768P, RESOLUTION_2K)

#: Aspect ratios the API accepts. ``1:1`` — our hard requirement — is present.
VALID_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

#: Duration contract: whole seconds, inclusive range.
DURATION_MIN = 4
DURATION_MAX = 15

#: Published pay-as-you-go price in **USD per second of output**. MiniMax bills
#: per second, not per token — see :meth:`MiniMaxProvider._record_usage` for why
#: that matters to the ledger.
PRICE_PER_SECOND_USD: dict[str, float] = {
    RESOLUTION_768P: 0.08,
    RESOLUTION_2K: 0.13,
}

#: Operational caveat worth repeating at the call site: in first-frame mode the
#: API derives the output aspect ratio from the supplied image, so a square clip
#: requires a **square seed frame** — passing ``--gen-ratio 1:1`` alone will not
#: square up a 16:9 seed.
RATIO_IGNORED_NOTE = (
    "MiniMax 首帧模式下画幅由首帧图片决定，ratio 字段会被忽略：想要 1:1 方图，"
    "首帧必须本身就是方图（video/seed_1x1_drone.png 已经是）。"
)

# ---------------------------------------------------------------------------
# CLI tier translation
# ---------------------------------------------------------------------------
# ``scripts/generate.py`` speaks Ark's tier names (480p/720p/1080p/4k) because
# Seedance was the first provider wired up. MiniMax has exactly two tiers, so
# the CLI values have to be translated. The translation is deliberately
# asymmetric:
#
#   * downwards / sideways (480p, 720p → 768P) is automatic — the operator asked
#     for something cheap and gets the cheap tier;
#   * upwards (1080p, 4k → 2K) is **refused**, because 2K costs 60% more per
#     second and "I typed the Seedance tier out of habit" must not silently
#     become a bigger bill. The error names the flag that opts in.

_CLI_RESOLUTION_ALIASES: dict[str, str] = {
    "480p": RESOLUTION_768P,
    "720p": RESOLUTION_768P,
    "768p": RESOLUTION_768P,
    "2k": RESOLUTION_2K,
}

#: Ark tiers that imply "expensive": mapping them automatically would spend more
#: money than the operator asked for, so they are rejected with a hint instead.
_CLI_RESOLUTION_REFUSED = ("1080p", "4k")

#: The tier used when nothing meaningful was requested — the cheap one, matching
#: the repo-wide quota discipline (the CLI default is Ark's ``480p``).
DEFAULT_RESOLUTION = RESOLUTION_768P

#: Default ratio. ``adaptive`` is accepted for image-to-video (our main usage);
#: text-to-video rejects it, which :func:`MiniMaxProvider._build_body` handles.
DEFAULT_RATIO = "adaptive"

DEFAULT_DURATION = 5

# ---------------------------------------------------------------------------
# Poll budget
# ---------------------------------------------------------------------------
# Same lesson as Seedance (#325): a too-short client budget turns a *paid*,
# still-running task into a "failure" that tempts the operator into paying
# twice. Budget scales with the two knobs that drive render time.

_BASE_POLL_SECONDS = 300
_BASELINE_DURATION = 5

RESOLUTION_POLL_FACTORS: dict[str, float] = {
    RESOLUTION_768P: 1.5,
    RESOLUTION_2K: 3.0,
}

#: Budget for :meth:`MiniMaxProvider.resume` when the original request shape is
#: unknown: the heaviest combination (2K / 15s). Resuming spends nothing, so
#: waiting long is the cheap side of the trade.
_RESUME_POLL_SECONDS = 3600

_PROGRESS_LOG_INTERVAL = 30.0

#: Task states that mean "still working".
_PENDING_STATES = ("queued", "preparing", "processing", "running", "pending")

#: Task states that mean "done, successfully".
_SUCCESS_STATES = ("succeeded", "success", "completed")

#: Task states that mean "done, unsuccessfully".
_FAILURE_STATES = ("failed", "fail", "cancelled", "canceled", "expired")

#: Documented ``base_resp.status_code`` values, mapped to something actionable.
_STATUS_CODE_HINTS: dict[int, str] = {
    1002: "触发限流（rate limit），稍后重试。",
    1004: "鉴权失败：MINIMAX_API_KEY 无效或未绑定到该分组。",
    1008: "余额不足：请在 MiniMax 控制台充值后重试（本次未生成）。",
    1026: "内容安全审核拦截：请修改 prompt 或首帧图片。",
    2013: "请求参数非法：检查 model / resolution / ratio / duration 组合。",
    2049: "API key 无效：请确认 MINIMAX_API_KEY 复制完整。",
}


def resolve_cli_resolution(requested: str | None) -> str:
    """Translate a ``--gen-resolution`` value into a MiniMax tier.

    ``"480p"`` / ``"720p"`` (Ark's cheap tiers, and the CLI default) map onto
    MiniMax's cheap tier; ``"768p"`` / ``"2k"`` are MiniMax's own names and pass
    through normalised to the API's casing.

    ``"1080p"`` / ``"4k"`` raise :class:`ValueError`: MiniMax has no such tier
    and silently promoting them to 2K would spend 60% more per second than was
    asked for. Raised locally, before any HTTP traffic, so nothing is spent.
    """
    if requested is None:
        return DEFAULT_RESOLUTION
    key = str(requested).strip().lower()
    if not key:
        return DEFAULT_RESOLUTION
    if key in _CLI_RESOLUTION_ALIASES:
        return _CLI_RESOLUTION_ALIASES[key]
    if key in _CLI_RESOLUTION_REFUSED:
        raise ValueError(
            f"MiniMax 没有 '{requested}' 档位，只有 {'/'.join(VALID_RESOLUTIONS)}。"
            f"想要 2K（短边 1440，约 ${PRICE_PER_SECOND_USD[RESOLUTION_2K]:g}/秒）请显式指定 "
            f"--gen-resolution 2k；想省钱用 --gen-resolution 768p。"
        )
    raise ValueError(f"Unknown MiniMax resolution {requested!r}. Supported: {', '.join(VALID_RESOLUTIONS)}")


def resolve_cli_model(requested: str | None) -> str:
    """Translate a ``--model`` value into a MiniMax model id.

    The CLI's ``--model`` default is an *Ark* model id (the flag predates this
    provider), so anything this provider does not recognise falls back to
    :data:`MODEL_H3` — the only model with a 2K tier. The substitution is
    logged rather than silent so an operator who really meant to pick a
    different MiniMax model can see that it did not take effect.
    """
    if requested and str(requested) in KNOWN_MODELS:
        return str(requested)
    if requested:
        log.info("MiniMax: 忽略非 MiniMax 模型 id %r，使用 %s（唯一支持 2K 的模型）。", requested, MODEL_H3)
    return MODEL_H3


def estimate_cost_usd(resolution: str, duration: object) -> float | None:
    """Cost of one clip in USD, or ``None`` for an unknown tier.

    MiniMax bills per second of output, so this is exact arithmetic rather than
    the history-based guess the token ledger has to make for Ark.
    """
    price = PRICE_PER_SECOND_USD.get(str(resolution))
    if price is None:
        return None
    try:
        seconds = float(duration)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return price * seconds


def poll_timeout_seconds(
    resolution: str = DEFAULT_RESOLUTION,
    duration: int | float | str = DEFAULT_DURATION,
    override: int | float | None = None,
) -> int:
    """Return the poll budget (seconds) for a *resolution* / *duration* pair.

    ``300s × tier factor × duration factor``, never below the 300s baseline.
    2K/10s lands at 1800s. *override* is the CLI's ``--gen-timeout``.
    """
    if override is not None:
        seconds = int(override)
        if seconds <= 0:
            raise ValueError(f"poll timeout must be a positive number of seconds, got {override!r}")
        return seconds

    factor = RESOLUTION_POLL_FACTORS.get(str(resolution), 1.0)
    try:
        duration_factor = float(duration) / _BASELINE_DURATION
    except (TypeError, ValueError):
        duration_factor = 1.0
    duration_factor = max(1.0, duration_factor)
    return max(_BASE_POLL_SECONDS, int(_BASE_POLL_SECONDS * factor * duration_factor))


def timeout_recovery_message(task_id: str, waited_seconds: int | float) -> str:
    """Operator-facing text for a poll timeout.

    A timeout is **not** a failure: the task keeps rendering on MiniMax's side
    and the seconds are already billed, so the one thing the operator must not
    do is re-run the generation.
    """
    return (
        f"MiniMax task {task_id} did not complete within {int(waited_seconds)}s.\n"
        f"  ⚠️  超时 ≠ 失败：任务很可能仍在 MiniMax 服务端继续跑，费用已经产生。\n"
        f"  ⚠️  不要重新生成 —— 重跑会再付一次钱。\n"
        f"  task_id: {task_id}\n"
        f"  取回成片: python -m scripts.generate --provider minimax "
        f"--resume-task {task_id} -o <output.mp4>\n"
        f"  下次想等更久: 加 --gen-timeout <秒>"
    )


class MiniMaxProvider(VideoGenProvider):
    """MiniMax H3 video generation (text-to-video and image-to-video).

    Requires ``MINIMAX_API_KEY``.
    """

    MODEL_H3 = MODEL_H3  # re-export for ``provider.generate(model=...)`` ergonomics

    def _load_api_key(self) -> str:
        api_key = os.environ.get(ENV_API_KEY, "")
        if not api_key:
            raise ValueError(
                f"{ENV_API_KEY} environment variable is not set. "
                "Create a key in the MiniMax console (https://platform.minimax.io/user-center/basic-information/interface-key)."
            )
        return api_key

    @property
    def base_url(self) -> str:
        """API host: ``MINIMAX_API_BASE`` if set, else the global host."""
        return os.environ.get(ENV_API_BASE, "").strip().rstrip("/") or BASE_URL_GLOBAL

    # ------------------------------------------------------------------
    # Public API — text-to-video
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        duration: int = DEFAULT_DURATION,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: str | int | float | bool | None,
    ) -> GenerationResult:
        """Generate a video from a text prompt.

        ``fps`` is accepted for a stable cross-provider surface but MiniMax
        renders at a fixed 24 fps and exposes no frame-rate field.
        """
        if not str(prompt or "").strip():
            raise ValueError("MiniMax text-to-video requires a non-empty prompt (or use generate_from_image).")
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        kwargs.setdefault("aspect_ratio", aspect_ratio)
        return self._run(content, duration=duration, has_first_frame=False, **kwargs)

    # ------------------------------------------------------------------
    # Image-to-video — the usage this card was opened for
    # ------------------------------------------------------------------

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = DEFAULT_DURATION,
        aspect_ratio: str = "16:9",
        **kwargs: str | int | float | bool | None,
    ) -> GenerationResult:
        """Generate a video from a starting frame.

        A local image is validated and inlined as a ``data:image/...;base64,…``
        URL; an ``http(s)`` URL is passed through verbatim.  MiniMax recommends
        a public URL for large assets — a 2K PNG base64-encodes to ~33% more
        bytes — but accepts data URLs, which keeps the local-file workflow
        working without an upload step.

        The prompt may be empty (image-only motion).
        """
        encoded = self._encode_image(image_path)
        content: list[dict[str, object]] = []
        if str(prompt or "").strip():
            content.append({"type": "text", "text": prompt})
        content.append({"type": "image_url", "image_url": {"url": encoded}, "role": "first_frame"})
        kwargs.setdefault("aspect_ratio", aspect_ratio)
        return self._run(content, duration=duration, has_first_frame=True, **kwargs)

    # ------------------------------------------------------------------
    # Internal: submit → poll → result
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _run(
        self,
        content: list[dict[str, object]],
        duration: int = DEFAULT_DURATION,
        has_first_frame: bool = False,
        **kwargs: str | int | float | bool | None,
    ) -> GenerationResult:
        # Client-side knobs and local guard rails — never body fields, so they
        # are popped before the body is assembled.
        poll_timeout = kwargs.pop("poll_timeout", None)
        budget_cap = kwargs.pop("budget_cap", None)
        budget_cap_tokens = kwargs.pop("budget_cap_tokens", None)
        # ``duration`` is both a named parameter and a legal kwarg on the
        # provider surface; popping it here keeps the two from colliding into a
        # "got multiple values for argument" TypeError.
        duration = kwargs.pop("duration", duration)  # type: ignore[assignment]

        body = self._build_body(
            content=content,
            duration=duration,
            has_first_frame=has_first_frame,
            **kwargs,
        )
        resolution = str(body["resolution"])
        timeout_seconds = poll_timeout_seconds(
            resolution=resolution,
            duration=body["duration"],  # type: ignore[arg-type]
            override=poll_timeout,  # type: ignore[arg-type]
        )

        # W-11 (#328): the budget soft-gate, evaluated *here* — before the httpx
        # client is even opened — so a blocked run cannot reach the submit POST.
        # Gating after submission would cost exactly as much as not gating.
        # ``resume()`` is intentionally NOT gated: it sends no POST, so it
        # spends nothing.
        usage_ledger.enforce_budget(
            cap=budget_cap,  # type: ignore[arg-type]
            cap_tokens=budget_cap_tokens,  # type: ignore[arg-type]
            model=str(body["model"]),
            resolution=resolution,
            duration=body["duration"],
        )

        cost = estimate_cost_usd(resolution, body["duration"])
        log.info(
            "MiniMax: submitting task (model=%s resolution=%s duration=%ss ratio=%s, 预估 %s, 轮询上限 %ds)",
            body["model"],
            resolution,
            body["duration"],
            body.get("ratio", "(未指定，API 默认 16:9)"),
            "未知" if cost is None else f"约 ${cost:.2f}",
            timeout_seconds,
        )

        headers = self._headers()
        with httpx.Client(base_url=self.base_url, timeout=30) as client:
            resp = client.post(_SUBMIT_PATH, json=body, headers=headers)
            task_id = self._handle_submit(resp)
            log.info("MiniMax: task submitted, task_id=%s", task_id)
            result = self._poll_task(client, task_id, headers, timeout_seconds)

        result.metadata.setdefault("estimated_cost_usd", cost)
        self._record_usage(
            result,
            model=body["model"],
            resolution=resolution,
            duration=body["duration"],
        )
        return result

    # ------------------------------------------------------------------
    # Resume: poll an already-submitted task
    # ------------------------------------------------------------------

    def resume(
        self,
        task_id: str,
        poll_timeout: int | float | None = None,
    ) -> GenerationResult:
        """Poll an **already submitted** MiniMax task and return its result.

        The recovery path for a client-side timeout, a dropped connection or a
        killed session: the task lives on MiniMax's side and its seconds are
        already billed, so re-submitting would simply pay for it twice.
        ``resume`` issues **no** ``POST`` — only the task-query ``GET`` — so it
        is always free, and therefore deliberately not budget-gated.
        """
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("resume() requires a non-empty MiniMax task id")

        timeout_seconds = _RESUME_POLL_SECONDS if poll_timeout is None else poll_timeout_seconds(override=poll_timeout)
        headers = self._headers()

        with httpx.Client(base_url=self.base_url, timeout=30) as client:
            log.info(
                "MiniMax: resuming task %s — 跳过提交，不产生新费用（轮询上限 %ds）",
                task_id,
                timeout_seconds,
            )
            result = self._poll_task(client, task_id, headers, timeout_seconds)

        self._record_usage(result)
        return result

    # ------------------------------------------------------------------
    # Internal: the poll loop, shared by _run and resume
    # ------------------------------------------------------------------

    def _poll_task(
        self,
        client: httpx.Client,
        task_id: str,
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> GenerationResult:
        """Poll *task_id* until it succeeds, fails, or the budget runs out.

        The first query happens immediately (before any sleep) so resuming an
        already-finished task returns straight away.
        """
        start = time.time()
        deadline = start + timeout_seconds
        next_progress_log = start

        while True:
            now = time.time()
            if now >= deadline:
                break

            poll_resp = client.get(_QUERY_PATH.format(task_id=task_id), headers=headers)
            self._raise_for_poll_status(poll_resp, task_id)
            poll_data = poll_resp.json()
            self._raise_for_base_resp(poll_data, f"MiniMax task {task_id}")

            status = str(self._extract_status(poll_data)).lower()

            if status in _SUCCESS_STATES:
                video_url = self._extract_video_url(poll_data)
                if not video_url:
                    raise RuntimeError(f"MiniMax task {task_id} completed but missing video URL: {poll_data}")
                log.info("MiniMax: task completed, url=%s", video_url)
                log.info("MiniMax: 下载链接有时效，请立刻保存成片。")
                return GenerationResult(
                    video_url=video_url,
                    provider=self.provider_name,
                    job_id=task_id,
                    metadata={"status": status, **poll_data},
                )
            if status in _FAILURE_STATES:
                raise RuntimeError(f"MiniMax task {task_id} {status}: {self._failure_reason(poll_data)}")

            if now >= next_progress_log:
                log.info(
                    "MiniMax: task %s status=%s, 已等待 %ds / 上限 %ds",
                    task_id,
                    status,
                    int(now - start),
                    timeout_seconds,
                )
                next_progress_log = now + _PROGRESS_LOG_INTERVAL

            time.sleep(_POLL_INTERVAL)

        raise RuntimeError(timeout_recovery_message(task_id, timeout_seconds))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _record_usage(
        self,
        result: GenerationResult,
        *,
        model: object = None,
        resolution: object = None,
        duration: object = None,
    ) -> None:
        """Append this task's usage to the local ledger (issue #328, W-11).

        Bookkeeping only, and it must stay that way: the generation has already
        succeeded and the money is already spent, so nothing here is allowed to
        propagate.

        **Known gap, stated rather than papered over:** the ledger is
        token-denominated because Ark bills in tokens, while MiniMax bills in
        USD per second.  There is no honest token number to write for a MiniMax
        clip, and inventing one would corrupt the very accounting the ledger
        exists for.  So a MiniMax run contributes to the *gate* (which is
        evaluated before every submit, exactly like Seedance) but not to the
        cumulative total, and the per-clip USD cost is logged instead.  Teaching
        the ledger a second currency is a ledger change, which is out of this
        card's scope — it needs its own card.
        """
        usage = result.metadata.get("usage") or {}
        cost = result.metadata.get("estimated_cost_usd")
        if cost is not None:
            log.info("MiniMax: 本次约 $%.2f（按 %s 每秒计价）。", float(cost), resolution)
        if not usage:
            log.info(
                "MiniMax 按秒计费、不返回 token 用量，因此本次不写入 token 账本"
                "（账本目前只记 Ark 的 token）。预算软闸仍在每次提交前生效。"
            )
            return
        try:
            usage_ledger.record_generation(
                provider=self.provider_name,
                usage=usage,
                task_id=result.job_id,
                model=model,
                resolution=resolution,
                duration=duration,
            )
        except Exception as exc:  # never fail a paid generation over bookkeeping
            log.warning("用量记账失败（已忽略，本次生成结果不受影响）：%s", exc)

    # ------------------------------------------------------------------
    # Request body
    # ------------------------------------------------------------------

    @staticmethod
    def _build_body(
        content: list[dict[str, object]],
        *,
        duration: int = DEFAULT_DURATION,
        has_first_frame: bool = False,
        **kwargs: str | int | float | bool | None,
    ) -> dict[str, object]:
        """Assemble the MiniMax request body.

        Only fields verified against the V2 reference are ever sent — the
        cross-provider CLI passes Seedance-only kwargs (``draft``, ``seed``,
        ``camera_fixed``, ``negative_prompt``…) to every provider, and
        forwarding an unknown field to MiniMax earns a ``2013 invalid
        parameters`` rejection. Unknown kwargs are therefore dropped, not
        passed through. ``extra_body`` is the deliberate escape hatch for a
        field this client does not know about yet.

        ``ratio`` handling mirrors the documented behaviour:

        * text-to-video does **not** accept ``adaptive``, so an adaptive
          request omits the field entirely and lets the API apply its 16:9
          default (warned, because an operator who wanted square would
          otherwise silently pay for a widescreen clip);
        * in first-frame mode the ratio is derived from the image, so an
          explicit ratio is sent but flagged as advisory.
        """
        model = resolve_cli_model(kwargs.get("model"))  # type: ignore[arg-type]
        resolution = resolve_cli_resolution(kwargs.get("resolution"))  # type: ignore[arg-type]
        duration = MiniMaxProvider._validate_duration(duration)

        ratio = kwargs.get("ratio")
        if ratio is None:
            # The cross-provider ``aspect_ratio`` surface. "16:9" is also the
            # API default, so it is forwarded as-is; no special-casing needed.
            ratio = kwargs.get("aspect_ratio", DEFAULT_RATIO)
        ratio = str(ratio)
        if ratio not in VALID_RATIOS:
            raise ValueError(f"MiniMax ratio {ratio!r} is not supported. Allowed: {', '.join(VALID_RATIOS)}")

        body: dict[str, object] = {
            "model": model,
            "content": content,
            "resolution": resolution,
            "duration": duration,
        }

        if has_first_frame:
            body["ratio"] = ratio
            if ratio != DEFAULT_RATIO:
                log.info("MiniMax: %s", RATIO_IGNORED_NOTE)
        elif ratio == DEFAULT_RATIO:
            log.warning(
                "MiniMax 文生视频不接受 ratio=adaptive，本次不发送 ratio 字段，"
                "API 将按默认 16:9 出片。要方图请显式 --gen-ratio 1:1。"
            )
        else:
            body["ratio"] = ratio

        extra = kwargs.get("extra_body")
        if isinstance(extra, dict):
            body.update(extra)
        return body

    @staticmethod
    def _validate_duration(duration: object) -> int:
        """Coerce and range-check ``duration``; raise before any HTTP traffic."""
        try:
            seconds = int(duration)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(f"MiniMax duration must be a whole number of seconds, got {duration!r}") from None
        if not (DURATION_MIN <= seconds <= DURATION_MAX):
            raise ValueError(
                f"MiniMax duration must be between {DURATION_MIN} and {DURATION_MAX} seconds, got {seconds}"
            )
        return seconds

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_submit(resp: httpx.Response) -> str:
        """Return the task id from a submit response, handling errors.

        MiniMax reports application-level errors in ``base_resp`` with HTTP
        200, so the body has to be inspected even on a "successful" response —
        an insufficient-balance rejection would otherwise look like a missing
        task id.
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = MiniMaxProvider._http_error_detail(exc)
            raise RuntimeError(f"MiniMax submit failed: {exc.response.status_code}{detail}") from None

        data = resp.json()
        MiniMaxProvider._raise_for_base_resp(data, "MiniMax submit")

        task_id = data.get("task_id") or data.get("id") or (data.get("task") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"MiniMax submit response missing task_id: {data}")
        return str(task_id)

    @staticmethod
    def _raise_for_base_resp(data: object, context: str) -> None:
        """Raise when the payload carries a non-zero ``base_resp.status_code``.

        Status code 0 means success. Anything else is an application error that
        arrives with HTTP 200, so it must be surfaced explicitly.
        """
        if not isinstance(data, dict):
            return
        base = data.get("base_resp")
        if not isinstance(base, dict):
            return
        try:
            code = int(base.get("status_code", 0))
        except (TypeError, ValueError):
            return
        if code == 0:
            return
        message = str(base.get("status_msg", "")).strip()
        hint = _STATUS_CODE_HINTS.get(code)
        parts = [f"{context} failed: status_code={code}"]
        if message:
            parts.append(message)
        if hint:
            parts.append(hint)
        raise RuntimeError(" — ".join(parts))

    @staticmethod
    def _raise_for_poll_status(resp: httpx.Response, task_id: str) -> None:
        """Convert a poll HTTP error into a ``RuntimeError`` naming the task.

        Callers (the CLI included) catch ``RuntimeError``/``ValueError``; an
        unwrapped ``httpx.HTTPStatusError`` would escape as a traceback. The
        common case is a resume with a typo'd or expired task id.
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = MiniMaxProvider._http_error_detail(exc)
            raise RuntimeError(f"MiniMax task {task_id} poll failed: {exc.response.status_code}{detail}") from None

    @staticmethod
    def _http_error_detail(exc: httpx.HTTPStatusError) -> str:
        """Best-effort human detail from an error response body."""
        try:
            body = exc.response.json()
        except Exception:
            return ""
        if not isinstance(body, dict):
            return ""
        base = body.get("base_resp")
        if isinstance(base, dict) and base.get("status_msg"):
            code = base.get("status_code")
            hint = _STATUS_CODE_HINTS.get(code) if isinstance(code, int) else None
            return f" — {base['status_msg']}" + (f" — {hint}" if hint else "")
        for key in ("message", "error_msg", "msg"):
            if body.get(key):
                return f" — {body[key]}"
        return ""

    @staticmethod
    def _extract_status(data: dict) -> str:
        """Locate the task status across the documented response nestings."""
        for bucket in (data, data.get("task") or {}):
            if isinstance(bucket, dict) and bucket.get("status"):
                return str(bucket["status"])
        return "unknown"

    @staticmethod
    def _extract_video_url(data: dict) -> str | None:
        """Locate the video download URL in a completed task response.

        The V2 task carries it at ``content.url``. The alternative spellings
        (``video_url``, a ``task`` wrapper, a flat top-level field) are
        defensive: the same defensiveness in the Seedance client is what let it
        survive Ark's response-shape drift.
        """
        for bucket in (data, data.get("task") or {}):
            if not isinstance(bucket, dict):
                continue
            content = bucket.get("content")
            if isinstance(content, dict):
                for key in ("url", "video_url", "download_url"):
                    if content.get(key):
                        return str(content[key])
            for key in ("video_url", "download_url"):
                if bucket.get(key):
                    return str(bucket[key])
        return None

    @staticmethod
    def _failure_reason(data: dict) -> str:
        """Render whatever the API said about a failed task."""
        for bucket in (data, data.get("task") or {}):
            if not isinstance(bucket, dict):
                continue
            error = bucket.get("error")
            if isinstance(error, dict):
                code = error.get("code", "")
                message = error.get("message", "")
                return f"{code}: {message}" if message else str(code)
            for key in ("status_msg", "message", "error_msg", "fail_reason"):
                if bucket.get(key):
                    return str(bucket[key])
        base = data.get("base_resp")
        if isinstance(base, dict) and base.get("status_msg"):
            return str(base["status_msg"])
        return "no reason reported by the API"

    # ------------------------------------------------------------------
    # Image encoding + validation
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Validate a local image and base64-encode it as a data URL; pass an
        ``http(s)`` URL through verbatim."""
        validate_image_for_i2v(image_path)
        lowered = image_path.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return image_path
        path = Path(image_path)
        data = path.read_bytes()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
