"""Seedance (火山方舟 / Volcengine Ark) video generation provider.

The Seedance video model is served on the **Volcengine Ark** platform, not on
a ``api.seedance.ai`` endpoint.  This provider speaks the real Ark
``/contents/generations/tasks`` contract that the lead calibrated with a live
key.  See the issue card for the measured request / response shape.

API base: https://ark.cn-beijing.volces.com/api/v3
Credentials: ``ARK_API_KEY`` env var (Volcengine Ark console).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path

import httpx

from integrations.base import GenerationResult, VideoGenProvider
from pipeline.image_prep import validate_image_for_i2v

log = logging.getLogger(__name__)

_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
_SUBMIT_PATH = "/contents/generations/tasks"
_QUERY_PATH = "/contents/generations/tasks/{task_id}"
_POLL_INTERVAL = 3.0

# ---------------------------------------------------------------------------
# Poll budget (issue #325)
# ---------------------------------------------------------------------------
# The client used to hard-code a 300s ceiling.  That covers the 480p/5s
# baseline but *guarantees* a false timeout for 4k/10s, which takes far longer
# on the Ark side.  A false timeout is not free: by the time the client gives
# up the quota has already been spent (the 4k/10s incident cost 1,952,100
# tokens), so "just run it again" burns a second copy.  The budget therefore
# scales with the two parameters that actually drive render time — resolution
# tier and duration — and any leftover overrun is recoverable via
# ``resume()`` instead of a re-submit.

#: Poll budget for the 480p / 5s baseline. Unchanged from the pre-#325 value
#: so the cheap default path behaves exactly as before.
_BASE_POLL_SECONDS = 300

#: Duration (seconds) that ``_BASE_POLL_SECONDS`` is calibrated for.
_BASELINE_DURATION = 5

#: Back-compat alias for the old module constant — still the 480p/5s budget.
_MAX_POLL_SECONDS = _BASE_POLL_SECONDS

#: Multiplier applied to the baseline budget per resolution tier. 4k is the
#: measured pain point: 4k/10s lands at 2400s, comfortably past the 300s that
#: aborted the live run.
RESOLUTION_POLL_FACTORS: dict[str, float] = {
    "480p": 1.0,
    "720p": 1.5,
    "1080p": 2.5,
    "4k": 4.0,
}

#: Budget used by :meth:`SeedanceProvider.resume` when the caller cannot say
#: what the original request looked like — the heaviest supported combination
#: (4k / 15s). Resuming costs no quota, so waiting long is the cheap side.
_RESUME_POLL_SECONDS = 3600

#: How often (seconds) to log "still running, waited Ns" while polling. Before
#: #325 the only sign of life was httpx's own GET logging, which says nothing
#: about elapsed time or the remaining budget.
_PROGRESS_LOG_INTERVAL = 30.0

# Model IDs (类常量). ``MODEL_FAST`` is the default (owner note: quota-limited,
# so we default to the lower-cost variant).
MODEL_FAST = "doubao-seedance-2-0-fast-260128"
MODEL_STD = "doubao-seedance-2-0-260128"

# Default generation parameters (low-spec per owner: limited quota).
_DEFAULT_RESOLUTION = "480p"
_DEFAULT_RATIO = "adaptive"
_DEFAULT_DURATION = 5

# Resolution tiers each model variant supports (issue #246). The fast variant
# caps at 720p; 4k (2880x2880 for 1:1) is standard-model-only. This is a
# pre-flight contract so callers see a clear, local error instead of the
# opaque Ark ``ModelNotOpen`` response after burning quota.
MODEL_RESOLUTIONS: dict[str, set[str]] = {
    MODEL_FAST: {"480p", "720p"},
    MODEL_STD: {"480p", "720p", "1080p", "4k"},
}

# Provider fields the CLI may request but which are sent into the Ark body
# only when explicitly supplied (default = omit, so the existing request
# shape is byte-for-byte unchanged for the common path).
PASSTHROUGH_FIELDS = ("draft", "return_last_frame", "seed", "camera_fixed")

# Canonical tier order, shared by both CLAs so ``choices`` never drift.
VALID_RESOLUTIONS = ("480p", "720p", "1080p", "4k")
VALID_RATIOS = ("adaptive", "16:9", "9:16", "1:1")


def poll_timeout_seconds(
    resolution: str = _DEFAULT_RESOLUTION,
    duration: int | float | str = _DEFAULT_DURATION,
    override: int | float | None = None,
) -> int:
    """Return the poll budget (seconds) for a *resolution* / *duration* pair.

    The budget is ``300s × resolution_factor × duration_factor`` and never
    drops below the 300s baseline, so the cheap 480p/5s path keeps its
    historical behaviour while 4k/10s gets 2400s instead of a guaranteed
    false timeout (issue #325).

    *override* (the CLI's ``--gen-timeout``) wins over the computed value;
    it must be a positive number of seconds.
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
    # Shorter-than-baseline clips never shrink the budget below 300s.
    duration_factor = max(1.0, duration_factor)
    return max(_BASE_POLL_SECONDS, int(_BASE_POLL_SECONDS * factor * duration_factor))


def timeout_recovery_message(task_id: str, waited_seconds: int | float) -> str:
    """Render the operator-facing text for a poll timeout.

    A timeout is **not** a failure: the Ark task usually keeps running and the
    quota is already spent.  The one thing an operator must not do is re-run
    the generation, so the recovery path (``--resume-task``) and the task id
    are stated up front rather than buried behind a generic "did not complete"
    line (issue #325).
    """
    return (
        f"Seedance task {task_id} did not complete within {int(waited_seconds)}s.\n"
        f"  ⚠️  超时 ≠ 失败：任务很可能仍在方舟服务端继续跑，额度已经消耗。\n"
        f"  ⚠️  不要重新生成 —— 重跑会再扣一份额度。\n"
        f"  task_id: {task_id}\n"
        f"  取回成片: python -m scripts.generate --provider seedance "
        f"--resume-task {task_id} -o <output.mp4>\n"
        f"  下次想等更久: 加 --gen-timeout <秒>"
    )


class SeedanceProvider(VideoGenProvider):
    """Seedance video generation on Volcengine Ark.

    Requires ``ARK_API_KEY`` environment variable.
    """

    # Model IDs exposed as class attributes for ergonomic override usage:
    # ``provider.generate(..., model=SeedanceProvider.MODEL_STD)``.
    MODEL_FAST = MODEL_FAST  # re-export module constant as class attribute
    MODEL_STD = MODEL_STD

    def _load_api_key(self) -> str:
        api_key = os.environ.get("ARK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ARK_API_KEY environment variable is not set. "
                "Generate one in the Volcengine Ark console "
                "(https://console.volcengine.com/ark/)."
            )
        return api_key

    # ------------------------------------------------------------------
    # Public API — text-to-video
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: str | int | float,
    ) -> GenerationResult:
        """Generate a video via the Ark Seedance text-to-video task endpoint.

        The Ark endpoint ignores ``aspect_ratio`` and ``fps`` (resolution/ratio/
        duration are native fields), but they are accepted to keep a stable
        cross-provider surface.
        """
        content = [{"type": "text", "text": prompt}]
        if "ratio" not in kwargs and aspect_ratio != "16:9":
            kwargs["aspect_ratio"] = aspect_ratio
        return self._run(content, duration=duration, **kwargs)

    # ------------------------------------------------------------------
    # Image-to-video
    # ------------------------------------------------------------------

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs: str | int | float,
    ) -> GenerationResult:
        """Generate a video from a starting image via the Ark Seedance task endpoint.

        A local image is validated, read, and base64-encoded into a
        ``data:image/png;base64,...`` data URL.  An ``http(s)`` URL is passed
        through verbatim.  Prompt may be empty (image-only motion).
        """
        encoded = self._encode_image(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encoded}},
        ]
        # Issue #246: aspect_ratio used to be a silent no-op on Ark.
        # Forward it through to _build_body (which maps it onto body["ratio"]),
        # but let an explicit ``ratio=`` kwarg override it. The cross-provider
        # default "16:9" is treated as "not set" so the Ark default "adaptive"
        # is preserved when nobody explicitly picks a ratio.
        if "ratio" not in kwargs and aspect_ratio != "16:9":
            kwargs["aspect_ratio"] = aspect_ratio
        return self._run(content, duration=duration, **kwargs)

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
        duration: int = 5,
        **kwargs: str | int | float | bool | None,
    ) -> GenerationResult:
        # ``poll_timeout`` is a client-side knob (CLI --gen-timeout), never a
        # body field — pop it before the body is assembled.
        poll_timeout = kwargs.pop("poll_timeout", None)
        body = self._build_body(
            content=content,
            duration=duration,
            **kwargs,  # type: ignore[arg-type]
        )
        timeout_seconds = poll_timeout_seconds(
            resolution=str(body["resolution"]),
            duration=body["duration"],  # type: ignore[arg-type]
            override=poll_timeout,  # type: ignore[arg-type]
        )
        headers = self._headers()

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            log.info(
                "Seedance: submitting task (resolution=%s duration=%ss, 轮询上限 %ds)",
                body["resolution"],
                body["duration"],
                timeout_seconds,
            )
            resp = client.post(_SUBMIT_PATH, json=body, headers=headers)
            task_id = self._handle_submit(resp)
            log.info("Seedance: task submitted, task_id=%s", task_id)
            return self._poll_task(client, task_id, headers, timeout_seconds)

    # ------------------------------------------------------------------
    # Resume: poll an already-submitted task (issue #325)
    # ------------------------------------------------------------------

    def resume(
        self,
        task_id: str,
        poll_timeout: int | float | None = None,
    ) -> GenerationResult:
        """Poll an **already submitted** Ark task and return its result.

        This is the recovery path for a client-side timeout, a dropped
        connection or a killed session: the task lives on the Ark side and its
        quota is already spent, so re-submitting the same generation would
        simply pay for it twice.  ``resume`` issues **no** ``POST`` — only the
        task-query ``GET`` — so it is always free.

        Parameters
        ----------
        task_id : str
            The Ark task id (``cgt-…``) echoed at submit time and repeated in
            the timeout message.
        poll_timeout : int | float | None
            Seconds to keep polling. Defaults to the heaviest supported
            combination's budget, since the original request parameters are
            not knowable from a task id alone.
        """
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("resume() requires a non-empty Seedance task id (e.g. cgt-20260908185831-69srb)")

        timeout_seconds = _RESUME_POLL_SECONDS if poll_timeout is None else poll_timeout_seconds(override=poll_timeout)
        headers = self._headers()

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            log.info(
                "Seedance: resuming task %s — 跳过提交，不消耗新额度（轮询上限 %ds）",
                task_id,
                timeout_seconds,
            )
            return self._poll_task(client, task_id, headers, timeout_seconds)

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

        The first query happens immediately (before any sleep) so a resume of
        an already-finished task returns straight away.
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

            # Real Ark shape: top-level ``status`` (running/succeeded/failed).
            # A ``usage.completion_tokens`` block may ride along — logged for observability.
            usage = poll_data.get("usage") or {}
            if usage:
                log.info("Seedance: task %s usage=%s", task_id, usage)
            status = poll_data.get("status") or poll_data.get("task", {}).get("status") or "unknown"

            if status in ("succeeded", "completed"):
                video_url = self._extract_video_url(poll_data)
                if not video_url:
                    raise RuntimeError(f"Seedance task {task_id} completed but missing video URL: {poll_data}")
                log.info("Seedance: task completed, url=%s", video_url)
                return GenerationResult(
                    video_url=video_url,
                    provider=self.provider_name,
                    job_id=task_id,
                    metadata={"status": status, **poll_data},
                )
            if status in ("failed", "expired"):
                error = poll_data.get("error") or poll_data.get("task", {}).get("error") or {}
                msg = self._error_message(error)
                raise RuntimeError(f"Seedance task {task_id} {status}: {msg}")

            if now >= next_progress_log:
                log.info(
                    "Seedance: task %s status=%s, 已等待 %ds / 上限 %ds",
                    task_id,
                    status,
                    int(now - start),
                    timeout_seconds,
                )
                next_progress_log = now + _PROGRESS_LOG_INTERVAL

            time.sleep(_POLL_INTERVAL)

        raise RuntimeError(timeout_recovery_message(task_id, timeout_seconds))

    @staticmethod
    def _raise_for_poll_status(resp: httpx.Response, task_id: str) -> None:
        """Convert a poll HTTP error into a ``RuntimeError`` naming the task.

        Callers (the CLI included) catch ``RuntimeError``/``ValueError``; an
        unwrapped ``httpx.HTTPStatusError`` would escape as a traceback. The
        common case is a resume with a typo'd or expired task id, which Ark
        answers with a ``ResourceNotFound`` error block.
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error = SeedanceProvider._extract_http_error(exc)
            if error:
                raise RuntimeError(f"Seedance task {task_id}: {SeedanceProvider._error_message(error)}") from None
            raise RuntimeError(f"Seedance task {task_id} poll failed: {exc.response.status_code} {exc}") from None

    @staticmethod
    def _build_body(
        content: list[dict[str, object]],
        *,
        duration: int = _DEFAULT_DURATION,
        **kwargs: str | int | float | bool,
    ) -> dict[str, object]:
        """Assemble the Ark request body from provider kwargs.

        The five base keys (model/content/resolution/ratio/duration) always
        appear, so the default request shape is unchanged.  Optional body
        fields from :data:`PASSTHROUGH_FIELDS` are copied through **only when
        explicitly supplied** — an absent flag never adds a key, so quota and
        pricing stay exactly what the pre-#246 client sent.

        ``aspect_ratio`` is the cross-provider surface name; the Ark body
        expects ``ratio``.  An explicit ``ratio`` wins, otherwise the caller's
        ``aspect_ratio`` is honoured (it is no longer silently dropped).

        Raises ``ValueError`` for a model/resolution pair the model variant
        does not support — raised *before* any HTTP traffic so the operator
        sees the reason locally instead of a later ``ModelNotOpen``.
        """
        model = kwargs.get("model", MODEL_FAST)
        resolution = kwargs.get("resolution", _DEFAULT_RESOLUTION)

        SeedanceProvider._validate_model_resolution(model, resolution)

        ratio = kwargs.get("ratio")
        if ratio is None:
            # Legacy alias: the cross-provider aspect_ratio was previously a
            # silent no-op on Ark. It now maps onto the real body field.
            ratio = kwargs.get("aspect_ratio", _DEFAULT_RATIO)

        body: dict[str, object] = {
            "model": model,
            "content": content,
            "resolution": resolution,
            "ratio": ratio,
            "duration": kwargs.get("duration", duration),
        }
        for field in PASSTHROUGH_FIELDS:
            if field in kwargs:
                body[field] = kwargs[field]
        return body

    @staticmethod
    def _validate_model_resolution(model: object, resolution: object) -> None:
        """Reject a resolution the chosen model variant cannot produce.

        The Ark fast variant tops out at 720p; requesting 4k with it yields
        ``ModelNotOpen`` from the API — an opaque failure that costs quota.
        Fail locally instead, naming both the unsupported tier and the model
        that does support it.
        """
        if not isinstance(model, str):
            model = str(model)
        if not isinstance(resolution, str):
            resolution = str(resolution)

        supported = MODEL_RESOLUTIONS.get(model)
        if supported is None:
            return  # unknown model id: let the API report it (misspelling case)
        if resolution in supported:
            return

        supported_str = ", ".join(sorted(supported))
        raise ValueError(
            f"Model {model} does not support resolution '{resolution}' "
            f"(supported: {supported_str}). "
            f"Use --model {MODEL_STD} for 4k / 1080p."
        )

    @staticmethod
    def _handle_submit(resp: httpx.Response) -> str:
        """Return the task id from a submit response, handling errors.

        Real Ark response shape: ``{"id": "cgt-...", ...}`` — the field is
        ``id``, **not** ``task_id``.  The legacy ``task_id`` lookups are kept as
        a defensive fallback only.
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error = SeedanceProvider._extract_http_error(exc)
            if error:
                raise RuntimeError(SeedanceProvider._error_message(error)) from None
            raise RuntimeError(f"Seedance submit failed: {exc.response.status_code} {exc}") from None

        data = resp.json()
        task_id: str | None = data.get("id") or (data.get("task") or {}).get("task_id") or data.get("task_id")
        if not task_id:
            error = data.get("error") or {}
            if error.get("code"):
                raise RuntimeError(SeedanceProvider._error_message(error))
            raise RuntimeError(f"Seedance submit response missing task_id: {data}")
        return task_id

    @staticmethod
    def _extract_video_url(data: dict) -> str | None:
        """Locate the video URL in an Ark task response.

        Real shape on success: a signed TOS download link at
        ``content.video_url``.  The older ``task.outputs[].video_url`` /
        ``content[].outputs[].video_url`` nesting is kept as a defensive
        fallback, plus a flat top-level ``video_url`` for shape drift.
        """
        content = data.get("content")
        if isinstance(content, dict) and content.get("video_url"):
            return str(content["video_url"])
        for bucket in (data.get("task") or {}, data):
            if not isinstance(bucket, dict):
                continue
            for item in bucket.get("outputs") or []:
                if isinstance(item, dict) and item.get("video_url"):
                    return str(item["video_url"])
            for item in bucket.get("content") or []:
                if isinstance(item, dict):
                    for out in item.get("outputs") or []:
                        if isinstance(out, dict) and out.get("video_url"):
                            return str(out["video_url"])
        return data.get("video_url")  # type: ignore[return-value]

    @staticmethod
    def _extract_http_error(exc: httpx.HTTPStatusError) -> dict | None:
        """Pull the ``error`` block out of an error HTTP response body."""
        try:
            body = exc.response.json()
        except Exception:
            return None
        error = body.get("error")
        return error if isinstance(error, dict) else None

    @staticmethod
    def _error_message(error: dict) -> str:
        """Render a user-actionable message from an Ark ``error`` block.

        Branches on the measured ``error.code`` values so callers see a hint
        they can act on rather than a raw code string.
        """
        code = error.get("code", "")
        message = error.get("message", "")
        msg = f"{code}: {message}" if message else str(code)
        if code == "InvalidEndpointOrModel.NotFound":
            return f"{code}: model not found ({message}). The configured model id may be invalid or misspelled."
        if code == "ModelNotOpen":
            model = error.get("model", "the configured model")
            return f"{code}: model {model} is not enabled for your project. 请在方舟控制台开通模型 {model}。"
        if code == "ResourceNotFound":
            return f"{code}: task/resource not found, may have expired ({message})"
        return msg

    # ------------------------------------------------------------------
    # Image encoding + validation
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Validate a local image and base64-encode it as a data URL; pass
        through an ``http(s)`` URL verbatim."""
        validate_image_for_i2v(image_path)
        if image_path.lower().startswith("http://") or image_path.lower().startswith("https://"):
            return image_path
        path = Path(image_path)
        data = path.read_bytes()
        ext = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
