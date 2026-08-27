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
_MAX_POLL_SECONDS = 300

# Model IDs (类常量). ``MODEL_FAST`` is the default (owner note: quota-limited,
# so we default to the lower-cost variant).
MODEL_FAST = "doubao-seedance-2-0-fast-260128"
MODEL_STD = "doubao-seedance-2-0-260128"

# Default generation parameters (low-spec per owner: limited quota).
_DEFAULT_RESOLUTION = "480p"
_DEFAULT_RATIO = "adaptive"
_DEFAULT_DURATION = 5


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
        return self._run(content, duration=duration, **kwargs)

    # ------------------------------------------------------------------
    # Internal: submit → poll → result
    # ------------------------------------------------------------------

    def _run(
        self,
        content: list[dict[str, object]],
        duration: int = 5,
        **kwargs: str | int | float,
    ) -> GenerationResult:
        body: dict[str, object] = {
            "model": kwargs.get("model", MODEL_FAST),
            "content": content,
            "resolution": kwargs.get("resolution", _DEFAULT_RESOLUTION),
            "ratio": kwargs.get("ratio", _DEFAULT_RATIO),
            "duration": kwargs.get("duration", duration),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            log.info("Seedance: submitting task")
            resp = client.post(_SUBMIT_PATH, json=body, headers=headers)
            task_id = self._handle_submit(resp)
            log.info("Seedance: task submitted, task_id=%s", task_id)

            deadline = time.time() + _MAX_POLL_SECONDS
            while time.time() < deadline:
                time.sleep(_POLL_INTERVAL)
                poll_resp = client.get(_QUERY_PATH.format(task_id=task_id), headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("task", {}).get("status")
                if status is None:
                    status = poll_data.get("status", "unknown")

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

            raise RuntimeError(f"Seedance task {task_id} did not complete within {_MAX_POLL_SECONDS}s")

    @staticmethod
    def _handle_submit(resp: httpx.Response) -> str:
        """Return the ``task_id`` from a submit response, handling errors."""
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error = SeedanceProvider._extract_http_error(exc)
            if error:
                raise RuntimeError(SeedanceProvider._error_message(error)) from None
            raise RuntimeError(f"Seedance submit failed: {exc.response.status_code} {exc}") from None

        data = resp.json()
        task_id: str | None = (data.get("task") or {}).get("task_id") or data.get("task_id")
        if not task_id:
            error = data.get("error") or {}
            if error.get("code"):
                raise RuntimeError(SeedanceProvider._error_message(error))
            raise RuntimeError(f"Seedance submit response missing task_id: {data}")
        return task_id

    @staticmethod
    def _extract_video_url(data: dict) -> str | None:
        """Locate the video URL in an Ark task response.

        The measured shape nests the result under ``task.outputs[].video_url``
        or flat ``content[].outputs[].video_url``; fall back to a top-level
        ``video_url`` to stay tolerant of shape drift.
        """
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
            return f"{code}: model {model} is not enabled for your project. "
            f"请在方舟控制台开通模型 {model}。"
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
