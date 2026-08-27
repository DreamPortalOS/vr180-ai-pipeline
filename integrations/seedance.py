"""Seedance (Google-backed) video generation provider.

API reference: https://docs.seedance.ai/
Credentials: ``SEEDANCE_API_KEY`` env var.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path

import httpx

from integrations.base import GenerationResult, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://api.seedance.ai/v1"
_SUBMIT_PATH = "/video/generations"
_I2V_SUBMIT_PATH = "/video/image-generations"
_QUERY_PATH = "/video/generations/{job_id}"
_POLL_INTERVAL = 3.0
_MAX_POLL_SECONDS = 300


class SeedanceProvider(VideoGenProvider):
    """Seedance video generation provider.

    Requires ``SEEDANCE_API_KEY`` environment variable.
    """

    # Image-to-video endpoint + field names.
    I2V_MODEL = "seedance-v1"
    I2V_IMAGE_FIELD = "image"
    I2V_IMAGE_FORMAT = "base64"

    def _load_api_key(self) -> str:
        api_key = os.environ.get("SEEDANCE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "SEEDANCE_API_KEY environment variable is not set. Generate one at https://docs.seedance.ai/"
            )
        return api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: str | int | float,
    ) -> GenerationResult:
        """Generate a video via Seedance API.

        Parameters
        ----------
        prompt : str
            Text description of the desired video.
        duration : int
            Target duration in seconds.
        aspect_ratio : str
            Aspect ratio, e.g. ``"16:9"``, ``"9:16"``, ``"1:1"``.
        fps : int
            Target frame rate.
        **kwargs
            Extra parameters (e.g. ``negative_prompt``, ``model``).

        Returns
        -------
        GenerationResult
            Result with the video URL.

        Raises
        ------
        RuntimeError
            If generation fails or polling times out.
        """
        body: dict[str, int | str | float] = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
        }
        if "negative_prompt" in kwargs:
            body["negative_prompt"] = kwargs["negative_prompt"]
        if "model" in kwargs:
            body["model"] = kwargs["model"]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            # 1. Submit
            log.info("Seedance: submitting job (prompt=%.50s...)", prompt)
            resp = client.post(_SUBMIT_PATH, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            job_id: str | None = data.get("id")
            if not job_id:
                raise RuntimeError(f"Seedance submit response missing id: {data}")
            log.info("Seedance: job submitted, job_id=%s", job_id)

            # 2. Poll until completion
            deadline = time.time() + _MAX_POLL_SECONDS
            while time.time() < deadline:
                time.sleep(_POLL_INTERVAL)
                path = _QUERY_PATH.format(job_id=job_id)
                poll_resp = client.get(path, headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("status", "unknown")
                if status == "completed":
                    output = poll_data.get("output", {})
                    video_url: str | None = output.get("video_url") if isinstance(output, dict) else None
                    if not video_url:
                        raise RuntimeError(f"Seedance job completed but missing video_url: {poll_data}")
                    log.info("Seedance: job completed, url=%s", video_url)
                    return GenerationResult(
                        video_url=video_url,
                        provider=self.provider_name,
                        job_id=job_id,
                        metadata={"status": status, **poll_data},
                    )
                elif status in ("failed", "cancelled"):
                    msg = poll_data.get("message", "unknown error")
                    raise RuntimeError(f"Seedance job {job_id} {status}: {msg}")
                # else: queued / processing

            raise RuntimeError(f"Seedance job {job_id} did not complete within {_MAX_POLL_SECONDS}s")

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs: str | int | float,
    ) -> GenerationResult:
        """Generate a video from a starting image via Seedance image-generations.

        A local image is base64-encoded into the request body; an ``http(s)``
        URL is passed through verbatim.  The skeleton mirrors :meth:`generate`.
        """
        body: dict[str, int | str | float | dict[str, str]] = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "fps": kwargs.get("fps", 24),
        }
        if "negative_prompt" in kwargs:
            body["negative_prompt"] = kwargs["negative_prompt"]
        if "model" in kwargs:
            body["model"] = kwargs["model"]

        body["model"] = body.get("model", self.I2V_MODEL)
        body["image_format"] = self.I2V_IMAGE_FORMAT
        body[self.I2V_IMAGE_FIELD] = self._encode_image(image_path)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            log.info("Seedance: submitting image-generation job (image=%s)", image_path)
            resp = client.post(_I2V_SUBMIT_PATH, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            job_id: str | None = data.get("id")
            if not job_id:
                raise RuntimeError(f"Seedance i2v submit response missing id: {data}")
            log.info("Seedance: i2v job submitted, job_id=%s", job_id)

            deadline = time.time() + _MAX_POLL_SECONDS
            while time.time() < deadline:
                time.sleep(_POLL_INTERVAL)
                poll_resp = client.get(_QUERY_PATH.format(job_id=job_id), headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("status", "unknown")
                if status == "completed":
                    output = poll_data.get("output", {})
                    video_url: str | None = output.get("video_url") if isinstance(output, dict) else None
                    if not video_url:
                        raise RuntimeError(f"Seedance i2v job completed but missing video_url: {poll_data}")
                    log.info("Seedance: i2v job completed, url=%s", video_url)
                    return GenerationResult(
                        video_url=video_url,
                        provider=self.provider_name,
                        job_id=job_id,
                        metadata={"status": status, **poll_data},
                    )
                elif status in ("failed", "cancelled"):
                    msg = poll_data.get("message", "unknown error")
                    raise RuntimeError(f"Seedance i2v job {job_id} {status}: {msg}")

            raise RuntimeError(f"Seedance i2v job {job_id} timed out within {_MAX_POLL_SECONDS}s")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Base64-encode a local image, or pass through an http(s) URL."""
        if image_path.lower().startswith("http://") or image_path.lower().startswith("https://"):
            return image_path
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image file not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")
