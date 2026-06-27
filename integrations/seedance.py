"""Seedance (Google-backed) video generation provider.

API reference: https://docs.seedance.ai/
Credentials: ``SEEDANCE_API_KEY`` env var.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from integrations.base import GenerationResult, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://api.seedance.ai/v1"
_SUBMIT_PATH = "/video/generations"
_QUERY_PATH = "/video/generations/{job_id}"
_POLL_INTERVAL = 3.0
_MAX_POLL_SECONDS = 300


class SeedanceProvider(VideoGenProvider):
    """Seedance video generation provider.

    Requires ``SEEDANCE_API_KEY`` environment variable.
    """

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
