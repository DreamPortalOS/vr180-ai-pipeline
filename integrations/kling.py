"""Kling (Kuaishou / 可灵) video generation provider.

API reference: https://docs.klingai.com/
Credentials: ``KLING_API_KEY`` env var.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from hashlib import sha256
from pathlib import Path

import httpx

from integrations.base import GenerationResult, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://api.klingai.com"
_SUBMIT_PATH = "/v1/videos/submit"
_I2V_SUBMIT_PATH = "/v1/videos/image2video"
_QUERY_PATH = "/v1/videos/{job_id}"
_FETCH_PATH = "/v1/videos/{job_id}/fetch"
_POLL_INTERVAL = 3.0  # seconds between poll attempts
_MAX_POLL_SECONDS = 300  # give up after 5 minutes


class KlingProvider(VideoGenProvider):
    """Kling (Kuaishou) video generation provider.

    Uses the Kling API to generate videos from text prompts.
    Requires ``KLING_API_KEY`` environment variable.
    """

    # Image-to-video model + request field names (API details to be
    # corrected against live docs once a key is available; structure first).
    I2V_MODEL = "kling-v1"
    I2V_IMAGE_FIELD = "image"
    I2V_IMAGE_TYPE = "base64"

    def _load_api_key(self) -> str:
        api_key = os.environ.get("KLING_API_KEY", "")
        if not api_key:
            raise ValueError("KLING_API_KEY environment variable is not set. Generate one at https://docs.klingai.com/")
        return api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(api_key: str, body: dict) -> tuple[str, str, str]:
        """Generate Kling API signature. Returns (signature, timestamp, nonce)."""
        ts = str(int(time.time()))
        nonce = uuid.uuid4().hex[:16]
        raw = f"{ts}{nonce}{json.dumps(body, separators=(',', ':'))}{api_key}"
        sig = sha256(raw.encode()).hexdigest()
        return sig, ts, nonce

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
        """Generate a video via Kling API.

        Parameters
        ----------
        prompt : str
            Text description of the desired video.
        duration : int
            Target duration in seconds (1-5 for Kling v1).
        aspect_ratio : str
            Aspect ratio, e.g. ``"16:9"``, ``"9:16"``, ``"1:1"``.
        fps : int
            Frames per second (24 or 30).
        **kwargs
            Extra parameters (e.g. ``cfg_scale``, ``model``).

        Returns
        -------
        GenerationResult
            Result with the video URL.

        Raises
        ------
        RuntimeError
            If generation fails or polling times out.
        """
        w, h = self._parse_aspect_ratio(aspect_ratio)
        body: dict[str, int | str | float] = {
            "model_name": kwargs.get("model", "kling-v1"),
            "prompt": prompt,
            "duration": duration,
            "width": w * 40,  # Kling uses 40px units
            "height": h * 40,
            "fps": fps,
        }
        if "cfg_scale" in kwargs:
            body["cfg_scale"] = kwargs["cfg_scale"]
        if "negative_prompt" in kwargs:
            body["negative_prompt"] = kwargs["negative_prompt"]

        sig, ts, nonce = self._sign(self._api_key, body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
        }

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            # 1. Submit
            log.info("Kling: submitting job (prompt=%.50s...)", prompt)
            resp = client.post(_SUBMIT_PATH, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            job_id: str | None = (data.get("data") or {}).get("job_id")
            if not job_id:
                raise RuntimeError(f"Kling submit response missing job_id: {data}")
            log.info("Kling: job submitted, job_id=%s", job_id)

            # 2. Poll until completion
            deadline = time.time() + _MAX_POLL_SECONDS
            query_headers = {
                "Authorization": f"Bearer {self._api_key}",
                "X-Timestamp": ts,
                "X-Nonce": nonce,
                "X-Signature": sig,
            }
            while time.time() < deadline:
                time.sleep(_POLL_INTERVAL)
                path = _QUERY_PATH.format(job_id=job_id)
                poll_resp = client.get(path, headers=query_headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json().get("data", {})

                status = poll_data.get("status", "unknown")
                if status == "succeed":
                    video_url: str | None = poll_data.get("video_url")
                    if not video_url:
                        raise RuntimeError(f"Kling job completed but missing video_url: {poll_data}")
                    log.info("Kling: job completed, downloading from %s", video_url)
                    return GenerationResult(
                        video_url=video_url,
                        provider=self.provider_name,
                        job_id=job_id,
                        metadata={"status": status, **poll_data},
                    )
                elif status == "failed":
                    msg = poll_data.get("message", "unknown error")
                    raise RuntimeError(f"Kling job {job_id} failed: {msg}")
                # else: still pending/processing

            raise RuntimeError(f"Kling job {job_id} did not complete within {_MAX_POLL_SECONDS}s")

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs: str | int | float,
    ) -> GenerationResult:
        """Generate a video from a starting image via the Kling image2video API.

        A local image is base64-encoded into the request body; an ``http(s)``
        URL is passed through verbatim.  The submit/poll skeleton mirrors
        :meth:`generate`.
        """
        w, h = self._parse_aspect_ratio(aspect_ratio)
        body: dict[str, int | str | float | dict[str, str]] = {
            "model_name": kwargs.get("model", self.I2V_MODEL),
            "prompt": prompt,
            "duration": duration,
            "width": w * 40,
            "height": h * 40,
            "fps": kwargs.get("fps", 24),
        }
        if "cfg_scale" in kwargs:
            body["cfg_scale"] = kwargs["cfg_scale"]
        if "negative_prompt" in kwargs:
            body["negative_prompt"] = kwargs["negative_prompt"]

        body["image_type"] = self.I2V_IMAGE_TYPE
        body[self.I2V_IMAGE_FIELD] = self._encode_image(image_path)

        sig, ts, nonce = self._sign(self._api_key, body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
        }

        with httpx.Client(base_url=_BASE_URL, timeout=30) as client:
            log.info("Kling: submitting image2video job (image=%s)", image_path)
            resp = client.post(_I2V_SUBMIT_PATH, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            job_id: str | None = (data.get("data") or {}).get("job_id")
            if not job_id:
                raise RuntimeError(f"Kling i2v submit response missing job_id: {data}")
            log.info("Kling: i2v job submitted, job_id=%s", job_id)

            deadline = time.time() + _MAX_POLL_SECONDS
            while time.time() < deadline:
                time.sleep(_POLL_INTERVAL)
                poll_resp = client.get(_QUERY_PATH.format(job_id=job_id), headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json().get("data", {})
                status = poll_data.get("status", "unknown")
                if status == "succeed":
                    video_url: str | None = poll_data.get("video_url")
                    if not video_url:
                        raise RuntimeError(f"Kling i2v job completed but missing video_url: {poll_data}")
                    log.info("Kling: i2v job completed, downloading from %s", video_url)
                    return GenerationResult(
                        video_url=video_url,
                        provider=self.provider_name,
                        job_id=job_id,
                        metadata={"status": status, **poll_data},
                    )
                elif status == "failed":
                    msg = poll_data.get("message", "unknown error")
                    raise RuntimeError(f"Kling i2v job {job_id} failed: {msg}")

            raise RuntimeError(f"Kling i2v job {job_id} timed out within {_MAX_POLL_SECONDS}s")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Base64-encode a local image, or pass through an http(s) URL."""
        if image_path.lower().startswith("http://") or image_path.lower().startswith("https://"):
            return image_path
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image file not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")
