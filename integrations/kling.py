"""Kling (快手可灵) video generation provider.

API reference: https://docs.klingai.com/
Credentials: ``KLING_API_KEY`` and ``KLING_SECRET_KEY`` env vars.

NOTE: Endpoint URLs and payload shapes are based on public documentation and
may need adjustment against the live API.  Marked with ``# TODO: verify``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from hashlib import sha256

import httpx

from integrations.base import GenerationParams, JobState, JobStatus, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://api.klingai.com"
_SUBMIT_PATH = "/v1/videos/submit"
_QUERY_PATH = "/v1/videos/{job_id}"
_FETCH_PATH = "/v1/videos/{job_id}/fetch"


class KlingProvider(VideoGenProvider):
    """Kling (Kuaishou Kling) video generation provider."""

    def _load_api_key(self) -> str:
        api_key = os.environ.get("KLING_API_KEY", "")
        if not api_key:
            log.warning("KLING_API_KEY not set; KlingProvider will fail at runtime")
        return api_key

    def _sign(self, body: dict) -> tuple[str, str, str]:
        """Generate Kling API signature.

        Returns (signature, timestamp, nonce) tuple.
        """
        ts = str(int(time.time()))
        nonce = uuid.uuid4().hex[:16]
        raw = f"{ts}{nonce}{json.dumps(body, separators=(',', ':'))}{self._api_key}"
        sig = sha256(raw.encode()).hexdigest()
        return sig, ts, nonce

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(base_url=_BASE_URL, timeout=120)
        return self._http_client

    async def submit(self, params: GenerationParams) -> str:
        # Build request body — TODO: verify against live API
        body = {
            "model_name": "kling-v1",
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "duration": params.duration_seconds,
            "resolution": params.resolution,
            "cfg_scale": 0.7,  # TODO: verify allowed values
        }
        sig, ts, nonce = self._sign(body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",  # TODO: verify auth scheme
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
        }
        client = await self._client()
        resp = await client.post(_SUBMIT_PATH, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        job_id: str = data.get("data", {}).get("job_id", "")
        if not job_id:
            raise ValueError(f"Kling submit response missing job_id: {data}")
        log.info("Kling submit OK, job_id=%s", job_id)
        return job_id

    async def poll(self, job_id: str) -> JobStatus:
        client = await self._client()
        sig, ts, nonce = self._sign({})
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
        }
        path = _QUERY_PATH.format(job_id=job_id)
        resp = await client.get(path, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", {})

        status = data.get("status", "unknown")
        state_map = {
            "pending": JobState.PENDING,
            "processing": JobState.PROCESSING,
            "succeed": JobState.COMPLETED,
            "failed": JobState.FAILED,
        }
        state = state_map.get(status, JobState.PENDING)

        return JobStatus(
            job_id=job_id,
            state=state,
            progress=data.get("progress", 0),
            message=data.get("message", ""),
            output_url=data.get("video_url"),
            metadata=data,
        )

    async def download(self, job_id: str, out_path: str) -> str:
        client = await self._client()
        sig, ts, nonce = self._sign({})
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
        }
        path = _FETCH_PATH.format(job_id=job_id)
        resp = await client.get(path, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        video_url = data.get("video_url")
        if not video_url:
            raise ValueError(f"Kling download response missing video_url: {data}")

        # Download the actual video binary
        dl_client = httpx.AsyncClient(timeout=300)
        async with dl_client:
            dl_resp = await dl_client.get(video_url)
            dl_resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(dl_resp.content)

        log.info("Kling download OK -> %s (%d bytes)", out_path, len(dl_resp.content))
        return out_path
