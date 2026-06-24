"""Seedance (Google-backed) video generation provider.

API reference: https://docs.seedance.ai/
Credentials: ``SEEDANCE_API_KEY`` env var.

NOTE: Endpoint URLs and payload shapes are based on public documentation and
may need adjustment against the live API.  Marked with ``# TODO: verify``.
"""

from __future__ import annotations

import logging
import os

import httpx

from integrations.base import GenerationParams, JobState, JobStatus, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://api.seedance.ai/v1"
_SUBMIT_PATH = "/video/generations"
_QUERY_PATH = "/video/generations/{job_id}"


class SeedanceProvider(VideoGenProvider):
    """Seedance video generation provider."""

    def _load_api_key(self) -> str:
        api_key = os.environ.get("SEEDANCE_API_KEY", "")
        if not api_key:
            log.warning("SEEDANCE_API_KEY not set; SeedanceProvider will fail at runtime")
        return api_key

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=_BASE_URL,
                timeout=120,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def submit(self, params: GenerationParams) -> str:
        body = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "duration": params.duration_seconds,
            "resolution": params.resolution,
            "fps": params.fps,
            # TODO: verify additional parameters against live API
        }
        client = await self._client()
        resp = await client.post(_SUBMIT_PATH, json=body)
        resp.raise_for_status()
        data = resp.json()
        job_id: str = data.get("id", "")
        if not job_id:
            raise ValueError(f"Seedance submit response missing id: {data}")
        log.info("Seedance submit OK, job_id=%s", job_id)
        return job_id

    async def poll(self, job_id: str) -> JobStatus:
        client = await self._client()
        path = _QUERY_PATH.format(job_id=job_id)
        resp = await client.get(path)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "unknown")
        state_map = {
            "queued": JobState.PENDING,
            "processing": JobState.PROCESSING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
            "cancelled": JobState.CANCELLED,
        }
        state = state_map.get(status, JobState.PENDING)

        return JobStatus(
            job_id=job_id,
            state=state,
            progress=data.get("progress", 0),
            message=data.get("message", ""),
            output_url=data.get("output", {}).get("video_url") if isinstance(data.get("output"), dict) else None,
            metadata=data,
        )

    async def download(self, job_id: str, out_path: str) -> str:
        status = await self.poll(job_id)
        if status.state != JobState.COMPLETED or not status.output_url:
            raise ValueError(f"Seedance job {job_id} not ready for download (state={status.state})")

        async with httpx.AsyncClient(timeout=300) as dl_client:
            resp = await dl_client.get(status.output_url)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)

        log.info("Seedance download OK -> %s (%d bytes)", out_path, len(resp.content))
        return out_path
