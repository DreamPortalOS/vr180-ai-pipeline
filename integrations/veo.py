"""Veo (Google DeepMind) video generation provider.

API reference: https://cloud.google.com/vertex-ai/generative-ai/docs/veo/overview
Credentials: ``VEO_API_KEY`` env var (or application-default credentials).

NOTE: Endpoint URLs and payload shapes are based on public documentation and
may need adjustment against the live API.  Marked with ``# TODO: verify``.
"""

from __future__ import annotations

import logging
import os

import httpx

from integrations.base import GenerationParams, JobStatus, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://us-central1-aiplatform.googleapis.com/v1"
_SUBMIT_PATH = "/projects/{project_id}/locations/us-central1/publishers/google/models/veo-001:predict"


class VeoProvider(VideoGenProvider):
    """Veo (Google DeepMind) video generation provider via Vertex AI."""

    def _load_api_key(self) -> str:
        api_key = os.environ.get("VEO_API_KEY", "")
        if not api_key:
            log.warning("VEO_API_KEY not set; VeoProvider will fail at runtime")
        return api_key

    @property
    def _project_id(self) -> str:
        return os.environ.get("GCP_PROJECT_ID", "my-project")

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=_BASE_URL,
                timeout=180,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def submit(self, params: GenerationParams) -> str:
        body = {
            "instances": [
                {
                    "prompt": params.prompt,
                    "negative_prompt": params.negative_prompt,
                    "properties": {
                        "duration_seconds": params.duration_seconds,
                        "resolution": params.resolution,
                        "fps": params.fps,
                        # TODO: verify additional properties against live API
                    },
                }
            ],
            "parameters": {
                "sampleCount": 1,
            },
        }
        client = await self._client()
        path = _SUBMIT_PATH.format(project_id=self._project_id)
        resp = await client.post(path, json=body)
        resp.raise_for_status()
        data = resp.json()

        # Veo returns a synchronous prediction result; treat as instant completion
        # TODO: verify if Veo supports async job submission
        predictions = data.get("predictions", [])
        if not predictions:
            raise ValueError(f"Veo submit response missing predictions: {data}")

        job_id = predictions[0].get("id", data.get("id", ""))
        log.info("Veo submit OK, job_id=%s", job_id)
        return job_id

    async def poll(self, job_id: str) -> JobStatus:
        raise NotImplementedError(
            "Veo currently returns synchronous predictions; "
            "poll is not yet supported for async jobs. "
            "TODO: implement once Veo async API is available."
        )

    async def download(self, job_id: str, out_path: str) -> str:
        raise NotImplementedError(
            "Veo download not implemented until async API is confirmed. "
            "TODO: implement once Veo async API is available."
        )
