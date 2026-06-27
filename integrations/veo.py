"""Veo (Google DeepMind / Vertex AI) video generation provider.

API reference: https://cloud.google.com/vertex-ai/generative-ai/docs/veo/overview
Credentials: ``VEO_API_KEY`` env var (API key for Vertex AI prediction).
"""

from __future__ import annotations

import logging
import os

import httpx

from integrations.base import GenerationResult, VideoGenProvider

log = logging.getLogger(__name__)

_BASE_URL = "https://us-central1-aiplatform.googleapis.com/v1"
_SUBMIT_PATH = "/projects/{project_id}/locations/us-central1/publishers/google/models/veo-001:predict"
_POLL_INTERVAL = 3.0
_MAX_POLL_SECONDS = 300


class VeoProvider(VideoGenProvider):
    """Veo (Google DeepMind) video generation provider via Vertex AI.

    Requires ``VEO_API_KEY`` environment variable. Optionally set
    ``GCP_PROJECT_ID`` for the target project (defaults to ``"my-project"``).

    .. note::

        Veo currently returns synchronous predictions. The ``generate()``
        method wraps the predict response into the standard
        :class:`GenerationResult` interface.
    """

    def _load_api_key(self) -> str:
        api_key = os.environ.get("VEO_API_KEY", "")
        if not api_key:
            raise ValueError(
                "VEO_API_KEY environment variable is not set. "
                "Generate one at https://console.cloud.google.com/apis/credentials"
            )
        return api_key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _project_id(self) -> str:
        return os.environ.get("GCP_PROJECT_ID", "my-project")

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
        """Generate a video via Veo (Vertex AI).

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
            Extra parameters (e.g. ``negative_prompt``, ``sample_count``).

        Returns
        -------
        GenerationResult
            Result with the video URL.

        Raises
        ------
        RuntimeError
            If the predict call fails.
        """
        properties: dict[str, int | str | float] = {
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
        }

        instance: dict[str, str | int | float | dict] = {
            "prompt": prompt,
            "properties": properties,
        }
        if "negative_prompt" in kwargs:
            instance["negative_prompt"] = kwargs["negative_prompt"]

        parameters = {
            "sampleCount": kwargs.get("sample_count", 1),
        }

        body = {
            "instances": [instance],
            "parameters": parameters,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(base_url=_BASE_URL, timeout=180) as client:
            log.info("Veo: calling predict (prompt=%.50s...)", prompt)
            path = _SUBMIT_PATH.format(project_id=self._project_id)
            resp = client.post(path, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            predictions = data.get("predictions", [])
            if not predictions:
                raise RuntimeError(f"Veo predict response missing predictions: {data}")

            prediction = predictions[0]
            # Veo returns videos as base64-encoded or Cloud Storage URIs
            # Try to extract a usable URL
            video_url: str | None = prediction.get("video_url") or prediction.get("videoUri") or prediction.get("uri")

            if not video_url:
                raise RuntimeError(f"Veo prediction missing video URL/URI: {prediction}")

            job_id = prediction.get("id", data.get("id", ""))

            log.info("Veo: predict OK, video_url=%s", video_url)
            return GenerationResult(
                video_url=video_url,
                provider=self.provider_name,
                job_id=job_id,
                metadata={
                    "predictions": predictions,
                    "project_id": self._project_id,
                    **data,
                },
            )
