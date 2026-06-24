"""Abstract base class for video generation providers.

Defines the ``VideoGenProvider`` interface that all external providers (Kling,
Seedance, Veo) must implement. Follows a three-phase lifecycle:

1. **submit(prompt, params) -> job_id** — Submit a generation job
2. **poll(job_id) -> JobStatus** — Poll for completion / progress
3. **download(job_id, out_path) -> path** — Download the finished video
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobState(str, Enum):
    """Possible states of a video generation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobStatus:
    """Status of a video generation job returned by ``poll()``."""

    job_id: str
    state: JobState
    progress: int = 0  # 0–100
    message: str = ""
    output_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationParams:
    """Parameters passed to ``submit()`` alongside the prompt."""

    prompt: str
    negative_prompt: str = ""
    duration_seconds: int = 5
    resolution: str = "1080p"
    fps: int = 24
    extra: dict[str, Any] = field(default_factory=dict)


class VideoGenProvider:
    """Abstract base for a video generation provider.

    Subclasses must implement ``submit``, ``poll``, and ``download``.
    Credentials are read from environment variables (e.g. ``KLING_API_KEY``)
    at instantiation time.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or self._load_api_key()
        self._http_client: Any = None  # lazily created httpx.AsyncClient

    # ── Subclass API ──────────────────────────────────────────────────────────

    def _load_api_key(self) -> str:
        """Load the API key from the environment.

        Subclasses MUST override this and provide the correct env var name.
        """
        raise NotImplementedError("Subclasses must implement _load_api_key")

    async def submit(self, params: GenerationParams) -> str:
        """Submit a generation job.

        Returns the job ID (string) that can be used with ``poll`` / ``download``.
        """
        raise NotImplementedError("Subclasses must implement submit")

    async def poll(self, job_id: str) -> JobStatus:
        """Poll a previously submitted job for its current status."""
        raise NotImplementedError("Subclasses must implement poll")

    async def download(self, job_id: str, out_path: str) -> str:
        """Download the completed video to *out_path*.

        Returns the path the video was saved to.
        """
        raise NotImplementedError("Subclasses must implement download")

    # ── Provider name ─────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"kling"``."""
        return type(self).__name__.lower().replace("provider", "")
