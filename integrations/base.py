"""Abstract base class for video generation providers.

Defines the :class:`VideoGenProvider` interface that all external providers
(Kling, Seedance, Veo) must implement.

Unlike the archived async submit/poll/download lifecycle, this modernised
interface exposes a single synchronous ``generate()`` call that blocks until
the video is ready, keeping the integration layer simple and composable with
the existing pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Result of a successful video generation."""

    video_url: str
    provider: str
    job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoGenProvider(ABC):
    """Abstract base for a video generation provider.

    Subclasses must implement :meth:`generate`. Credentials are read from
    environment variables at instantiation time (e.g. ``KLING_API_KEY``).
    If a required key is missing the constructor **must** raise ``ValueError``
    with a clear message — no silent fallback to an empty string.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or self._load_api_key()

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_api_key(self) -> str:
        """Read the API key from the environment.

        Returns the key value.

        Raises
        ------
        ValueError
            If the required environment variable is not set.
        """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a video from a text prompt.

        Parameters
        ----------
        prompt : str
            The text prompt describing the desired video.
        duration : int
            Target duration in seconds (provider-dependent).
        aspect_ratio : str
            Aspect ratio string, e.g. ``"16:9"``, ``"9:16"``, ``"1:1"``.
        fps : int
            Target frame rate.
        **kwargs
            Provider-specific extra parameters (e.g. ``cfg_scale``, ``model``).

        Returns
        -------
        GenerationResult
            Object containing the video URL and metadata.

        Raises
        ------
        RuntimeError
            If the generation fails or the API returns an error.
        """

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"kling"``."""
        return type(self).__name__.lower().replace("provider", "")

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
        """Parse ``"16:9"`` → ``(16, 9)``."""
        parts = aspect_ratio.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio!r} (expected 'W:H')")
        return int(parts[0]), int(parts[1])
