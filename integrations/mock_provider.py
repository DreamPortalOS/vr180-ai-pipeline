"""Mock video generation provider.

Synthesises a small, real, playable video locally via ``ffmpeg`` lavfi
(``testsrc2``).  No API key, no network, no model — this is the key to
running the full image→video→VR180 chain on CI (ubuntu, CPU-only, no keys)
and in local development.

Both :meth:`generate` and :meth:`generate_from_image` produce the same kind
of short test-card video; the only difference is that the image variant
notes the input image path in metadata.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import uuid
from typing import Any

from integrations.base import GenerationResult, VideoGenProvider

log = logging.getLogger(__name__)

_FFMPEG = os.environ.get("FFMPEG_BINARY", "ffmpeg")
# Default output geometry / rate — small enough to be fast on CI.
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 24


class MockProvider(VideoGenProvider):
    """Local mock provider that renders a test-card video with ffmpeg.

    Requires no credentials.  ``ffmpeg`` must be on ``PATH`` (CI has it).
    The produced file is a genuine, ffprobe-readable ``.mp4``.
    """

    def _load_api_key(self) -> str:
        # No key needed for the mock provider.
        return "mock"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_geometry(aspect_ratio: str, fps: int) -> tuple[int, int, int]:
        """Map an aspect ratio string to (width, height, fps).

        Uses the mock default geometry, picking the smallest of the standard
        widths that matches the requested aspect ratio.  Falls back to
        1280×720 on any unparseable ratio.
        """
        try:
            w, h = VideoGenProvider._parse_aspect_ratio(aspect_ratio)
        except ValueError:
            w, h = 16, 9
        # Scale so the shorter side keeps the mock default height; cap to keep
        # CI fast.  Use a fixed height of 720 and derive width from ratio.
        height = _DEFAULT_HEIGHT
        width = round(height * w / h)
        # Keep width even (ffmpeg yuv420p requirement).
        if width % 2:
            width += 1
        return width, height, int(fps)

    @staticmethod
    def _render(out_path: str, duration: int, fps: int, width: int, height: int) -> str:
        """Render a test-card video to *out_path* via ffmpeg lavfi.

        Returns the output path.  Raises :class:`RuntimeError` if ffmpeg is
        missing or exits non-zero.
        """
        src = f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"
        cmd = [
            _FFMPEG,
            "-y",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            src,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        log.info("MockProvider: rendering %ds @ %dx%d/%dfps -> %s", duration, width, height, fps, out_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg not found on PATH (set FFMPEG_BINARY or install ffmpeg): {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg lavfi render failed (exit {result.returncode}):\nstderr:\n{result.stderr[-2000:]}"
            )
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ffmpeg produced no output at {out_path}")
        log.info("MockProvider: rendered %d bytes", os.path.getsize(out_path))
        return out_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: Any,
    ) -> GenerationResult:
        """Render a mock test-card video from a (ignored) text prompt."""
        del prompt  # prompt is ignored for the mock; present for API parity
        width, height, rate = self._resolve_geometry(aspect_ratio, fps)
        out_path = self._out_path("mock")
        self._render(out_path, duration, rate, width, height)
        return GenerationResult(
            video_url=out_path,
            provider=self.provider_name,
            job_id=f"mock-{uuid.uuid4().hex[:8]}",
            metadata={
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "fps": rate,
                "width": width,
                "height": height,
                "source": "lavfi:testsrc2",
            },
        )

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> GenerationResult:
        """Render a mock test-card video as if seeded by *image_path*.

        The image itself is not composited (no real generation happens) —
        this provider only guarantees a playable file.  ``image_path`` is
        recorded in metadata for downstream provenance.
        """
        del prompt
        fps = int(kwargs.get("fps", 24))
        width, height, rate = self._resolve_geometry(aspect_ratio, fps)
        out_path = self._out_path("mock_i2v")
        self._render(out_path, duration, rate, width, height)
        return GenerationResult(
            video_url=out_path,
            provider=self.provider_name,
            job_id=f"mock-i2v-{uuid.uuid4().hex[:8]}",
            metadata={
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "fps": rate,
                "width": width,
                "height": height,
                "source": "lavfi:testsrc2",
                "image_path": image_path,
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _out_path(prefix: str) -> str:
        """Build a unique output path inside the provider's output directory.

        The output directory can be set explicitly by the caller via the
        ``MOCK_PROVIDER_OUTPUT_DIR`` env var (``image_to_vr180`` / the
        ``--image`` CLI and all tests pass it, so artifacts land in
        ``tmp_path`` / the job workdir).

        **Important:** if no directory is configured we fall back to the OS
        temp directory (`tempfile.mkdtemp`), **never** the repo ``video/``
        folder.  ``video/`` is the lead's local-media / deliverable directory
        (git-ignored) — automation must never write into it.  A bare
        ``generate(...)`` call with no env var therefore produces a
        non-persistent tempfile that does not pollute the workspace.
        """
        video_dir = os.environ.get("MOCK_PROVIDER_OUTPUT_DIR")
        if not video_dir:
            video_dir = tempfile.mkdtemp(prefix="mock_provider_")
        os.makedirs(video_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return os.path.join(video_dir, f"{prefix}_{timestamp}_{unique}.mp4")
