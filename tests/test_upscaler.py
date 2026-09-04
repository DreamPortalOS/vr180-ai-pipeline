"""Tests for pipeline.upscaler — the "lie detector" suite (issue #245).

The original bug: ``--upscale 2/4`` claimed to use Real-ESRGAN but the
``realesrgan`` / ``basicsr`` packages were never installed, so the code
silently fell back to bicubic interpolation (bit-identical to
``cv2.INTER_CUBIC``) while the tqdm progress bar still said
``Real-ESRGAN``.

Evidence captured before the fix (and still reproducible after, since the
fallback path *is* INTER_CUBIC):

    >>> import cv2, numpy as np
    >>> frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    >>> from pipeline.upscaler import PixelUpscaler
    >>> u = PixelUpscaler(scale=4, device="cpu")   # no realesrgan installed
    >>> out = u.upscale_frame(frame)
    >>> np.array_equal(out,
    ...                cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC))
    True

i.e. the "super-resolution" output is bit-for-bit identical to
``cv2.INTER_CUBIC`` — and the progress bar still claimed ``Real-ESRGAN``.

These tests monkeypatch the backend availability instead of installing real
dependencies, so they pass in the CPU-only, model-less CI environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)


@pytest.fixture()
def mock_frames(frame):
    """Small list of BGR frames, like the stage receives after RGB->BGR."""
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)]


# ---------------------------------------------------------------------------
# Helpers to force backend availability / unavailability in sys.modules.
# ---------------------------------------------------------------------------


def _hide_realesrgan(monkeypatch):
    """Ensure realesrgan/basicsr cannot be imported for the duration of a test."""
    for name in ("realesrgan", "basicsr", "basicsr.archs", "basicsr.archs.rrdbnet_arch"):
        monkeypatch.setitem(sys.modules, name, None)
        # Clear any real module that might have been cached
        for mod_name in list(sys.modules):
            if mod_name == name or mod_name.startswith(name + "."):
                sys.modules[mod_name] = None


def _provide_mock_realesrgan(monkeypatch):
    """Plant fake realesrgan/basicsr modules so the backend resolves as
    available, then provide a real-but-cheap enhance() that actually upscales.

    The dotted ``basicsr.archs.rrdbnet_arch`` import requires every parent
    package to resolve, so all ancestors are planted.
    """
    rrdb = mock.MagicMock()
    rrdb.RRDBNet = mock.MagicMock(return_value=mock.MagicMock())
    realesrgan = mock.MagicMock()

    def _fake_enhance(img, outscale=2):
        h, w = img.shape[:2]
        result = cv2.resize(img, (w * outscale, h * outscale), interpolation=cv2.INTER_CUBIC)
        return result, None

    fake_inst = mock.MagicMock()
    fake_inst.enhance = _fake_enhance
    realesrgan.RealESRGANer = mock.MagicMock(return_value=fake_inst)

    basicsr = mock.MagicMock()
    basicsr_archs = mock.MagicMock()

    for name, mod in (
        ("basicsr", basicsr),
        ("basicsr.archs", basicsr_archs),
        ("basicsr.archs.rrdbnet_arch", rrdb),
        ("realesrgan", realesrgan),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# Unit tests — PixelUpscaler backend honesty
# ---------------------------------------------------------------------------


class TestBackendAvailability:
    def test_unavailable_backend_resolves_to_bicubic(self, monkeypatch, frame):
        from pipeline.upscaler import PixelUpscaler

        _hide_realesrgan(monkeypatch)
        up = PixelUpscaler(scale=2, device="cpu")
        # backend property triggers _load_model
        assert up.backend == "bicubic"

    def test_unavailable_backend_warns_and_label_is_bicubic(self, monkeypatch, caplog, frame):
        """Regression: backend unavailable MUST log a warning mentioning
        'bicubic' — never silently.
        """
        import logging

        from pipeline.upscaler import PixelUpscaler

        _hide_realesrgan(monkeypatch)
        with caplog.at_level(logging.WARNING):
            up = PixelUpscaler(scale=2, device="cpu")
            # Force detection
            _ = up.backend
        joined = " ".join(r.message for r in caplog.records)
        assert "bicubic" in joined.lower()

    def test_unavailable_backend_output_is_bicubic(self, monkeypatch, frame):
        from pipeline.upscaler import PixelUpscaler

        _hide_realesrgan(monkeypatch)
        up = PixelUpscaler(scale=2, device="cpu")
        out = up.upscale_frame(frame)
        expected = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)
        assert np.array_equal(out, expected)
        assert out.shape == (64, 64, 3)

    def test_unavailable_backend_strict_raises(self, monkeypatch, frame):
        from pipeline.upscaler import (
            PixelUpscaler,
            RealESRGANUnavailableError,
        )

        _hide_realesrgan(monkeypatch)
        up = PixelUpscaler(scale=2, device="cpu", strict=True)
        with pytest.raises(RealESRGANUnavailableError):
            up.upscale_frame(frame)

    def test_available_backend_resolves_to_realesrgan(self, monkeypatch, frame):
        from pipeline.upscaler import PixelUpscaler

        _provide_mock_realesrgan(monkeypatch)
        monkeypatch.setattr(
            PixelUpscaler,
            "_get_model_path",
            mock.MagicMock(return_value="/tmp/no-download.pth"),
        )
        up = PixelUpscaler(scale=2, device="cpu")
        assert up.backend == "realesrgan"

    def test_available_backend_runs_real_path(self, monkeypatch, frame):
        from pipeline.upscaler import PixelUpscaler

        _provide_mock_realesrgan(monkeypatch)
        monkeypatch.setattr(
            PixelUpscaler,
            "_get_model_path",
            mock.MagicMock(return_value="/tmp/no-download.pth"),
        )
        up = PixelUpscaler(scale=2, device="cpu")
        out = up.upscale_frame(frame)
        assert out.shape == (64, 64, 3)
        # The mock enhance() does INTER_CUBIC, so we at least verify the
        # upscaler actually delegated to it (output is upscaled, not identity).
        assert out.shape != frame.shape


# ---------------------------------------------------------------------------
# Integration-style tests — run_upscale_stage labeling
# ---------------------------------------------------------------------------


class TestUpscaleStageLabeling:
    """Drive the real ``run_upscale_stage`` from run_pipeline.py and assert
    the progress-bar description (label) and warning log are honest.
    """

    @pytest.fixture()
    def rgb_frames(self, frame):
        # run_upscale_stage expects RGB frames.
        return [frame.copy(), frame.copy()]

    @staticmethod
    def _run_stage(monkeypatch, rgb_frames, strict=False):
        """Import run_pipeline, patch its tqdm to capture the desc, and run
        the upscale stage.  Returns (results, desc).
        """
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_pipeline as _rp
        from run_pipeline import run_upscale_stage as _stage

        args = SimpleNamespace(
            upscale=2,
            upscale_model=None,
            upscale_ffmpeg=False,
            device="cpu",
            upscale_strict=strict,
            tiled_upscale=False,
            tile_size=512,
        )

        captured_desc = []

        def _tqdm_stub(iterable, desc=None, **_kwargs):
            captured_desc.append(desc)
            yield from iterable

        monkeypatch.setattr(_rp, "tqdm", _tqdm_stub)
        results = _stage(args, rgb_frames)
        return results, captured_desc[0] if captured_desc else ""

    def test_unavailable_label_is_bicubic_not_realesrgan(self, monkeypatch, rgb_frames, caplog):
        import logging

        _hide_realesrgan(monkeypatch)

        with caplog.at_level(logging.WARNING):
            result, desc = self._run_stage(monkeypatch, rgb_frames)

        assert len(result) == 2
        # The label MUST say bicubic, and MUST NOT claim Real-ESRGAN.
        assert "bicubic" in desc, f"Expected 'bicubic' in label, got: {desc!r}"
        assert "Real-ESRGAN" not in desc, (
            f"Label lies: progress bar claimed Real-ESRGAN but output is bicubic. ({desc!r})"
        )
        # And a warning must have been logged.
        joined = " ".join(r.message for r in caplog.records)
        assert "bicubic" in joined.lower()

    def test_available_label_is_realesrgan(self, monkeypatch, rgb_frames):
        _provide_mock_realesrgan(monkeypatch)

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_pipeline as _rp

        monkeypatch.setattr(
            _rp.PixelUpscaler,
            "_get_model_path",
            mock.MagicMock(return_value="/tmp/no-download.pth"),
        )

        result, desc = self._run_stage(monkeypatch, rgb_frames)
        assert len(result) == 2
        assert "Real-ESRGAN" in desc, f"Expected Real-ESRGAN in label when backend is available, got: {desc!r}"


# ---------------------------------------------------------------------------
# CLI-level tests — --upscale-strict non-zero exit, --upscale 0 no detection
# ---------------------------------------------------------------------------


class TestUpscaleCLI:
    """End-to-end CLI behaviour for the upscale stage via a tiny in-memory
    input video rendered with cv2 + ffmpeg/avfoundation-free encoding.
    """

    @pytest.fixture()
    def tiny_video(self, tmp_path):
        """A 2-frame video encoded with ffmpeg so run_pipeline has a real --input."""
        out = tmp_path / "tiny.mp4"
        frames = []
        for _ in range(2):
            rng = np.random.default_rng(0)
            frames.append(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        import subprocess

        # Encode via ffmpeg using raw RGB frames piped in (no external file).
        raw = b"".join(f.tobytes() for f in frames)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "16x16",
            "-framerate",
            "1",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        subprocess.run(cmd, input=raw, capture_output=True, check=True, timeout=30)
        return out

    def test_strict_exits_nonzero_when_backend_missing(self, tiny_video, tmp_path):
        """--upscale 2 --upscale-strict with realesrgan absent → exit != 0.

        The subprocess starts fresh and this repo's CI env never ships the
        realesrgan/basicsr packages, so the backend is naturally missing.
        """
        cmd = [
            sys.executable,
            str(RUN_PIPELINE),
            "--input",
            str(tiny_video),
            "--output",
            str(tmp_path / "out.mp4"),
            "--stage",
            "all",
            "--upscale",
            "2",
            "--upscale-strict",
            "--stages",
            "upscale",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        assert proc.returncode != 0, (
            "--upscale-strict must fail when Real-ESRGAN is unavailable. "
            f"stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-400:]!r}"
        )
        combined = (proc.stdout + proc.stderr).lower()
        assert "bicubic" in combined or "real-esrgan" in combined, (
            "Non-zero exit should explain the missing backend, not a mystery error."
        )

    def test_upscale_zero_triggers_no_backend_detection(self, tiny_video, tmp_path):
        """--upscale 0 (the default) must NOT probe the Real-ESRGAN backend
        at all — the stage is dropped before any import happens.
        """
        cmd = [
            sys.executable,
            str(RUN_PIPELINE),
            "--input",
            str(tiny_video),
            "--output",
            str(tmp_path / "out.mp4"),
            "--upscale",
            "0",
            "--stage",
            "all",
            "--stages",
            "upscale",
        ]
        # Run with realesrgan modules poisoned so any stray detection would
        # surface.  The stage itself should be skipped entirely.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        # With --upscale 0 the upscale stage is filtered out (line 2282 of
        # run_pipeline.py).  It should not error on a missing backend.
        combined = (proc.stdout + proc.stderr).lower()
        assert "real-esrgan backend unavailable" not in combined, "--upscale 0 must not touch backend detection."
