"""Tests for the VideoGen provider abstraction layer.

Covers base types, factory, each provider's initialisation and generate()
method.  No live API calls are made — all HTTP responses are mocked.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from integrations.base import GenerationResult, VideoGenProvider
from integrations.factory import get_provider, list_providers
from integrations.kling import KlingProvider
from integrations.seedance import SeedanceProvider
from integrations.veo import VeoProvider

# ══════════════════════════════════════════════════════════════════════════════
# Base types
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerationResult:
    def test_defaults(self) -> None:
        r = GenerationResult(video_url="https://example.com/v.mp4", provider="kling")
        assert r.video_url == "https://example.com/v.mp4"
        assert r.provider == "kling"
        assert r.job_id is None
        assert r.metadata == {}

    def test_full_construction(self) -> None:
        r = GenerationResult(
            video_url="https://example.com/v.mp4",
            provider="kling",
            job_id="job-001",
            metadata={"duration": 5},
        )
        assert r.job_id == "job-001"
        assert r.metadata["duration"] == 5


class TestVideoGenProviderABC:
    def test_abc_raises_on_init_without_env(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VideoGenProvider()

    def test_subclass_must_implement_generate(self) -> None:
        """Subclass that doesn't implement generate can't be instantiated."""

        class Minimal(VideoGenProvider):
            def _load_api_key(self) -> str:
                return "test-key"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Minimal()

    def test_provider_name_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "key")
        monkeypatch.setenv("ARK_API_KEY", "key")
        assert KlingProvider().provider_name == "kling"
        assert SeedanceProvider().provider_name == "seedance"

    def test_parse_aspect_ratio(self) -> None:
        w, h = VideoGenProvider._parse_aspect_ratio("16:9")
        assert (w, h) == (16, 9)

    def test_parse_aspect_ratio_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid aspect ratio"):
            VideoGenProvider._parse_aspect_ratio("invalid")


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════


class TestFactory:
    def test_list_providers(self) -> None:
        providers = list_providers()
        assert "kling" in providers
        assert "local-svd" in providers
        assert "seedance" in providers
        assert "veo" in providers
        assert "mock" in providers
        assert len(providers) == 5

    def test_get_provider_kling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "key")
        instance = get_provider("kling")
        assert isinstance(instance, KlingProvider)

    def test_get_provider_seedance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "key")
        instance = get_provider("seedance")
        assert isinstance(instance, SeedanceProvider)

    def test_get_provider_veo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "key")
        instance = get_provider("veo")
        assert isinstance(instance, VeoProvider)

    def test_get_provider_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "key")
        instance = get_provider("KLING")
        assert isinstance(instance, KlingProvider)

    def test_get_provider_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_get_provider_with_api_key(self) -> None:
        instance = get_provider("kling", api_key="custom-key")
        assert instance._api_key == "custom-key"


# ══════════════════════════════════════════════════════════════════════════════
# KlingProvider
# ══════════════════════════════════════════════════════════════════════════════


class TestKlingProvider:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "env-key")
        provider = KlingProvider()
        assert provider._api_key == "env-key"

    def test_load_api_key_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="KLING_API_KEY"):
            KlingProvider()

    def test_provider_name(self) -> None:
        provider = KlingProvider(api_key="key")
        assert provider.provider_name == "kling"

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-job-001"}}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "data": {
                "status": "succeed",
                "video_url": "https://cdn.kling.com/video.mp4",
                "duration": 5,
            }
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
        ):
            result = provider.generate("fly over mountains", duration=5)

        assert isinstance(result, GenerationResult)
        assert result.video_url == "https://cdn.kling.com/video.mp4"
        assert result.job_id == "kling-job-001"
        assert result.provider == "kling"
        mock_client.post.assert_called_once()
        mock_client.get.assert_called_once()

    def test_generate_submit_missing_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {}}
        submit_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing job_id"),
        ):
            provider.generate("test")

    def test_generate_poll_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-job-002"}}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"data": {"status": "failed", "message": "content rejected"}}
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="content rejected"),
        ):
            provider.generate("violence")


# ══════════════════════════════════════════════════════════════════════════════
# SeedanceProvider
# ══════════════════════════════════════════════════════════════════════════════


class TestSeedanceProvider:
    """Seedance now runs on Volcengine Ark (ARK_API_KEY)."""

    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "env-key")
        provider = SeedanceProvider()
        assert provider._api_key == "env-key"

    def test_load_api_key_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="ARK_API_KEY"):
            SeedanceProvider()
        # Old key name must NOT be accepted — it referred to a fake endpoint.
        with pytest.raises(ValueError, match="ARK_API_KEY"), pytest.MonkeyPatch().context() as mp:
            mp.setenv("SEEDANCE_API_KEY", "legacy-key")
            SeedanceProvider()

    def test_model_constants(self) -> None:
        from integrations.seedance import MODEL_FAST, MODEL_STD

        assert MODEL_FAST == "doubao-seedance-2-0-fast-260128"
        assert MODEL_STD == "doubao-seedance-2-0-260128"

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828004522-rxht8"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828004522-rxht8",
            "status": "succeeded",
            "content": {"video_url": "https://ark-cdn.volces.com/video.mp4"},
            "usage": {"completion_tokens": 512},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            result = provider.generate("flying")

        assert result.video_url == "https://ark-cdn.volces.com/video.mp4"
        assert result.job_id == "cgt-20260828004522-rxht8"
        assert result.provider == "seedance"

        # Verify the request body used the Ark content-list contract.
        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == "doubao-seedance-2-0-fast-260128"
        assert body["content"] == [{"type": "text", "text": "flying"}]
        assert body["resolution"] == "480p"
        assert body["ratio"] == "adaptive"
        assert body["duration"] == 5
        assert mock_client.get.call_args[0][0] == "/contents/generations/tasks/cgt-20260828004522-rxht8"

    def test_generate_custom_kwargs_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828010000-abcde"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828010000-abcde",
            "status": "running",
        }
        poll_resp.raise_for_status.return_value = None

        poll_resp2 = MagicMock(spec=httpx.Response)
        poll_resp2.json.return_value = {
            "id": "cgt-20260828010000-abcde",
            "status": "succeeded",
            "content": {"video_url": "https://x/v.mp4"},
        }
        poll_resp2.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [poll_resp, poll_resp2]

        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            provider.generate("test", model=SeedanceProvider.MODEL_STD, resolution="720p", ratio="16:9", duration=8)

        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == SeedanceProvider.MODEL_STD
        assert body["resolution"] == "720p"
        assert body["ratio"] == "16:9"
        assert body["duration"] == 8

    def test_generate_submit_missing_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {}
        submit_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing task_id"),
        ):
            provider.generate("test")

    def test_generate_submit_http_error_with_error_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        err_body = {"error": {"code": "InvalidEndpointOrModel.NotFound", "message": "model not found"}}
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.json.return_value = err_body
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=err_resp
        )

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = err_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match=r"InvalidEndpointOrModel\.NotFound"),
        ):
            provider.generate("test")

    def test_generate_poll_failed_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828020000-fail01"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828020000-fail01",
            "status": "running",
        }
        poll_resp.raise_for_status.return_value = None

        poll_resp2 = MagicMock(spec=httpx.Response)
        poll_resp2.json.return_value = {
            "id": "cgt-20260828020000-fail01",
            "status": "failed",
            "error": {"code": "ModelNotOpen", "message": "model not enabled", "model": "doubao-seedance-2-0-260128"},
        }
        poll_resp2.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [poll_resp, poll_resp2]

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="ModelNotOpen"),
        ):
            provider.generate("test")

    def test_generate_poll_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828030000-timeout1"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"id": "cgt-20260828030000-timeout1", "status": "running"}
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
            patch("integrations.seedance.time.time", side_effect=[1.0, 1e9]),
            pytest.raises(RuntimeError, match="did not complete"),
        ):
            provider.generate("test")


# ══════════════════════════════════════════════════════════════════════════════
# VeoProvider
# ══════════════════════════════════════════════════════════════════════════════


class TestVeoProvider:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "env-key")
        provider = VeoProvider()
        assert provider._api_key == "env-key"

    def test_load_api_key_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="VEO_API_KEY"):
            VeoProvider()

    def test_project_id_default(self) -> None:
        provider = VeoProvider(api_key="key")
        assert provider._project_id == "my-project"

    def test_project_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GCP_PROJECT_ID", "my-real-project")
        provider = VeoProvider(api_key="key")
        assert provider._project_id == "my-real-project"

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {
            "predictions": [
                {
                    "id": "veo-job-001",
                    "video_url": "https://storage.googleapis.com/veo/video.mp4",
                }
            ],
            "id": "pred-001",
        }
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with patch("integrations.veo.httpx.Client", return_value=mock_client):
            result = provider.generate("flying")

        assert result.video_url == "https://storage.googleapis.com/veo/video.mp4"
        assert result.job_id == "veo-job-001"
        assert result.provider == "veo"

    def test_generate_missing_predictions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with (
            patch("integrations.veo.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing predictions"),
        ):
            provider.generate("test")

    def test_generate_missing_video_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        # Prediction with no video_url, no videoUri, no uri — only an id
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {
            "predictions": [{"id": "veo-job-002"}],
        }
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with (
            patch("integrations.veo.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing video URL"),
        ):
            provider.generate("test")


# ══════════════════════════════════════════════════════════════════════════════
# Integration with prompt_builder (end-to-end flow)
# ══════════════════════════════════════════════════════════════════════════════


class TestPromptToGenerationFlow:
    """Verify that wrap_prompt produces strings compatible with VideoGenProvider."""

    def test_positive_prompt_used_in_generate(self) -> None:
        from pipeline.prompt_builder import wrap_prompt_for_vr180

        wrapped = wrap_prompt_for_vr180("fly over mountains", scene_type="fpv")
        prompt = wrapped["positive"]
        negative = wrapped["negative"]

        assert "fly over mountains" in prompt
        assert isinstance(negative, str)
        assert len(negative) > 0
        assert "rapid turns" in negative

        # The prompt is a plain string — compatible with generate


# ══════════════════════════════════════════════════════════════════════════════
# Image-to-video (G-1): base default + per-provider i2v + MockProvider
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseGenerateFromImage:
    def test_default_raises_not_implemented(self) -> None:
        """A provider that does not override i2v must raise NotImplementedError."""

        class StubProvider(VideoGenProvider):
            def _load_api_key(self) -> str:
                return "test-key"

            def generate(
                self,
                prompt: str,
                duration: int = 5,
                aspect_ratio: str = "16:9",
                fps: int = 24,
                **kwargs,
            ) -> GenerationResult:
                return GenerationResult(video_url="https://x.com/v.mp4", provider=self.provider_name)

        p = StubProvider()
        with pytest.raises(NotImplementedError, match="does not support image-to-video"):
            p.generate_from_image("photo.png")

    def test_not_implemented_mentions_provider_name(self) -> None:
        class StubProvider(VideoGenProvider):
            def _load_api_key(self) -> str:
                return "test-key"

            def generate(
                self,
                prompt: str,
                duration: int = 5,
                aspect_ratio: str = "16:9",
                fps: int = 24,
                **kwargs,
            ) -> GenerationResult:
                return GenerationResult(video_url="https://x.com/v.mp4", provider=self.provider_name)

        p = StubProvider()
        with pytest.raises(NotImplementedError, match=type(p).__name__.lower().replace("provider", "")):
            p.generate_from_image("photo.png")


def _mock_httpx_client():
    """Return a MagicMock httpx.Client usable as both value and context manager."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.__enter__.return_value = mock_client
    return mock_client


class TestKlingImageToVideo:
    def test_i2v_sends_image_payload_and_polls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-i2v-001"}}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"data": {"status": "succeed", "video_url": "https://cdn.kling.com/i2v.mp4"}}
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            patch("integrations.kling.time.sleep", return_value=None),
        ):
            result = provider.generate_from_image("https://example.com/photo.png", duration=5, aspect_ratio="16:9")

        assert result.video_url == "https://cdn.kling.com/i2v.mp4"
        assert result.job_id == "kling-i2v-001"
        assert result.provider == "kling"

        # The image URL was passed through verbatim into the request body
        body = mock_client.post.call_args[1]["json"]
        assert body["image"] == "https://example.com/photo.png"
        mock_client.get.assert_called_once()

    def test_i2v_encodes_local_image_as_base64(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNGfake-image-bytes")

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-i2v-002"}}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"data": {"status": "succeed", "video_url": "https://cdn.kling.com/i2v.mp4"}}
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            patch("integrations.kling.time.sleep", return_value=None),
        ):
            provider.generate_from_image(str(img))

        body = mock_client.post.call_args[1]["json"]
        import base64

        assert base64.b64decode(body["image"]) == b"\x89PNGfake-image-bytes"
        assert body["image_type"] == KlingProvider.I2V_IMAGE_TYPE

    def test_i2v_poll_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-i2v-003"}}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"data": {"status": "failed", "message": "unsafe content"}}
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            patch("integrations.kling.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="unsafe content"),
        ):
            provider.generate_from_image("https://x.com/p.png")

    def test_i2v_poll_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"data": {"job_id": "kling-i2v-004"}}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"data": {"status": "processing"}}
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        # Freeze the clock so the deadline is reached immediately.
        with (
            patch.object(provider, "_sign", return_value=("sig", "1", "nonce")),
            patch("integrations.kling.httpx.Client", return_value=mock_client),
            patch("integrations.kling.time.sleep", return_value=None),
            patch(
                "integrations.kling.time.time",
                side_effect=[1.0, 1e9],
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            provider.generate_from_image("https://x.com/p.png")

    def test_i2v_missing_image_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        provider = KlingProvider()
        with pytest.raises(RuntimeError, match="Image file not found"):
            provider.generate_from_image("/nonexistent/photo.png")


class TestSeedanceImageToVideo:
    def _submit_poll(self, task_id="cgt-20260828040000-i2v01", status="succeeded"):
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": task_id}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": task_id,
            "status": status,
            "content": {"video_url": "https://ark-cdn.volces.com/i2v.mp4"},
        }
        poll_resp.raise_for_status.return_value = None
        return submit_resp, poll_resp

    def test_i2v_http_url_passed_through_in_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An http(s) image URL is treated as pass-through and placed in the
        content list's image_url item."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828050000-http01"}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828050000-http01",
            "status": "running",
        }
        poll_resp.raise_for_status.return_value = None
        poll_resp2 = MagicMock(spec=httpx.Response)
        poll_resp2.json.return_value = {
            "id": "cgt-20260828050000-http01",
            "status": "succeeded",
            "content": {"video_url": "https://ark-cdn.volces.com/i2v.mp4"},
        }
        poll_resp2.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [poll_resp, poll_resp2]

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            result = provider.generate_from_image("https://example.com/p.png", duration=5, prompt="pan left")

        assert result.video_url == "https://ark-cdn.volces.com/i2v.mp4"
        assert result.job_id == "cgt-20260828050000-http01"
        assert result.provider == "seedance"

        body = mock_client.post.call_args[1]["json"]
        content = body["content"]
        assert content[0] == {"type": "text", "text": "pan left"}
        assert content[1] == {"type": "image_url", "image_url": {"url": "https://example.com/p.png"}}
        assert mock_client.get.call_count == 2

    def test_i2v_local_image_encoded_as_base64_data_url(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """A local image is validated and base64-encoded as a data URL."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        import cv2 as _cv2

        img = tmp_path / "p.png"
        _cv2.imwrite(str(img), _cv2.resize(np.zeros((400, 600, 3), dtype=np.uint8), (600, 400)))

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828060000-base01"}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828060000-base01",
            "status": "running",
        }
        poll_resp.raise_for_status.return_value = None
        poll_resp2 = MagicMock(spec=httpx.Response)
        poll_resp2.json.return_value = {
            "id": "cgt-20260828060000-base01",
            "status": "succeeded",
            "content": {"video_url": "https://ark-cdn.volces.com/i2v.mp4"},
        }
        poll_resp2.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [poll_resp, poll_resp2]

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            provider.generate_from_image(str(img), prompt="")

        import base64

        body = mock_client.post.call_args[1]["json"]
        url = body["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        raw = base64.b64decode(url.split(",", 1)[1])
        assert raw == img.read_bytes()

    def test_i2v_poll_error_resource_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ResourceNotFound code surfaces as a readable error message."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828070000-err01"}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828070000-err01",
            "status": "running",
        }
        poll_resp.raise_for_status.return_value = None
        poll_resp2 = MagicMock(spec=httpx.Response)
        poll_resp2.json.return_value = {
            "id": "cgt-20260828070000-err01",
            "status": "expired",
            "error": {"code": "ResourceNotFound", "message": "task expired"},
        }
        poll_resp2.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [poll_resp, poll_resp2]

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="ResourceNotFound"),
        ):
            provider.generate_from_image("https://x.com/p.png")

    def test_i2v_rejects_invalid_format(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()
        bad = tmp_path / "x.tif"
        bad.write_bytes(b"not-an-image")
        with pytest.raises(ValueError, match="unsupported format"):
            provider.generate_from_image(str(bad))

    def test_i2v_rejects_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()
        with pytest.raises(ValueError, match="Image file not found"):
            provider.generate_from_image("/nonexistent/photo.png")

    def test_i2v_empty_prompt_allows_image_only(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Seedance allows an empty prompt — image-only motion."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        provider = SeedanceProvider()
        img = tmp_path / "p.png"
        import cv2 as _cv2

        _cv2.imwrite(str(img), _cv2.resize(np.zeros((400, 600, 3), dtype=np.uint8), (600, 400)))

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260828080000-empty01"}
        submit_resp.raise_for_status.return_value = None
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-20260828080000-empty01",
            "status": "succeeded",
            "content": {"video_url": "https://ark-cdn.volces.com/i2v.mp4"},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            result = provider.generate_from_image(str(img), prompt="")
        assert result.provider == "seedance"
        assert mock_client.post.call_args[1]["json"]["content"][0] == {"type": "text", "text": ""}


class TestVeoImageToVideo:
    def test_i2v_sends_image_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {
            "predictions": [{"id": "veo-i2v-001", "video_url": "https://storage.googleapis.com/veo/i2v.mp4"}]
        }
        mock_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = mock_resp

        with patch("integrations.veo.httpx.Client", return_value=mock_client):
            result = provider.generate_from_image("https://example.com/p.png", duration=5, aspect_ratio="16:9")

        assert result.video_url == "https://storage.googleapis.com/veo/i2v.mp4"
        assert result.job_id == "veo-i2v-001"
        assert result.provider == "veo"

        body = mock_client.post.call_args[1]["json"]
        inst = body["instances"][0]
        assert inst["image"][VeoProvider.I2V_IMAGE_BYTES_FIELD] == "https://example.com/p.png"

    def test_i2v_missing_predictions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = mock_resp

        with (
            patch("integrations.veo.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing predictions"),
        ):
            provider.generate_from_image("https://x.com/p.png")

    def test_i2v_missing_video_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {"predictions": [{"id": "veo-i2v-002"}]}
        mock_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = mock_resp

        with (
            patch("integrations.veo.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing video URL"),
        ):
            provider.generate_from_image("https://x.com/p.png")


# ══════════════════════════════════════════════════════════════════════════════
# MockProvider: real ffmpeg lavfi render, verified with ffprobe
# ══════════════════════════════════════════════════════════════════════════════


def _probe(out_path):
    """Run ffprobe against *out_path*; return the parsed JSON stream info."""
    import json as _json
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    assert r.returncode == 0, f"ffprobe failed: {r.stderr}"
    return _json.loads(r.stdout)


class TestMockProvider:
    def test_factory_registers_mock(self) -> None:
        assert "mock" in list_providers()
        instance = get_provider("mock")
        from integrations.mock_provider import MockProvider

        assert isinstance(instance, MockProvider)

    def test_mock_provider_no_env_needed(self) -> None:
        from integrations.mock_provider import MockProvider

        # Should not raise even with no environment keys set.
        p = MockProvider()
        assert p._api_key == "mock"

    def test_mock_generate_produces_readable_video(self, tmp_path) -> None:
        from integrations.mock_provider import MockProvider

        provider = MockProvider()
        out = tmp_path / "mock_gen.mp4"
        out_path = provider._render(str(out), duration=1, fps=24, width=128, height=72)
        assert out_path == str(out)

        info = _probe(out_path)
        assert info.get("format") and info["format"].get("duration")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))

    def test_mock_generate_returns_result(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from integrations.mock_provider import MockProvider

        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path))
        provider = MockProvider()

        result = provider.generate("mock flyover", duration=1)

        assert result.provider == "mock"
        assert result.job_id.startswith("mock-")
        assert result.video_url.endswith(".mp4")
        info = _probe(result.video_url)
        assert info.get("format")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))

    def test_mock_generate_from_image_returns_result(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from integrations.mock_provider import MockProvider

        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path))
        provider = MockProvider()

        result = provider.generate_from_image("https://example.com/cat.png", duration=1)

        assert result.provider == "mock"
        assert result.job_id.startswith("mock-i2v-")
        assert result.metadata["image_path"] == "https://example.com/cat.png"
        info = _probe(result.video_url)
        assert info.get("format")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))


# ══════════════════════════════════════════════════════════════════════════════
# LocalSVDProvider (G-4): local diffusers/SVD image-to-video, pluggable backend
# ══════════════════════════════════════════════════════════════════════════════


class TestLocalSVDFactory:
    def test_factory_registers_local_svd(self) -> None:
        assert "local-svd" in list_providers()
        from integrations.local_svd import LocalSVDProvider

        instance = get_provider("local-svd")
        assert isinstance(instance, LocalSVDProvider)

    def test_provider_name(self) -> None:
        from integrations.local_svd import LocalSVDProvider

        # No env var / key required — local generation needs none.
        provider = LocalSVDProvider()
        assert provider.provider_name == "localsvd"

    def test_api_key_is_local_no_env_needed(self) -> None:
        from integrations.local_svd import LocalSVDProvider

        provider = LocalSVDProvider()
        assert provider._api_key == "local"

    def test_generate_raises_not_implemented(self) -> None:
        from integrations.local_svd import LocalSVDProvider

        provider = LocalSVDProvider()
        with pytest.raises(NotImplementedError, match="image-to-video"):
            provider.generate("a flying scene")


class TestLocalSVDParameterSelection:
    """12GB/MPS parameter selection logic (acceptance: inject fake device info).

    The provider picks resolution / CPU-offload from the device.  We monkey-patch
    ``detect_best_device`` and set ``_device`` / ``vram_gb`` directly to exercise
    each branch without touching any real GPU.
    """

    def _make_provider(self, device: str):
        from integrations.local_svd import LocalSVDProvider

        provider = LocalSVDProvider()
        provider._device = device
        return provider

    def test_cuda_low_vram_12gb_low_res_and_offload(self) -> None:

        provider = self._make_provider("cuda")
        params = provider._select_params()
        assert params["width"] == 576
        assert params["height"] == 320
        assert params["enable_cpu_offload"] is True
        assert params["motion_amplitude"] == 6.0
        assert params["fps"] == 7

    def test_cuda_low_vram_injected_8gb_stays_low_res(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit vram_gb below the 12GB threshold stays low-res + offload."""

        provider = self._make_provider("cuda")
        params = provider._select_params(vram_gb=8)
        assert params["width"] == 576
        assert params["height"] == 320
        assert params["enable_cpu_offload"] is True
        assert params["fps"] == 7

    def test_cuda_high_vram_24gb_full_res_no_offload(self) -> None:

        provider = self._make_provider("cuda")
        params = provider._select_params(vram_gb=24)
        assert params["width"] == 1024
        assert params["height"] == 576
        assert params["enable_cpu_offload"] is False

    def test_mps_full_res_no_offload(self) -> None:
        """MPS (M2 Max 96GB unified memory) → full resolution, no offload."""

        provider = self._make_provider("mps")
        params = provider._select_params()
        assert params["width"] == 1024
        assert params["height"] == 576
        assert params["enable_cpu_offload"] is False

    def test_cpu_fallback_low_res(self) -> None:

        provider = self._make_provider("cpu")
        params = provider._select_params()
        assert params["width"] == 576
        assert params["height"] == 320
        assert params["enable_cpu_offload"] is False

    def test_explicit_kwargs_override_device_defaults(self) -> None:

        provider = self._make_provider("mps")
        params = provider._select_params(width=384, height=256, fps=6, motion_amplitude=4.5, enable_cpu_offload=True)
        assert params["width"] == 384
        assert params["height"] == 256
        assert params["fps"] == 6
        assert params["motion_amplitude"] == 4.5
        assert params["enable_cpu_offload"] is True


class TestLocalSVDMockBackend:
    """Full-path test with the fake backend — no model, no network."""

    def test_generate_from_image_full_path_with_mock(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        import cv2
        import numpy as np
        from integrations.local_svd import LocalSVDProvider, MockSVDBackend

        monkeypatch.setenv("SVD_PROVIDER_OUTPUT_DIR", str(tmp_path))

        img = tmp_path / "input.png"
        cv2.imwrite(str(img), np.zeros((400, 600, 3), dtype=np.uint8))

        provider = LocalSVDProvider(backend=MockSVDBackend(), device="cuda")
        result = provider.generate_from_image(str(img), duration=1, aspect_ratio="16:9")

        assert result.provider == "localsvd"
        assert result.job_id.startswith("local-svd-")
        assert result.video_url.endswith(".mp4")
        assert os.path.exists(result.video_url)
        info = _probe(result.video_url)
        assert info.get("format")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))

        meta = result.metadata
        assert meta["device"] == "cuda"
        assert meta["backend"] == "MockSVDBackend"
        assert meta["model_id"] == "stabilityai/stable-video-diffusion-img2vid-xt"
        assert meta["image_path"] == str(img)

    def test_generate_from_image_uses_selected_resolution(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Mock backend must be called with the params the provider chose."""
        import cv2
        import numpy as np
        from integrations.local_svd import LocalSVDProvider, MockSVDBackend

        monkeypatch.setenv("SVD_PROVIDER_OUTPUT_DIR", str(tmp_path))

        img = tmp_path / "input.png"
        cv2.imwrite(str(img), np.zeros((400, 600, 3), dtype=np.uint8))

        class _RecordingBackend(MockSVDBackend):
            def __init__(self) -> None:
                super().__init__()
                self.last_call: dict = {}

            def generate(
                self,
                image_path: str,
                width: int,
                height: int,
                fps: int,
                motion_amplitude: float,
                num_frames: int,
                output_dir: str,
            ) -> list[str]:
                self.last_call = {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "num_frames": num_frames,
                }
                # Defer to the real MockSVDBackend.generate (this same
                # instance) so the _loaded guard and frame writing run.
                return MockSVDBackend.generate(
                    self, image_path, width, height, fps, motion_amplitude, num_frames, output_dir
                )

        recorder = _RecordingBackend()
        provider = LocalSVDProvider(backend=recorder, device="cuda")

        result = provider.generate_from_image(str(img), duration=1)

        assert result.video_url.endswith(".mp4")
        # 12GB CUDA defaults: 576x320, 7 fps.  1s * 7fps = 7 frames.
        assert recorder.last_call["width"] == 576
        assert recorder.last_call["height"] == 320
        assert recorder.last_call["fps"] == 7
        assert recorder.last_call["num_frames"] == 7

    def test_mock_backend_loads_before_generate(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Backend.load() must run before backend.generate()."""
        import cv2
        import numpy as np
        from integrations.local_svd import LocalSVDProvider, MockSVDBackend

        monkeypatch.setenv("SVD_PROVIDER_OUTPUT_DIR", str(tmp_path))

        img = tmp_path / "input.png"
        cv2.imwrite(str(img), np.zeros((400, 600, 3), dtype=np.uint8))

        backend = MockSVDBackend()
        provider = LocalSVDProvider(backend=backend, device="mps")

        provider.generate_from_image(str(img), duration=1)
        assert backend._loaded is True

    def test_generate_from_image_validates_image_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing image → ValidationError from image_prep before the backend loads."""
        from integrations.local_svd import LocalSVDProvider, MockSVDBackend

        backend = MockSVDBackend()
        provider = LocalSVDProvider(backend=backend, device="cuda")
        with pytest.raises(ValueError, match="Image file not found"):
            provider.generate_from_image("/nonexistent/photo.png")
        assert backend._loaded is False

    def test_model_id_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from integrations.local_svd import LocalSVDProvider

        monkeypatch.setenv("SVD_MODEL_ID", "owner/my-svd-fine-tune")
        provider = LocalSVDProvider()
        assert provider._model_id == "owner/my-svd-fine-tune"


class TestLocalSVDRealBackendDependency:
    """Real backend only tested for the actionable missing-dependency error.

    CI has no diffusers and no weights, so we verify the *error text* the
    operator sees rather than any inference.  These tests deliberately do
    not import diffusers at the top of the module.
    """

    def _real_backend(self):
        from integrations.local_svd import _DiffusersSVDBackend

        return _DiffusersSVDBackend()

    def test_load_missing_diffusers_error_is_actionable(self) -> None:
        """Missing diffusers class → actionable RuntimeError naming the pip install command."""
        import sys
        import types

        backend = self._real_backend()
        # Inject a fake `diffusers` module that has no pipeline class, so the
        # `from diffusers import StableVideoDiffusionPipeline` inside load()
        # raises ImportError — which the backend re-wraps as RuntimeError.
        fake_mod = types.ModuleType("diffusers")
        original = sys.modules.get("diffusers")
        try:
            sys.modules["diffusers"] = fake_mod
            with pytest.raises(RuntimeError, match="pip install diffusers torch"):
                backend.load("stabilityai/stable-video-diffusion-img2vid-xt", "cuda", True)
        finally:
            if original is None:
                sys.modules.pop("diffusers", None)
            else:
                sys.modules["diffusers"] = original

    def test_missing_dependency_message_naming_pip_install(self) -> None:
        """The dependency error message tells the user how to fix it."""
        from integrations.local_svd import _DiffusersSVDBackend

        backend = _DiffusersSVDBackend()
        msg = str(backend._missing_dependency_error(ImportError("no module named diffusers")))
        assert "pip install diffusers torch" in msg
        assert "diffusers" in msg

    def test_missing_model_message_naming_vram(self) -> None:
        """The missing-weights error names VRAM requirements and the install command."""
        from integrations.local_svd import _DiffusersSVDBackend

        backend = _DiffusersSVDBackend()
        msg = str(
            backend._missing_model_error("stabilityai/stable-video-diffusion-img2vid-xt", RuntimeError("no weights"))
        )
        assert "pip install diffusers torch" in msg
        assert "12 GB VRAM" in msg
        assert "stabilityai/stable-video-diffusion-img2vid-xt" in msg

    def test_get_backend_builds_real_when_no_injection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no backend injected, _get_backend() constructs the real one."""
        from integrations.local_svd import LocalSVDProvider, _DiffusersSVDBackend

        provider = LocalSVDProvider()
        backend = provider._get_backend()
        assert isinstance(backend, _DiffusersSVDBackend)
        # The real backend is stored for reuse.
        assert provider._get_backend() is backend


class TestLocalSVDHelperFunctions:
    def test_frames_for_duration_capped_at_25(self) -> None:
        from integrations.local_svd import _frames_for_duration

        assert _frames_for_duration(1, 7) == 7
        # Long durations cap at 25 frames (SVD-XT produces ~25).
        assert _frames_for_duration(5, 7) == 25
        # Zero duration still yields at least one frame.
        assert _frames_for_duration(0, 7) >= 1

    def test_frames_for_duration_rounding(self) -> None:
        from integrations.local_svd import _frames_for_duration

        assert _frames_for_duration(3, 7) == 21
