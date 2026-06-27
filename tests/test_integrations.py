"""Tests for the VideoGen provider abstraction layer.

Covers base types, factory, each provider's initialisation and generate()
method.  No live API calls are made — all HTTP responses are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
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
        monkeypatch.setenv("SEEDANCE_API_KEY", "key")
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
        assert "seedance" in providers
        assert "veo" in providers
        assert len(providers) == 3

    def test_get_provider_kling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "key")
        instance = get_provider("kling")
        assert isinstance(instance, KlingProvider)

    def test_get_provider_seedance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "key")
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
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "env-key")
        provider = SeedanceProvider()
        assert provider._api_key == "env-key"

    def test_load_api_key_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="SEEDANCE_API_KEY"):
            SeedanceProvider()

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "seedance-job-001"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "status": "completed",
            "output": {"video_url": "https://cdn.seedance.ai/video.mp4"},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            result = provider.generate("flying")

        assert result.video_url == "https://cdn.seedance.ai/video.mp4"
        assert result.job_id == "seedance-job-001"
        assert result.provider == "seedance"

    def test_generate_submit_missing_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {}
        submit_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="missing id"),
        ):
            provider.generate("test")

    def test_generate_poll_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "seedance-job-003"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"status": "failed", "message": "credit exhausted"}
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="credit exhausted"),
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
