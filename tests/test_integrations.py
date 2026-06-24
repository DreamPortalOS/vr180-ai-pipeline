"""Tests for the VideoGen provider abstraction layer.

Covers base types, factory, and each provider's initialisation + signature
methods.  Live API calls are NOT made — HTTP responses are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.base import GenerationParams, JobState, JobStatus, VideoGenProvider
from integrations.factory import get_provider, list_providers
from integrations.kling import KlingProvider
from integrations.seedance import SeedanceProvider
from integrations.veo import VeoProvider

# ═══════════════════════════════════════════════════════════════════════════════
# Base types
# ═══════════════════════════════════════════════════════════════════════════════


class TestJobState:
    def test_enum_values(self) -> None:
        assert JobState.PENDING.value == "pending"
        assert JobState.PROCESSING.value == "processing"
        assert JobState.COMPLETED.value == "completed"
        assert JobState.FAILED.value == "failed"
        assert JobState.CANCELLED.value == "cancelled"

    def test_all_members_covered(self) -> None:
        assert len(JobState) == 5


class TestJobStatus:
    def test_defaults(self) -> None:
        s = JobStatus(job_id="abc", state=JobState.PENDING)
        assert s.job_id == "abc"
        assert s.state == JobState.PENDING
        assert s.progress == 0
        assert s.message == ""
        assert s.output_url is None
        assert s.metadata == {}

    def test_full_construction(self) -> None:
        s = JobStatus(
            job_id="abc",
            state=JobState.COMPLETED,
            progress=100,
            message="done",
            output_url="https://example.com/video.mp4",
            metadata={"duration": 5},
        )
        assert s.output_url == "https://example.com/video.mp4"
        assert s.metadata["duration"] == 5


class TestGenerationParams:
    def test_defaults(self) -> None:
        p = GenerationParams(prompt="test")
        assert p.prompt == "test"
        assert p.negative_prompt == ""
        assert p.duration_seconds == 5
        assert p.resolution == "1080p"
        assert p.fps == 24
        assert p.extra == {}

    def test_full_construction(self) -> None:
        p = GenerationParams(
            prompt="fly over mountains",
            negative_prompt="blur, shake",
            duration_seconds=10,
            resolution="4k",
            fps=30,
            extra={"cfg_scale": 0.7},
        )
        assert p.duration_seconds == 10
        assert p.extra["cfg_scale"] == 0.7


class TestVideoGenProviderABC:
    """Verify that the abstract base class raises on direct usage."""

    def test_abc_raises_notimplemented_on_init(self) -> None:
        """ABC __init__ calls _load_api_key which raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            VideoGenProvider()

    @pytest.mark.asyncio
    async def test_subclass_must_implement(self) -> None:
        class Complete(VideoGenProvider):
            def _load_api_key(self) -> str:
                return ""

            async def submit(self, params):
                raise NotImplementedError("submit")

            async def poll(self, job_id):
                raise NotImplementedError("poll")

            async def download(self, job_id, out_path):
                raise NotImplementedError("download")

        instance = Complete()
        with pytest.raises(NotImplementedError, match="submit"):
            await instance.submit(GenerationParams(prompt="x"))
        with pytest.raises(NotImplementedError, match="poll"):
            await instance.poll("abc")
        with pytest.raises(NotImplementedError, match="download"):
            await instance.download("abc", "/tmp/x.mp4")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactory:
    def test_list_providers(self) -> None:
        providers = list_providers()
        assert "kling" in providers
        assert "seedance" in providers
        assert "veo" in providers
        assert len(providers) == 3

    def test_get_provider_kling(self) -> None:
        instance = get_provider("kling")
        assert isinstance(instance, KlingProvider)

    def test_get_provider_seedance(self) -> None:
        instance = get_provider("seedance")
        assert isinstance(instance, SeedanceProvider)

    def test_get_provider_veo(self) -> None:
        instance = get_provider("veo")
        assert isinstance(instance, VeoProvider)

    def test_get_provider_case_insensitive(self) -> None:
        instance = get_provider("KLING")
        assert isinstance(instance, KlingProvider)

    def test_get_provider_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_provider_name_property(self) -> None:
        assert get_provider("kling").provider_name == "kling"
        assert get_provider("seedance").provider_name == "seedance"
        assert get_provider("veo").provider_name == "veo"

    def test_get_provider_with_api_key(self) -> None:
        instance = get_provider("kling", api_key="custom-key")
        assert instance._api_key == "custom-key"


# ═══════════════════════════════════════════════════════════════════════════════
# KlingProvider
# ═══════════════════════════════════════════════════════════════════════════════


class TestKlingProvider:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "env-key")
        provider = KlingProvider()
        assert provider._api_key == "env-key"

    def test_load_api_key_empty_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KLING_API_KEY", raising=False)
        provider = KlingProvider()
        assert provider._api_key == ""

    def test_sign_returns_expected_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")
        provider = KlingProvider()
        sig, ts, nonce = provider._sign({"prompt": "hello"})
        assert isinstance(sig, str) and len(sig) == 64  # sha256 hex
        assert isinstance(ts, str) and ts.isdigit()
        assert isinstance(nonce, str) and len(nonce) == 16

    @pytest.mark.asyncio
    async def test_submit_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")

        provider = KlingProvider()
        # Use MagicMock for response (json/raise_for_status are sync in httpx)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"job_id": "kling-job-001"}}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        params = GenerationParams(prompt="fly over mountains")
        job_id = await provider.submit(params)
        assert job_id == "kling-job-001"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_missing_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")

        provider = KlingProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        with pytest.raises(ValueError, match="missing job_id"):
            await provider.submit(GenerationParams(prompt="x"))

    @pytest.mark.asyncio
    async def test_poll_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")

        provider = KlingProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "status": "succeed",
                "progress": 100,
                "video_url": "https://example.com/video.mp4",
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        status = await provider.poll("kling-job-001")
        assert status.state == JobState.COMPLETED
        assert status.progress == 100
        assert status.output_url == "https://example.com/video.mp4"

    @pytest.mark.asyncio
    async def test_download_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("KLING_API_KEY", "test-key")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")

        provider = KlingProvider()

        # Mock the fetch endpoint response (sync methods)
        mock_fetch_resp = MagicMock()
        mock_fetch_resp.json.return_value = {"data": {"video_url": "https://cdn.example.com/video.mp4"}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_fetch_resp)
        provider._http_client = mock_client

        out_path = str(tmp_path / "output.mp4")

        # Mock the download client separately
        mock_dl_resp = MagicMock()
        mock_dl_resp.content = b"fake-video-binary"

        with patch("httpx.AsyncClient", autospec=True) as mock_httpx_cls:
            mock_dl_client = AsyncMock()
            mock_dl_client.get = AsyncMock(return_value=mock_dl_resp)
            mock_httpx_cls.return_value = mock_dl_client
            result = await provider.download("kling-job-001", out_path)

        assert result == out_path
        with open(out_path, "rb") as f:
            assert f.read() == b"fake-video-binary"


# ═══════════════════════════════════════════════════════════════════════════════
# SeedanceProvider
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeedanceProvider:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "env-key")
        provider = SeedanceProvider()
        assert provider._api_key == "env-key"

    @pytest.mark.asyncio
    async def test_submit_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "seedance-job-001"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        job_id = await provider.submit(GenerationParams(prompt="flying"))
        assert job_id == "seedance-job-001"

    @pytest.mark.asyncio
    async def test_submit_missing_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        with pytest.raises(ValueError, match="missing id"):
            await provider.submit(GenerationParams(prompt="x"))

    @pytest.mark.asyncio
    async def test_poll_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "completed",
            "progress": 100,
            "output": {"video_url": "https://cdn.seedance.ai/video.mp4"},
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        status = await provider.poll("seedance-job-001")
        assert status.state == JobState.COMPLETED
        assert status.output_url == "https://cdn.seedance.ai/video.mp4"

    @pytest.mark.asyncio
    async def test_poll_state_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEEDANCE_API_KEY", "test-key")
        provider = SeedanceProvider()

        # Test queued -> PENDING
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "queued"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        status = await provider.poll("job-001")
        assert status.state == JobState.PENDING


# ═══════════════════════════════════════════════════════════════════════════════
# VeoProvider
# ═══════════════════════════════════════════════════════════════════════════════


class TestVeoProvider:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "env-key")
        provider = VeoProvider()
        assert provider._api_key == "env-key"

    def test_project_id_default(self) -> None:
        provider = VeoProvider()
        assert provider._project_id == "my-project"

    def test_project_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GCP_PROJECT_ID", "my-real-project")
        provider = VeoProvider()
        assert provider._project_id == "my-real-project"

    @pytest.mark.asyncio
    async def test_submit_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "predictions": [{"id": "veo-job-001"}],
            "id": "pred-001",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        job_id = await provider.submit(GenerationParams(prompt="flying"))
        assert job_id == "veo-job-001"

    @pytest.mark.asyncio
    async def test_submit_missing_predictions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEO_API_KEY", "test-key")
        monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
        provider = VeoProvider()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        provider._http_client = mock_client

        with pytest.raises(ValueError, match="missing predictions"):
            await provider.submit(GenerationParams(prompt="x"))

    @pytest.mark.asyncio
    async def test_poll_raises_not_implemented(self) -> None:
        provider = VeoProvider()
        with pytest.raises(NotImplementedError):
            await provider.poll("veo-job-001")

    @pytest.mark.asyncio
    async def test_download_raises_not_implemented(self) -> None:
        provider = VeoProvider()
        with pytest.raises(NotImplementedError):
            await provider.download("veo-job-001", "/tmp/x.mp4")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with prompt_builder (end-to-end flow)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptToGenerationFlow:
    """Verify that wrap_prompt_for_vr180 produces strings usable with GenerationParams."""

    def test_positive_prompt_used_in_params(self) -> None:
        from pipeline.prompt_builder import wrap_prompt_for_vr180

        wrapped = wrap_prompt_for_vr180("fly over mountains", scene_type="fpv")

        params = GenerationParams(
            prompt=wrapped["positive"],
            negative_prompt=wrapped["negative"],
            duration_seconds=5,
            resolution="1080p",
            fps=24,
        )

        assert "fly over mountains" in params.prompt
        assert isinstance(params.negative_prompt, str)
        assert len(params.negative_prompt) > 0
        assert "rapid turns" in params.negative_prompt
