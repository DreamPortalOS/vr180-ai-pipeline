"""Tests for integrations/minimax.py — the MiniMax H3 2K lane (issue #353).

**Nothing in this file touches the network or the operator's ledger.**  Every
test patches ``integrations.minimax.httpx.Client``; the two that exercise the
budget gate point ``VR180_LEDGER_PATH`` at ``tmp_path``.  A MiniMax 2K/10s clip
costs real money ($0.13/s → ~$1.30), so a test that accidentally submitted one
would be a bug with an invoice attached.

The load-bearing test in here is
``TestBudgetGate::test_gate_blocks_before_the_submit_post``.  Its value is
entirely in the ``post.call_count == 0`` assertion: a gate that fires *after*
the POST costs exactly as much as no gate at all.  See that class's docstring
for the mutation check that keeps the assertion honest.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from PIL import Image

from integrations import minimax
from integrations.factory import get_provider
from integrations.minimax import (
    DURATION_MAX,
    DURATION_MIN,
    MODEL_H3,
    PRICE_PER_SECOND_USD,
    RESOLUTION_2K,
    RESOLUTION_768P,
    VALID_RATIOS,
    MiniMaxProvider,
    estimate_cost_usd,
    poll_timeout_seconds,
    resolve_cli_model,
    resolve_cli_resolution,
    timeout_recovery_message,
)
from integrations.usage_ledger import BudgetExceededError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider(monkeypatch: pytest.MonkeyPatch) -> MiniMaxProvider:
    """A provider with a fake key. The key is never sent anywhere real."""
    monkeypatch.setenv(minimax.ENV_API_KEY, "test-key-not-a-real-credential")
    return MiniMaxProvider()


def _response(payload: dict) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _client(
    *,
    poll_payloads: list[dict] | None = None,
    submit_payload: dict | None = None,
) -> MagicMock:
    """A mocked ``httpx.Client`` usable as a context manager.

    GETs yield *poll_payloads* in order; the POST returns *submit_payload*.
    """
    if poll_payloads is None:
        poll_payloads = [_succeeded()]
    if submit_payload is None:
        submit_payload = {"task_id": "mm-task-0001"}

    client = MagicMock(spec=httpx.Client)
    client.__enter__.return_value = client
    client.post.return_value = _response(submit_payload)

    polls = [_response(p) for p in poll_payloads]
    if len(polls) == 1:
        client.get.return_value = polls[0]
    else:
        client.get.side_effect = polls
    return client


def _succeeded(url: str = "https://cdn.minimax.io/out.mp4") -> dict:
    """A V2 query payload for a finished task (URL lives at ``content.url``)."""
    return {"status": "succeeded", "content": {"url": url}}


def _square_png(path: Path, size: int = 512) -> Path:
    """A real square PNG that passes ``validate_image_for_i2v``.

    Must be a genuine decodable image: the provider validates dimensions and
    aspect ratio before encoding, so fake bytes would fail for the wrong reason.
    """
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, size, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _run_generate(provider: MiniMaxProvider, client: MagicMock, **kwargs):
    """Drive ``generate()`` with the poll loop's sleep stubbed out."""
    with (
        patch("integrations.minimax.httpx.Client", return_value=client),
        patch("integrations.minimax.time.sleep", return_value=None),
    ):
        return provider.generate("a drone shot over a river", **kwargs)


def _submitted_body(client: MagicMock) -> dict:
    """The JSON body of the single submit POST."""
    assert client.post.call_count == 1
    return client.post.call_args[1]["json"]


# ══════════════════════════════════════════════════════════════════════════════
# Credentials / factory wiring
# ══════════════════════════════════════════════════════════════════════════════


class TestCredentials:
    def test_load_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(minimax.ENV_API_KEY, "mm-key")
        assert MiniMaxProvider()._api_key == "mm-key"

    def test_missing_api_key_raises_naming_the_variable(self) -> None:
        """The conftest ``_isolated_env`` fixture guarantees the var is unset."""
        with pytest.raises(ValueError, match=minimax.ENV_API_KEY):
            MiniMaxProvider()

    def test_provider_name_is_minimax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _provider(monkeypatch).provider_name == "minimax"

    def test_factory_returns_a_minimax_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(minimax.ENV_API_KEY, "mm-key")
        assert isinstance(get_provider("minimax"), MiniMaxProvider)

    def test_base_url_defaults_to_the_global_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _provider(monkeypatch).base_url == minimax.BASE_URL_GLOBAL

    def test_base_url_override_switches_to_the_mainland_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        monkeypatch.setenv(minimax.ENV_API_BASE, minimax.BASE_URL_CN + "/")
        assert provider.base_url == minimax.BASE_URL_CN


# ══════════════════════════════════════════════════════════════════════════════
# Budget soft-gate (#328, W-11) — the reason this card exists
# ══════════════════════════════════════════════════════════════════════════════


class TestBudgetGate:
    """The gate must fire **before** the submit POST, not after.

    Mutation check (run manually when touching this class — it is the only
    thing that proves the tests are load-bearing rather than decorative):
    delete the ``usage_ledger.enforce_budget(...)`` call from
    ``MiniMaxProvider._run`` and re-run this class.  Both
    ``test_gate_blocks_before_the_submit_post`` and
    ``test_gate_blocks_image_to_video_too`` must go **red** on the
    ``post.call_count == 0`` assertion — verified 2026-09-16.  If they stay
    green the gate is not being exercised and the assertions are worthless.
    """

    @staticmethod
    def _ledger_over_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point the ledger at tmp_path and put it well past a token cap.

        Token mode rather than 元 mode on purpose: it needs no unit price, so
        the gate under test is the cap arithmetic and not price configuration.
        """
        ledger = tmp_path / "usage_ledger.jsonl"
        ledger.write_text('{"total_tokens": 5000}\n', encoding="utf-8")
        monkeypatch.setenv("VR180_LEDGER_PATH", str(ledger))
        monkeypatch.setenv("VR180_BUDGET_CAP_TOKENS", "1000")
        return ledger

    def test_gate_blocks_before_the_submit_post(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Over cap: raise, and send nothing. The POST is what costs money."""
        provider = _provider(monkeypatch)
        self._ledger_over_cap(monkeypatch, tmp_path)
        client = _client()

        with pytest.raises(BudgetExceededError):
            _run_generate(provider, client, duration=10, resolution="2k")

        assert client.post.call_count == 0, "budget gate fired after the submit POST — the money is already spent"
        assert client.get.call_count == 0

    def test_gate_blocks_image_to_video_too(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """i2v is the expensive path this card was opened for — gate it too."""
        provider = _provider(monkeypatch)
        self._ledger_over_cap(monkeypatch, tmp_path)
        image = _square_png(tmp_path / "seed.png")
        client = _client()

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
            pytest.raises(BudgetExceededError),
        ):
            provider.generate_from_image(str(image), duration=10, resolution="2k", aspect_ratio="1:1")

        assert client.post.call_count == 0

    def test_gate_never_opens_an_http_client_at_all(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Stronger than call_count: the client is not even constructed.

        Guards the ordering directly — ``enforce_budget`` sits above the
        ``with httpx.Client(...)`` block, so a refactor that moves it inside
        would still pass a POST-count assertion if the body were restructured.
        """
        provider = _provider(monkeypatch)
        self._ledger_over_cap(monkeypatch, tmp_path)

        # A fully-configured client, so that if the gate were ever removed this
        # test fails on the assertion below rather than spinning in the poll
        # loop against an unconfigured mock.
        with (
            patch("integrations.minimax.httpx.Client", return_value=_client()) as client_cls,
            pytest.raises(BudgetExceededError),
        ):
            _run_generate(provider, client_cls.return_value, duration=10, resolution="2k")

        assert client_cls.call_count == 0

    def test_under_the_cap_the_submit_goes_through(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The negative control: same wiring, cap not reached, POST happens."""
        provider = _provider(monkeypatch)
        ledger = tmp_path / "usage_ledger.jsonl"
        ledger.write_text('{"total_tokens": 10}\n', encoding="utf-8")
        monkeypatch.setenv("VR180_LEDGER_PATH", str(ledger))
        monkeypatch.setenv("VR180_BUDGET_CAP_TOKENS", "1000000")

        client = _client()
        result = _run_generate(provider, client, duration=10, resolution="2k")

        assert client.post.call_count == 1
        assert result.video_url == "https://cdn.minimax.io/out.mp4"

    def test_budget_cap_kwarg_is_not_sent_as_a_body_field(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``budget_cap`` is a local guard rail; MiniMax would reject it (2013)."""
        provider = _provider(monkeypatch)
        monkeypatch.setenv("VR180_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
        client = _client()

        _run_generate(provider, client, budget_cap=999, budget_cap_tokens=999_999, poll_timeout=60)

        body = _submitted_body(client)
        for leaked in ("budget_cap", "budget_cap_tokens", "poll_timeout"):
            assert leaked not in body

    def test_resume_is_deliberately_not_gated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Resume issues no POST, so it spends nothing — blocking it would only
        strand an already-paid-for clip behind a cap it cannot influence."""
        provider = _provider(monkeypatch)
        self._ledger_over_cap(monkeypatch, tmp_path)
        client = _client()

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            result = provider.resume("mm-task-0001")

        assert client.post.call_count == 0
        assert result.video_url == "https://cdn.minimax.io/out.mp4"

    def test_a_test_run_never_writes_the_real_ledger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """#329's interlock, re-asserted from the MiniMax side."""
        from integrations import usage_ledger

        monkeypatch.delenv("VR180_LEDGER_PATH", raising=False)
        assert usage_ledger._writes_blocked(None) is True


# ══════════════════════════════════════════════════════════════════════════════
# Request body — the verified V2 contract
# ══════════════════════════════════════════════════════════════════════════════


class TestRequestBody:
    """Pins the contract verified against platform.minimax.io on 2026-09-16."""

    def test_text_to_video_body_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client, duration=10, resolution="2k", aspect_ratio="1:1")

        body = _submitted_body(client)
        assert body["model"] == MODEL_H3
        assert body["resolution"] == RESOLUTION_2K
        assert body["duration"] == 10
        assert body["ratio"] == "1:1"
        assert body["content"] == [{"type": "text", "text": "a drone shot over a river"}]

    def test_submit_hits_the_documented_v2_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client)
        assert client.post.call_args[0][0] == "/v2/video_generation"

    def test_poll_hits_the_documented_v2_query_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(submit_payload={"task_id": "mm-abc"})
        _run_generate(provider, client)
        assert client.get.call_args[0][0] == "/v2/query/video_generation/mm-abc"

    def test_auth_is_a_bearer_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client)
        headers = client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key-not-a-real-credential"
        assert headers["Content-Type"] == "application/json"

    def test_one_to_one_is_an_accepted_ratio(self) -> None:
        """Our hard requirement. Documented enum, confirmed in the V2 reference."""
        assert "1:1" in VALID_RATIOS

    def test_unsupported_ratio_is_refused_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        with pytest.raises(ValueError, match="not supported"):
            _run_generate(provider, client, aspect_ratio="5:4")
        assert client.post.call_count == 0

    def test_text_to_video_omits_adaptive_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """t2v rejects ``adaptive`` server-side, so the field is dropped."""
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client, aspect_ratio="adaptive")
        assert "ratio" not in _submitted_body(client)

    def test_seedance_only_kwargs_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cross-provider CLI sprays Ark kwargs at every provider; forwarding
        one to MiniMax earns a ``2013 invalid parameters`` rejection."""
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client, draft=True, seed=42, camera_fixed=True, negative_prompt="blurry")

        body = _submitted_body(client)
        assert set(body) == {"model", "content", "resolution", "duration", "ratio"}

    def test_extra_body_is_the_escape_hatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client, extra_body={"callback_url": "https://example.test/hook"})
        assert _submitted_body(client)["callback_url"] == "https://example.test/hook"

    def test_empty_prompt_is_refused_for_text_to_video(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        with pytest.raises(ValueError, match="non-empty prompt"):
            provider.generate("   ")

    @pytest.mark.parametrize("duration", [DURATION_MIN, 10, DURATION_MAX])
    def test_durations_inside_the_documented_range_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, duration: int
    ) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        _run_generate(provider, client, duration=duration)
        assert _submitted_body(client)["duration"] == duration

    @pytest.mark.parametrize("duration", [DURATION_MIN - 1, DURATION_MAX + 1, 0])
    def test_durations_outside_the_range_raise_before_any_http(
        self, monkeypatch: pytest.MonkeyPatch, duration: int
    ) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        with pytest.raises(ValueError, match="between"):
            _run_generate(provider, client, duration=duration)
        assert client.post.call_count == 0

    def test_non_integer_duration_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        with pytest.raises(ValueError, match="whole number"):
            _run_generate(provider, client, duration="ten")
        assert client.post.call_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# Image-to-video — the square-seed workflow
# ══════════════════════════════════════════════════════════════════════════════


class TestImageToVideo:
    def test_local_image_becomes_a_first_frame_data_url(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        provider = _provider(monkeypatch)
        image = _square_png(tmp_path / "seed.png")
        client = _client()

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            provider.generate_from_image(str(image), "drift forward", duration=10, aspect_ratio="1:1")

        content = _submitted_body(client)["content"]
        text_part = next(p for p in content if p["type"] == "text")
        image_part = next(p for p in content if p["type"] == "image_url")

        assert text_part["text"] == "drift forward"
        assert image_part["role"] == "first_frame"
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == image.read_bytes()

    def test_http_url_first_frame_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        remote = "https://example.test/seed.png"

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            provider.generate_from_image(remote, duration=10)

        content = _submitted_body(client)["content"]
        assert content[0]["image_url"]["url"] == remote

    def test_prompt_may_be_omitted_for_image_only_motion(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        provider = _provider(monkeypatch)
        image = _square_png(tmp_path / "seed.png")
        client = _client()

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            provider.generate_from_image(str(image))

        content = _submitted_body(client)["content"]
        assert [p["type"] for p in content] == ["image_url"]

    def test_a_missing_seed_frame_raises_before_any_http(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(ValueError, match="not found"),
        ):
            provider.generate_from_image(str(tmp_path / "nope.png"))
        assert client.post.call_count == 0

    def test_square_output_needs_a_square_seed_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """In first-frame mode the API derives the ratio from the image and
        ignores ``ratio``. An operator asking for 1:1 must be told that."""
        provider = _provider(monkeypatch)
        image = _square_png(tmp_path / "seed.png")
        client = _client()

        with (
            caplog.at_level(logging.INFO, logger="integrations.minimax"),
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            provider.generate_from_image(str(image), duration=10, aspect_ratio="1:1")

        assert "首帧必须本身就是方图" in caplog.text


# ══════════════════════════════════════════════════════════════════════════════
# CLI tier / model translation
# ══════════════════════════════════════════════════════════════════════════════


class TestCliTranslation:
    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("480p", RESOLUTION_768P),
            ("720p", RESOLUTION_768P),
            ("768p", RESOLUTION_768P),
            ("2k", RESOLUTION_2K),
            ("2K", RESOLUTION_2K),
            (None, RESOLUTION_768P),
            ("", RESOLUTION_768P),
        ],
    )
    def test_cheap_and_explicit_tiers_map_through(self, requested: str | None, expected: str) -> None:
        assert resolve_cli_resolution(requested) == expected

    @pytest.mark.parametrize("requested", ["1080p", "4k"])
    def test_ark_only_tiers_are_refused_rather_than_promoted(self, requested: str) -> None:
        """Silently promoting an Ark tier to 2K would spend 60% more per second
        than the operator asked for."""
        with pytest.raises(ValueError, match="--gen-resolution 2k"):
            resolve_cli_resolution(requested)

    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown MiniMax resolution"):
            resolve_cli_resolution("8k")

    def test_ark_model_default_falls_back_to_h3(self) -> None:
        assert resolve_cli_model("doubao-seedance-1-0-lite-i2v-250428") == MODEL_H3

    def test_known_minimax_model_is_kept(self) -> None:
        assert resolve_cli_model(MODEL_H3) == MODEL_H3

    def test_no_model_falls_back_to_h3(self) -> None:
        assert resolve_cli_model(None) == MODEL_H3


# ══════════════════════════════════════════════════════════════════════════════
# Pricing
# ══════════════════════════════════════════════════════════════════════════════


class TestPricing:
    def test_published_per_second_rates(self) -> None:
        """First-party pay-as-you-go table (platform.minimax.io), 2026-09-16."""
        assert PRICE_PER_SECOND_USD[RESOLUTION_2K] == 0.13
        assert PRICE_PER_SECOND_USD[RESOLUTION_768P] == 0.08

    def test_the_headline_number_from_the_card(self) -> None:
        """2K × 10s ≈ $1.30 ≈ 9.5 元 — the figure the card is built on."""
        assert estimate_cost_usd(RESOLUTION_2K, 10) == pytest.approx(1.30)

    def test_cheap_tier_ten_seconds(self) -> None:
        assert estimate_cost_usd(RESOLUTION_768P, 10) == pytest.approx(0.80)

    def test_unknown_tier_has_no_price(self) -> None:
        assert estimate_cost_usd("4k", 10) is None

    def test_unparseable_duration_has_no_price(self) -> None:
        assert estimate_cost_usd(RESOLUTION_2K, "ten") is None


# ══════════════════════════════════════════════════════════════════════════════
# Poll budget + timeout recovery
# ══════════════════════════════════════════════════════════════════════════════


class TestPollBudget:
    def test_baseline_is_never_below_300s(self) -> None:
        assert poll_timeout_seconds(RESOLUTION_768P, 1) >= 300

    def test_two_k_ten_seconds_gets_1800s(self) -> None:
        assert poll_timeout_seconds(RESOLUTION_2K, 10) == 1800

    def test_budget_grows_with_the_tier(self) -> None:
        cheap = poll_timeout_seconds(RESOLUTION_768P, 10)
        assert cheap < poll_timeout_seconds(RESOLUTION_2K, 10)

    def test_override_wins(self) -> None:
        assert poll_timeout_seconds(RESOLUTION_2K, 10, override=90) == 90

    def test_override_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            poll_timeout_seconds(override=0)

    def test_timeout_message_says_do_not_regenerate(self) -> None:
        message = timeout_recovery_message("mm-task-9", 1800)
        assert "mm-task-9" in message
        assert "不要重新生成" in message
        assert "--resume-task mm-task-9" in message

    def test_a_timeout_raises_the_recovery_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A client-side timeout is not a failure: the clip is already billed."""
        provider = _provider(monkeypatch)
        client = _client(poll_payloads=[{"status": "processing"}])

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="不要重新生成"),
        ):
            provider.generate("x", poll_timeout=1, duration=10)


# ══════════════════════════════════════════════════════════════════════════════
# Response handling
# ══════════════════════════════════════════════════════════════════════════════


class TestResponseHandling:
    def test_success_reads_the_url_from_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(poll_payloads=[_succeeded("https://cdn.minimax.io/final.mp4")])
        result = _run_generate(provider, client, duration=10)

        assert result.video_url == "https://cdn.minimax.io/final.mp4"
        assert result.provider == "minimax"
        assert result.job_id == "mm-task-0001"

    def test_estimated_cost_is_attached_to_the_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()
        result = _run_generate(provider, client, duration=10, resolution="2k")
        assert result.metadata["estimated_cost_usd"] == pytest.approx(1.30)

    def test_it_polls_until_the_task_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(poll_payloads=[{"status": "queued"}, {"status": "processing"}, _succeeded()])
        _run_generate(provider, client, duration=10)
        assert client.get.call_count == 3

    def test_missing_task_id_is_a_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(submit_payload={"nothing": "useful"})
        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(RuntimeError, match="missing task_id"),
        ):
            provider.generate("x", duration=10)

    def test_insufficient_balance_arrives_as_http_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MiniMax reports application errors in ``base_resp`` with HTTP 200, so
        a 1008 would otherwise look like a merely missing task id."""
        provider = _provider(monkeypatch)
        client = _client(submit_payload={"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(RuntimeError, match="余额不足"),
        ):
            provider.generate("x", duration=10)

    def test_auth_failure_hint_names_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(submit_payload={"base_resp": {"status_code": 1004, "status_msg": "auth failed"}})

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(RuntimeError, match=minimax.ENV_API_KEY),
        ):
            provider.generate("x", duration=10)

    def test_a_failed_task_surfaces_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(poll_payloads=[{"status": "failed", "base_resp": {"status_msg": "moderation"}}])

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="moderation"),
        ):
            provider.generate("x", duration=10)

    def test_a_succeeded_task_without_a_url_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client(poll_payloads=[{"status": "succeeded", "content": {}}])

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
            pytest.raises(RuntimeError, match="missing video URL"),
        ):
            provider.generate("x", duration=10)

    def test_submit_http_error_is_wrapped_not_leaked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Callers catch RuntimeError/ValueError; a bare HTTPStatusError would
        escape the CLI as a traceback."""
        provider = _provider(monkeypatch)
        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 401
        error_resp.json.return_value = {"base_resp": {"status_code": 2049, "status_msg": "invalid api key"}}
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError("401", request=MagicMock(), response=error_resp)

        client = MagicMock(spec=httpx.Client)
        client.__enter__.return_value = client
        client.post.return_value = error_resp

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(RuntimeError, match="MiniMax submit failed: 401"),
        ):
            provider.generate("x", duration=10)


# ══════════════════════════════════════════════════════════════════════════════
# Resume
# ══════════════════════════════════════════════════════════════════════════════


class TestResume:
    def test_resume_sends_no_submit_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _client()

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            patch("integrations.minimax.time.sleep", return_value=None),
        ):
            result = provider.resume("mm-task-0001")

        assert client.post.call_count == 0
        assert result.video_url == "https://cdn.minimax.io/out.mp4"

    def test_resume_rejects_an_empty_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        with pytest.raises(ValueError, match="non-empty"):
            provider.resume("  ")

    def test_resume_of_an_unknown_task_is_a_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 404
        error_resp.json.return_value = {}
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError("404", request=MagicMock(), response=error_resp)

        client = MagicMock(spec=httpx.Client)
        client.__enter__.return_value = client
        client.get.return_value = error_resp

        with (
            patch("integrations.minimax.httpx.Client", return_value=client),
            pytest.raises(RuntimeError, match="poll failed: 404"),
        ):
            provider.resume("mm-does-not-exist")

        assert client.post.call_count == 0
