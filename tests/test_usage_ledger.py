"""Tests for the generation usage ledger + budget soft-gate (issue #328, W-11).

The load-bearing assertion in this file is ``post.call_count == 0``: a budget
gate that fires *after* the submit request has been sent is worth exactly
nothing, because the quota is spent the moment Ark accepts the task.  Every
budget test therefore asserts on the mocked client's POST counter, not just on
the exception.

Everything here is mocked — no HTTP, no Ark, no key.  The autouse
``ledger_file`` fixture redirects the ledger into ``tmp_path`` and clears any
budget/price environment the developer's own shell might have set, so the
suite can never read or write the operator's real
``~/.vr180/usage_ledger.jsonl``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from integrations import usage_ledger
from integrations.seedance import SeedanceProvider
from integrations.usage_ledger import (
    ENV_BUDGET_CAP,
    ENV_BUDGET_CAP_TOKENS,
    ENV_LEDGER_PATH,
    ENV_PRICE_PER_MTOKEN,
    BudgetExceededError,
)

_TASK_ID = "cgt-20260909120000-test1"
_VIDEO_URL = "https://ark-cdn.volces.com/v.mp4"

# A plausible unit price in 元/百万token. Only ever set via the environment —
# the production code deliberately has no default (the Ark list price moves).
_PRICE = 37.0


@pytest.fixture(autouse=True)
def ledger_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the ledger into tmp_path and clear inherited budget config.

    Autouse on purpose: forgetting it in a single test would silently point
    that test at the operator's real ledger under ``~/.vr180``.
    """
    path = tmp_path / "ledger" / "usage_ledger.jsonl"
    monkeypatch.setenv(ENV_LEDGER_PATH, str(path))
    for name in (ENV_PRICE_PER_MTOKEN, ENV_BUDGET_CAP, ENV_BUDGET_CAP_TOKENS):
        monkeypatch.delenv(name, raising=False)
    return path


def _seed(path: Path, *entries: dict) -> None:
    """Append raw records to the ledger, as a previous run would have."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _past(tokens: int, *, resolution: str = "480p", duration: int = 5, task_id: str = "cgt-old") -> dict:
    return {
        "timestamp": "2026-09-08T20:00:00+08:00",
        "provider": "seedance",
        "model": "doubao-seedance-2-0-fast-260128",
        "resolution": resolution,
        "duration": duration,
        "task_id": task_id,
        "completion_tokens": tokens,
        "total_tokens": tokens,
    }


def _mock_client(*, usage: dict | None = None, task_id: str = _TASK_ID) -> MagicMock:
    """A mocked httpx client whose submit succeeds and whose first poll is done."""
    submit = MagicMock(spec=httpx.Response)
    submit.json.return_value = {"id": task_id}
    submit.raise_for_status.return_value = None

    payload: dict = {"id": task_id, "status": "succeeded", "content": {"video_url": _VIDEO_URL}}
    if usage is not None:
        payload["usage"] = usage
    poll = MagicMock(spec=httpx.Response)
    poll.json.return_value = payload
    poll.raise_for_status.return_value = None

    client = MagicMock(spec=httpx.Client)
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.post.return_value = submit
    client.get.return_value = poll
    return client


def _provider(monkeypatch: pytest.MonkeyPatch) -> SeedanceProvider:
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    return SeedanceProvider()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestLedgerRecording:
    def test_successful_generation_appends_a_complete_record(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400, "total_tokens": 48400})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            result = provider.generate("flying over a temple", duration=5, resolution="480p")

        assert result.video_url == _VIDEO_URL

        records = usage_ledger.read_records(ledger_file)
        assert len(records) == 1
        record = records[0]
        for field in (
            "timestamp",
            "provider",
            "model",
            "resolution",
            "duration",
            "task_id",
            "completion_tokens",
            "total_tokens",
            "price_per_mtoken",
            "estimated_cost",
            "cumulative_tokens",
            "cumulative_cost",
        ):
            assert field in record, f"ledger record is missing {field!r}"

        assert record["provider"] == "seedance"
        assert record["task_id"] == _TASK_ID
        assert record["model"] == "doubao-seedance-2-0-fast-260128"
        assert record["resolution"] == "480p"
        assert record["duration"] == 5
        assert record["completion_tokens"] == 48400
        assert record["total_tokens"] == 48400
        assert record["estimated_cost"] == pytest.approx(48400 / 1_000_000 * _PRICE)
        assert record["cumulative_tokens"] == 48400

    def test_cumulative_tokens_accumulate_across_runs(self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed(ledger_file, _past(100_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 50_000})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            provider.generate("flying")

        records = usage_ledger.read_records(ledger_file)
        assert len(records) == 2
        assert records[-1]["cumulative_tokens"] == 150_000

    def test_a_task_without_usage_writes_nothing(self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No usage block reported means no consumption to account for."""
        provider = _provider(monkeypatch)
        client = _mock_client(usage=None)

        with patch("integrations.seedance.httpx.Client", return_value=client):
            provider.generate("flying")

        assert usage_ledger.read_records(ledger_file) == []

    def test_resume_of_an_already_recorded_task_does_not_double_charge(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--resume-task polls the same task and sees the same usage block."""
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1_952_100})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            provider.resume(_TASK_ID)
            provider.resume(_TASK_ID)

        records = usage_ledger.read_records(ledger_file)
        assert len(records) == 1
        assert records[0]["total_tokens"] == 1_952_100

    def test_default_ledger_is_never_written_from_a_test_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Safety interlock: no VR180_LEDGER_PATH under pytest → no write."""
        monkeypatch.delenv(ENV_LEDGER_PATH, raising=False)
        assert usage_ledger._writes_blocked(None) is True
        assert usage_ledger.record_generation(provider="seedance", usage={"completion_tokens": 10}) is None


# ---------------------------------------------------------------------------
# Pricing: never guessed
# ---------------------------------------------------------------------------


class TestPricing:
    def test_price_is_unset_by_default(self) -> None:
        assert usage_ledger.price_per_mtoken() is None

    @pytest.mark.parametrize("bad", ["", "  ", "not-a-number", "0", "-3"])
    def test_invalid_price_is_ignored_rather_than_guessed(self, bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, bad)
        assert usage_ledger.price_per_mtoken() is None

    def test_without_a_price_tokens_are_recorded_but_no_amount_is_estimated(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
        ):
            result = provider.generate("flying")

        assert result.video_url == _VIDEO_URL
        record = usage_ledger.read_records(ledger_file)[0]
        assert record["total_tokens"] == 48400
        assert record["price_per_mtoken"] is None
        assert record["estimated_cost"] is None
        assert record["cumulative_cost"] is None
        assert "未配置单价" in caplog.text


# ---------------------------------------------------------------------------
# The gate — everything here asserts the POST never happened
# ---------------------------------------------------------------------------


class TestBudgetGate:
    def test_at_100_percent_the_request_is_refused_before_submit(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core value of this card: blocked *before* any money is spent."""
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, _past(3_000_000))  # 3M × 37/1e6 = 111 元 > 100 元
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1})

        with (
            patch("integrations.seedance.httpx.Client", return_value=client),
            pytest.raises(BudgetExceededError) as excinfo,
        ):
            provider.generate("flying", budget_cap=100.0)

        assert client.post.call_count == 0
        assert client.get.call_count == 0
        message = str(excinfo.value)
        assert "已用" in message
        assert "上限" in message
        assert "提交前" in message

    def test_the_env_var_gates_just_like_the_flag(self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        monkeypatch.setenv(ENV_BUDGET_CAP, "100")
        _seed(ledger_file, _past(3_000_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1})

        with (
            patch("integrations.seedance.httpx.Client", return_value=client),
            pytest.raises(BudgetExceededError),
        ):
            provider.generate("flying")

        assert client.post.call_count == 0

    def test_at_80_percent_it_warns_and_still_generates(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, _past(2_400_000))  # 88.80 元 of 100 → 88.8%
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
        ):
            result = provider.generate("flying", budget_cap=100.0)

        assert result.video_url == _VIDEO_URL
        assert client.post.call_count == 1
        warning = caplog.text
        assert "已用" in warning
        assert "上限" in warning
        assert "本次预估" in warning
        assert "88.80 元" in warning
        assert "100.00 元" in warning

    def test_well_under_the_cap_stays_quiet(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, _past(100_000))  # 3.70 元 of 100
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1000})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
        ):
            provider.generate("flying", budget_cap=100.0)

        assert client.post.call_count == 1
        assert "预算软闸" not in caplog.text

    def test_no_cap_configured_never_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 9_999_999})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            result = provider.generate("flying")

        assert result.video_url == _VIDEO_URL
        assert client.post.call_count == 1

    def test_resume_is_never_gated(self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recovery must stay possible when over budget — it spends nothing."""
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        monkeypatch.setenv(ENV_BUDGET_CAP, "1")
        _seed(ledger_file, _past(9_000_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1_952_100})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            result = provider.resume(_TASK_ID)

        assert result.video_url == _VIDEO_URL
        assert client.post.call_count == 0


class TestTokenModeFallback:
    def test_token_cap_blocks_when_no_price_is_configured(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_BUDGET_CAP_TOKENS, "1000000")
        _seed(ledger_file, _past(2_000_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1})

        with (
            patch("integrations.seedance.httpx.Client", return_value=client),
            pytest.raises(BudgetExceededError) as excinfo,
        ):
            provider.generate("flying")

        assert client.post.call_count == 0
        message = str(excinfo.value)
        assert "tokens" in message
        assert "已用" in message
        assert "上限" in message

    def test_token_cap_warns_at_80_percent(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(ENV_BUDGET_CAP_TOKENS, "1000000")
        _seed(ledger_file, _past(850_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1000})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
        ):
            provider.generate("flying")

        assert client.post.call_count == 1
        assert "850,000 tokens" in caplog.text
        assert "1,000,000 tokens" in caplog.text

    def test_an_amount_cap_without_a_price_falls_back_to_the_token_gate(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(ENV_BUDGET_CAP_TOKENS, "1000000")
        _seed(ledger_file, _past(2_000_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
            pytest.raises(BudgetExceededError),
        ):
            provider.generate("flying", budget_cap=100.0)

        assert client.post.call_count == 0
        assert "未配置单价" in caplog.text
        assert "token 模式" in caplog.text

    def test_an_amount_cap_without_a_price_and_without_a_token_cap_does_not_block(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No usable gate must degrade to "warn loudly", never to a false block."""
        _seed(ledger_file, _past(9_000_000))
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            result = provider.generate("flying", budget_cap=100.0)

        assert result.video_url == _VIDEO_URL
        assert client.post.call_count == 1


# ---------------------------------------------------------------------------
# A bookkeeping failure must never destroy a paid result
# ---------------------------------------------------------------------------


class TestLedgerFailureIsNonFatal:
    def test_a_write_error_does_not_fail_the_generation(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400})

        with (
            caplog.at_level(logging.WARNING, logger="integrations.usage_ledger"),
            patch("integrations.seedance.httpx.Client", return_value=client),
            patch("integrations.usage_ledger._record_generation", side_effect=OSError("No space left")),
        ):
            result = provider.generate("flying")

        assert result.video_url == _VIDEO_URL
        assert result.job_id == _TASK_ID
        assert "账本写入失败" in caplog.text

    def test_an_unwritable_ledger_path_does_not_fail_the_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parent is a regular file, so mkdir/open cannot succeed."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("i am a file", encoding="utf-8")
        monkeypatch.setenv(ENV_LEDGER_PATH, str(blocker / "usage_ledger.jsonl"))

        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            result = provider.generate("flying")

        assert result.video_url == _VIDEO_URL

    def test_a_patched_record_generation_that_raises_is_still_survivable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Belt-and-braces: the provider guards the call site too."""
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 48400})

        with (
            patch("integrations.seedance.httpx.Client", return_value=client),
            patch("integrations.usage_ledger.record_generation", side_effect=RuntimeError("boom")),
        ):
            result = provider.generate("flying")

        assert result.video_url == _VIDEO_URL

    def test_a_corrupt_ledger_line_is_skipped_not_fatal(self, ledger_file: Path) -> None:
        ledger_file.parent.mkdir(parents=True, exist_ok=True)
        ledger_file.write_text(
            json.dumps(_past(1000)) + "\n{not json at all\n" + json.dumps(_past(2000)) + "\n",
            encoding="utf-8",
        )
        records = usage_ledger.read_records(ledger_file)
        assert len(records) == 2
        assert usage_ledger.total_tokens(records) == 3000

    def test_a_missing_ledger_reads_as_empty(self, tmp_path: Path) -> None:
        assert usage_ledger.read_records(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# --summary
# ---------------------------------------------------------------------------


class TestSummaryCli:
    def test_summary_reports_totals_and_the_last_ten_records(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, *[_past(100_000, task_id=f"cgt-{i:02d}") for i in range(12)])

        assert usage_ledger.main(["--summary"]) == 0
        out = capsys.readouterr().out

        assert str(ledger_file) in out
        assert "记录条数 : 12" in out
        assert "1,200,000" in out
        assert "44.40 元" in out  # 1.2M × 37/1e6
        assert "最近 10 条记录" in out
        assert "cgt-11" in out
        assert "cgt-00" not in out  # trimmed by the 10-record window

    def test_summary_says_so_when_no_price_is_configured(
        self, ledger_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed(ledger_file, _past(48_400))
        assert usage_ledger.main([]) == 0
        out = capsys.readouterr().out
        assert "未配置单价" in out
        assert "48,400" in out

    def test_summary_on_an_empty_ledger_is_still_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert usage_ledger.main(["--summary"]) == 0
        out = capsys.readouterr().out
        assert "账本为空" in out
        assert "未配置" in out

    def test_summary_reports_the_configured_cap(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        monkeypatch.setenv(ENV_BUDGET_CAP, "100")
        _seed(ledger_file, _past(2_400_000))

        usage_ledger.main(["--summary"])
        out = capsys.readouterr().out
        assert "100.00 元" in out
        assert "88.8%" in out

    def test_summary_honours_an_explicit_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        other = tmp_path / "elsewhere.jsonl"
        _seed(other, _past(777))
        usage_ledger.main(["--summary", "--path", str(other)])
        assert "777" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI wiring (scripts/generate.py)
# ---------------------------------------------------------------------------


class TestGenerateCliBudgetFlag:
    def test_budget_cap_refuses_before_submit_and_exits_non_zero(
        self, tmp_path: Path, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, _past(3_000_000))
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        client = _mock_client(usage={"completion_tokens": 1})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "flying over a temple",
                    "--provider",
                    "seedance",
                    "--budget-cap",
                    "100",
                    "--output",
                    str(tmp_path / "out.mp4"),
                ]
            )

        assert rc != 0
        assert client.post.call_count == 0
        assert not (tmp_path / "out.mp4").exists()

    def test_a_run_under_the_cap_still_succeeds(
        self, tmp_path: Path, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_PRICE_PER_MTOKEN, str(_PRICE))
        _seed(ledger_file, _past(100_000))
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        client = _mock_client(usage={"completion_tokens": 48400})
        client.get.return_value.json.return_value["content"]["video_url"] = str(source)

        with patch("integrations.seedance.httpx.Client", return_value=client):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "flying over a temple",
                    "--provider",
                    "seedance",
                    "--budget-cap",
                    "100",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        assert client.post.call_count == 1
        assert out.read_bytes() == b"fake-mp4"
        assert len(usage_ledger.read_records(ledger_file)) == 2

    def test_a_non_positive_budget_cap_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        import scripts.generate as gen

        assert gen.main(["fly", "--provider", "seedance", "--budget-cap", "0"]) == 2

    def test_the_flag_is_never_sent_in_the_request_body(
        self, ledger_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--budget-cap is a local guard rail, not an Ark body field."""
        provider = _provider(monkeypatch)
        client = _mock_client(usage={"completion_tokens": 1000})

        with patch("integrations.seedance.httpx.Client", return_value=client):
            provider.generate("flying", budget_cap=1000.0, budget_cap_tokens=1_000_000)

        body = client.post.call_args[1]["json"]
        assert "budget_cap" not in body
        assert "budget_cap_tokens" not in body


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestEstimation:
    def test_estimate_prefers_the_same_tier(self) -> None:
        records = [
            _past(50_000, resolution="480p", duration=5),
            _past(70_000, resolution="480p", duration=5),
            _past(1_952_100, resolution="4k", duration=10),
        ]
        estimate = usage_ledger.estimate_next_tokens(
            records,
            model="doubao-seedance-2-0-fast-260128",
            resolution="480p",
            duration=5,
        )
        assert estimate == pytest.approx(60_000)

    def test_estimate_is_none_without_history(self) -> None:
        assert usage_ledger.estimate_next_tokens([], resolution="480p", duration=5) is None

    def test_check_budget_is_off_without_a_cap(self, ledger_file: Path) -> None:
        _seed(ledger_file, _past(1000))
        status = usage_ledger.check_budget()
        assert status.mode == usage_ledger.MODE_OFF
        assert status.cap is None
        assert status.blocked is False
        assert status.warn is False
        assert status.used_tokens == 1000
