"""End-to-end tests for scripts/generate.py, including the --image path.

Uses the mock provider so no network, keys, or models are needed.  All
artifacts land under ``tmp_path`` via the MOCK_PROVIDER_OUTPUT_DIR env var.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _probe(out_path):
    import json

    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    assert r.returncode == 0, f"ffprobe failed: {r.stderr}"
    return json.loads(r.stdout)


class TestGenerateCliImageMode:
    def test_image_mode_with_mock_provider(self, tmp_path, monkeypatch) -> None:
        """``generate.py --image x.png --provider mock`` produces a playable mp4."""
        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path / "out"))

        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNGfake")
        out = tmp_path / "out.mp4"

        cmd = [
            sys.executable,
            "-m",
            "scripts.generate",
            "--image",
            str(img),
            "--provider",
            "mock",
            "--duration",
            "1",
            "--output",
            str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 0, f"CLI exited {r.returncode}; stdout={r.stdout}\nstderr={r.stderr}"

        assert out.exists(), "No mp4 artifact produced at --output path"
        assert "Video saved to" in r.stdout
        info = _probe(str(out))
        assert info.get("format")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))
        assert r.stdout.find("Provider: mock") >= 0

    def test_image_mode_no_prompt_still_works(self, tmp_path, monkeypatch) -> None:
        """--image with no prompt falls back to an empty prompt rather than erroring."""
        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path / "out"))

        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNGfake")
        out = tmp_path / "out.mp4"

        cmd = [
            sys.executable,
            "-m",
            "scripts.generate",
            "--image",
            str(img),
            "--provider",
            "mock",
            "--duration",
            "1",
            "--output",
            str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"

    def test_text_mode_requires_prompt(self, tmp_path, monkeypatch) -> None:
        """Without --image, a prompt is required; calling with no prompt errors.

        The mock provider requires no env, and produces a real file, so the
        only failure mode here is the prompt-missing guard (exit code 2).
        """
        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path / "out"))

        cmd = [sys.executable, "-m", "scripts.generate", "--provider", "mock", "--duration", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert r.returncode != 0
        combined = (r.stdout + r.stderr).lower()
        assert "prompt" in combined


def _mock_httpx_client():
    """Return a MagicMock httpx.Client usable as both value and context manager."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.__enter__.return_value = mock_client
    return mock_client


class _ExpiredClock:
    """Stand-in for ``integrations.seedance.time`` that is always out of budget.

    The first reading is the poll loop's start stamp; every later reading is
    far past any deadline, so the loop times out on its first check without a
    real wait.  Patching the *module attribute* (rather than ``time.time``
    itself) keeps the fake clock away from ``logging``, which also reads the
    wall clock and would otherwise consume the fake readings.
    """

    def __init__(self) -> None:
        self._readings = 0

    def time(self) -> float:
        self._readings += 1
        return 1.0 if self._readings == 1 else 1e9

    def sleep(self, _seconds: float) -> None:
        return None


class TestGenerateCliGenTierPassthrough:
    """H-2: --gen-resolution / --gen-ratio / --duration reach the seedance
    request body (CLI → provider kwargs → HTTP body). No real network: the
    httpx.Client is patched and the poll returns a local file so the download
    step copies a real artefact rather than fetching.
    """

    def _seedance_httpx(self, video_path: str):
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-cli-passthrough-01"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-cli-passthrough-01",
            "status": "succeeded",
            "content": {"video_url": video_path},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp
        return mock_client

    def test_high_tier_reaches_request_body(self, tmp_path, monkeypatch) -> None:
        """--gen-resolution 720p --gen-ratio 16:9 --duration 8 land in the body."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        # The poll returns a local path so _download_video copies it (no network).
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--duration",
                    "8",
                    "--gen-resolution",
                    "720p",
                    "--gen-ratio",
                    "16:9",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["resolution"] == "720p"
        assert body["ratio"] == "16:9"
        assert body["duration"] == 8
        assert out.exists()

    def test_defaults_are_480p_5s_adaptive(self, tmp_path, monkeypatch) -> None:
        """With no --gen-* flags the body still reflects 480p / 5s / adaptive."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["resolution"] == "480p"
        assert body["ratio"] == "adaptive"
        assert body["duration"] == 5

    def test_parser_defaults(self) -> None:
        """The argparse defaults keep the quota discipline (480p / 5s / adaptive / fast)."""
        import scripts.generate as gen

        args = gen.build_parser().parse_args(["--image", "x.png", "--provider", "mock"])
        assert args.gen_resolution == "480p"
        assert args.gen_ratio == "adaptive"
        assert args.duration == 5

        # P-1 (#246): default model stays the fast (low-cost) variant —
        # existing behaviour is unchanged.
        from integrations.seedance import MODEL_FAST

        assert args.model == MODEL_FAST

    def test_no_model_flag_uses_fast(self, tmp_path, monkeypatch) -> None:
        """Regression: without --model the body still carries the fast model."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == "doubao-seedance-2-0-fast-260128"


class TestGenerateCliSeedanceModelResolution:
    """P-1 (#246): --model unlocks 4k; model/resolution mismatches fail fast.

    Every HTTP call is mocked; the poll returns a *local* file path so the
    download step copies a real artefact rather than hitting the network.
    """

    def _seedance_httpx(self, video_path: str):
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-cli-model-01"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-cli-model-01",
            "status": "succeeded",
            "content": {"video_url": video_path},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp
        return mock_client

    def test_std_model_4k_1x1_reaches_request_body(self, tmp_path, monkeypatch) -> None:
        """--model <std> + --gen-resolution 4k + --gen-ratio 1:1 → correct body."""
        from integrations.seedance import MODEL_STD

        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--model",
                    MODEL_STD,
                    "--gen-resolution",
                    "4k",
                    "--gen-ratio",
                    "1:1",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == MODEL_STD
        assert body["resolution"] == "4k"
        assert body["ratio"] == "1:1"
        assert out.exists()

    def test_fast_model_with_4k_fails_before_any_request(self, tmp_path, monkeypatch) -> None:
        """fast + 4k → the reason is explained and NO HTTP request is sent."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--gen-resolution",
                    "4k",
                    "--output",
                    str(out),
                ]
            )

        assert rc != 0
        # The pre-flight error must fire before any HTTP traffic.
        assert mock_client.post.call_count == 0

    def test_fast_model_with_4k_error_names_fast(
        self, caplog: pytest.LogCaptureFixture, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error text names that fast cannot do 4k, not just the code."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
            caplog.at_level(logging.ERROR),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--gen-resolution",
                    "4k",
                    "--output",
                    str(out),
                ]
            )

        assert rc != 0
        msg = "\n".join(record.message for record in caplog.records)
        assert "fast" in msg.lower()
        assert "4k" in msg.lower()

    def test_duration_range_rejected_by_seedance(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--duration outside [4, 15] with --provider seedance is rejected."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            caplog.at_level(logging.ERROR),
        ):
            import scripts.generate as gen

            rc3 = gen.main(
                [
                    "p",
                    "--provider",
                    "seedance",
                    "--duration",
                    "3",
                    "--output",
                    "/tmp/out.mp4",
                ]
            )
            rc20 = gen.main(
                [
                    "p",
                    "--provider",
                    "seedance",
                    "--duration",
                    "20",
                    "--output",
                    "/tmp/out.mp4",
                ]
            )

        assert rc3 != 0
        assert rc20 != 0
        msg = "\n".join(record.message for record in caplog.records)
        assert "4" in msg and "15" in msg
        # The check fires before any provider is constructed — no HTTP.
        assert mock_client.__enter__.call_count == 0

    def test_duration_inside_range_accepted(self, monkeypatch, tmp_path) -> None:
        """--duration 10 passes with seedance; 4 and 15 are the valid edges."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "p",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--duration",
                    "10",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["duration"] == 10

    def test_gen_ratio_choices_rejected(self) -> None:
        """--gen-ratio with an unknown value is rejected by argparse."""
        import scripts.generate as gen

        parser = gen.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["p", "--gen-ratio", "4:3"])
        assert parser.parse_args(["p", "--gen-ratio", "1:1"]).gen_ratio == "1:1"

    def test_passthrough_flags_reach_body_when_set(self, tmp_path, monkeypatch) -> None:
        """--draft / --return-last-frame / --seed / --camera-fixed land in the body."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--draft",
                    "--return-last-frame",
                    "--seed",
                    "42",
                    "--camera-fixed",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["draft"] is True
        assert body["return_last_frame"] is True
        assert body["seed"] == 42
        assert body["camera_fixed"] is True

    def test_passthrough_flags_absent_from_body_by_default(self, tmp_path, monkeypatch) -> None:
        """Regression: without the flags the request body carries none of them."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert "draft" not in body
        assert "return_last_frame" not in body
        assert "seed" not in body
        assert "camera_fixed" not in body
        # The pre-#246 five-key shape is still the default.
        assert set(body) == {"model", "content", "resolution", "ratio", "duration"}

    def test_aspect_ratio_reaches_ratio_field(self, tmp_path, monkeypatch) -> None:
        """--aspect-ratio is no longer a silent no-op: it maps onto body["ratio"]."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "pan left",
                    "--image",
                    "https://example.com/p.png",
                    "--provider",
                    "seedance",
                    "--aspect-ratio",
                    "9:16",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        body = mock_client.post.call_args[1]["json"]
        assert body["ratio"] == "9:16"


class TestGenerateCliResumeTask:
    """W-9 (#325): ``--resume-task`` retrieves a task without re-submitting it.

    The production incident: a 4k/10s run timed out client-side at the
    hard-coded 300s while the Ark task kept running.  1,952,100 tokens were
    already spent, and the error told the operator to re-run — which would
    have burned a second copy.  ``--resume-task`` is the way out, so the
    load-bearing assertion here is **zero POST calls**.
    """

    def _resume_httpx(self, video_path: str, task_id: str = "cgt-20260908185831-69srb"):
        """A mocked client whose GET reports the task as already succeeded."""
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": task_id,
            "status": "succeeded",
            "content": {"video_url": video_path},
            "usage": {"completion_tokens": 1952100},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.get.return_value = poll_resp
        return mock_client

    def test_resume_task_downloads_without_submitting(self, tmp_path, monkeypatch) -> None:
        """--resume-task polls + downloads and sends NO submit request."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "recovered.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._resume_httpx(str(src))
        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            import scripts.generate as gen

            rc = gen.main(
                [
                    "--provider",
                    "seedance",
                    "--resume-task",
                    "cgt-20260908185831-69srb",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        # The whole point of the flag: no submission, so no extra quota.
        assert mock_client.post.call_count == 0
        assert mock_client.get.call_count == 1
        assert mock_client.get.call_args[0][0] == "/contents/generations/tasks/cgt-20260908185831-69srb"
        assert out.exists()
        assert out.read_bytes() == b"fake-mp4"

    def test_resume_task_needs_no_prompt(self, tmp_path, monkeypatch) -> None:
        """Resuming without a prompt must not trip the 'prompt required' guard."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "recovered.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._resume_httpx(str(src))
        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            import scripts.generate as gen

            rc = gen.main(["--provider", "seedance", "--resume-task", "cgt-x", "--output", str(out)])

        assert rc == 0
        assert mock_client.post.call_count == 0

    def test_resume_task_prints_the_task_id(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "recovered.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        mock_client = self._resume_httpx(str(src), task_id="cgt-resume-print")
        with patch("integrations.seedance.httpx.Client", return_value=mock_client):
            import scripts.generate as gen

            rc = gen.main(["--provider", "seedance", "--resume-task", "cgt-resume-print", "--output", str(out)])

        assert rc == 0
        assert "cgt-resume-print" in capsys.readouterr().out

    def test_resume_task_still_running_is_polled_then_delivered(self, tmp_path, monkeypatch) -> None:
        """A task that is still running when resumed is waited on, not dropped."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")

        src = tmp_path / "recovered.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        running = MagicMock(spec=httpx.Response)
        running.json.return_value = {"id": "cgt-x", "status": "running"}
        running.raise_for_status.return_value = None
        done = MagicMock(spec=httpx.Response)
        done.json.return_value = {"id": "cgt-x", "status": "succeeded", "content": {"video_url": str(src)}}
        done.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.get.side_effect = [running, done]

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
        ):
            import scripts.generate as gen

            rc = gen.main(["--provider", "seedance", "--resume-task", "cgt-x", "--output", str(out)])

        assert rc == 0
        assert mock_client.post.call_count == 0
        assert mock_client.get.call_count == 2
        assert out.exists()

    def test_resume_task_failure_is_reported_not_crashed(
        self, caplog: pytest.LogCaptureFixture, tmp_path, monkeypatch
    ) -> None:
        """An unknown task id exits non-zero with a message, not a traceback."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        err_resp = MagicMock(spec=httpx.Response)
        err_resp.status_code = 404
        err_resp.json.return_value = {"error": {"code": "ResourceNotFound", "message": "task not found"}}
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=err_resp
        )

        mock_client = _mock_httpx_client()
        mock_client.get.return_value = err_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            caplog.at_level(logging.ERROR),
        ):
            import scripts.generate as gen

            rc = gen.main(["--provider", "seedance", "--resume-task", "cgt-nope", "--output", str(out)])

        assert rc != 0
        assert mock_client.post.call_count == 0
        assert "ResourceNotFound" in "\n".join(record.getMessage() for record in caplog.records)
        assert not out.exists()

    def test_resume_task_on_unsupported_provider_is_rejected(
        self, caplog: pytest.LogCaptureFixture, tmp_path, monkeypatch
    ) -> None:
        """--resume-task with a provider that has no resume() names seedance."""
        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path / "out"))
        out = tmp_path / "out.mp4"

        with caplog.at_level(logging.ERROR):
            import scripts.generate as gen

            rc = gen.main(["--provider", "mock", "--resume-task", "cgt-x", "--output", str(out)])

        assert rc == 2
        assert "seedance" in "\n".join(record.getMessage() for record in caplog.records).lower()
        assert not out.exists()

    def test_parser_defaults_have_no_resume_or_timeout(self) -> None:
        """Both new flags are opt-in; the default behaviour is untouched."""
        import scripts.generate as gen

        args = gen.build_parser().parse_args(["p", "--provider", "seedance"])
        assert args.resume_task is None
        assert args.gen_timeout is None


class TestGenerateCliGenTimeout:
    """W-9 (#325): --gen-timeout overrides the computed poll budget."""

    def _seedance_httpx(self, video_path: str):
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-cli-timeout-01"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {
            "id": "cgt-cli-timeout-01",
            "status": "succeeded",
            "content": {"video_url": video_path},
        }
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp
        return mock_client

    def _run_and_capture_budget(self, argv: list[str], tmp_path) -> tuple[int, int]:
        """Run the CLI with a spied poll loop; return (exit code, poll budget)."""
        from integrations.seedance import SeedanceProvider

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")

        captured: dict[str, int] = {}
        original = SeedanceProvider._poll_task

        def _spy(self, client, task_id, headers, timeout_seconds):
            captured["timeout_seconds"] = timeout_seconds
            return original(self, client, task_id, headers, timeout_seconds)

        mock_client = self._seedance_httpx(str(src))
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time.sleep", return_value=None),
            patch.object(SeedanceProvider, "_poll_task", _spy),
        ):
            import scripts.generate as gen

            rc = gen.main(argv)
        return rc, captured["timeout_seconds"]

    def test_gen_timeout_overrides_the_computed_budget(self, tmp_path, monkeypatch) -> None:
        from integrations.seedance import MODEL_STD

        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        rc, budget = self._run_and_capture_budget(
            [
                "pan left",
                "--provider",
                "seedance",
                "--model",
                MODEL_STD,
                "--gen-resolution",
                "4k",
                "--duration",
                "10",
                "--gen-timeout",
                "77",
                "--output",
                str(out),
            ],
            tmp_path,
        )
        assert rc == 0
        assert budget == 77

    def test_4k_10s_without_the_flag_scales_past_1800s(self, tmp_path, monkeypatch) -> None:
        """The incident combination must no longer be capped at 300s."""
        from integrations.seedance import MODEL_STD

        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        rc, budget = self._run_and_capture_budget(
            [
                "pan left",
                "--provider",
                "seedance",
                "--model",
                MODEL_STD,
                "--gen-resolution",
                "4k",
                "--duration",
                "10",
                "--output",
                str(out),
            ],
            tmp_path,
        )
        assert rc == 0
        assert budget >= 1800

    def test_default_run_keeps_the_300s_budget(self, tmp_path, monkeypatch) -> None:
        """Regression: 480p/5s is still exactly 300s."""
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        rc, budget = self._run_and_capture_budget(
            ["pan left", "--provider", "seedance", "--output", str(out)],
            tmp_path,
        )
        assert rc == 0
        assert budget == 300

    def test_non_positive_gen_timeout_is_rejected(
        self, caplog: pytest.LogCaptureFixture, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        mock_client = _mock_httpx_client()
        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            caplog.at_level(logging.ERROR),
        ):
            import scripts.generate as gen

            rc = gen.main(
                ["p", "--provider", "seedance", "--gen-timeout", "0", "--output", str(out)],
            )

        assert rc == 2
        assert "gen-timeout" in "\n".join(record.getMessage() for record in caplog.records)
        assert mock_client.post.call_count == 0


class TestGenerateCliTimeoutGuidance:
    """W-9 (#325): what the operator reads when the client gives up.

    Asserting the *wording* rather than the exception type is deliberate — the
    text is the control that stops somebody re-running a 1.95M-token job.
    """

    def test_timeout_error_tells_the_operator_not_to_regenerate(
        self, caplog: pytest.LogCaptureFixture, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("ARK_API_KEY", "test-key")
        out = tmp_path / "out.mp4"

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.json.return_value = {"id": "cgt-20260908185831-69srb"}
        submit_resp.raise_for_status.return_value = None

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.json.return_value = {"id": "cgt-20260908185831-69srb", "status": "running"}
        poll_resp.raise_for_status.return_value = None

        mock_client = _mock_httpx_client()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch("integrations.seedance.httpx.Client", return_value=mock_client),
            patch("integrations.seedance.time", _ExpiredClock()),
            caplog.at_level(logging.ERROR),
        ):
            import scripts.generate as gen

            rc = gen.main(["p", "--provider", "seedance", "--output", str(out)])

        assert rc != 0
        msg = "\n".join(record.getMessage() for record in caplog.records)
        assert "cgt-20260908185831-69srb" in msg
        assert "不要重新生成" in msg
        assert "--resume-task cgt-20260908185831-69srb" in msg
        assert not out.exists()
