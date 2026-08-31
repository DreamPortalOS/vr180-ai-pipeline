"""End-to-end tests for scripts/generate.py, including the --image path.

Uses the mock provider so no network, keys, or models are needed.  All
artifacts land under ``tmp_path`` via the MOCK_PROVIDER_OUTPUT_DIR env var.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx


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
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == 0, f"CLI exited {r.returncode}; stdout={r.stdout}\nstderr={r.stderr}"

        # The CLI prints the saved path; grab the produced mp4 from tmp_path.
        mp4s = list(tmp_path.glob("**/*.mp4"))
        assert mp4s, "No mp4 artifact produced under tmp_path"
        produced = mp4s[0]
        assert "Video saved to" in r.stdout
        info = _probe(str(produced))
        assert info.get("format")
        assert any(s.get("codec_type") == "video" for s in info.get("streams", []))
        assert r.stdout.find("Provider: mock") >= 0

    def test_image_mode_no_prompt_still_works(self, tmp_path, monkeypatch) -> None:
        """--image with no prompt falls back to an empty prompt rather than erroring."""
        monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path / "out"))

        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNGfake")

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
        """The argparse defaults keep the quota discipline (480p / 5s / adaptive)."""
        import scripts.generate as gen

        args = gen.build_parser().parse_args(["--image", "x.png", "--provider", "mock"])
        assert args.gen_resolution == "480p"
        assert args.gen_ratio == "adaptive"
        assert args.duration == 5
