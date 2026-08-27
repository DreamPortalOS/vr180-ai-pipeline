"""End-to-end tests for scripts/generate.py, including the --image path.

Uses the mock provider so no network, keys, or models are needed.  All
artifacts land under ``tmp_path`` via the MOCK_PROVIDER_OUTPUT_DIR env var.
"""

from __future__ import annotations

import subprocess
import sys


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
