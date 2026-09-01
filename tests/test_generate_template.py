"""Tests for the --template / --template-field / --list-templates wiring
in ``scripts/generate.py`` (issue #195).

All tests inject a fake provider (``MagicMock``) — no real generation API is
called, no quota is spent (CLAUDE.md red line).  The fake provider's
``generate`` return value points ``video_url`` at a local file so the
download step copies bytes instead of fetching over the network.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from pipeline.prompt_library import TEMPLATES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_provider(video_path: str) -> MagicMock:
    """A provider mock whose ``generate`` returns a result pointing at a local file.

    The returned object records the kwargs passed to ``generate`` via the
    normal MagicMock call-args machinery, so tests can assert on the prompt
    that reached the provider.
    """
    result = MagicMock()
    result.video_url = video_path
    result.provider = "fake"
    result.job_id = "fake-job"

    provider = MagicMock()
    provider.generate.return_value = result
    provider.generate_from_image.return_value = result
    return provider


def _patch_get_provider(provider: MagicMock):
    """Patch scripts.generate.get_provider to return *provider*."""
    return patch("scripts.generate.get_provider", return_value=provider)


# ---------------------------------------------------------------------------
# --list-templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_list_templates_prints_all_keys_and_exits_zero(self, capsys) -> None:
        """``--list-templates`` lists every key and returns 0 without calling a provider."""
        import scripts.generate as gen

        provider = _make_fake_provider("ignored")
        with _patch_get_provider(provider):
            rc = gen.main(["--list-templates"])

        assert rc == 0
        out = capsys.readouterr().out
        for key in TEMPLATES:
            assert key in out, f"template key {key!r} missing from --list-templates output"
        # provider must NOT have been touched — no quota, no generation call
        provider.generate.assert_not_called()
        provider.generate_from_image.assert_not_called()

    def test_list_templates_real_subprocess(self) -> None:
        """Run ``python -m scripts.generate --list-templates`` as a real subprocess."""
        r = subprocess.run(
            [sys.executable, "-m", "scripts.generate", "--list-templates"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        for key in TEMPLATES:
            assert key in r.stdout, f"template key {key!r} missing from subprocess output"


# ---------------------------------------------------------------------------
# --template renders into the provider prompt
# ---------------------------------------------------------------------------


class TestTemplateRender:
    def test_template_body_reaches_provider(self, tmp_path) -> None:
        """``--template <key> --template-field ...`` puts template body features in the prompt."""
        import scripts.generate as gen

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        provider = _make_fake_provider(str(src))
        with _patch_get_provider(provider):
            rc = gen.main(
                [
                    "--template",
                    "slow_dolly_in",
                    "--template-field",
                    "subject=a sailboat",
                    "--template-field",
                    "setting=a calm harbour at dawn",
                    "--provider",
                    "mock",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        # the rendered body must carry both the supplied values and the
        # constraint backbone that makes the shot VR180-friendly
        sent_prompt = provider.generate.call_args.kwargs["prompt"]
        assert "a sailboat" in sent_prompt
        assert "a calm harbour at dawn" in sent_prompt
        # characteristic template-body phrase (slow constant-speed dolly)
        assert "constant-speed" in sent_prompt
        assert "no cuts" in sent_prompt


# ---------------------------------------------------------------------------
# --template + --target-aware composition
# ---------------------------------------------------------------------------


class TestTemplateAndTargetAware:
    def test_template_body_and_wrap_suffix_both_present(self, tmp_path) -> None:
        """With --template + --target-aware, both template body and wrap suffix survive."""
        import scripts.generate as gen

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-mp4")
        out = tmp_path / "out.mp4"

        provider = _make_fake_provider(str(src))
        with _patch_get_provider(provider):
            rc = gen.main(
                [
                    "--template",
                    "orbit_left",
                    "--template-field",
                    "subject=a marble statue",
                    "--template-field",
                    "setting=a sunlit plaza",
                    "--target-aware",
                    "--scene",
                    "orbit",
                    "--provider",
                    "mock",
                    "--output",
                    str(out),
                ]
            )

        assert rc == 0
        sent_prompt = provider.generate.call_args.kwargs["prompt"]
        # template-body features survive (the layer that decides "what to say")
        assert "a marble statue" in sent_prompt
        assert "a sunlit plaza" in sent_prompt
        assert "constant-speed" in sent_prompt
        # wrap suffix survives (the layer that appends target-projection constraints)
        # wrap_prompt's positive = <prompt>, <scene motion...>, <composition...>, <quality...>
        # the orbit scene composition phrase "main subject at centre of frame" is a wrap addition
        assert "main subject" in sent_prompt
        # negative prompt is a separate kwarg produced by wrapping
        sent_kwargs = provider.generate.call_args.kwargs
        assert sent_kwargs.get("negative_prompt")


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------


class TestTemplateErrors:
    def test_missing_placeholder_exits_nonzero_and_prints_message(self, tmp_path, caplog) -> None:
        """A missing placeholder must exit non-zero and print the PromptLibraryError message."""
        import scripts.generate as gen

        out = tmp_path / "out.mp4"
        provider = _make_fake_provider("ignored")
        with _patch_get_provider(provider), caplog.at_level("ERROR"):
            rc = gen.main(
                [
                    "--template",
                    "slow_dolly_in",
                    "--template-field",
                    "subject=a sailboat",
                    "--provider",
                    "mock",
                    "--output",
                    str(out),
                ]
            )

        assert rc != 0
        # the error log must mention the missing field (PromptLibraryError names it)
        combined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "setting" in combined, f"missing-field name not surfaced in logs: {combined!r}"
        # provider must not have been called — we failed before generation
        provider.generate.assert_not_called()

    def test_template_and_positional_prompt_mutually_exclusive(self, tmp_path, caplog) -> None:
        """Giving both --template and a positional prompt exits non-zero."""
        import scripts.generate as gen

        out = tmp_path / "out.mp4"
        provider = _make_fake_provider("ignored")
        with _patch_get_provider(provider), caplog.at_level("ERROR"):
            rc = gen.main(
                [
                    "fly over mountains",
                    "--template",
                    "slow_dolly_in",
                    "--template-field",
                    "subject=a sailboat",
                    "--template-field",
                    "setting=a harbour",
                    "--provider",
                    "mock",
                    "--output",
                    str(out),
                ]
            )

        assert rc != 0
        provider.generate.assert_not_called()

    def test_malformed_template_field_exits_nonzero_and_names_it(self, tmp_path, caplog) -> None:
        """``--template-field subject`` (missing ``=``) exits non-zero and points at the bad item."""
        import scripts.generate as gen

        out = tmp_path / "out.mp4"
        provider = _make_fake_provider("ignored")
        with _patch_get_provider(provider), caplog.at_level("ERROR"):
            rc = gen.main(
                [
                    "--template",
                    "slow_dolly_in",
                    "--template-field",
                    "subject",
                    "--template-field",
                    "setting=a harbour",
                    "--provider",
                    "mock",
                    "--output",
                    str(out),
                ]
            )

        assert rc != 0
        combined = " ".join(rec.getMessage() for rec in caplog.records)
        # the offending item must be named in the error
        assert "subject" in combined, f"malformed item not named in logs: {combined!r}"
        assert "NAME=VALUE" in combined
        provider.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Parser-level unit checks
# ---------------------------------------------------------------------------


class TestParserTemplateArgs:
    def test_template_field_is_append_action(self) -> None:
        import scripts.generate as gen

        args = gen.build_parser().parse_args(
            [
                "--template",
                "slow_dolly_in",
                "--template-field",
                "subject=x",
                "--template-field",
                "setting=y",
            ]
        )
        assert args.template == "slow_dolly_in"
        assert args.template_field == ["subject=x", "setting=y"]

    def test_no_template_flags_defaults(self) -> None:
        import scripts.generate as gen

        args = gen.build_parser().parse_args(["a prompt"])
        assert args.template is None
        assert args.template_field is None
        assert args.list_templates is False

    def test_parse_template_fields_ok(self) -> None:
        import scripts.generate as gen

        fields = gen._parse_template_fields(["subject=a", "setting=b"])
        assert fields == {"subject": "a", "setting": "b"}

    def test_parse_template_fields_missing_equals(self) -> None:
        import scripts.generate as gen

        with pytest.raises(ValueError, match="subject"):
            gen._parse_template_fields(["subject"])
