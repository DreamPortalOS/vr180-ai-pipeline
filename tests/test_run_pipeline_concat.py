"""C-1b (#191): --inputs concat pre-stage wiring into run_pipeline.

All tests are fully mocked — ``concat_segments`` / ``probe_segment`` are
monkeypatched in ``pipeline.segment_concat``, and the rest of ``main()``
is stubbed after the concat pre-stage so no real ffmpeg, depth model, or
stereo renderer ever runs.  These are wiring tests: they assert the CLI
argument plumbing and the order/value of calls into the concat layer,
not the concat layer itself (that is owned by
``tests/test_segment_concat.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_pipeline as rp  # noqa: E402

from pipeline.segment_concat import ConcatError  # noqa: E402

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_inputs_accepted(self) -> None:
        args = rp.parse_args(["--inputs", "a.mp4", "b.mp4", "-o", "out.mp4"])
        assert args.inputs == ["a.mp4", "b.mp4"]
        assert args.input is None
        assert args.concat_crossfade == 0.0
        assert args.concat_mode == "demux"

    def test_crossfade_default_zero(self) -> None:
        args = rp.parse_args(["--inputs", "a.mp4", "b.mp4", "-o", "out.mp4"])
        assert args.concat_crossfade == 0.0

    def test_mode_default_demux(self) -> None:
        args = rp.parse_args(["--inputs", "a.mp4", "b.mp4", "-o", "out.mp4"])
        assert args.concat_mode == "demux"

    def test_input_and_inputs_mutually_exclusive(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            rp.parse_args(["--input", "a.mp4", "--inputs", "b.mp4", "c.mp4", "-o", "out.mp4"])
        assert exc_info.value.code != 0

    def test_neither_input_nor_inputs_exits_in_main(self, capsys, monkeypatch) -> None:
        # argparse only enforces "not both" via the mutual-exclusion group;
        # the "at least one" guard lives in main().  Stub out the rest of
        # main after the guard so it never touches real pipelines.  We also
        # override parse_args so main() does not read sys.argv (pytest args).
        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=None,
                concat_crossfade=0.0,
                concat_mode="demux",
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "either --input/-i or --inputs is required" in (captured.out + captured.err)


# ---------------------------------------------------------------------------
# Concat pre-stage wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_concat(tmp_path, monkeypatch):
    """Inject fake concat_segments + probe_segment + downstream stubs.

    Returns a dict with the call log so tests can assert order/values.
    """
    calls = {"concat": [], "probe": []}

    def _fake_concat_segments(segments, output_path, *, mode="demux", crossfade=0.0, **kw):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake concat video")
        calls["concat"].append(
            {
                "segments": [str(Path(s.path)) for s in segments],
                "output_path": str(out),
                "mode": mode,
                "crossfade": crossfade,
            }
        )
        return out

    def _fake_probe_segment(path):
        calls["probe"].append(str(Path(path)))
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.5, "has_audio": True}

    monkeypatch.setattr(rp, "concat_segments", _fake_concat_segments)
    # probe_segment is re-imported locally inside _concat_segments_preprocess,
    # so patch it at the source module to prevent any real ffprobe subprocess.
    import pipeline.segment_concat as _sc

    monkeypatch.setattr(_sc, "probe_segment", _fake_probe_segment)
    monkeypatch.setattr(rp, "apply_quality_preset", lambda args: None)
    monkeypatch.setattr(rp, "_apply_comfort_preset", lambda args: None)

    # Short-circuit main() after the concat pre-stage: everything that follows
    # (fps inheritance, SBS detection, stage loop) touches real video files /
    # models.  Raising SystemExit(0) here lets the wiring tests assert the
    # concat call log and then exit cleanly.  Consumers that need main() to
    # continue should NOT use this fixture.
    def _stop_after_concat(args):
        raise SystemExit(0)

    monkeypatch.setattr(rp, "_manifest_prepare", _stop_after_concat)
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    return calls, tmp_path


class TestConcatWiring:
    def test_concat_called_once_in_command_order(self, fake_concat, monkeypatch, capsys) -> None:
        calls, _ = fake_concat

        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["seg1.mp4", "seg2.mp4", "seg3.mp4"],
                concat_crossfade=0.0,
                concat_mode="demux",
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 0
        assert len(calls["concat"]) == 1
        segs = calls["concat"][0]["segments"]
        # The order must match the command line.  Path form (absolute vs
        # relative) depends on how concat_segments was invoked, so compare
        # the leaf name in sequence to assert ordering without a cwd coupling.
        assert [Path(s).name for s in segs] == ["seg1.mp4", "seg2.mp4", "seg3.mp4"]

    def test_concat_output_passed_as_input(self, fake_concat, monkeypatch) -> None:
        _calls, tmp_path = fake_concat

        received_input = {}

        def _apply_quality_preset(args):
            received_input["input"] = args.input
            received_input["output"] = args.output

        monkeypatch.setattr(rp, "apply_quality_preset", _apply_quality_preset)
        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["a.mp4", "b.mp4"],
                concat_crossfade=0.0,
                concat_mode="demux",
                temp_dir=str(tmp_path),
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 0
        assert received_input["input"] is not None
        # The intermediate lives under --temp-dir/concat/
        assert "concat" in received_input["input"]
        assert received_input["input"].startswith(str(tmp_path))

    def test_crossfade_transmitted(self, fake_concat, monkeypatch) -> None:
        calls, _ = fake_concat

        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["a.mp4", "b.mp4"],
                concat_crossfade=0.5,
                concat_mode="demux",
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 0
        assert calls["concat"][0]["crossfade"] == 0.5

    def test_mode_transmitted(self, fake_concat, monkeypatch) -> None:
        calls, _ = fake_concat

        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["a.mp4", "b.mp4"],
                concat_crossfade=0.0,
                concat_mode="filter",
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 0
        assert calls["concat"][0]["mode"] == "filter"

    def test_intermediate_lands_under_temp_dir(self, fake_concat, monkeypatch, tmp_path) -> None:
        calls, _ = fake_concat
        temp = tmp_path / "my_temp"

        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["a.mp4", "b.mp4"],
                concat_crossfade=0.0,
                concat_mode="demux",
                temp_dir=str(temp),
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 0
        out_path = calls["concat"][0]["output_path"]
        assert out_path.startswith(str(temp))
        # Must NOT land under video/
        assert "video" not in Path(out_path).parts


class TestCompatFailure:
    def test_concat_error_prints_message_and_exits_nonzero(self, monkeypatch, capsys) -> None:
        err = ConcatError(
            "incompatible segment resolution/fps for concat:\n"
            "  reference a.mp4: 1920x1080 @ 30fps\n"
            "  b.mp4: 1280x720 @ 25fps (expected 1920x1080 @ 30fps)"
        )

        def _boom(*a, **kw):
            raise err

        monkeypatch.setattr(rp, "concat_segments", _boom)
        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input=None,
                inputs=["a.mp4", "b.mp4"],
                concat_crossfade=0.0,
                concat_mode="demux",
                validate_input=False,
                output=None,
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            rp.main()
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "incompatible segment resolution/fps" in combined
        # The offending segment's actual values must be surfaced (acceptance).
        assert "b.mp4" in combined
        assert "1280x720 @ 25fps" in combined


class TestNoConcatWhenInputGiven:
    def test_single_input_bypasses_concat(self, monkeypatch, capsys) -> None:
        """Regression: --input (single file) must keep its pre-C-1b behaviour.

        concat_segments must NOT be called when --input is used.
        """
        concat_called = {"count": 0}

        def _boom(*a, **kw):
            concat_called["count"] += 1
            raise RuntimeError("concat should not run for single --input")

        monkeypatch.setattr(rp, "concat_segments", _boom)
        monkeypatch.setattr(
            rp,
            "parse_args",
            lambda: MagicMock(
                input="/tmp/clip.mp4",
                inputs=None,
                concat_crossfade=0.0,
                concat_mode="demux",
                validate_input=False,
                output=None,
                stage="depth",
                streaming=False,
                projection="vr180",
                video_upscale="none",
                fps=30,
                device=None,
                stream=False,
                max_frames=None,
                upscale=0,
                stages=None,
                manifest=None,
                resume_from=None,
            ),
        )
        monkeypatch.setattr(rp, "apply_quality_preset", lambda args: None)
        monkeypatch.setattr(rp, "_apply_comfort_preset", lambda args: None)
        monkeypatch.setattr(rp, "apply_playback_preset", lambda args: None)
        monkeypatch.setattr(rp, "_manifest_prepare", lambda args: (None, None, None))
        monkeypatch.setattr(rp, "detect_best_device", lambda: "cpu")
        fake_exit = MagicMock()
        monkeypatch.setattr(rp.sys, "exit", fake_exit)
        # Stub the depth-stage intake + stage so the single-input path exits
        # cleanly without touching real video files or models; the assertion
        # is purely that concat_segments was never invoked.
        monkeypatch.setattr(rp, "_intake_frames", lambda *a, **kw: ([], 0))
        monkeypatch.setattr(rp, "run_depth_stage", lambda args, frames: [])

        rp.main()
        assert concat_called["count"] == 0
