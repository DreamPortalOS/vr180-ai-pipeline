"""F-9 (#285): VR180 output is edge-feathered 165->180 by default.

The owner's 09-07 headset comparison settled on ``feather`` (black beyond the
source FOV, faded 165->180) and retired ``accum`` (SphereAccumulator: duplicated
/ stretched artefacts in the outer ring).  ``scripts/run_pipeline.py`` now:

* hands :class:`StreamingPipeline` ``edge_feather_start/end = 165/180`` for
  ``--projection vr180`` when neither angle flag is given;
* honours ``--no-edge-feather`` by handing it ``None/None`` -- the pre-#285
  default, i.e. the byte-identical off path;
* lets an explicit ``--edge-feather-start/--edge-feather-end`` win untouched;
* leaves ``--projection fulldome`` alone (it has no feather layer);
* keeps ``--sphere-accumulate on`` running exactly as before but logs one
  ``DEPRECATED`` warning.

Everything below is CLI wiring: ``parse_args`` is real (so the real argparse
defaults and the real F-9 policy run), while ``StreamingPipeline`` /
``FulldomeMapper`` / metadata injection / audio / sidecar are mocked -- no
model, no ffmpeg, no file writes.  CPU-only, CI-safe.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_LOGGER = "vr180-pipeline"

# A minimal VR180 streaming invocation: --streaming pins the branch (the
# default --quality standard would flip it on anyway), --preflight off keeps
# the host-resource probe out, --fps skips the cv2 fps inheritance read.
_STREAM_ARGV = ["--input", "in.mp4", "--output", "out.mp4", "--fps", "30", "--preflight", "off", "--streaming"]


@pytest.fixture
def rp():
    scripts = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts)
    try:
        import run_pipeline

        yield run_pipeline
    finally:
        sys.modules.pop("run_pipeline", None)
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts)


def _drive_streaming_main(rp, argv: list[str], monkeypatch) -> dict:
    """Run ``main()`` for *argv* up to the ``StreamingPipeline(...)`` call and return its kwargs."""
    args = rp.parse_args(argv)
    captured: dict = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        inst = MagicMock()
        inst.process_stream.return_value = "out.mp4"
        return inst

    monkeypatch.setattr(rp, "parse_args", lambda: args)
    monkeypatch.setattr(rp, "detect_best_device", lambda: "cpu")
    monkeypatch.setattr(rp, "build_depth_backend", lambda a, **kw: (None, "depth-anything"))
    monkeypatch.setattr(rp, "build_stereo_backend", lambda a, **kw: (None, "default"))
    monkeypatch.setattr(rp, "StreamingPipeline", fake_ctor)
    monkeypatch.setattr(rp, "_copy_audio_to_output", lambda *a, **kw: None)
    monkeypatch.setattr(rp, "_maybe_copy_audio_from_input", lambda *a, **kw: None)
    monkeypatch.setattr(rp, "_write_sidecar_from_args", lambda *a, **kw: None)
    with patch("pipeline.spherical_injector.inject_spherical_metadata"), patch("os.replace"):
        rp.main()
    assert captured, "main() never reached the StreamingPipeline(...) call"
    return captured


def _feather(kwargs: dict) -> tuple:
    return kwargs["edge_feather_start"], kwargs["edge_feather_end"]


def _deprecation_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "DEPRECATED" in r.getMessage()]


# ---------------------------------------------------------------------------
#  VR180 default: 165/180 reaches the stream; --no-edge-feather hands it None/None
# ---------------------------------------------------------------------------


class TestVr180FeatherDefault:
    def test_no_flags_hands_the_stream_165_180(self, rp, monkeypatch):
        assert _feather(_drive_streaming_main(rp, [*_STREAM_ARGV], monkeypatch)) == (165.0, 180.0)

    def test_explicit_projection_vr180_is_the_same_default(self, rp, monkeypatch):
        kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV, "--projection", "vr180"], monkeypatch)
        assert _feather(kwargs) == (165.0, 180.0) == rp.DEFAULT_VR180_EDGE_FEATHER

    def test_no_edge_feather_hands_the_stream_none_none(self, rp, monkeypatch, caplog):
        """The explicit off switch must reproduce the pre-#285 default call exactly (None/None)."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV, "--no-edge-feather"], monkeypatch)
        assert _feather(kwargs) == (None, None)
        # The switch is a supported streaming knob: the K-22 swallowed-arg
        # detector must not report it as ignored.
        assert "does NOT honour" not in caplog.text
        assert "no-edge-feather" not in caplog.text

    def test_no_edge_feather_is_registered_as_a_streaming_knob(self, rp):
        assert rp._STREAMING_SUPPORTED["no_edge_feather"]

    def test_explicit_start_wins_over_the_default(self, rp, monkeypatch):
        kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV, "--edge-feather-start", "110"], monkeypatch)
        # end stays None: the omitted bound keeps its #244 default inside the stream.
        assert _feather(kwargs) == (110.0, None)

    def test_explicit_end_wins_over_the_default(self, rp, monkeypatch):
        kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV, "--edge-feather-end", "170"], monkeypatch)
        assert _feather(kwargs) == (None, 170.0)

    def test_explicit_both_win_over_the_default(self, rp, monkeypatch):
        argv = [*_STREAM_ARGV, "--edge-feather-start", "150", "--edge-feather-end", "175"]
        assert _feather(_drive_streaming_main(rp, argv, monkeypatch)) == (150.0, 175.0)

    def test_no_edge_feather_rejects_explicit_angles(self, rp, capsys):
        with pytest.raises(SystemExit):
            rp.parse_args(["--no-edge-feather", "--edge-feather-start", "110"])
        assert "--no-edge-feather" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            rp.parse_args(["--no-edge-feather", "--edge-feather-end", "170"])

    def test_invalid_angles_are_still_rejected_at_parse_time(self, rp):
        with pytest.raises(SystemExit):
            rp.parse_args(["--edge-feather-start", "170", "--edge-feather-end", "165"])

    def test_parser_level_defaults_are_untouched(self, rp):
        """The argparse defaults stay None/None + off: the #244 batch outpaint gate
        (``run_outpaint_stage`` is a no-op with parsed defaults, pinned in
        tests/test_outpainter.py) and ``parse_args()`` are byte-identical to
        before -- the VR180 default lives at the streaming call site only."""
        args = rp.parse_args([])
        assert args.projection == "vr180"
        assert args.edge_feather_start is None and args.edge_feather_end is None
        assert args.no_edge_feather is False

    def test_help_documents_the_default_and_the_switch(self, rp):
        # argparse re-wraps help text, so compare on whitespace-normalised text.
        text = " ".join(rp._build_parser().format_help().split())
        assert "--no-edge-feather" in text
        assert "165->180 by default" in text

    def test_resolver_policy_table(self, rp):
        resolve = rp.resolve_cli_edge_feather
        assert resolve(rp.parse_args([])) == (165.0, 180.0)
        assert resolve(rp.parse_args(["--no-edge-feather"])) == (None, None)
        assert resolve(rp.parse_args(["--edge-feather-start", "110"])) == (110.0, None)
        assert resolve(rp.parse_args(["--edge-feather-end", "170"])) == (None, 170.0)

    def test_resolver_ignores_a_mock_off_switch(self, rp):
        """Older wiring tests drive ``main()`` with a MagicMock ``args`` whose unset
        attributes are truthy Mocks -- a Mock ``no_edge_feather`` must not read
        as "off" or those tests would silently lose their explicit angles."""
        args = MagicMock()
        args.projection = "vr180"
        args.edge_feather_start, args.edge_feather_end = 165.0, None
        assert rp.resolve_cli_edge_feather(args) == (165.0, None)


# ---------------------------------------------------------------------------
#  --projection fulldome: no feather injected (regression)
# ---------------------------------------------------------------------------


class TestFulldomeUnaffected:
    def test_resolver_leaves_fulldome_alone(self, rp):
        resolve = rp.resolve_cli_edge_feather
        assert resolve(rp.parse_args(["--projection", "fulldome"])) == (None, None)
        # An explicitly typed angle passes through untouched, exactly as before #285.
        assert resolve(rp.parse_args(["--projection", "fulldome", "--edge-feather-start", "110"])) == (110.0, None)

    def test_fulldome_main_injects_no_feather_and_never_builds_the_stream(self, rp, monkeypatch):
        argv = ["--input", "in.mp4", "--output", "out.mp4", "--fps", "30", "--preflight", "off"]
        args = rp.parse_args([*argv, "--projection", "fulldome"])
        dome_kwargs: dict = {}

        def fake_mapper(**kwargs):
            dome_kwargs.update(kwargs)
            inst = MagicMock()
            inst.convert.return_value = "out_dome.mp4"
            return inst

        stream_ctor = MagicMock(side_effect=AssertionError("fulldome must not construct StreamingPipeline"))
        monkeypatch.setattr(rp, "parse_args", lambda: args)
        monkeypatch.setattr(rp, "detect_best_device", lambda: "cpu")
        monkeypatch.setattr(rp, "FulldomeMapper", fake_mapper)
        monkeypatch.setattr(rp, "StreamingPipeline", stream_ctor)
        monkeypatch.setattr(rp, "_write_sidecar_from_args", lambda *a, **kw: None)

        rp.main()

        assert dome_kwargs, "main() never reached the FulldomeMapper(...) call"
        assert not any("feather" in key for key in dome_kwargs), dome_kwargs
        assert not stream_ctor.called
        # main() must not have grown the args either (fulldome path untouched).
        assert args.edge_feather_start is None and args.edge_feather_end is None


# ---------------------------------------------------------------------------
#  --sphere-accumulate on: DEPRECATED warning, behaviour unchanged
# ---------------------------------------------------------------------------


class TestSphereAccumulateDeprecated:
    def test_on_warns_deprecated_and_still_reaches_the_stream(self, rp, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV, "--sphere-accumulate", "on"], monkeypatch)
        # Behaviour unchanged: the flag (and its radial scale) reach the constructor as before.
        assert kwargs["sphere_accumulate"] == "on"
        assert kwargs["sphere_radial_scale"] == rp.DEFAULT_SPHERE_RADIAL_SCALE
        warnings = _deprecation_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one DEPRECATED warning, got {warnings}"
        assert "--sphere-accumulate" in warnings[0]

    def test_off_does_not_warn(self, rp, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            kwargs = _drive_streaming_main(rp, [*_STREAM_ARGV], monkeypatch)
        assert kwargs["sphere_accumulate"] == "off"
        assert _deprecation_warnings(caplog) == []

    def test_parser_still_accepts_on_and_defaults_off(self, rp):
        assert rp.parse_args([]).sphere_accumulate == "off"
        assert rp.parse_args(["--sphere-accumulate", "on"]).sphere_accumulate == "on"

    def test_help_marks_it_deprecated(self, rp):
        text = rp._build_parser().format_help()
        # rindex: the flag also appears in the usage line; the last occurrence is
        # the option's own entry, and the tag must sit in *that* entry's help.
        start = text.rindex("--sphere-accumulate {off,on}")
        end = text.index("--sphere-radial-scale", start)
        assert "[DEPRECATED" in text[start:end]


# ---------------------------------------------------------------------------
#  #272 guard: the constructor / call-site kwarg sets still match
# ---------------------------------------------------------------------------


def test_streaming_constructor_and_call_site_still_agree():
    """F-9 rewired two kwarg *values* at the single ``StreamingPipeline(...)`` call
    site without adding or renaming a kwarg; the #272 set-equality guard in
    tests/test_streaming_entry.py must therefore stay green.  Re-asserted here
    so this card's own file fails loudly if that ever drifts."""
    import ast
    import inspect

    from pipeline.streaming_pipeline import StreamingPipeline

    with open(os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    calls = [
        {kw.arg for kw in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None))
        == "StreamingPipeline"
    ]
    assert len(calls) == 1
    assert calls[0] == set(inspect.signature(StreamingPipeline.__init__).parameters) - {"self"}
