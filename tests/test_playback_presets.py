"""Tests for D-2 downstream playback presets (issue #79) and their CLI wiring.

Covers:

  1. ``resolve_playback`` is a pure function — three preset values, explicit
     override precedence, invalid name raises ``ValueError``.
  2. ``preset_encode_args`` builds the ffmpeg args for the documented
     playback-side constraints (fixed 1s GOP / segment-head IDR / +faststart).
  3. ``scripts.run_pipeline`` accepts ``--preset {pcvr,standalone,source}``,
     default is ``source``, and explicit ``--codec`` / ``--crf`` / ``--bitrate``
     / ``--gop`` win over the preset (the "override always wins" invariant).
  4. The encode stages (``StreamingPipeline``, ``RawFrameFFmpegWriter``,
     ``VRMetadataEmbedder``) thread the resolved knobs into their ffmpeg
     command lists.
  5. The sidecar ``generation`` block records the chosen preset.

No GPU, models, cv2, or ffmpeg are touched here.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pipeline.playback_presets import (
    DEFAULT_GOP_SECONDS,
    DEFAULT_PLAYBACK,
    PLAYBACK_PRESETS,
    resolve_playback,
)
from pipeline.streaming_pipeline import (
    RawFrameFFmpegWriter,
    StreamingPipeline,
    preset_encode_args,
)
from pipeline.vr_metadata import VRMetadataEmbedder

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
RUN_PIPELINE = SCRIPTS / "run_pipeline.py"


def _load_run_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline_playback", RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. resolve_playback pure-function tests
# ---------------------------------------------------------------------------


class TestResolvePlayback:
    """Preset values, explicit override precedence, invalid-name rejection."""

    def test_pcvr_tier_values(self) -> None:
        resolved = resolve_playback("pcvr")
        assert resolved["codec"] == "h265"
        assert resolved["crf"] == 18
        assert resolved["bitrate"] is None
        assert resolved["gop_seconds"] == 1
        assert resolved["force_idr"] is True
        assert resolved["faststart"] is True

    def test_standalone_tier_values(self) -> None:
        resolved = resolve_playback("standalone")
        assert resolved["codec"] == "h265"
        assert resolved["crf"] == 23
        assert resolved["gop_seconds"] == 1
        assert resolved["force_idr"] is True
        assert resolved["faststart"] is True

    def test_source_is_passthrough(self) -> None:
        """source fills in nothing — every knob is None/False."""
        resolved = resolve_playback("source")
        assert resolved["codec"] is None
        assert resolved["crf"] is None
        assert resolved["bitrate"] is None
        assert resolved["gop_seconds"] is None
        assert resolved["force_idr"] is False
        assert resolved["faststart"] is None

    def test_none_uses_default_preset(self) -> None:
        """``None`` name collapses to :data:`DEFAULT_PLAYBACK` (source)."""
        assert resolve_playback(None) == resolve_playback(DEFAULT_PLAYBACK)

    def test_explicit_codec_overrides(self) -> None:
        """An explicit codec always wins, even from the HEVC tiers."""
        assert resolve_playback("pcvr", {"codec": "h264"})["codec"] == "h264"
        # The non-overridden keys keep their preset values.
        assert resolve_playback("pcvr", {"codec": "h264"})["crf"] == 18

    def test_explicit_crf_overrides(self) -> None:
        assert resolve_playback("standalone", {"crf": 20})["crf"] == 20
        assert resolve_playback("standalone", {"crf": 20})["codec"] == "h265"

    def test_explicit_bitrate_overrides(self) -> None:
        """An explicit bitrate wins over the preset's None."""
        assert resolve_playback("pcvr", {"bitrate": "80M"})["bitrate"] == "80M"

    def test_explicit_gop_seconds_overrides(self) -> None:
        assert resolve_playback("pcvr", {"gop_seconds": 2})["gop_seconds"] == 2

    def test_none_explicit_is_a_noop(self) -> None:
        """A ``None`` explicit dict behaves like no overrides."""
        assert resolve_playback("pcvr", None) == resolve_playback("pcvr")
        assert resolve_playback("pcvr", {}) == resolve_playback("pcvr")

    def test_returns_fresh_dict(self) -> None:
        """Mutating the result must not mutate the preset (defensive copy)."""
        a = resolve_playback("pcvr")
        a["crf"] = -1
        assert resolve_playback("pcvr")["crf"] == 18

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown playback preset"):
            resolve_playback("ultra")
        with pytest.raises(ValueError, match="choose from"):
            resolve_playback("pcvr ")

    def test_unknown_explicit_keys_pass_through(self) -> None:
        """Unknown override keys pass through so the dict stays a drop-in."""
        assert resolve_playback("pcvr", {"extra": 1})["extra"] == 1

    def test_default_gop_seconds_is_one(self) -> None:
        """The documented seek-precision GOP is 1 second."""
        assert DEFAULT_GOP_SECONDS == 1
        for tier in ("pcvr", "standalone"):
            assert PLAYBACK_PRESETS[tier]["gop_seconds"] == 1


# ---------------------------------------------------------------------------
# 2. preset_encode_args — ffmpeg arg builder
# ---------------------------------------------------------------------------


class TestPresetEncodeArgs:
    """The ffmpeg args for the documented playback-side constraints."""

    def test_none_gop_no_args(self) -> None:
        assert preset_encode_args(None, False, None) == []

    def test_zero_gop_no_args(self) -> None:
        """0 = let ffmpeg pick (no GOP args)."""
        assert preset_encode_args(0, False, None) == []

    def test_gop_sets_fixed_gop_and_sc_threshold(self) -> None:
        """A fixed 1s GOP = -g / -keyint_min / -sc_threshold 0."""
        args = preset_encode_args(30, False, None)
        assert "-g" in args
        assert args[args.index("-g") + 1] == "30"
        assert args[args.index("-keyint_min") + 1] == "30"
        assert args[args.index("-sc_threshold") + 1] == "0"

    def test_force_idr_adds_force_key_frames(self) -> None:
        """Segment-head IDR forces a keyframe at the GOP interval."""
        args = preset_encode_args(30, True, None)
        assert "-force_key_frames" in args
        expr = args[args.index("-force_key_frames") + 1]
        assert "n_forced" in expr
        assert "30" in expr

    def test_no_force_idr_omits_force_key_frames(self) -> None:
        args = preset_encode_args(30, False, None)
        assert "-force_key_frames" not in args

    def test_faststart_true_adds_movflags(self) -> None:
        assert "+faststart" in preset_encode_args(None, False, True)
        assert "-movflags" in preset_encode_args(None, False, True)

    def test_faststart_false_drops_faststart(self) -> None:
        args = preset_encode_args(None, False, False)
        assert "-movflags" in args
        assert "+faststart" not in args

    def test_faststart_none_no_movflags(self) -> None:
        """None = leave the caller's movflags untouched."""
        assert "-movflags" not in preset_encode_args(None, False, None)


# ---------------------------------------------------------------------------
# 3. CLI wiring — run_pipeline.py
# ---------------------------------------------------------------------------


class _FakeArgs:
    """Minimal args namespace for apply_*_preset tests."""

    def __init__(self, **overrides):
        self.quality = "preview"
        self.max_bitrate = 200.0
        self.output_width = None
        self.output_height = None
        self.streaming = False
        self.projection = "vr180"
        self.bitrate = None
        self.preset = "source"
        self.codec = None
        self.crf = None
        self.gop = None
        self.fps = 30
        self.no_temporal = False
        self.comfort = "balanced"
        self.max_disparity = None
        self.convergence = None
        for k, v in overrides.items():
            setattr(self, k, v)


class TestRunPipelinePresetCLI:
    """--preset defaults to source; explicit flags override the preset."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        mod = _load_run_pipeline()
        return mod.parse_args(argv)

    def test_help_lists_preset(self, capsys: pytest.CaptureFixture) -> None:
        mod = _load_run_pipeline()
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--preset" in out
        assert "pcvr" in out
        assert "standalone" in out
        assert "source" in out

    def test_default_preset_is_source(self) -> None:
        args = self._parse(["--input", "x.mp4"])
        assert args.preset == "source"

    def test_codec_crf_defaults_are_none_sentinel(self) -> None:
        """Before preset resolution codec/crf are None (unset)."""
        args = self._parse(["--input", "x.mp4"])
        assert args.codec is None
        assert args.crf is None
        assert args.gop is None

    def test_preset_choices_enforced(self) -> None:
        with pytest.raises(SystemExit):
            self._parse(["--input", "x.mp4", "--preset", "ultra"])

    def test_source_resolves_to_legacy_defaults(self) -> None:
        """source = passthrough → h264/23 (pre-D-2 behaviour)."""
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="source")
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.codec == "h264"
        assert args.crf == 23
        assert args.gop is None
        assert args._preset_faststart is None

    def test_pcvr_resolves_to_hevc_crf18_gop(self) -> None:
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="pcvr", fps=30)
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.codec == "h265"
        assert args.crf == 18
        assert args.gop == 30  # 1s × 30fps
        assert args._preset_faststart is True
        assert args._preset_force_idr is True

    def test_standalone_resolves_to_hevc_crf23(self) -> None:
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="standalone", fps=60)
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.codec == "h265"
        assert args.crf == 23
        assert args.gop == 60  # 1s × 60fps
        assert args._preset_faststart is True

    def test_explicit_codec_wins_over_preset(self) -> None:
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="pcvr", codec="h264")
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.codec == "h264"
        assert args.crf == 18  # preset value kept

    def test_explicit_crf_wins_over_preset(self) -> None:
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="pcvr", crf=25)
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.crf == 25
        assert args.codec == "h265"  # preset codec kept

    def test_explicit_bitrate_wins_over_preset(self) -> None:
        """The quality path's adaptive bitrate must win over the preset None."""
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="pcvr")
        mod.apply_quality_preset(args)  # sets args.bitrate from adaptive scaling
        mod.apply_playback_preset(args)
        assert args.bitrate is not None  # not clobbered to None

    def test_explicit_gop_wins_over_preset_seconds(self) -> None:
        """An explicit --gop frame count skips the preset's 1s translation."""
        mod = _load_run_pipeline()
        args = _FakeArgs(preset="pcvr", fps=30, gop=15)
        mod.apply_quality_preset(args)
        mod.apply_playback_preset(args)
        assert args.gop == 15  # explicit, not 30


# ---------------------------------------------------------------------------
# 4. encode-stage ffmpeg command threading
# ---------------------------------------------------------------------------


class TestStreamingPipelinePresetCmd:
    """StreamingPipeline._build_ffmpeg_cmd honours the preset knobs."""

    def _make(self, **kw):
        with (
            patch("pipeline.streaming_pipeline.DepthEstimator"),
            patch("pipeline.streaming_pipeline.StereoRenderer"),
            patch("pipeline.streaming_pipeline.EquirectangularMapper"),
        ):
            return StreamingPipeline(
                model_size="small",
                device="cpu",
                codec=kw.get("codec", "h264"),
                crf=kw.get("crf", 23),
                fps=kw.get("fps", 30),
                bitrate=kw.get("bitrate"),
                gop=kw.get("gop"),
                force_idr=kw.get("force_idr", False),
                faststart=kw.get("faststart"),
            )

    def test_gop_idr_faststart_in_cmd(self) -> None:
        p = self._make(codec="h265", crf=18, gop=30, force_idr=True, faststart=True)
        cmd = p._build_ffmpeg_cmd("out.mp4", 7680, 3840)
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "30"
        assert "-force_key_frames" in cmd
        assert "+faststart" in cmd

    def test_no_preset_keeps_faststart_default(self) -> None:
        """faststart=None (no preset) still emits +faststart (pre-D-2)."""
        p = self._make(faststart=None)
        cmd = p._build_ffmpeg_cmd("out.mp4", 1920, 1920)
        assert "+faststart" in cmd
        assert "-g" not in cmd  # no GOP without a preset


class TestRawFrameFFmpegWriterPresetCmd:
    """RawFrameFFmpegWriter._build_cmd honours the preset knobs."""

    def test_gop_idr_in_cmd(self) -> None:
        w = RawFrameFFmpegWriter(
            "out.mp4",
            1920,
            1920,
            codec="h265",
            crf=18,
            fps=30,
            gop=30,
            force_idr=True,
            faststart=True,
        )
        cmd = w._build_cmd()
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "30"
        assert "-force_key_frames" in cmd
        assert "+faststart" in cmd

    def test_no_preset_keeps_faststart(self) -> None:
        w = RawFrameFFmpegWriter("out.mp4", 100, 50, codec="h264")
        cmd = w._build_cmd()
        assert "+faststart" in cmd
        assert "-g" not in cmd


class TestVRMetadataEmbedderPresetCmd:
    """VRMetadataEmbedder threads the preset knobs into its encode cmd."""

    def test_gop_idr_in_encode_cmd(self) -> None:
        e = VRMetadataEmbedder(
            codec="h265",
            crf=18,
            fps=30,
            gop=30,
            force_idr=True,
            faststart=True,
        )
        # Build the encode cmd via the internal path (no real ffmpeg run):
        # construct the same list the embedder builds, minus the io.
        frames = [np.zeros((4, 8, 3), dtype=np.uint8)]
        # Patch inject_spherical_metadata + Popen so we can capture the cmd.
        with patch("pipeline.vr_metadata.inject_spherical_metadata") as inj:
            inj.return_value = None
            with patch("subprocess.Popen") as popen:
                popen.return_value.communicate.return_value = (b"", b"")
                popen.return_value.returncode = 0
                with contextlib.suppress(Exception):
                    e.embed_single_frame_batch(frames, "out.mp4")
                cmd = popen.call_args.args[0]
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "30"
        assert "-force_key_frames" in cmd
        assert "+faststart" in cmd

    def test_default_faststart_preserved(self) -> None:
        e = VRMetadataEmbedder(codec="h264", crf=23)  # no preset knobs
        assert e.gop is None
        assert e.faststart is None
        frames = [np.zeros((4, 8, 3), dtype=np.uint8)]
        with patch("pipeline.vr_metadata.inject_spherical_metadata") as inj:
            inj.return_value = None
            with patch("subprocess.Popen") as popen:
                popen.return_value.communicate.return_value = (b"", b"")
                popen.return_value.returncode = 0
                with contextlib.suppress(Exception):
                    e.embed_single_frame_batch(frames, "out.mp4")
                cmd = popen.call_args.args[0]
        assert "+faststart" in cmd
        assert "-g" not in cmd


# ---------------------------------------------------------------------------
# 5. sidecar generation.preset
# ---------------------------------------------------------------------------


class TestSidecarPresetField:
    """The sidecar generation block records the chosen preset."""

    def test_run_pipeline_sidecar_records_preset(self, tmp_path: Path) -> None:
        mod = _load_run_pipeline()

        # Build a tiny synthetic mp4 the sidecar can probe (mocked ffprobe).
        mp4 = tmp_path / "out.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x08ftyp")

        args = argparse.Namespace(
            output_width=1920,
            output_height=1920,
            preset="pcvr",
        )
        # write_sidecar is imported from pipeline.sidecar inside the helper,
        # so patch it there to capture the generation block.
        captured: dict = {}

        def fake_write_sidecar(path, *, immersive, generation, **kw):
            captured["generation"] = generation

        with patch("pipeline.sidecar.write_sidecar", side_effect=fake_write_sidecar):
            mod._write_sidecar_from_args(str(mp4), "vr180", args)
        assert captured["generation"]["preset"] == "pcvr"
        assert captured["generation"]["route"] == "vr180"

    def test_source_preset_omitted_when_unset(self, tmp_path: Path) -> None:
        """When no preset is on args, the field is absent (not 'None')."""
        mod = _load_run_pipeline()
        mp4 = tmp_path / "out.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x08ftyp")
        # args with no preset attribute at all (e.g. legacy callers).
        args = argparse.Namespace(output_width=1920, output_height=1920)
        captured: dict = {}

        def fake_write_sidecar(path, *, immersive, generation, **kw):
            captured["generation"] = generation

        with patch("pipeline.sidecar.write_sidecar", side_effect=fake_write_sidecar):
            mod._write_sidecar_from_args(str(mp4), "vr180", args)
        assert "preset" not in captured["generation"]
        assert captured["generation"]["route"] == "vr180"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
