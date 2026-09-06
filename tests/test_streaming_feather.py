"""Tests for issue #261 (F-1) — edge feather on the streaming path.

#244 / PR #259 landed the angle-weighted edge feather (``--edge-feather-start``
/ ``--edge-feather-end``) for the batch path only.  On the streaming path
(``--quality standard|high`` ⇒ :class:`StreamingPipeline`, the production
default) the flags were named by the #243 swallowed-arg detector and ignored,
so a real render kept a hard edge at the source FOV.  This card threads the
feather into ``StreamingPipeline`` reusing the #259 geometry
(:func:`pipeline.outpainter.sbs_edge_feather_weights`) — the only new code is
the per-frame *application* of those constant weights.

Covered here (fake depth/stereo backends, the real OpenCV mapper at 192²/eye,
no ffmpeg, no model):

  - feather off ⇒ the bytes handed to the encoder are exactly the mapper's
    output, frame for frame, and no alpha plane is ever derived;
  - 165→180 on the production geometry (126° source) ⇒ brightness along a
    radial scan line is monotone down to 0 at the content edge (the #259
    assertions), on both fuse-loop branches;
  - the streaming frame equals the batch path's ``Outpainter`` result for the
    same frame (MAE < 2 per the card; in fact bit-exact);
  - ``--quality standard --edge-feather-start 165`` no longer trips the
    swallowed-arg warning, and the flags reach the constructor;
  - the sparse per-frame apply is bit-identical to ``apply_edge_feather``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.equirectangular_mapper import EquirectangularMapper  # noqa: E402
from pipeline.outpainter import Outpainter, apply_edge_feather, sbs_edge_feather_weights  # noqa: E402
from pipeline.streaming_pipeline import StreamingPipeline, _apply_feather_plan, _build_feather_plan  # noqa: E402

_EYE = 192  # per-eye size (same tiny-but-real geometry as the #259 fixtures)
_SRC_H, _SRC_W = 36, 64  # 16:9 synthetic source
_SRC_RGB = (200, 150, 100)
_HFOV = 126.0  # the production geometry: hole on all four sides


# ---------------------------------------------------------------------------
#  Fakes / harness
# ---------------------------------------------------------------------------


class _FakeDepth:
    """Per-frame depth estimator (no ``estimate_video`` ⇒ never whole-clip)."""

    def estimate(self, rgb):
        return np.full(rgb.shape[:2], 0.5, dtype=np.float32)


class _FakeStereo:
    """Per-frame stereo renderer handing the source to both eyes (pure geometry)."""

    def render(self, rgb, depth):
        return rgb.copy(), rgb.copy()


class _FakeWholeClipStereo:
    """Whole-clip stereo (``render_video``); the L/R frames come back through the patched loader."""

    def render_video(self, input_path, depth_dir, output_left, output_right):
        return output_left, output_right


def _src_rgb() -> np.ndarray:
    return np.full((_SRC_H, _SRC_W, 3), _SRC_RGB, dtype=np.uint8)


def _fake_cap(num_frames: int):
    bgr = _src_rgb()[:, :, ::-1].copy()
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {3: _SRC_W, 4: _SRC_H, 7: float(num_frames), 5: 30.0}.get(prop, 0.0)
    cap.read.side_effect = [(True, bgr.copy()) for _ in range(num_frames)] + [(False, None)]
    return cap


def _make_pipeline(**kwargs) -> StreamingPipeline:
    kwargs.setdefault("depth_estimator", _FakeDepth())
    kwargs.setdefault("stereo_renderer", _FakeStereo())
    return StreamingPipeline(
        device="cpu",
        output_width=_EYE,
        output_height=_EYE,
        src_hfov=_HFOV,
        use_ffmpeg=False,
        hw_encoder=False,
        **kwargs,
    )


def _run(p: StreamingPipeline, num_frames: int = 2) -> list[np.ndarray]:
    """Drive ``process_stream`` with cv2/ffmpeg mocked; return the SBS frames written to the encoder pipe."""
    proc = MagicMock()
    proc.returncode = 0
    chunks: list[bytes] = []
    proc.stdin.write.side_effect = chunks.append
    with (
        patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=_fake_cap(num_frames)),
        patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
    ):
        p.process_stream("in.mp4", "out.mp4")
    return [np.frombuffer(b, dtype=np.uint8).reshape(_EYE, 2 * _EYE, 3) for b in chunks]


def _reference() -> tuple[np.ndarray, np.ndarray]:
    """What the batch path sees: the mapper's SBS frame and its FOV alpha side-car."""
    mapper = EquirectangularMapper(output_width=_EYE, output_height=_EYE, src_hfov=_HFOV, use_ffmpeg=False)
    src = _src_rgb()
    sbs = mapper.map_stereo_pair(src, src)
    alpha = mapper.map_single(src, with_alpha=True)[:, :, 3]
    return sbs, np.concatenate([alpha, alpha], axis=1)


def _batch_feathered(start: float, end: float) -> np.ndarray:
    """The non-streaming result for the same frame (Stage 3.5, run_outpaint_stage)."""
    sbs, alpha = _reference()
    return Outpainter(mode="none", edge_feather_start=start, edge_feather_end=end).process([sbs], alpha=alpha)[0]


def _equator_ray_inside_content(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Brightness (channel 0) along the left eye's equator, centre → last content pixel (#259)."""
    content = np.flatnonzero(alpha[_EYE // 2, _EYE // 2 : _EYE] > 0)
    return frame[_EYE // 2, _EYE // 2 : _EYE, 0].astype(int)[: content[-1] + 1]


def _alpha_calls(mapper) -> list:
    return [c for c in mapper.map_single.call_args_list if c.kwargs.get("with_alpha")]


# ---------------------------------------------------------------------------
#  Regression: feather off ⇒ pixel-identical to the pre-#261 stream
# ---------------------------------------------------------------------------


class TestFeatherOffIsByteIdentical:
    def test_default_is_off(self):
        assert _make_pipeline().edge_feather is None

    def test_explicit_none_is_off(self):
        assert _make_pipeline(edge_feather_start=None, edge_feather_end=None).edge_feather is None

    def test_encoder_receives_exactly_the_mapper_output(self):
        p = _make_pipeline()
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=2)
        sbs, _ = _reference()
        assert len(frames) == 2
        for f in frames:
            assert np.array_equal(f, sbs)
        # No alpha plane is derived when the feather is off — no extra mapper work at all.
        assert _alpha_calls(p.eq_mapper) == []


# ---------------------------------------------------------------------------
#  Feather on: the fade is real, anchored at the content edge, per frame
# ---------------------------------------------------------------------------


class TestFeatherOnStreaming:
    def test_radial_scan_is_monotone_to_zero_at_the_content_edge(self):
        p = _make_pipeline(edge_feather_start=165, edge_feather_end=180)
        assert p.edge_feather == (165.0, 180.0)
        out = _run(p, num_frames=1)[0]
        sbs, alpha = _reference()
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        ray = _equator_ray_inside_content(out, alpha)
        assert ray[0] == orig, "centre untouched"
        assert ray[-1] == 0, "last content pixel before the hole must be black"
        assert np.all(np.diff(ray) <= 0), "monotone from centre to the content edge"
        assert (-np.diff(ray)).max() <= 0.2 * orig, "no step — a hard edge would be a full-brightness drop"
        assert 0 < ray[-4] < orig, "the ramp lives inside the content (anchored at ~126°, not 180°)"
        assert np.all(out[alpha == 0] == 0), "hole stays black"
        assert np.array_equal(out[:, :_EYE], out[:, _EYE:]), "both eyes treated alike"
        assert np.count_nonzero(np.any(out != sbs, axis=2)) > 0, "the stream really changed pixels"

    def test_matches_the_batch_path_outpainter(self):
        p = _make_pipeline(edge_feather_start=165, edge_feather_end=180)
        out = _run(p, num_frames=1)[0]
        batch = _batch_feathered(165, 180)
        mae = float(np.abs(out.astype(int) - batch.astype(int)).mean())
        assert mae < 2.0, f"MAE {mae}"
        assert np.array_equal(out, batch), "same geometry function on the same alpha ⇒ bit-exact"

    def test_wide_feather_110_matches_batch_and_darkens_more(self):
        wide = _run(_make_pipeline(edge_feather_start=110), num_frames=1)[0]
        assert np.array_equal(wide, _batch_feathered(110, 180))
        default = _run(_make_pipeline(edge_feather_start=165), num_frames=1)[0]
        sbs, _ = _reference()
        changed = lambda f: np.count_nonzero(np.any(f != sbs, axis=2))  # noqa: E731
        assert changed(wide) > changed(default) > 0

    def test_single_bound_enables_with_the_244_defaults(self):
        assert _make_pipeline(edge_feather_start=110).edge_feather == (110.0, 180.0)
        assert _make_pipeline(edge_feather_end=170).edge_feather == (165.0, 170.0)

    @pytest.mark.parametrize(("start", "end"), [(170, 165), (-1, 180), (0, 181)])
    def test_invalid_angles_raise_at_construction(self, start, end):
        with pytest.raises(ValueError):
            _make_pipeline(edge_feather_start=start, edge_feather_end=end)

    def test_weights_are_built_once_for_the_whole_stream(self):
        p = _make_pipeline(edge_feather_start=165)
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=3)
        assert len(_alpha_calls(p.eq_mapper)) == 1, "alpha/weights are geometry-only: derived once, not per frame"
        assert len(frames) == 3
        assert all(np.array_equal(f, frames[0]) for f in frames)

    def test_wholeclip_stereo_branch_is_feathered_too(self, tmp_path):
        """The precomputed-L/R fuse loop (StereoCrafter) goes through the same Stage 3 as the per-frame loop."""
        src = _src_rgb()
        p = _make_pipeline(
            edge_feather_start=165,
            edge_feather_end=180,
            stereo_renderer=_FakeWholeClipStereo(),
            temp_dir=str(tmp_path),
        )
        left = [src.copy(), src.copy()]
        right = [src.copy(), src.copy()]
        with patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]):
            frames = _run(p, num_frames=2)
        batch = _batch_feathered(165, 180)
        assert len(frames) == 2
        for f in frames:
            assert np.array_equal(f, batch)

    def test_equirect_stage_timing_still_reported(self):
        p = _make_pipeline(edge_feather_start=165)
        _run(p, num_frames=2)
        assert p.stage_timings["equirect"] > 0.0


# ---------------------------------------------------------------------------
#  The sparse per-frame apply ≡ apply_edge_feather (same weights)
# ---------------------------------------------------------------------------


def _synthetic(eye: int = 96, coverage: str = "disc", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random SBS content with a black hole, plus the alpha plane describing that hole."""
    rng = np.random.default_rng(seed)
    if coverage == "full":
        alpha_eye = np.full((eye, eye), 255, dtype=np.uint8)
    else:
        yy, xx = np.mgrid[:eye, :eye]
        r = ((xx - eye / 2 + 0.5) / (0.7 * eye / 2)) ** 2 + ((yy - eye / 2 + 0.5) / (0.5 * eye / 2)) ** 2
        alpha_eye = np.where(r <= 1.0, 255, 0).astype(np.uint8)
    alpha = np.concatenate([alpha_eye, alpha_eye], axis=1)
    sbs = rng.integers(0, 256, (eye, 2 * eye, 3), dtype=np.uint8)
    sbs[alpha == 0] = 0  # the mapper renders the hole black (#255/#258)
    return sbs, alpha


class TestSparseApplyMatchesApplyEdgeFeather:
    @pytest.mark.parametrize("coverage", ["disc", "full"])
    @pytest.mark.parametrize(("start", "end"), [(165, 180), (110, 180), (180, 180), (150, 160), (0, 180)])
    def test_bit_exact(self, coverage, start, end):
        sbs, alpha = _synthetic(coverage=coverage)
        weights = sbs_edge_feather_weights(alpha, start, end)
        plan = _build_feather_plan(weights, alpha, eye_shape=(_SRC_H, _SRC_W))
        assert np.array_equal(_apply_feather_plan(sbs.copy(), plan), apply_edge_feather(sbs, weights))

    def test_boxes_cover_every_changed_pixel(self):
        _, alpha = _synthetic()
        weights = sbs_edge_feather_weights(alpha, 165, 180)
        plan = _build_feather_plan(weights, alpha, eye_shape=(_SRC_H, _SRC_W))
        assert plan.eye_shape == (_SRC_H, _SRC_W) and plan.sbs_shape == alpha.shape
        assert 1 <= len(plan.rois) <= 2
        covered = np.zeros(alpha.shape, dtype=bool)
        for r0, r1, c0, c1, block in plan.rois:
            covered[r0:r1, c0:c1] = True
            assert block.dtype == np.float32
            assert block.shape == (r1 - r0, (c1 - c0) * 3)
            assert block.flags.c_contiguous
        assert np.all(covered[(weights < 1.0) & (alpha > 0)])
        assert plan.roi_pct < 100.0, "a partial-FOV source leaves the hole's corners out of the boxes"
        assert plan.rois[0][4] is plan.rois[1][4], "identical eyes share one weight block"

    def test_applies_in_place_on_a_contiguous_frame(self):
        sbs, alpha = _synthetic()
        plan = _build_feather_plan(sbs_edge_feather_weights(alpha, 165, 180), alpha, eye_shape=(_SRC_H, _SRC_W))
        assert _apply_feather_plan(sbs, plan) is sbs

    def test_shape_mismatch_raises(self):
        sbs, alpha = _synthetic()
        plan = _build_feather_plan(sbs_edge_feather_weights(alpha, 165, 180), alpha, eye_shape=(_SRC_H, _SRC_W))
        with pytest.raises(RuntimeError, match="does not match"):
            _apply_feather_plan(sbs[:, :-2], plan)


# ---------------------------------------------------------------------------
#  CLI wiring: the swallowed-arg warning is gone and the flags reach the stream
# ---------------------------------------------------------------------------


class TestCliWiring:
    @pytest.fixture
    def rp(self):
        scripts = os.path.join(PROJECT_ROOT, "scripts")
        sys.path.insert(0, scripts)
        try:
            import run_pipeline

            yield run_pipeline
        finally:
            sys.modules.pop("run_pipeline", None)
            with contextlib.suppress(ValueError):
                sys.path.remove(scripts)

    @staticmethod
    def _standard_streaming_args(rp, *extra):
        """``--quality standard --edge-feather-start 165`` exactly as the operator runs it."""
        args = rp.parse_args(["--quality", "standard", "--edge-feather-start", "165", *extra])
        rp.apply_quality_preset(args)  # the preset is what flips the run to streaming
        assert args.streaming and args.stage == "all"
        return args

    def test_registry_lists_both_feather_flags(self, rp):
        assert rp._STREAMING_SUPPORTED["edge_feather_start"]
        assert rp._STREAMING_SUPPORTED["edge_feather_end"]

    def test_quality_standard_with_feather_no_longer_trips_the_swallowed_arg_warning(self, rp, caplog):
        args = self._standard_streaming_args(rp)
        with caplog.at_level(logging.WARNING, logger="vr180-pipeline"):
            rp._warn_streaming_unsupported_args(args)
        assert "does NOT honour" not in caplog.text
        assert "edge-feather" not in caplog.text

    def test_detector_is_still_live_for_a_genuinely_ignored_flag(self, rp, caplog):
        """Negative control: the silence above is not the detector having gone quiet."""
        args = self._standard_streaming_args(rp, "--upscale", "2")
        with caplog.at_level(logging.WARNING, logger="vr180-pipeline"):
            rp._warn_streaming_unsupported_args(args)
        assert "does NOT honour" in caplog.text
        assert "--upscale" in caplog.text
        assert "edge-feather" not in caplog.text

    def test_streaming_branch_forwards_the_flags_to_the_constructor(self, rp):
        from tests.test_streaming_passthrough import _make_streaming_magic_args

        args = _make_streaming_magic_args(edge_feather_start=165.0, edge_feather_end=None)
        captured: dict = {}

        def fake_ctor(**kwargs):
            captured.update(kwargs)
            inst = MagicMock()
            inst.process_stream.return_value = "out.mp4"
            return inst

        with (
            patch.object(rp, "parse_args", return_value=args),
            patch.object(rp, "apply_quality_preset"),
            patch.object(rp, "build_depth_backend", return_value=(None, "depth-anything")),
            patch.object(rp, "build_stereo_backend", return_value=(None, "default")),
            patch.object(rp, "StreamingPipeline", side_effect=fake_ctor),
            patch.object(rp, "_copy_audio_to_output"),
            patch.object(rp, "_maybe_copy_audio_from_input"),
            patch.object(rp, "_write_sidecar_from_args"),
            patch("pipeline.spherical_injector.inject_spherical_metadata"),
            patch("os.replace"),
        ):
            rp.main()
        assert captured["edge_feather_start"] == 165.0
        assert captured["edge_feather_end"] is None
