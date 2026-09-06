"""Tests for issue #274 (F-7) — ``SphereAccumulator`` wired into the streaming path.

#262 / PR #263 landed :class:`pipeline.sphere_accumulator.SphereAccumulator`
(temporal spherical accumulation: what earlier frames showed before it flowed
out of the frame fills the 126°→180° outer ring, faded 165→180) as a module
only.  This card wires it into :class:`StreamingPipeline` behind
``--sphere-accumulate {off,on}`` (default **off**) and
``--sphere-radial-scale`` — a preparation, not a new default.

Covered here (the #266 harness: fake depth/stereo backends, the real OpenCV
mapper at 192²/eye, cv2/ffmpeg mocked, no model):

  - ``off`` (default) ⇒ the bytes handed to the encoder are exactly the
    mapper's output, frame for frame, no alpha plane is derived and the
    accumulator class is never instantiated (spy on the module-global name);
    the #261 feather and the #267 fill are untouched when off;
  - ``on`` ⇒ frame 0 is the cold-start feathered frame (#263 ⇔ #266,
    byte-exact), the outer ring (the mapper's ``alpha == 0`` hole) has
    non-zero pixels by frame 5 and its coverage never decreases — with a
    constant source and with the #263 concentric-ring outflow fixture as
    per-frame source content; each eye equals a ``SphereAccumulator`` driven
    directly with the same RGBA (the "replace the eye with ``update()``'s
    output" contract); one accumulator per eye, fresh per stream; both
    fuse-loop branches; ``radial_scale <= 1`` composites only;
  - ``on`` + ``--edge-feather-*`` ⇒ the accumulator takes the angles and the
    per-frame feather is **not** applied on top: bytes identical to ``on``
    alone, a second attenuation would have changed them, and the equator ray
    descends exactly once to zero;
  - ``on`` + ``--outpaint gradient`` ⇒ one WARNING, fill skipped, output equal
    to ``on`` alone;
  - validation (mode, square canvas, radial scale) and the CLI wiring
    (defaults, parse-time rejection, swallowed-arg registry, forwarding).
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.outpainter import apply_edge_feather, sbs_edge_feather_weights  # noqa: E402
from pipeline.sphere_accumulator import SphereAccumulator  # noqa: E402
from pipeline.streaming_pipeline import DEFAULT_SPHERE_RADIAL_SCALE, StreamingPipeline  # noqa: E402
from tests.test_sphere_accumulator import _ring_pattern  # noqa: E402
from tests.test_streaming_feather import (  # noqa: E402
    _EYE,
    _HFOV,
    _SRC_H,
    _SRC_W,
    _alpha_calls,
    _FakeDepth,
    _FakeStereo,
    _FakeWholeClipStereo,
    _make_pipeline,
    _reference,
    _run,
    _src_rgb,
)
from tests.test_streaming_outpaint import _batch  # noqa: E402

_STREAM_LOGGER = "vr180-streaming"
_N = 8  # frames per accumulation run


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _spy():
    """Pass-through spy on the module-global ``SphereAccumulator`` the stream instantiates.

    ``wraps`` keeps the real class behind the mock, so instances are genuine
    accumulators while ``call_count`` counts constructions.
    """
    return patch("pipeline.streaming_pipeline.SphereAccumulator", wraps=SphereAccumulator)


def _fake_cap_seq(frames_rgb: list[np.ndarray]):
    """Like the #266 ``_fake_cap`` but yields a different source frame per index (BGR, as cv2 would)."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    h, w = frames_rgb[0].shape[:2]
    cap.get.side_effect = lambda prop: {3: w, 4: h, 7: float(len(frames_rgb)), 5: 30.0}.get(prop, 0.0)
    cap.read.side_effect = [(True, f[:, :, ::-1].copy()) for f in frames_rgb] + [(False, None)]
    return cap


def _run_seq(p: StreamingPipeline, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
    """Drive ``process_stream`` over *frames_rgb*; return the SBS frames written to the encoder pipe."""
    proc = MagicMock()
    proc.returncode = 0
    chunks: list[bytes] = []
    proc.stdin.write.side_effect = chunks.append
    with (
        patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=_fake_cap_seq(frames_rgb)),
        patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
    ):
        p.process_stream("in.mp4", "out.mp4")
    return [np.frombuffer(b, dtype=np.uint8).reshape(_EYE, 2 * _EYE, 3) for b in chunks]


def _outflow_sources(n: int = _N) -> list[np.ndarray]:
    """The #263 concentric-ring forward-flight fixture as 36×64 source frames.

    ``_ring_pattern(k)`` is ``_ring_pattern(k - 1)`` expanded radially by
    1.05 about the centre; the 16:9 crop keeps the flow centred.
    """
    size = _SRC_W
    r0 = (size - _SRC_H) // 2
    return [_ring_pattern(k, size)[r0 : r0 + _SRC_H].copy() for k in range(n)]


def _hole() -> np.ndarray:
    """The outer ring: the mapper's ``alpha == 0`` region of the SBS frame (126° source)."""
    _, alpha = _reference()
    return alpha == 0


def _hole_nonzero_fraction(frame: np.ndarray) -> float:
    """Fraction of the outer ring (original alpha == 0) that carries a non-zero pixel."""
    return float(np.mean(frame[_hole()].max(axis=1) > 0))


def _left_eye_rgba(sbs: np.ndarray) -> np.ndarray:
    """The left eye of a mapper SBS frame tiled with the mapper's alpha plane — what the stream feeds ``update``."""
    _, alpha = _reference()
    return np.dstack((sbs[:, :_EYE], alpha[:, :_EYE]))


def _equator_ray(frame: np.ndarray) -> np.ndarray:
    """Channel-0 brightness along the left eye's equator, centre → 180° rim."""
    return frame[_EYE // 2, _EYE // 2 : _EYE, 0].astype(int)


def _accumulate_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == _STREAM_LOGGER and "sphere-accumulate" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
#  Regression: off (default) ⇒ pixel-identical, accumulator never built
# ---------------------------------------------------------------------------


class TestOffIsByteIdentical:
    def test_default_is_off(self):
        p = _make_pipeline()
        assert p.sphere_accumulate == "off"
        assert p._accumulate is False
        assert p._sphere_fade is None and p._accumulators is None
        assert p.sphere_radial_scale == DEFAULT_SPHERE_RADIAL_SCALE == 1.05
        assert not p._stage35_enabled

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"sphere_accumulate": "off"}, {"sphere_accumulate": "off", "sphere_radial_scale": 1.5}],
    )
    def test_encoder_receives_exactly_the_mapper_output_and_no_accumulator_exists(self, kwargs):
        with _spy() as spy:
            p = _make_pipeline(**kwargs)
            p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
            frames = _run(p, num_frames=3)
        sbs, _ = _reference()
        assert len(frames) == 3
        for f in frames:
            assert np.array_equal(f, sbs), "off: bytes are exactly the mapper's output, frame for frame"
        assert _alpha_calls(p.eq_mapper) == [], "off: no alpha plane is derived — no extra mapper work at all"
        spy.assert_not_called()
        assert p._accumulators is None and p._boundary_plan is None

    def test_off_leaves_the_261_feather_and_the_267_fill_exactly_as_before(self):
        with _spy() as spy:
            feathered = _run(_make_pipeline(edge_feather_start=165, edge_feather_end=180), num_frames=2)
            filled = _run(_make_pipeline(outpaint="gradient"), num_frames=2)
        spy.assert_not_called()
        for f in feathered:
            assert np.array_equal(f, _batch("none", edge_feather_start=165, edge_feather_end=180))
        for f in filled:
            assert np.array_equal(f, _batch("gradient"))


# ---------------------------------------------------------------------------
#  on: the outer ring is filled from history, per eye, per frame
# ---------------------------------------------------------------------------


class TestOnFillsTheOuterRing:
    def test_frame0_is_the_cold_start_feathered_frame(self):
        """#263 cold start (input + 165→180 fade, byte-exact) ⇔ the #266 feathered stream."""
        p = _make_pipeline(sphere_accumulate="on")
        assert p._accumulate and p._stage35_enabled and p._sphere_fade == (165.0, 180.0)
        out = _run(p, num_frames=1)[0]
        assert np.array_equal(out, _batch("none", edge_feather_start=165, edge_feather_end=180))
        assert _hole_nonzero_fraction(out) == 0.0, "nothing to remember yet — the ring is black"

    def test_ring_fills_and_coverage_never_decreases_with_a_constant_source(self):
        frames = _run(_make_pipeline(sphere_accumulate="on"), num_frames=_N)
        nz = [_hole_nonzero_fraction(f) for f in frames]
        assert nz[0] == 0.0
        assert nz[4] > 0.0, f"the outer ring must carry content by frame 5: {nz}"
        assert all(b >= a for a, b in itertools.pairwise(nz)), f"coverage must never decrease: {nz}"
        assert nz[-1] > nz[1] > 0.0, f"and it really grows with the flow: {nz}"
        sbs, alpha = _reference()
        interior = cv2.erode((alpha > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        assert np.array_equal(frames[-1][interior], sbs[interior]), "deep inside the FOV the live frame wins"
        for f in frames:
            assert np.array_equal(f[:, :_EYE], f[:, _EYE:]), "identical eyes in ⇒ identical eyes out"

    def test_ring_coverage_grows_monotonically_with_the_263_outflow_fixture(self):
        """The #263 concentric-ring forward flight as source content: acceptance criterion 2."""
        frames = _run_seq(_make_pipeline(sphere_accumulate="on"), _outflow_sources())
        assert len(frames) == _N
        nz = [_hole_nonzero_fraction(f) for f in frames]
        assert nz[0] == 0.0, "frame 0: the hole is empty"
        assert nz[4] > 0.0, f"non-zero pixels in the outer ring by frame 5: {nz}"
        assert all(b >= a for a, b in itertools.pairwise(nz)), f"coverage monotone non-decreasing: {nz}"
        assert nz[-1] > 0.25, f"after {_N} frames at ×1.05 a real share of the ring is remembered: {nz[-1]:.3f}"
        for f in frames:
            assert np.array_equal(f[:, :_EYE], f[:, _EYE:])

    def test_each_eye_equals_the_accumulator_driven_directly(self):
        """The card's contract: alpha from ``map_single(with_alpha=True)``, ``update()`` output replaces the eye."""
        sources = _outflow_sources()
        frames = _run_seq(_make_pipeline(sphere_accumulate="on"), sources)
        mapper = _make_pipeline().eq_mapper
        ref = SphereAccumulator(_EYE, 165.0, 180.0)
        for src, out in zip(sources, frames, strict=True):
            expected = ref.update(_left_eye_rgba(mapper.map_stereo_pair(src, src)), DEFAULT_SPHERE_RADIAL_SCALE)
            assert np.array_equal(out[:, :_EYE], expected), "left eye == SphereAccumulator.update, byte-exact"
            assert np.array_equal(out[:, _EYE:], expected), "right eye too (its own accumulator, same input)"

    def test_radial_scale_one_composites_only_and_the_ring_stays_black(self):
        """#263 semantics: ``radial_scale <= 1`` never expands the buffer."""
        p = _make_pipeline(sphere_accumulate="on", sphere_radial_scale=1.0)
        frames = _run(p, num_frames=_N)
        assert all(np.array_equal(f, frames[0]) for f in frames), "a static composite is stable"
        assert all(_hole_nonzero_fraction(f) == 0.0 for f in frames)
        assert all(not acc.alpha[_hole()[:, :_EYE]].any() for acc in p._accumulators), "the buffer never grows"

    def test_radial_scale_reaches_the_accumulator(self):
        """A larger scale flows content outward faster ⇒ more of the ring covered after the same frames."""
        slow = _run(_make_pipeline(sphere_accumulate="on", sphere_radial_scale=1.02), num_frames=4)[-1]
        fast = _run(_make_pipeline(sphere_accumulate="on", sphere_radial_scale=1.10), num_frames=4)[-1]
        assert _hole_nonzero_fraction(fast) > _hole_nonzero_fraction(slow) > 0.0

    def test_one_accumulator_per_eye_fresh_per_stream(self):
        with _spy() as spy:
            p = _make_pipeline(sphere_accumulate="on")
            assert p._accumulators is None, "nothing is built at construction"
            _run(p, num_frames=3)
            assert spy.call_count == 2, "exactly one accumulator per eye"
            first = p._accumulators
            left, right = first
            assert isinstance(left, SphereAccumulator) and isinstance(right, SphereAccumulator)
            assert left is not right
            assert left.frames_seen == right.frames_seen == 3
            assert left.fade == right.fade == (165.0, 180.0) and left.size == right.size == _EYE
            _run(p, num_frames=2)
            assert spy.call_count == 4, "a second stream builds fresh accumulators"
            assert p._accumulators[0] is not first[0] and p._accumulators[1] is not first[1]
            assert p._accumulators[0].frames_seen == 2, "a new clip must not inherit the previous ring"

    def test_alpha_is_derived_once_and_no_feather_or_fill_plan_is_built(self):
        p = _make_pipeline(sphere_accumulate="on")
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=3)
        assert len(_alpha_calls(p.eq_mapper)) == 1, "the valid region is geometry-only: derived once, not per frame"
        assert len(frames) == 3
        plan = p._boundary_plan
        assert plan is not None
        _, alpha = _reference()
        assert plan.alpha_eye is not None and np.array_equal(plan.alpha_eye, alpha[:, :_EYE])
        assert plan.alpha_eye.flags.c_contiguous
        assert plan.feather is None and plan.fill_mask is None

    def test_wholeclip_stereo_branch_is_accumulated_too(self, tmp_path):
        """The precomputed-L/R fuse loop (StereoCrafter) goes through the same Stage 3 as the per-frame loop."""
        src = _src_rgb()
        p = _make_pipeline(sphere_accumulate="on", stereo_renderer=_FakeWholeClipStereo(), temp_dir=str(tmp_path))
        left = [src.copy() for _ in range(3)]
        right = [src.copy() for _ in range(3)]
        with patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]):
            frames = _run(p, num_frames=3)
        expected = _run(_make_pipeline(sphere_accumulate="on"), num_frames=3)
        assert len(frames) == 3
        for f, e in zip(frames, expected, strict=True):
            assert np.array_equal(f, e)
        assert _hole_nonzero_fraction(frames[-1]) > 0.0

    def test_equirect_stage_timing_still_reported(self):
        p = _make_pipeline(sphere_accumulate="on")
        _run(p, num_frames=2)
        assert p.stage_timings["equirect"] > 0.0
        assert set(p.stage_timings) >= {"depth", "stereo", "equirect", "metadata", "encode", "_total"}


# ---------------------------------------------------------------------------
#  on + --edge-feather-*: the accumulator owns the fade — attenuated once
# ---------------------------------------------------------------------------


class TestOnPlusFeatherIsASingleFade:
    def test_external_feather_is_not_applied_a_second_time(self):
        on = _run(_make_pipeline(sphere_accumulate="on"), num_frames=6)
        p = _make_pipeline(sphere_accumulate="on", edge_feather_start=165, edge_feather_end=180)
        both = _run(p, num_frames=6)
        assert p.edge_feather == (165.0, 180.0) and p._sphere_fade == (165.0, 180.0)
        assert p._boundary_plan.feather is None, "the per-frame feather plan is not built"
        assert all(acc.fade == (165.0, 180.0) for acc in p._accumulators)
        for a, b in zip(on, both, strict=True):
            assert np.array_equal(a, b), "on + feather ≡ on: the 165→180 fade is applied exactly once"
        # Negative control: had the #261 feather run on top, the ring would have darkened again.
        _, alpha = _reference()
        twice = apply_edge_feather(on[-1], sbs_edge_feather_weights(alpha, 165.0, 180.0))
        assert np.count_nonzero(np.any(twice != both[-1], axis=2)) > 0, "a second attenuation is detectable"
        assert twice.sum() < both[-1].sum()

    def test_equator_ray_descends_once_to_zero_beyond_the_source_edge(self):
        """One fade only: full brightness (± the mapper's own edge ringing) out to a single monotone ramp, then 0.

        The mapper's interpolation leaves ±2 ringing on the last source
        pixels (the "anti-aliased last pixel" of #266); the accumulator
        remembers those pixels as they are and carries them outward, so the
        ring is flat only to within that tolerance.  A *second* fade would show
        up as a second run below full brightness — anchored at the 126° source
        edge — which is exactly what is excluded here.
        """
        out = _run(_make_pipeline(sphere_accumulate="on", edge_feather_start=165, edge_feather_end=180), num_frames=6)[
            -1
        ]
        sbs, alpha = _reference()
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        tol = 3
        ray = _equator_ray(out)
        edge = int(np.flatnonzero(alpha[_EYE // 2, _EYE // 2 : _EYE] > 0)[-1])  # last source pixel (126°)
        assert ray[0] == orig, "centre untouched"
        assert ray[-1] == 0, "black at the 180° rim"
        assert ray[edge] >= orig - tol, "the source edge stays bright: the fade anchors at the *accumulated* edge"
        assert ray[edge + 1] > 0, "the first former-hole pixel is remembered content, not black"
        below = np.flatnonzero(ray < orig - tol)
        assert below.size >= 3, "a real ramp, not a hard cut"
        assert np.array_equal(below, np.arange(below[0], below[-1] + 1)), "exactly one run below full brightness"
        assert below[0] > edge + 1, "…and it starts beyond the source edge, inside the remembered ring"
        assert np.all(ray[: below[0]] >= orig - tol), "flat (to the mapper's ringing) all the way to the ramp"
        ramp = ray[below[0] :]
        assert np.all(np.diff(ramp) <= 0), "the ramp is monotone non-increasing to the rim — one descent"
        assert (-np.diff(ramp)).max() <= 0.5 * orig, "no step — a hard edge would be a full-brightness drop"
        assert ramp[-1] == 0 and not ray[below[-1] + 1 :].any(), "and it ends in black that stays black"

    def test_custom_feather_angles_parametrise_the_accumulator(self):
        p = _make_pipeline(sphere_accumulate="on", edge_feather_start=150)
        wide = _run(p, num_frames=4)
        assert p.edge_feather == (150.0, 180.0) and p._sphere_fade == (150.0, 180.0)
        assert all(acc.fade == (150.0, 180.0) for acc in p._accumulators)
        default = _run(_make_pipeline(sphere_accumulate="on"), num_frames=4)
        sbs, _ = _reference()
        changed = lambda f: np.count_nonzero(np.any(f != sbs, axis=2))  # noqa: E731
        assert changed(wide[0]) > changed(default[0]) > 0, "150→180 darkens more of frame 0 than 165→180"
        assert not np.array_equal(wide[-1], default[-1])
        assert np.array_equal(wide[0], _batch("none", edge_feather_start=150, edge_feather_end=180))

    @pytest.mark.parametrize(("start", "end"), [(170, 165), (-1, 180), (0, 181)])
    def test_invalid_angles_still_raise_at_construction(self, start, end):
        with pytest.raises(ValueError):
            _make_pipeline(sphere_accumulate="on", edge_feather_start=start, edge_feather_end=end)


# ---------------------------------------------------------------------------
#  on + --outpaint gradient: nothing to add — skipped loudly, once
# ---------------------------------------------------------------------------


class TestOnPlusOutpaintDegrades:
    def test_gradient_is_skipped_with_one_warning_and_output_equals_on(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_STREAM_LOGGER):
            p = _make_pipeline(sphere_accumulate="on", outpaint="gradient")
            frames = _run(p, num_frames=3)
        warnings = _accumulate_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one degrade warning, got {warnings}"
        assert "outpaint gradient" in warnings[0] and "skipping" in warnings[0]
        assert p.outpaint == "gradient", "the requested mode is preserved (#243 passthrough contract)"
        assert p._fill_mode == "none"
        assert p._boundary_plan.fill_mask is None
        expected = _run(_make_pipeline(sphere_accumulate="on"), num_frames=3)
        for f, e in zip(frames, expected, strict=True):
            assert np.array_equal(f, e)

    def test_on_alone_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_STREAM_LOGGER):
            _run(_make_pipeline(sphere_accumulate="on"), num_frames=2)
        assert _accumulate_warnings(caplog) == []


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_mode_raises_at_construction(self):
        with pytest.raises(ValueError, match="sphere_accumulate"):
            _make_pipeline(sphere_accumulate="auto")

    @staticmethod
    def _rect(**kwargs) -> StreamingPipeline:
        return StreamingPipeline(
            device="cpu",
            output_width=_EYE,
            output_height=_EYE // 2,
            src_hfov=_HFOV,
            use_ffmpeg=False,
            hw_encoder=False,
            depth_estimator=_FakeDepth(),
            stereo_renderer=_FakeStereo(),
            **kwargs,
        )

    def test_on_needs_a_square_canvas(self):
        with pytest.raises(ValueError, match="square"):
            self._rect(sphere_accumulate="on")
        assert self._rect().sphere_accumulate == "off", "off does not care about the canvas shape"

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_radial_scale_must_be_finite_positive_when_on(self, scale):
        with pytest.raises(ValueError, match="sphere_radial_scale"):
            _make_pipeline(sphere_accumulate="on", sphere_radial_scale=scale)
        p = _make_pipeline(sphere_radial_scale=scale)  # off: the value is unused, never validated
        assert p.sphere_accumulate == "off"

    def test_accumulate_sbs_guards(self):
        p = _make_pipeline(sphere_accumulate="on")
        _, alpha = _reference()
        alpha_eye = alpha[:, :_EYE]
        with pytest.raises(RuntimeError, match="no live accumulators"):
            p._accumulate_sbs(np.zeros((_EYE, 2 * _EYE, 3), np.uint8), alpha_eye)
        p._accumulators = (SphereAccumulator(_EYE), SphereAccumulator(_EYE))
        with pytest.raises(RuntimeError, match="does not match"):
            p._accumulate_sbs(np.zeros((_EYE, 2 * _EYE - 2, 3), np.uint8), alpha_eye)


# ---------------------------------------------------------------------------
#  CLI wiring (mirrors the #266 checks): defaults, parse-time rejection,
#  swallowed-arg registry, forwarding into the constructor
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

    def test_parser_defaults_are_off(self, rp):
        args = rp.parse_args([])
        assert args.sphere_accumulate == "off"
        assert args.sphere_radial_scale == DEFAULT_SPHERE_RADIAL_SCALE == 1.05

    def test_flags_parse(self, rp):
        args = rp.parse_args(["--sphere-accumulate", "on", "--sphere-radial-scale", "1.1"])
        assert args.sphere_accumulate == "on" and args.sphere_radial_scale == 1.1

    def test_unknown_mode_is_rejected_at_parse_time(self, rp):
        with pytest.raises(SystemExit):
            rp.parse_args(["--sphere-accumulate", "auto"])

    @pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
    def test_bad_radial_scale_is_rejected_at_parse_time_when_on(self, rp, scale):
        with pytest.raises(SystemExit):
            rp.parse_args(["--sphere-accumulate", "on", "--sphere-radial-scale", scale])
        args = rp.parse_args(["--sphere-radial-scale", scale])  # off: unused, accepted
        assert args.sphere_accumulate == "off"

    def test_registry_lists_both_flags(self, rp):
        assert rp._STREAMING_SUPPORTED["sphere_accumulate"]
        assert rp._STREAMING_SUPPORTED["sphere_radial_scale"]

    @staticmethod
    def _standard_streaming_args(rp, *extra):
        args = rp.parse_args(
            ["--quality", "standard", "--sphere-accumulate", "on", "--sphere-radial-scale", "1.1", *extra]
        )
        rp.apply_quality_preset(args)
        assert args.streaming and args.stage == "all"
        return args

    def test_quality_standard_with_accumulate_does_not_trip_the_swallowed_arg_warning(self, rp, caplog):
        args = self._standard_streaming_args(rp)
        with caplog.at_level(logging.WARNING, logger="vr180-pipeline"):
            rp._warn_streaming_unsupported_args(args)
        assert "does NOT honour" not in caplog.text
        assert "sphere" not in caplog.text

    def test_detector_is_still_live_for_a_genuinely_ignored_flag(self, rp, caplog):
        args = self._standard_streaming_args(rp, "--upscale", "2")
        with caplog.at_level(logging.WARNING, logger="vr180-pipeline"):
            rp._warn_streaming_unsupported_args(args)
        assert "does NOT honour" in caplog.text and "--upscale" in caplog.text
        assert "sphere" not in caplog.text

    def test_streaming_branch_forwards_the_flags_to_the_constructor(self, rp):
        from tests.test_streaming_passthrough import _make_streaming_magic_args

        args = _make_streaming_magic_args(sphere_accumulate="on", sphere_radial_scale=1.1)
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
        assert captured["sphere_accumulate"] == "on"
        assert captured["sphere_radial_scale"] == 1.1
