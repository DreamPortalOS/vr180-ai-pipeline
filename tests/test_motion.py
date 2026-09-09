"""Tests for camera-motion measurement (#331) — ``pipeline/motion.py`` + ``scripts/analyze_motion.py``.

Three of these tests exist because a specific mistake was already made and
must not be repeatable:

``test_sign_convention_*``
    #327's predecessor scripts mixed two opposite roll conventions, which made
    two estimators that agreed to 0.03° report a correlation of -0.861 and read
    as contradicting each other.  A flipped sign shipped to a motion seat leans
    the chair the wrong way, so the convention is nailed down here — once
    against a hand-written matrix (no OpenCV semantics involved) and once
    end-to-end through the real estimator.

``test_cross_check_*`` / ``test_biased_front_end_is_caught_end_to_end``
    #327 reported 21.16° of camera roll that was really 1.37°, because it
    integrated a biased per-frame estimate and never checked the sum against a
    direct measurement.  The end-to-end test feeds the incremental path a
    deliberately biased front end while the direct path stays honest, and
    asserts the warning fires.  Delete the cross-check and that test goes red.

``test_*_never_zero`` / ``test_flat_frames_*``
    A frame whose motion cannot be measured must report ``nan``, not 0.0.
    Zero reads as "the camera was level", which is the single most misleading
    thing this module could say about a clip that flies straight forward.

Everything synthetic: numpy textures warped by known matrices, no models, no
GPU, no ffmpeg, no repo writes (``tmp_path`` only).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import analyze_motion

from pipeline.motion import (
    VERDICT_CONSISTENT,
    VERDICT_DRIFT_SUSPECTED,
    VERDICT_UNAVAILABLE,
    BaselineSegment,
    ConfidenceGates,
    FrameMotion,
    LongBaselineCheck,
    MotionTrack,
    TrackingConfig,
    band_label,
    band_rms_deg,
    cross_check_roll,
    dominant_band,
    estimate_motion,
    estimate_pairwise,
    inlier_spread,
    integrate_roll,
    roll_deg_from_affine,
)

# ---------------------------------------------------------------------------
# Synthetic scene: a textured plate warped by an exactly-known similarity
# ---------------------------------------------------------------------------

FPS = 24.0


def _texture(size: int, seed: int = 7) -> np.ndarray:
    """Band-limited noise: enough corners for Shi-Tomasi, smooth enough for sub-pixel LK."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    acc = np.zeros((size, size), np.float32)
    for sigma in (2, 4, 8, 16, 32):
        acc += cv2.GaussianBlur(noise, (0, 0), sigma) * sigma
    acc -= acc.min()
    return (255.0 * acc / max(float(acc.max()), 1e-6)).astype(np.uint8)


def _warp_matrix(big: int, pad: int, angle_deg: float, zoom: float) -> np.ndarray:
    """The exact 2x3 similarity that renders one frame from the padded plate.

    Rotation/zoom happen about the *plate* centre, then the result is shifted
    so that the plate centre lands on the output centre.  Because every frame
    is rendered from the same plate at an absolute angle (never recursively
    from its predecessor), the ground truth carries no accumulated error of
    its own — which is the whole point of a calibration rig.
    """
    matrix = cv2.getRotationMatrix2D((big / 2.0, big / 2.0), angle_deg, zoom)
    matrix[0, 2] -= pad
    matrix[1, 2] -= pad
    return matrix


def _render(plate: np.ndarray, out: int, pad: int, angle_deg: float, zoom: float = 1.0) -> np.ndarray:
    big = out + 2 * pad
    return cv2.warpAffine(plate, _warp_matrix(big, pad, angle_deg, zoom), (out, out), flags=cv2.INTER_LINEAR)


def _sequence(
    n: int,
    *,
    out: int,
    pad: int,
    roll_per_frame: float,
    zoom_per_frame: float = 0.0,
    seed: int = 7,
) -> list[np.ndarray]:
    plate = _texture(out + 2 * pad, seed=seed)
    return [_render(plate, out, pad, roll_per_frame * i, 1.0 + zoom_per_frame * i) for i in range(n)]


# -- module-scoped rigs (each is a fraction of a second, but reused a lot) ---

ROLL_RAMP_N, ROLL_RAMP_STEP = 80, 0.05
SLOW_ROLL_N, SLOW_ROLL_STEP = 100, 0.02
FORWARD_N, FORWARD_ZOOM = 60, 0.010


@pytest.fixture(scope="module")
def roll_ramp_frames() -> list[np.ndarray]:
    """Steady 0.05°/frame roll *plus* forward zoom — the realistic calibration case."""
    return _sequence(ROLL_RAMP_N, out=448, pad=160, roll_per_frame=ROLL_RAMP_STEP, zoom_per_frame=0.004)


@pytest.fixture(scope="module")
def roll_ramp_track(roll_ramp_frames) -> MotionTrack:
    return estimate_motion(roll_ramp_frames, fps=FPS, stride=30)


@pytest.fixture(scope="module")
def slow_roll_frames() -> list[np.ndarray]:
    """0.02°/frame — the regime an ORB front end measures at a gain of x0.01."""
    return _sequence(SLOW_ROLL_N, out=512, pad=120, roll_per_frame=SLOW_ROLL_STEP, seed=11)


@pytest.fixture(scope="module")
def slow_roll_track(slow_roll_frames) -> MotionTrack:
    return estimate_motion(slow_roll_frames, fps=FPS, stride=40)


@pytest.fixture(scope="module")
def no_roll_track() -> MotionTrack:
    """The noise floor for the slow-roll test: same rig, zero true roll."""
    frames = _sequence(SLOW_ROLL_N, out=512, pad=120, roll_per_frame=0.0, zoom_per_frame=0.002, seed=11)
    return estimate_motion(frames, fps=FPS, stride=40)


@pytest.fixture(scope="module")
def forward_frames() -> list[np.ndarray]:
    """Pure radial zoom, zero rotation — where ORB hallucinates 2.62° of roll."""
    return _sequence(FORWARD_N, out=448, pad=200, roll_per_frame=0.0, zoom_per_frame=FORWARD_ZOOM)


@pytest.fixture(scope="module")
def forward_track(forward_frames) -> MotionTrack:
    return estimate_motion(forward_frames, fps=FPS, stride=30)


# ---------------------------------------------------------------------------
# Sign convention — the thing that must never drift
# ---------------------------------------------------------------------------


def test_sign_convention_from_hand_written_matrix() -> None:
    """A content rotation east->north (counter-clockwise on screen) is +roll.

    Written out by hand so the assertion does not inherit any OpenCV
    convention: the matrix maps the unit vector (1, 0) — pointing right — onto
    (0, -1), which with pixel y pointing *down* points up the screen.  Right
    turning into up is counter-clockwise, and the module defines that as the
    camera having rolled clockwise, i.e. a positive reading.
    """
    ccw_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    moved = ccw_90[:, :2] @ np.array([1.0, 0.0])
    assert moved[1] < 0, "sanity: the sample point must end up higher on screen (y down)"
    assert roll_deg_from_affine(ccw_90) == pytest.approx(90.0)

    cw_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    assert (cw_90[:, :2] @ np.array([1.0, 0.0]))[1] > 0, "sanity: this one moves the point down"
    assert roll_deg_from_affine(cw_90) == pytest.approx(-90.0)

    assert roll_deg_from_affine(np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])) == pytest.approx(0.0)


@pytest.mark.parametrize("angle", [-6.0, -1.5, 1.5, 6.0])
def test_sign_convention_end_to_end(angle: float) -> None:
    """``getRotationMatrix2D(centre, +A)`` between two frames is recovered as ``roll_deg = +A``.

    This is the identity downstream code should hold in its head, so it is
    asserted rather than described.  It also fixes the magnitude: no factor of
    two, no radians, no half-angle.
    """
    out, pad = 384, 140
    plate = _texture(out + 2 * pad)
    first = _render(plate, out, pad, 0.0)
    second = _render(plate, out, pad, angle)

    motion = estimate_pairwise(first, second)
    assert motion.trusted, motion.reason
    assert motion.roll_deg == pytest.approx(angle, abs=0.05)
    assert math.copysign(1.0, motion.roll_deg) == math.copysign(1.0, angle)


def test_positive_roll_means_the_picture_turns_counter_clockwise() -> None:
    """Tie the sign to something visible: +roll moves a right-of-centre point upward."""
    out, pad = 384, 140
    matrix = _warp_matrix(out + 2 * pad, pad, angle_deg=5.0, zoom=1.0)
    centre = np.array([out / 2.0, out / 2.0])
    # Where the plate point that started level with the centre, to its right,
    # lands after the +5 deg render.
    plate_point = np.array([(out + 2 * pad) / 2.0 + 100.0, (out + 2 * pad) / 2.0])
    landed = matrix[:, :2] @ plate_point + matrix[:, 2]

    assert landed[1] < centre[1], "positive roll must lift a right-of-centre point (y is down)"
    assert roll_deg_from_affine(matrix) == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Accuracy against synthetic ground truth
# ---------------------------------------------------------------------------


def test_synthetic_ground_truth_rms_and_drift(roll_ramp_track: MotionTrack) -> None:
    """Per-frame RMS <= 0.2 deg and accumulated drift <= 0.5 deg / 100 frames.

    The acceptance numbers of #331.  Both are met with two orders of magnitude
    to spare, which is the margin that makes the long-baseline cross-check a
    real check rather than a noise detector.
    """
    truth_increments = np.full(ROLL_RAMP_N - 1, ROLL_RAMP_STEP)
    estimated = np.array([m.roll_deg for m in roll_ramp_track.increments], dtype=float)
    assert np.isfinite(estimated).all(), "every frame of a clean synthetic clip should be measurable"

    rms = float(np.sqrt(np.mean((estimated - truth_increments) ** 2)))
    assert rms <= 0.2, f"per-frame RMS {rms:.4f}deg exceeds the 0.2deg budget"

    truth_total = ROLL_RAMP_STEP * (ROLL_RAMP_N - 1)
    drift_per_100 = (roll_ramp_track.net_roll_deg - truth_total) / (ROLL_RAMP_N - 1) * 100.0
    assert abs(drift_per_100) <= 0.5, f"drift {drift_per_100:.4f}deg/100f exceeds the 0.5deg budget"

    ratio = roll_ramp_track.cumulative_p2p_deg / truth_total
    assert 0.95 <= ratio <= 1.10, f"peak-to-peak gain {ratio:.3f} outside [0.95, 1.10]"


def test_pure_forward_motion_does_not_hallucinate_roll(forward_track: MotionTrack) -> None:
    """A clip that only flies forward must report ~zero roll (ORB reports 2.62 deg)."""
    assert forward_track.reliable_ratio == 1.0
    assert abs(forward_track.net_roll_deg) <= 0.5
    assert forward_track.cumulative_p2p_deg <= 0.5
    # The zoom is real and must still be seen, otherwise "no roll" is trivial.
    scales = np.array([m.scale for m in forward_track.increments])
    assert float(np.median(scales)) == pytest.approx(1.0 + FORWARD_ZOOM, rel=0.05)


def test_slow_roll_is_detected(slow_roll_track: MotionTrack, no_roll_track: MotionTrack) -> None:
    """0.02 deg/frame must be measured, not lost in the noise.

    This is the case that disqualified ORB: on slow roll with no translation
    to push features apart, integer-precision keypoints give a gain of x0.01,
    i.e. the rotation is simply invisible.  The comparison against a zero-roll
    rig of the same texture and size makes "detected" mean something.
    """
    truth = SLOW_ROLL_STEP * (SLOW_ROLL_N - 1)
    measured = slow_roll_track.net_roll_deg
    assert 0.80 * truth <= measured <= 1.25 * truth, f"gain {measured / truth:.3f} on a {truth:.2f}deg roll"

    per_frame_mean = float(np.mean(slow_roll_track.trusted_roll_increments))
    assert per_frame_mean == pytest.approx(SLOW_ROLL_STEP, rel=0.25)

    noise_floor = abs(no_roll_track.net_roll_deg)
    assert measured > 10.0 * max(noise_floor, 1e-3), (
        f"a {truth:.2f}deg roll ({measured:.3f}deg measured) must stand clear of the "
        f"{noise_floor:.4f}deg zero-roll noise floor"
    )


# ---------------------------------------------------------------------------
# Confidence: unmeasurable is not the same as zero
# ---------------------------------------------------------------------------


def test_flat_frames_are_rejected_and_never_read_as_zero() -> None:
    """Texture-free frames yield ``trusted=False`` and ``roll_deg=nan``, without raising."""
    out, pad = 320, 120
    plate = _texture(out + 2 * pad, seed=3)
    frames = [_render(plate, out, pad, 0.08 * i) for i in range(12)]
    for blind in (5, 6):
        frames[blind] = np.full((out, out), 128, dtype=np.uint8)

    track = estimate_motion(frames, fps=FPS, stride=4)

    blind_steps = [4, 5, 6]  # the pairs that touch a flat frame
    for step in blind_steps:
        motion = track.increments[step]
        assert not motion.trusted
        assert math.isnan(motion.roll_deg), "an unmeasurable step must be nan, never 0.0"
        assert motion.reason, "a rejected frame must say which gate rejected it"

    assert track.untrusted_indices == blind_steps
    assert track.n_untrusted == 3
    assert track.reliable_ratio == pytest.approx(1.0 - 3 / 11)
    assert track.cumulative_filled[[s + 1 for s in blind_steps]].all()
    assert not track.cumulative_filled[[1, 2, 3, 11]].any()
    assert np.isfinite(track.cumulative_roll_deg).all(), "the bridged curve stays usable"


def test_integrate_roll_bridges_gaps_instead_of_zeroing_them() -> None:
    """An untrusted step is interpolated from its neighbours, and flagged."""
    increments = [
        _fake_increment(0, 1.0),
        _fake_increment(1, 1.0),
        _fake_increment(2, None),  # unmeasurable
        _fake_increment(3, 1.0),
    ]
    cumulative, filled = integrate_roll(increments)

    np.testing.assert_allclose(cumulative, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert filled.tolist() == [False, False, False, True, False]
    # The zeroing bug would have produced 3.0 here, understating the roll by a
    # whole step and reporting the clip as steadier than it is.
    assert cumulative[-1] == pytest.approx(4.0)


def test_integrate_roll_returns_nan_when_nothing_is_measurable() -> None:
    """No trusted step anywhere means "unknown", not "no motion"."""
    cumulative, filled = integrate_roll([_fake_increment(i, None) for i in range(4)])
    assert np.isnan(cumulative).all()
    assert filled.all()

    track = MotionTrack(
        increments=[_fake_increment(i, None) for i in range(4)],
        cumulative_roll_deg=cumulative,
        cumulative_filled=filled,
        long_baseline=_unavailable_check(),
        n_frames=5,
    )
    assert track.reliable_ratio == 0.0
    assert math.isnan(track.net_roll_deg)
    assert math.isnan(track.cumulative_p2p_deg)


def test_confidence_gate_names_the_failing_criterion() -> None:
    """Each gate is reported by name so a failure is actionable, not a shrug."""
    out, pad = 320, 120
    plate = _texture(out + 2 * pad, seed=5)
    pair = (_render(plate, out, pad, 0.0), _render(plate, out, pad, 1.0))

    impossible = ConfidenceGates(min_tracked=10**6)
    motion = estimate_pairwise(*pair, gates=impossible)
    assert not motion.trusted
    assert "n_tracked" in motion.reason
    assert math.isnan(motion.roll_deg)
    # The rejected estimate is still preserved for diagnosis...
    assert motion.raw_roll_deg == pytest.approx(1.0, abs=0.05)
    # ...but must not leak into the trusted reading.
    assert estimate_pairwise(*pair).roll_deg == pytest.approx(1.0, abs=0.05)


def test_inlier_spread_separates_uniform_coverage_from_a_clump() -> None:
    rng = np.random.default_rng(0)
    uniform = rng.uniform(0, 400, size=(500, 2))
    clump = rng.uniform(190, 210, size=(500, 2))
    assert inlier_spread(uniform, (400, 400)) == pytest.approx(0.577, abs=0.08)
    assert inlier_spread(clump, (400, 400)) < 0.05
    assert inlier_spread(np.zeros((1, 2)), (400, 400)) == 0.0


def test_estimate_motion_rejects_degenerate_input() -> None:
    frame = np.zeros((32, 32), np.uint8)
    with pytest.raises(ValueError, match="at least 2 frames"):
        estimate_motion([frame])
    with pytest.raises(ValueError, match="stride"):
        estimate_motion([frame, frame], stride=0)


# ---------------------------------------------------------------------------
# Long-baseline cross-check — the #327 trap
# ---------------------------------------------------------------------------


def test_cross_check_flags_the_issue_327_numbers() -> None:
    """The exact readings that fooled #327 must come back as a warning.

    Integrated 21.16 deg over 240 frames of the drone clip; direct
    registration of frame 0 against frame 239 gave 1.37 deg.  A cross-check
    that does not fire here is decoration.
    """
    endpoint = BaselineSegment(
        start=0,
        end=239,
        direct_roll_deg=1.37,
        integrated_roll_deg=21.16,
        diff_deg=21.16 - 1.37,
        direct_trusted=True,
        direct_n_tracked=400,
        direct_inlier_ratio=0.255,
    )
    check = cross_check_roll([], endpoint, stride=60, threshold_deg=2.0)

    assert check.verdict == VERDICT_DRIFT_SUSPECTED
    assert check.drift_warning is True
    assert check.available is True
    assert check.max_abs_diff_deg == pytest.approx(19.79)
    assert "21.16" in check.message and "1.37" in check.message


def test_cross_check_is_quiet_when_the_two_readings_agree() -> None:
    segments = [
        BaselineSegment(0, 60, 1.10, 1.20, 0.10, True),
        BaselineSegment(60, 120, -0.40, -0.35, -0.05, True),
    ]
    check = cross_check_roll(segments, None, threshold_deg=2.0)
    assert check.verdict == VERDICT_CONSISTENT
    assert check.drift_warning is False
    assert check.max_abs_diff_deg == pytest.approx(0.10)


def test_cross_check_reports_unavailable_rather_than_passing_silently() -> None:
    """Two of four real clips cannot be registered across 10 s; that is not agreement.

    Counting an untrusted direct registration as "consistent" would be the
    same silent-pass failure the card forbids for per-frame estimates.
    """
    segments = [
        BaselineSegment(0, 60, float("nan"), 4.0, float("nan"), False),
        BaselineSegment(60, 120, float("nan"), 5.0, float("nan"), False),
    ]
    check = cross_check_roll(segments, None, threshold_deg=2.0)
    assert check.verdict == VERDICT_UNAVAILABLE
    assert check.drift_warning is False
    assert check.available is False
    assert math.isnan(check.max_abs_diff_deg)
    assert "unavailable" in check.message and "UNVERIFIED" in check.message


def test_cross_check_ignores_untrusted_segments_but_still_uses_the_good_one() -> None:
    segments = [
        BaselineSegment(0, 60, float("nan"), 40.0, float("nan"), False),
        BaselineSegment(60, 120, 0.20, 9.00, 8.80, True),
    ]
    check = cross_check_roll(segments, None, threshold_deg=2.0)
    assert check.verdict == VERDICT_DRIFT_SUSPECTED
    assert check.max_abs_diff_deg == pytest.approx(8.80)


def test_honest_track_passes_the_cross_check(roll_ramp_track: MotionTrack) -> None:
    check = roll_ramp_track.long_baseline
    assert check.verdict == VERDICT_CONSISTENT, check.message
    assert check.max_abs_diff_deg < 0.5
    assert check.endpoint is not None and check.endpoint.direct_trusted
    assert check.segments, "a clip longer than one stride must produce segment checks"


def test_biased_front_end_is_caught_end_to_end(forward_frames) -> None:
    """Reproduce the 21 deg illusion on frames with zero true rotation, and catch it.

    The incremental path gets a front end with a +0.1 deg/frame bias bolted on
    — the same failure mode ORB exhibits at up to +1.13 deg/100 frames — while
    the direct path stays honest.  The integrated curve then claims ~5.9 deg of
    roll on a clip that never rotates, and the cross-check has to say so.
    """

    def biased(prev, cur, index):
        honest = estimate_pairwise(prev, cur, index)
        return dataclasses.replace(honest, roll_deg=honest.roll_deg + 0.1)

    track = estimate_motion(forward_frames, fps=FPS, stride=30, incremental_estimator=biased)

    assert track.net_roll_deg == pytest.approx(0.1 * (FORWARD_N - 1), abs=0.2)
    assert track.long_baseline.verdict == VERDICT_DRIFT_SUSPECTED, track.long_baseline.message
    assert track.long_baseline.max_abs_diff_deg > 2.0
    assert "#327" in track.long_baseline.message


def test_the_same_frames_pass_when_the_front_end_is_honest(forward_track: MotionTrack) -> None:
    """Control for the test above: without the injected bias there is no warning."""
    assert forward_track.long_baseline.verdict == VERDICT_CONSISTENT
    assert forward_track.long_baseline.max_abs_diff_deg < 0.5


# ---------------------------------------------------------------------------
# Frequency bands
# ---------------------------------------------------------------------------


def test_band_labels() -> None:
    assert band_label(None, 0.3) == "lt_0.3hz"
    assert band_label(0.3, 1.0) == "0.3_1hz"
    assert band_label(3.0, None) == "gt_3hz"


@pytest.mark.parametrize(
    ("freq_hz", "expected_band"),
    [(0.1, "lt_0.3hz"), (0.5, "0.3_1hz"), (2.0, "1_3hz"), (6.0, "gt_3hz")],
)
def test_band_rms_of_a_known_sine(freq_hz: float, expected_band: str) -> None:
    """A pure tone of known amplitude lands in one band at A/sqrt(2), within 10%."""
    n, amplitude = 240, 2.0
    t = np.arange(n) / FPS
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)

    bands = band_rms_deg(signal, FPS)
    assert dominant_band(bands) == expected_band
    assert bands[expected_band] == pytest.approx(amplitude / math.sqrt(2), rel=0.10)
    for name, value in bands.items():
        if name != expected_band:
            assert value < 0.10 * bands[expected_band], f"leakage into {name}"


def test_band_rms_separates_a_slow_drift_from_a_shake() -> None:
    """The measured reality: >95% of the roll energy sits below 0.3 Hz.

    This is what made #327's "1-second smoothing window removes >=90%" plan
    impossible — a 1 s window cuts around 1 Hz and the energy is nowhere near
    there.
    """
    n = 240
    t = np.arange(n) / FPS
    drift_plus_shake = 3.0 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.sin(2 * np.pi * 2.0 * t)
    bands = band_rms_deg(drift_plus_shake, FPS)
    assert bands["lt_0.3hz"] > 20 * bands["1_3hz"]
    assert dominant_band(bands) == "lt_0.3hz"


def test_band_rms_is_nan_without_a_usable_sample_rate() -> None:
    bands = band_rms_deg([0.0, 1.0, 2.0, 3.0], float("nan"))
    assert all(math.isnan(v) for v in bands.values())
    assert dominant_band(bands) == ""
    assert all(math.isnan(v) for v in band_rms_deg([1.0, 2.0], FPS).values())


def test_band_rms_ignores_the_mean_tilt() -> None:
    """Bands describe variation about the average tilt, not the tilt itself."""
    signal = np.full(64, 12.0)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in band_rms_deg(signal, FPS).values())


# ---------------------------------------------------------------------------
# scripts/analyze_motion.py — report and CLI
# ---------------------------------------------------------------------------


def _fake_increment(index: int, roll: float | None, *, n_tracked: int = 900) -> FrameMotion:
    """A FrameMotion with the shape of a real one; ``roll=None`` means unmeasurable."""
    trusted = roll is not None
    return FrameMotion(
        index=index,
        roll_deg=float("nan") if roll is None else roll,
        raw_roll_deg=float("nan") if roll is None else roll,
        scale=1.0,
        tx=0.0,
        ty=0.0,
        n_tracked=n_tracked if trusted else 0,
        fb_err_px=0.01,
        inlier_ratio=0.5,
        resid_px=1.0,
        spread=0.5,
        trusted=trusted,
        reason="" if trusted else "tracking failed: too few surviving correspondences",
    )


def _unavailable_check() -> LongBaselineCheck:
    return LongBaselineCheck(
        stride=60,
        threshold_deg=2.0,
        segments=(),
        endpoint=None,
        max_abs_diff_deg=float("nan"),
        verdict=VERDICT_UNAVAILABLE,
        message="long-baseline cross-check unavailable: UNVERIFIED",
    )


def _track_from_rolls(rolls, *, fps: float = FPS, check: LongBaselineCheck | None = None) -> MotionTrack:
    increments = [_fake_increment(i, r) for i, r in enumerate(rolls)]
    cumulative, filled = integrate_roll(increments)
    return MotionTrack(
        increments=increments,
        cumulative_roll_deg=cumulative,
        cumulative_filled=filled,
        long_baseline=check
        or LongBaselineCheck(
            stride=60,
            threshold_deg=2.0,
            segments=(BaselineSegment(0, len(rolls), float(cumulative[-1]), float(cumulative[-1]), 0.0, True),),
            endpoint=None,
            max_abs_diff_deg=0.0,
            verdict=VERDICT_CONSISTENT,
            message="agrees",
        ),
        n_frames=len(rolls) + 1,
        frame_shape=(960, 960),
        fps=fps,
    )


def test_verdict_steady() -> None:
    track = _track_from_rolls(np.full(240, 0.001).tolist())
    report = analyze_motion.analyse_track(track)
    assert report.verdict == analyze_motion.VERDICT_STEADY
    assert "基本平稳" in report.summary


def test_verdict_low_frequency_drift() -> None:
    t = np.arange(240) / FPS
    rolls = np.diff(4.0 * np.sin(2 * np.pi * 0.1 * t), prepend=0.0)[1:]
    report = analyze_motion.analyse_track(_track_from_rolls(rolls.tolist()))
    assert report.verdict == analyze_motion.VERDICT_LOW_FREQ_DRIFT
    assert report.dominant_band == "lt_0.3hz"
    assert "低频缓慢漂移" in report.summary


def test_verdict_high_frequency_shake() -> None:
    t = np.arange(240) / FPS
    rolls = np.diff(2.0 * np.sin(2 * np.pi * 5.0 * t), prepend=0.0)[1:]
    report = analyze_motion.analyse_track(_track_from_rolls(rolls.tolist()))
    assert report.verdict == analyze_motion.VERDICT_HIGH_FREQ_SHAKE
    assert "高频抖动" in report.summary


def test_verdict_unmeasured_when_most_frames_fail() -> None:
    """Too few trusted frames must suppress the verdict, not produce "steady"."""
    rolls = [0.5 if i % 4 == 0 else None for i in range(80)]
    track = _track_from_rolls(rolls, check=_unavailable_check())
    report = analyze_motion.analyse_track(track)

    assert report.reliable_ratio == pytest.approx(0.25)
    assert report.verdict == analyze_motion.VERDICT_UNMEASURED
    assert "无法可信测量" in report.summary
    assert report.n_untrusted == 60
    assert len(report.untrusted) == analyze_motion.MAX_LISTED_FAILURES
    assert any("UNVERIFIED" in w for w in report.warnings)
    assert any("NOT zero motion" in w for w in report.warnings)


def test_report_carries_both_readings_and_serialises(roll_ramp_track: MotionTrack) -> None:
    report = analyze_motion.analyse_track(roll_ramp_track, video="synthetic.mp4")
    payload = json.loads(json.dumps(analyze_motion.report_to_dict(report), ensure_ascii=False))

    assert payload["roll"]["cum_p2p_deg"] > 0
    assert payload["long_baseline"]["verdict"] == VERDICT_CONSISTENT
    assert payload["long_baseline"]["segments"], "the integration-free reading must be in the JSON"
    assert payload["long_baseline"]["endpoint"]["end"] == ROLL_RAMP_N - 1
    assert payload["confidence"]["reliable_ratio"] == 1.0
    assert set(payload["bands"]["rms_deg"]) == {"lt_0.3hz", "0.3_1hz", "1_3hz", "gt_3hz"}


def test_report_to_dict_replaces_non_finite_floats_with_null() -> None:
    """``json.dumps`` would happily emit bare ``NaN``, which is not valid JSON."""
    track = _track_from_rolls([None, None, None], check=_unavailable_check())
    payload = analyze_motion.report_to_dict(analyze_motion.analyse_track(track))
    assert payload["roll"]["cum_p2p_deg"] is None
    assert payload["long_baseline"]["max_abs_diff_deg"] is None
    assert "NaN" not in json.dumps(payload)


def test_render_report_shows_both_readings_and_the_warning() -> None:
    endpoint = BaselineSegment(0, 239, 1.37, 21.16, 19.79, True, 400, 0.255)
    check = cross_check_roll([], endpoint, stride=60, threshold_deg=2.0)
    track = _track_from_rolls(np.full(239, 21.16 / 239).tolist(), check=check)
    text = analyze_motion.render_report(analyze_motion.analyse_track(track, video="drone.mp4"))

    assert "drone.mp4" in text
    assert "Long-baseline cross-check" in text
    assert "[WARN]" in text
    assert "1.370" in text and "21.160" in text
    assert "camera rolled clockwise" in text, "the sign convention must travel with the numbers"


def test_classify_thresholds_are_configurable() -> None:
    assert (
        analyze_motion.classify(cum_p2p_deg=3.0, low_freq_rms_deg=1.0, high_freq_rms_deg=0.1, reliable_ratio=1.0)
        == analyze_motion.VERDICT_LOW_FREQ_DRIFT
    )
    assert (
        analyze_motion.classify(
            cum_p2p_deg=3.0,
            low_freq_rms_deg=1.0,
            high_freq_rms_deg=0.1,
            reliable_ratio=1.0,
            steady_p2p_deg=10.0,
        )
        == analyze_motion.VERDICT_STEADY
    )
    assert (
        analyze_motion.classify(
            cum_p2p_deg=3.0,
            low_freq_rms_deg=1.0,
            high_freq_rms_deg=0.1,
            reliable_ratio=0.9,
            min_reliable_ratio=0.95,
        )
        == analyze_motion.VERDICT_UNMEASURED
    )


# -- decoding ---------------------------------------------------------------


class _FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` so the decode path needs no codec."""

    def __init__(self, frames: list[np.ndarray], fps: float = 24.0, opened: bool = True):
        self._frames = frames
        self._fps = fps
        self._opened = opened
        self._pos = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1]) if self._frames else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0]) if self._frames else 0.0
        return 0.0

    def read(self):
        if self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame

    def release(self):
        self.released = True


def _bgr(n: int, size: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_load_gray_frames_downscales_and_greys(monkeypatch) -> None:
    captures: list[_FakeCapture] = []

    def factory(_path):
        capture = _FakeCapture(_bgr(5, 1024))
        captures.append(capture)
        return capture

    monkeypatch.setattr(analyze_motion.cv2, "VideoCapture", factory)
    frames, fps = analyze_motion.load_gray_frames("whatever.mp4", work_size=256)

    assert len(frames) == 5
    assert fps == pytest.approx(24.0)
    assert all(f.shape == (256, 256) and f.ndim == 2 for f in frames)
    assert captures[0].released, "the capture must be released even on the happy path"


def test_load_gray_frames_honours_max_frames(monkeypatch) -> None:
    monkeypatch.setattr(analyze_motion.cv2, "VideoCapture", lambda _p: _FakeCapture(_bgr(9, 128)))
    frames, _ = analyze_motion.load_gray_frames("whatever.mp4", work_size=960, max_frames=3)
    assert len(frames) == 3
    assert frames[0].shape == (128, 128), "work_size must never upscale"


def test_load_gray_frames_raises_on_unopenable_and_on_too_few_frames(monkeypatch) -> None:
    monkeypatch.setattr(analyze_motion.cv2, "VideoCapture", lambda _p: _FakeCapture([], opened=False))
    with pytest.raises(OSError, match="cannot open video"):
        analyze_motion.load_gray_frames("missing.mp4")

    monkeypatch.setattr(analyze_motion.cv2, "VideoCapture", lambda _p: _FakeCapture(_bgr(1, 64)))
    with pytest.raises(OSError, match="need at least 2"):
        analyze_motion.load_gray_frames("tooshort.mp4")


# -- CLI --------------------------------------------------------------------


def test_cli_writes_json_into_tmp_path(monkeypatch, tmp_path: Path, capsys) -> None:
    frames = _sequence(12, out=256, pad=96, roll_per_frame=0.1)
    monkeypatch.setattr(analyze_motion, "load_gray_frames", lambda *_a, **_k: (frames, FPS))

    out_json = tmp_path / "nested" / "report.json"
    exit_code = analyze_motion.main(["--video", str(tmp_path / "clip.mp4"), "--json", str(out_json), "--stride", "4"])

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_frames"] == 12
    assert payload["long_baseline"]["segments"]
    assert "Camera Motion Report" in capsys.readouterr().out


def test_cli_json_without_a_path_goes_to_stdout(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    frames = _sequence(8, out=256, pad=96, roll_per_frame=0.1)
    monkeypatch.setattr(analyze_motion, "load_gray_frames", lambda *_a, **_k: (frames, FPS))

    assert analyze_motion.main(["--video", "clip.mp4", "--json", "--quiet", "--stride", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]
    # No --json PATH was given, so nothing may land on disk — not next to the
    # input, not in the CWD (the repo root in a normal run).
    assert list(tmp_path.glob("*.json")) == [], "the CLI must not write anywhere it was not told to"


def test_cli_returns_1_when_the_video_cannot_be_read(tmp_path: Path) -> None:
    assert analyze_motion.main(["--video", str(tmp_path / "does_not_exist.mp4"), "--quiet"]) == 1


def test_cli_strict_exits_2_on_an_unmeasurable_clip(monkeypatch, capsys) -> None:
    flat = [np.full((128, 128), 128, np.uint8) for _ in range(6)]
    monkeypatch.setattr(analyze_motion, "load_gray_frames", lambda *_a, **_k: (flat, FPS))

    assert analyze_motion.main(["--video", "flat.mp4", "--quiet"]) == 0
    assert analyze_motion.main(["--video", "flat.mp4", "--quiet", "--strict"]) == 2
    capsys.readouterr()


def test_cli_help_runs_without_pythonpath() -> None:
    """K-15 discipline: direct invocation must work with no PYTHONPATH set.

    ``tests/test_scripts_importable.py`` owns the shared list of entry scripts
    and is off-limits for this card, so the same guarantee is asserted here for
    the one script this card adds.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = Path(__file__).resolve().parent.parent / "scripts" / "analyze_motion.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "--drift-threshold" in proc.stdout


def test_tracking_config_is_overridable_without_touching_defaults() -> None:
    """Config is data, so a caller can trade accuracy for speed without a fork."""
    out, pad = 256, 96
    plate = _texture(out + 2 * pad, seed=9)
    pair = (_render(plate, out, pad, 0.0), _render(plate, out, pad, 2.0))

    cheap = TrackingConfig(max_corners=120, min_distance=16)
    motion = estimate_pairwise(*pair, cfg=cheap, gates=ConfidenceGates(min_tracked=40))
    assert motion.trusted, motion.reason
    assert motion.n_tracked <= 120
    assert motion.roll_deg == pytest.approx(2.0, abs=0.1)
    assert TrackingConfig().max_corners == 2500, "the shared default must be untouched"
