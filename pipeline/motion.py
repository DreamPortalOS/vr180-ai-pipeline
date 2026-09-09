#!/usr/bin/env python3
"""Monocular camera-motion measurement — measured, cross-checked, never assumed.

Why this module exists
----------------------
Issue #327 reported that two generated clips rolled by 8.35° and 21.16° and
concluded from that "prompt wording cannot control camera stability".  Both
numbers were wrong, and so was the conclusion.  They came from *integrating
per-frame estimates*: each frame's estimate carries a small systematic bias,
integration compounds it, and 240 frames of a +0.09°/frame bias is +21° of
camera roll that never happened.  Registering frame 0 directly against frame
239 — one measurement, no integration, therefore immune to per-frame bias —
gives **+1.37°** (drone) and **+1.86°** (crystal).  The real clips are an
order of magnitude steadier than the report claimed, and the clip whose prompt
forbade roll was in fact the steadier of the two.

That mistake is cheap to make and expensive to act on, so this module is built
around not making it again:

* the front end is **Shi-Tomasi + Lucas-Kanade**, not ORB.  Synthetic
  ground-truth calibration (``D:\\Github\\_research\\imu``) measured ORB at
  2x the per-frame noise of LK, up to **+1.13°/100 frames** of accumulation
  *bias*, **2.62°** of roll hallucinated on a clip that only moves forward,
  and — worst of all — a gain of **x0.01** on slow roll, i.e. it simply cannot
  see the 0.02-0.09°/frame regime that real footage lives in.  LK measured
  0.03-0.04° per-frame RMS and <=0.2°/100 frames of drift on the same clips.
* every reading is paired with a **long-baseline cross-check**
  (:class:`LongBaselineCheck`): the same rotation measured *again* by direct
  registration over a long stride, with no integration in between.  When the
  two disagree by more than a threshold the track says so out loud.  Run
  against #327's data this check fires immediately — that is its whole job.
* a frame the estimator cannot trust reports ``roll_deg = nan``, **never
  0.0**.  Zero is a claim ("the camera was level across this step"); on a
  purely-forward shot, where rotation is genuinely hard to observe, that claim
  is both wrong and reassuring, which is the worst combination.  Untrusted
  steps are reconstructed for the cumulative curve by interpolation between
  their nearest trusted neighbours and every one of them is flagged in
  :attr:`MotionTrack.cumulative_filled`.

Scope: **measurement and diagnosis only.**  Nothing here stabilises, warps or
corrects anything, and nothing here does I/O — decoding lives in
``scripts/analyze_motion.py``.  Focus-of-expansion / forward-direction
estimation is deliberately absent: under a monocular camera it depends on an
assumed FOV and the angle error reaches +-50% when that guess is 40° off.

Sign convention — locked by tests, do not "tidy" it
---------------------------------------------------
``roll_deg > 0`` means, equivalently:

* the **image content rotates counter-clockwise** on screen (pixel y down);
* the **camera rolled clockwise** about its optical axis, seen from behind the
  camera looking forward — the horizon in the picture tips counter-clockwise;
* a frame pair synthesised with ``cv2.getRotationMatrix2D(centre, +A, 1.0)``
  is recovered as ``roll_deg = +A``.

This paragraph exists because the research scripts that preceded this module
mixed two opposite conventions, which made two estimators that agreed to 0.03°
report a correlation of **-0.861** and read as "the two methods contradict each
other".  They did not; one of them was negated.  The same sign, shipped to a
motion seat, leans the chair the wrong way — so the convention is asserted in
``tests/test_motion.py`` rather than merely described here.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Long-baseline stride, in frames.  60 frames is 2.5 s at 24 fps: long enough
#: that per-frame bias has had time to compound into something visible, short
#: enough that the two endpoints still share enough scene content to register.
#: The whole-clip endpoint check (frame 0 vs frame N-1) is always run as well;
#: on real 10 s clips it is the strongest evidence when it succeeds and simply
#: fails to register when the shot has changed too much.
DEFAULT_BASELINE_STRIDE: int = 60

#: Integrated-vs-direct disagreement, in degrees, above which the track is
#: flagged.  #327's drone clip disagreed by ~19.8°; a healthy LK track on the
#: same footage disagrees by a few tenths of a degree.  2° sits far from both.
DEFAULT_DRIFT_WARNING_DEG: float = 2.0

#: Frequency band edges (Hz) used by :func:`band_rms_deg`.  Chosen to separate
#: "slow drift the viewer reads as a tilting horizon" (<0.3 Hz, where all four
#: measured clips put >95% of their roll energy) from "shake" (>1 Hz).
DEFAULT_BAND_EDGES: tuple[float, ...] = (0.3, 1.0, 3.0)

VERDICT_CONSISTENT = "consistent"
VERDICT_DRIFT_SUSPECTED = "drift_suspected"
VERDICT_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TrackingConfig:
    """Front-end parameters: Shi-Tomasi corners + pyramidal Lucas-Kanade.

    Defaults follow the synthetic-calibration run that qualified this front
    end.  ``fb_tol_px`` is the forward-backward consistency tolerance: a point
    is kept only if tracking it forward and then back lands within that many
    pixels of where it started, which is what removes the mis-tracks that
    would otherwise become the systematic bias this module exists to avoid.
    """

    max_corners: int = 2500
    quality_level: float = 0.01
    min_distance: int = 8
    block_size: int = 7
    lk_window: int = 21
    lk_levels: int = 4
    fb_tol_px: float = 0.6
    ransac_thresh_px: float = 1.5
    ransac_iters: int = 5000
    #: Below this many surviving correspondences the pair is not estimated at
    #: all (distinct from :attr:`ConfidenceGates.min_tracked`, which decides
    #: whether an estimate that *was* produced may be trusted).
    min_points: int = 20


@dataclass(frozen=True)
class ConfidenceGates:
    """Per-frame trust gates.  All five must hold for ``trusted`` to be True.

    Calibrated against four real clips: healthy frames track 1600-1900 points
    with 0.004-0.008 px forward-backward error, 0.45-0.66 inlier ratio and
    0.9-2.7 px residual; the one degraded clip fell to 529 points / 0.024 px
    and its pass rate dropped to 57.5%.

    ``min_inlier_ratio`` is deliberately **low**.  A forward-flying shot has
    real parallax, so a single global 2-D similarity genuinely cannot explain
    near and far content at once and 0.45-0.66 is what a *good* frame looks
    like.  Raising this to 0.8 would reject every usable frame.
    """

    min_tracked: int = 200
    max_fb_err_px: float = 0.35
    min_inlier_ratio: float = 0.25
    max_resid_px: float = 6.0
    #: Spatial spread of the inliers, normalised so that points spread evenly
    #: over the whole frame score ~0.58 (see :func:`inlier_spread`).  Guards
    #: against a solution fitted to one small clump, which is free to rotate.
    min_spread: float = 0.30


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameMotion:
    """One pairwise registration.

    ``roll_deg`` is ``nan`` whenever ``trusted`` is False — see the module
    docstring on why an untrusted step must not read as 0.0.  ``raw_roll_deg``
    keeps whatever the estimator produced (possibly garbage) so a diagnostic
    report can show *what* the rejected frame claimed.
    """

    index: int
    roll_deg: float
    raw_roll_deg: float
    scale: float
    tx: float
    ty: float
    n_tracked: int
    fb_err_px: float
    inlier_ratio: float
    resid_px: float
    spread: float
    trusted: bool
    #: Empty when trusted; otherwise the gate(s) that rejected the frame.
    reason: str = ""


@dataclass(frozen=True)
class BaselineSegment:
    """One long-baseline comparison: direct registration vs integrated sum."""

    start: int
    end: int
    direct_roll_deg: float
    integrated_roll_deg: float
    diff_deg: float
    direct_trusted: bool
    direct_n_tracked: int = 0
    direct_inlier_ratio: float = float("nan")

    @property
    def span(self) -> int:
        """Number of frame steps this segment spans."""
        return self.end - self.start


@dataclass(frozen=True)
class LongBaselineCheck:
    """The cross-check that would have caught #327's 21°.

    ``verdict`` is one of :data:`VERDICT_CONSISTENT`,
    :data:`VERDICT_DRIFT_SUSPECTED` or :data:`VERDICT_UNAVAILABLE`.  The third
    state is not cosmetic: over a 10 s baseline two of four real clips could
    not be registered directly at all (the scene had changed too much), and
    reporting that as "consistent" would be the same silent-pass failure this
    module is built to avoid.
    """

    stride: int
    threshold_deg: float
    segments: tuple[BaselineSegment, ...]
    endpoint: BaselineSegment | None
    max_abs_diff_deg: float
    verdict: str
    message: str

    @property
    def drift_warning(self) -> bool:
        """True only for :data:`VERDICT_DRIFT_SUSPECTED`."""
        return self.verdict == VERDICT_DRIFT_SUSPECTED

    @property
    def available(self) -> bool:
        """True when at least one direct registration could be compared."""
        return self.verdict != VERDICT_UNAVAILABLE


@dataclass
class MotionTrack:
    """A whole clip's roll measurement, in both readings the card demands.

    ``increments`` is the per-frame reading (what a spectrum needs);
    ``cumulative_roll_deg`` is its running sum; ``long_baseline`` is the
    independent, integration-free reading of the same rotation.  Consumers
    that look only at the cumulative curve are exactly the consumers #327
    burned, so ``long_baseline`` is not optional and not lazily computed.
    """

    increments: list[FrameMotion]
    cumulative_roll_deg: np.ndarray
    cumulative_filled: np.ndarray
    long_baseline: LongBaselineCheck
    n_frames: int
    frame_shape: tuple[int, int] = (0, 0)
    fps: float = float("nan")
    gates: ConfidenceGates = field(default_factory=ConfidenceGates)

    # -- confidence ---------------------------------------------------------

    @property
    def untrusted_indices(self) -> list[int]:
        """Indices (into ``increments``) of steps that failed the gates."""
        return [i for i, m in enumerate(self.increments) if not m.trusted]

    @property
    def n_untrusted(self) -> int:
        """How many steps were rejected."""
        return len(self.untrusted_indices)

    @property
    def reliable_ratio(self) -> float:
        """Fraction of steps that passed every gate (0.0 for an empty track)."""
        if not self.increments:
            return 0.0
        return 1.0 - self.n_untrusted / len(self.increments)

    # -- roll readings ------------------------------------------------------

    @property
    def trusted_roll_increments(self) -> np.ndarray:
        """Per-frame roll of the trusted steps only (no nan, no fill)."""
        vals = np.array([m.roll_deg for m in self.increments], dtype=float)
        return vals[np.isfinite(vals)]

    @property
    def net_roll_deg(self) -> float:
        """Integrated roll from the first frame to the last."""
        if self.cumulative_roll_deg.size == 0:
            return float("nan")
        return float(self.cumulative_roll_deg[-1])

    @property
    def cumulative_p2p_deg(self) -> float:
        """Peak-to-peak of the cumulative curve — the "how far did it tip" number."""
        cum = self.cumulative_roll_deg
        finite = cum[np.isfinite(cum)]
        if finite.size == 0:
            return float("nan")
        return float(finite.max() - finite.min())

    @property
    def per_frame_std_deg(self) -> float:
        """Standard deviation of the trusted per-frame increments."""
        vals = self.trusted_roll_increments
        return float(vals.std()) if vals.size else float("nan")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def roll_deg_from_affine(matrix: np.ndarray) -> float:
    """Camera roll, in degrees, from a 2x3 partial-affine (similarity) matrix.

    ``matrix`` maps points of the *previous* frame onto the *current* one, so
    ``atan2(m10, m00)`` is the rotation the **image content** underwent.  The
    camera rotated the other way, hence the negation — see the module
    docstring's sign-convention block, which ``tests/test_motion.py`` asserts.
    """
    theta_image = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return -math.degrees(theta_image)


def inlier_spread(points: np.ndarray, shape: tuple[int, int]) -> float:
    """How widely the correspondences are spread over the frame.

    ``min(std_x / width, std_y / height) * 2``: points spread evenly over the
    whole frame score ~0.577 (a uniform distribution has std = extent/sqrt(12)),
    a tight clump scores ~0.  A similarity fitted to a clump is under-
    constrained in rotation, which is precisely how a confident-looking wrong
    angle gets produced.
    """
    if points is None or len(points) < 2:
        return 0.0
    height, width = shape
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    sx = float(pts[:, 0].std()) / max(width, 1)
    sy = float(pts[:, 1].std()) / max(height, 1)
    return float(min(sx, sy) * 2.0)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Coerce a frame to single-channel uint8 without copying when possible."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Front end: Shi-Tomasi + Lucas-Kanade
# ---------------------------------------------------------------------------


def track_lk(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    cfg: TrackingConfig | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Track corners from ``prev_gray`` into ``gray``.

    Returns ``(p0, p1, fb_err_px)`` where ``p0``/``p1`` are the surviving
    Nx2 float32 correspondences and ``fb_err_px`` is the median
    forward-backward error of the survivors.  ``(None, None, nan)`` when the
    pair cannot be tracked (flat frames, too few corners, LK failure).

    The forward-backward filter is what separates this front end from the ORB
    one it replaces: descriptor matching happily returns confident wrong pairs
    at integer-pixel precision, and those are what integrate into a phantom
    21° of roll.  A point that does not survive the round trip is dropped
    before it can vote.
    """
    cfg = cfg or TrackingConfig()
    prev_gray = to_gray(prev_gray)
    gray = to_gray(gray)

    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        blockSize=cfg.block_size,
    )
    if p0 is None or len(p0) < cfg.min_points:
        return None, None, float("nan")

    lk_kwargs = {
        "winSize": (cfg.lk_window, cfg.lk_window),
        "maxLevel": cfg.lk_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    p1, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_kwargs)
    if p1 is None:
        return None, None, float("nan")
    p0_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_kwargs)
    if p0_back is None:
        return None, None, float("nan")

    fb = np.linalg.norm((p0 - p0_back).reshape(-1, 2), axis=1)
    keep = (status_fwd.ravel() == 1) & (status_bwd.ravel() == 1) & (fb < cfg.fb_tol_px)
    if int(keep.sum()) < cfg.min_points:
        return None, None, float(np.median(fb)) if fb.size else float("nan")
    return (
        p0.reshape(-1, 2)[keep],
        p1.reshape(-1, 2)[keep],
        float(np.median(fb[keep])),
    )


def _rejected(
    index: int,
    reason: str,
    *,
    n_tracked: int = 0,
    fb_err_px: float = float("nan"),
) -> FrameMotion:
    """Build an untrusted :class:`FrameMotion` — ``roll_deg`` stays ``nan``."""
    nan = float("nan")
    return FrameMotion(
        index=index,
        roll_deg=nan,
        raw_roll_deg=nan,
        scale=nan,
        tx=nan,
        ty=nan,
        n_tracked=n_tracked,
        fb_err_px=fb_err_px,
        inlier_ratio=nan,
        resid_px=nan,
        spread=0.0,
        trusted=False,
        reason=reason,
    )


def estimate_pairwise(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    index: int = 0,
    *,
    cfg: TrackingConfig | None = None,
    gates: ConfidenceGates | None = None,
) -> FrameMotion:
    """Register ``gray`` against ``prev_gray`` and grade the result.

    The two frames need not be adjacent — passing frame 0 and frame 239 is
    exactly how the long-baseline cross-check gets its second opinion.

    ``index`` is carried through untouched so callers can label the reading
    (for adjacent pairs the convention used here is the index of the *second*
    frame minus one, i.e. the step number).
    """
    cfg = cfg or TrackingConfig()
    gates = gates or ConfidenceGates()

    p0, p1, fb_err = track_lk(prev_gray, gray, cfg)
    if p0 is None or p1 is None:
        return _rejected(index, "tracking failed: too few surviving correspondences", fb_err_px=fb_err)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_thresh_px,
        maxIters=cfg.ransac_iters,
    )
    if matrix is None:
        return _rejected(index, "similarity fit failed", n_tracked=len(p0), fb_err_px=fb_err)

    predicted = (matrix[:, :2] @ p0.T).T + matrix[:, 2]
    resid_px = float(np.median(np.linalg.norm(predicted - p1, axis=1)))
    inlier_ratio = float(inlier_mask.mean()) if inlier_mask is not None else float("nan")
    inliers = p0 if inlier_mask is None else p0[inlier_mask.ravel().astype(bool)]
    spread = inlier_spread(inliers, to_gray(prev_gray).shape[:2])

    raw_roll = roll_deg_from_affine(matrix)
    scale = float(math.hypot(float(matrix[0, 0]), float(matrix[1, 0])))

    failures: list[str] = []
    if len(p0) < gates.min_tracked:
        failures.append(f"n_tracked {len(p0)} < {gates.min_tracked}")
    if not (fb_err <= gates.max_fb_err_px):
        failures.append(f"fb_err {fb_err:.4f}px > {gates.max_fb_err_px}px")
    if not (inlier_ratio >= gates.min_inlier_ratio):
        failures.append(f"inlier_ratio {inlier_ratio:.3f} < {gates.min_inlier_ratio}")
    if not (resid_px <= gates.max_resid_px):
        failures.append(f"resid {resid_px:.3f}px > {gates.max_resid_px}px")
    if not (spread >= gates.min_spread):
        failures.append(f"spread {spread:.3f} < {gates.min_spread}")

    trusted = not failures
    return FrameMotion(
        index=index,
        roll_deg=raw_roll if trusted else float("nan"),
        raw_roll_deg=raw_roll,
        scale=scale,
        tx=float(matrix[0, 2]),
        ty=float(matrix[1, 2]),
        n_tracked=len(p0),
        fb_err_px=fb_err,
        inlier_ratio=inlier_ratio,
        resid_px=resid_px,
        spread=spread,
        trusted=trusted,
        reason="" if trusted else "; ".join(failures),
    )


PairwiseEstimator = Callable[[np.ndarray, np.ndarray, int], FrameMotion]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def integrate_roll(increments: Sequence[FrameMotion]) -> tuple[np.ndarray, np.ndarray]:
    """Running sum of per-frame roll, plus a mask of the reconstructed steps.

    Returns ``(cumulative, filled)`` both of length ``len(increments) + 1``;
    ``cumulative[0]`` is 0.0 by definition (the first frame is the reference).

    An untrusted step is **not** treated as zero.  It is linearly interpolated
    from its nearest trusted neighbours (held flat past the ends), and
    ``filled[k]`` marks every sample whose step was reconstructed that way.
    The distinction matters most on the exact footage where it is tempting to
    skip it: on a shot flying straight forward, rotation is the hardest thing
    to observe, so the frames that fail the gates cluster there — and filling
    them with 0.0 would report that shot as unusually steady.

    If nothing at all could be trusted the cumulative curve is all-``nan``:
    "no measurement" rather than "no motion".
    """
    n = len(increments)
    if n == 0:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=bool)

    raw = np.array([m.roll_deg for m in increments], dtype=float)
    valid = np.isfinite(raw)
    filled_steps = np.concatenate(([False], ~valid))

    if not valid.any():
        return np.full(n + 1, np.nan, dtype=float), np.ones(n + 1, dtype=bool)

    bridged = raw.copy()
    idx = np.arange(n, dtype=float)
    bridged[~valid] = np.interp(idx[~valid], idx[valid], raw[valid])
    cumulative = np.concatenate(([0.0], np.cumsum(bridged)))
    return cumulative, filled_steps


# ---------------------------------------------------------------------------
# Long-baseline cross-check
# ---------------------------------------------------------------------------


def cross_check_roll(
    segments: Sequence[BaselineSegment],
    endpoint: BaselineSegment | None,
    *,
    stride: int = DEFAULT_BASELINE_STRIDE,
    threshold_deg: float = DEFAULT_DRIFT_WARNING_DEG,
) -> LongBaselineCheck:
    """Compare each direct registration against the integrated sum over the same span.

    This is the function that would have stopped #327: feed it the drone
    clip's numbers — integrated 21.16°, directly registered 1.37° — and the
    verdict is :data:`VERDICT_DRIFT_SUSPECTED`, not a headline.

    Segments whose *direct* registration was itself untrusted are excluded
    from the comparison rather than counted as agreement; if that leaves
    nothing to compare, the verdict is :data:`VERDICT_UNAVAILABLE`.
    """
    pool = [seg for seg in segments if seg.direct_trusted and np.isfinite(seg.diff_deg)]
    if endpoint is not None and endpoint.direct_trusted and np.isfinite(endpoint.diff_deg):
        pool.append(endpoint)

    if not pool:
        return LongBaselineCheck(
            stride=stride,
            threshold_deg=threshold_deg,
            segments=tuple(segments),
            endpoint=endpoint,
            max_abs_diff_deg=float("nan"),
            verdict=VERDICT_UNAVAILABLE,
            message=(
                "long-baseline cross-check unavailable: no direct registration over a "
                f"{stride}-frame span could be trusted, so the integrated roll is "
                "UNVERIFIED — treat it as a lower bound on uncertainty, not as a measurement"
            ),
        )

    worst = max(pool, key=lambda seg: abs(seg.diff_deg))
    max_abs = float(abs(worst.diff_deg))
    if max_abs > threshold_deg:
        return LongBaselineCheck(
            stride=stride,
            threshold_deg=threshold_deg,
            segments=tuple(segments),
            endpoint=endpoint,
            max_abs_diff_deg=max_abs,
            verdict=VERDICT_DRIFT_SUSPECTED,
            message=(
                f"integrated roll disagrees with direct registration by {max_abs:.2f}° "
                f"over frames {worst.start}-{worst.end} "
                f"(integrated {worst.integrated_roll_deg:+.2f}°, direct {worst.direct_roll_deg:+.2f}°, "
                f"threshold {threshold_deg:.2f}°). The integrated curve is accumulating "
                "per-frame bias — trust the direct reading, not the sum (see #327)"
            ),
        )
    return LongBaselineCheck(
        stride=stride,
        threshold_deg=threshold_deg,
        segments=tuple(segments),
        endpoint=endpoint,
        max_abs_diff_deg=max_abs,
        verdict=VERDICT_CONSISTENT,
        message=(
            f"integrated roll agrees with direct registration to {max_abs:.2f}° "
            f"(threshold {threshold_deg:.2f}°) over {len(pool)} baseline(s)"
        ),
    )


def _baseline_segment(
    frames: Sequence[np.ndarray],
    cumulative: np.ndarray,
    start: int,
    end: int,
    estimator: PairwiseEstimator,
) -> BaselineSegment:
    """Register ``frames[start]`` against ``frames[end]`` and diff it against the sum."""
    direct = estimator(frames[start], frames[end], start)
    integrated = float(cumulative[end] - cumulative[start])
    direct_roll = direct.roll_deg if direct.trusted else float("nan")
    diff = float(integrated - direct_roll) if direct.trusted else float("nan")
    return BaselineSegment(
        start=start,
        end=end,
        direct_roll_deg=direct_roll,
        integrated_roll_deg=integrated,
        diff_deg=diff,
        direct_trusted=direct.trusted,
        direct_n_tracked=direct.n_tracked,
        direct_inlier_ratio=direct.inlier_ratio,
    )


# ---------------------------------------------------------------------------
# Whole-clip estimation
# ---------------------------------------------------------------------------


def estimate_motion(
    frames: Sequence[np.ndarray],
    *,
    fps: float = float("nan"),
    cfg: TrackingConfig | None = None,
    gates: ConfidenceGates | None = None,
    stride: int = DEFAULT_BASELINE_STRIDE,
    drift_threshold_deg: float = DEFAULT_DRIFT_WARNING_DEG,
    incremental_estimator: PairwiseEstimator | None = None,
    direct_estimator: PairwiseEstimator | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MotionTrack:
    """Measure a clip's camera roll both ways and cross-check them.

    ``frames`` is a sequence of decoded frames (grayscale or BGR); this
    function does no I/O.  Both readings the card requires come back in one
    :class:`MotionTrack`: ``increments`` / ``cumulative_roll_deg`` for the
    integrated view and ``long_baseline`` for the integration-free one.

    ``incremental_estimator`` and ``direct_estimator`` are separate injection
    points on purpose.  The front end is the component under suspicion here,
    so a test must be able to hand the incremental path a deliberately biased
    estimator while the direct path stays honest — that is the only way to
    prove the cross-check actually fires on the failure it was built for,
    rather than merely existing.
    """
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames to measure motion, got {len(frames)}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    cfg = cfg or TrackingConfig()
    gates = gates or ConfidenceGates()
    incremental = incremental_estimator or functools.partial(estimate_pairwise, cfg=cfg, gates=gates)
    direct = direct_estimator or functools.partial(estimate_pairwise, cfg=cfg, gates=gates)

    grays = [to_gray(f) for f in frames]
    n = len(grays)

    increments: list[FrameMotion] = []
    for i in range(1, n):
        increments.append(incremental(grays[i - 1], grays[i], i - 1))
        if progress is not None:
            progress(i, n - 1)

    cumulative, filled = integrate_roll(increments)

    segments = [
        _baseline_segment(grays, cumulative, start, start + stride, direct) for start in range(0, n - stride, stride)
    ]
    # The whole-clip endpoint is the strongest single piece of evidence when it
    # registers, and it is also the only baseline available on clips shorter
    # than one stride — so it is computed whenever it spans more than one step.
    endpoint = _baseline_segment(grays, cumulative, 0, n - 1, direct) if n >= 3 else None
    check = cross_check_roll(segments, endpoint, stride=stride, threshold_deg=drift_threshold_deg)

    return MotionTrack(
        increments=increments,
        cumulative_roll_deg=cumulative,
        cumulative_filled=filled,
        long_baseline=check,
        n_frames=n,
        frame_shape=(int(grays[0].shape[0]), int(grays[0].shape[1])),
        fps=float(fps),
        gates=gates,
    )


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def band_label(low: float | None, high: float | None) -> str:
    """Human-readable key for one frequency band, e.g. ``"0.3_1hz"``."""
    if low is None:
        return f"lt_{high:g}hz"
    if high is None:
        return f"gt_{low:g}hz"
    return f"{low:g}_{high:g}hz"


def band_rms_deg(
    signal: Sequence[float] | np.ndarray,
    fps: float,
    edges: Sequence[float] = DEFAULT_BAND_EDGES,
) -> dict[str, float]:
    """RMS of ``signal`` (degrees) within each frequency band, via Parseval.

    Applied to the *cumulative* roll curve this answers the question #327 got
    wrong in the other direction: is the tilt a slow drift (energy below
    0.3 Hz — a horizon that leans, which no 1-second smoothing window can
    remove) or actual shake (above 1 Hz)?  On the four measured clips the
    sub-0.3 Hz band held up to 92x the energy of the 1-3 Hz band, which is why
    the "de-shake it" plan could only ever have removed ~6% of it.

    The mean is removed first, so the returned bands describe variation about
    the average tilt, not the average tilt itself.
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    edge_list = [float(e) for e in edges]
    labels = [band_label(None, edge_list[0])]
    labels += [band_label(edge_list[i], edge_list[i + 1]) for i in range(len(edge_list) - 1)]
    labels.append(band_label(edge_list[-1], None))

    if x.size < 4 or not np.isfinite(fps) or fps <= 0:
        return dict.fromkeys(labels, float("nan"))

    x = x - x.mean()
    n = x.size
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    # Parseval: one-sided power per bin, doubled except for DC and Nyquist.
    power = (np.abs(spectrum) / n) ** 2
    weight = np.full(power.shape, 2.0)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0
    power = power * weight

    bounds = [0.0, *edge_list, float("inf")]
    out: dict[str, float] = {}
    for label, low, high in zip(labels, bounds[:-1], bounds[1:], strict=True):
        mask = (freqs >= low) & (freqs < high)
        out[label] = float(np.sqrt(power[mask].sum()))
    return out


def dominant_band(bands: dict[str, float]) -> str:
    """Key of the band holding the most energy (empty string if none is finite)."""
    finite = {k: v for k, v in bands.items() if np.isfinite(v)}
    if not finite:
        return ""
    return max(finite, key=lambda k: finite[k])
