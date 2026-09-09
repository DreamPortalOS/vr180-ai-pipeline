"""Tests for the pre-pipeline source health check (scripts/check_source_quality.py, W-1 #305).

The tool answers one question before a 40-minute conversion starts: *is this
clip worth converting?*  Six checks — ``decodable`` / ``square`` /
``forward_motion`` / ``edges`` / ``detail_ratio`` / ``anchor`` — and an exit
code, so it can gate a run.  Hand it an image instead of a clip and it answers
the cheaper question *is this keyframe worth animating?* (G-3a #336), which is
where the money is: a 4k/10s generation costs ≈¥50, a keyframe costs almost
nothing, and ``forward_motion`` — the only check a single frame cannot support —
is reported ``skipped`` rather than guessed at.

Where the value is
------------------
``forward_motion`` is the check the card exists for.  The VR180 route converts
*parallax*; a locked-off camera has none, so a static clip produces a flat
picture pasted on a sphere after the full GPU spend.  A generation prompt that
says "camera flies forward" is a wish — models return static shots and lateral
pans routinely — so the motion has to be **measured**.

The discriminator, and why all three synthetic cases are here
------------------------------------------------------------
Dense Farnebäck flow, projected onto the radial direction from the frame
centre.  Three fixtures, built with numpy/OpenCV only (no real video, no
network, no model), pin the three cases the operator actually hits:

===============  ==================================  ====================
fixture          what it is                          must be judged
===============  ==================================  ====================
``radial_outflow_frames``  texture zooming out of the centre   PASS
``static_frames``          one frame + noise                   FAIL, 提示含「静态」
``panning_frames``         the whole frame sliding sideways    FAIL
===============  ==================================  ====================

The pan is the fixture that earns the method.  A naive "is there motion?"
test passes a pan happily — its flow magnitude is *large*.  What separates it
is the **sign structure**: a lateral move projects to ``+|v|`` on the leading
half of the frame and ``-|v|`` on the trailing half, so the mean radial
component cancels to ~0 while ``flow_mean`` stays big.  Forward motion instead
has the flow scale with radius (``≈ (s-1)·r``), which is the second condition —
outer ring above inner disc.

Measured on these fixtures at 240×240 with 8 pairs, before any assertion was
written (this is the margin the assertions are allowed to rely on):

* radial outflow — ``radial ≈ +3.61 px``, inner ``+1.37`` → outer ``+4.78``
* static —         ``radial ≈ -0.0002 px``
* pan —            ``radial ≈ -0.003 px`` with ``flow ≈ 6 px``

Three orders of magnitude between the good case and both bad ones, so the
thresholds are nowhere near the fixtures' noise and the tests are not
knife-edge.

The verdict is taken on a **scale-free rate** (radial flow ÷ the frame's
half-diagonal ``R``) rather than on raw pixels, so the same threshold holds for
a 480 px analysis frame and a 2880 px source.  ``test_radial_rate_is_scale_free``
pins that property.

The composition pair, and the margin they were given
----------------------------------------------------
``detail_ratio`` and ``anchor`` encode two rules measured off 34 finished
competitor films and 613 of their stills (``_research/redraion/ANALYSIS.md``
§3): detail concentrated in the middle (ratio 1.58–1.60) and a subject up front
in 80 % of frames (median 11.3 % of the picture, eccentricity 0.236).

``anchor`` is the one that could easily have become decoration, so its
discriminator was measured on the fixtures **before** any assertion was written:

==============================  ====================
fixture                         subject area found
==============================  ====================
11 % disc, centred              10.99 %
40 % disc, centred              39.15 %
11 % disc, off-centre           11.10 %
uniform texture (8 seeds)       0.14 – 0.35 %
flat gray + sensor noise        0.00 %
rim-damped / empty-centre       0.00 – 0.17 %
==============================  ====================

A factor of ~31 between the quietest true subject and the loudest false one,
with :data:`~scripts.check_source_quality.ANCHOR_MIN_AREA` (1.5 %) sitting
between them — so "no anchor" is a verdict the check can actually reach, not a
branch that never fires.  ``test_subject_and_subjectless_frames_are_separated``
pins that margin directly.

Everything here runs on synthetic arrays: no test reads ``video/``, decodes a
real file, or shells out to ffmpeg.  The ffprobe/ffmpeg layer is covered by
parsing captured ffprobe JSON and by an AST sweep
(``test_subprocess_calls_are_list_form_without_shell``) that holds every
``subprocess.run`` in the module to list form with no ``shell=True``.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import check_source_quality as csq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_source_quality.py"

FRAME_SIZE = 240
N_FRAMES = 9  # 9 frames -> 8 adjacent pairs, the CLI default


# ---------------------------------------------------------------------------
# Synthetic footage (numpy/OpenCV only — no video files, no ffmpeg)
# ---------------------------------------------------------------------------


def _texture(height: int, width: int, seed: int = 7) -> np.ndarray:
    """A blobby grayscale texture: enough structure for Farnebäck to track.

    Built by upsampling coarse noise rather than using per-pixel noise — flow
    needs a *trackable* pattern, and white noise has no correspondence between
    frames at all.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(max(2, height // 8), max(2, width // 8)), dtype=np.uint8)
    img = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0)


def radial_outflow_frames(
    n: int = N_FRAMES,
    size: int = FRAME_SIZE,
    zoom_per_frame: float = 1.04,
) -> list[np.ndarray]:
    """Positive sample: texture streaming outward from the centre.

    Each frame is the previous scene scaled up about the frame centre, which is
    exactly what a forward-flying camera does to a static world — everything
    moves away from the vanishing point, and the further out it already is, the
    faster it goes.  Rendered from a 2× oversized canvas so the growing crop
    never runs out of source pixels and never has to invent border content.
    """
    base = _texture(size * 2, size * 2)
    frames: list[np.ndarray] = []
    for i in range(n):
        matrix = cv2.getRotationMatrix2D((size, size), 0, zoom_per_frame**i)
        big = cv2.warpAffine(base, matrix, (2 * size, 2 * size), flags=cv2.INTER_LINEAR)
        frames.append(big[size // 2 : size // 2 + size, size // 2 : size // 2 + size].copy())
    return frames


def static_frames(n: int = N_FRAMES, size: int = FRAME_SIZE, sigma: float = 3.0) -> list[np.ndarray]:
    """Negative sample 1: a locked-off camera — one scene plus sensor noise.

    The noise matters: a pixel-identical repeat would give an exactly-zero flow
    field, which is an easier problem than reality.  A little noise makes
    Farnebäck return a small *random* field, which is what a real static shot
    looks like and what the threshold has to survive.
    """
    rng = np.random.default_rng(11)
    base = _texture(size, size, seed=3).astype(np.int16)
    return [np.clip(base + rng.normal(0.0, sigma, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def panning_frames(n: int = N_FRAMES, size: int = FRAME_SIZE, dx: int = 6) -> list[np.ndarray]:
    """Negative sample 2: the whole frame sliding sideways.

    The hard case.  There is plenty of motion — more raw flow than the positive
    sample has — so anything that merely asks "does this move?" passes it.  Only
    the radial projection sees that the motion is not *outward*.
    """
    base = _texture(size, size + dx * n)
    return [base[:, i * dx : i * dx + size].copy() for i in range(n)]


def _stats_for(frames: list[np.ndarray]) -> list[csq.RadialStats]:
    return [csq.radial_flow_stats(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


# ---------------------------------------------------------------------------
# Synthetic compositions (detail_ratio / anchor)
# ---------------------------------------------------------------------------


def _fine_texture(height: int, width: int, seed: int = 7) -> np.ndarray:
    """Like :func:`_texture` but on a 4× finer grid, returned as float.

    The grid size matters for the composition fixtures in a way it does not for
    the flow ones.  With ``//8`` cells a 240² frame holds only 30×30 blobs, and
    the sampling noise on "is the middle busier than the rim?" is large enough
    that some seeds land above 1.2 by luck — measured across 20 seeds, ``//8``
    spans 0.88–1.23 while ``//4`` spans 0.96–1.07.  A fixture whose verdict
    depends on the seed proves nothing, so the uniform-texture case uses the
    tighter one.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(max(2, height // 4), max(2, width // 4)), dtype=np.uint8)
    img = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)


def subjectless_frame(size: int = FRAME_SIZE, seed: int = 4) -> np.ndarray:
    """Even texture edge to edge: no anchor, and no centre/rim difference."""
    return np.clip(_fine_texture(size, size, seed), 0, 255).astype(np.uint8)


def rim_damped_frame(rim_factor: float, size: int = FRAME_SIZE, seed: int = 4) -> np.ndarray:
    """The same texture with the rim's *contrast* scaled by ``rim_factor``.

    Scaling contrast rather than brightness is what makes this a detail fixture
    instead of a vignette one: mean luminance is untouched, so ``edges`` stays
    clean and only the Laplacian energy moves.
    """
    base = _fine_texture(size, size, seed)
    damped = 128.0 + (base - 128.0) * rim_factor
    third = size // 3
    damped[third : 2 * third, third : 2 * third] = base[third : 2 * third, third : 2 * third]
    return np.clip(damped, 0, 255).astype(np.uint8)


def empty_centre_frame(size: int = FRAME_SIZE, seed: int = 4) -> np.ndarray:
    """The inverse composition: a flat middle inside a busy rim."""
    frame = np.clip(_fine_texture(size, size, seed), 0, 255).astype(np.uint8)
    third = size // 3
    frame[third : 2 * third, third : 2 * third] = 128
    return frame


def flat_frame(size: int = FRAME_SIZE) -> np.ndarray:
    """Featureless gray plus sensor noise — nothing to anchor on anywhere."""
    rng = np.random.default_rng(1)
    return np.clip(128 + rng.normal(0.0, 4.0, (size, size)), 0, 255).astype(np.uint8)


def _paint_subject(
    frame: np.ndarray,
    area_frac: float,
    cx_frac: float = 0.5,
    cy_frac: float = 0.5,
    level: float = 215.0,
    amplitude: float = 26.0,
    seed: int = 17,
) -> np.ndarray:
    """Paint a bright *textured* disc of exactly ``area_frac`` of the frame.

    Textured, not flat-filled, on purpose: a uniform blob is the one case a
    percentile threshold happens to measure correctly, so a flat fixture would
    have hidden the very failure that put Otsu in the implementation.
    """
    height, width = frame.shape
    radius = round(math.sqrt(area_frac * height * width / math.pi))
    subject = level + (_fine_texture(height, width, seed) - 128.0) / 127.0 * amplitude
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (int(width * cx_frac), int(height * cy_frac)), radius, 1, -1)
    out = frame.astype(np.float32).copy()
    out[mask == 1] = subject[mask == 1]
    return np.clip(out, 0, 255).astype(np.uint8)


def subject_frame(
    area_frac: float = 0.11,
    cx_frac: float = 0.5,
    size: int = FRAME_SIZE,
    seed: int = 3,
) -> np.ndarray:
    """A bright subject of a given size and position on a dark textured field."""
    background = np.clip(45.0 + (_fine_texture(size, size, seed) - 128.0) / 127.0 * 30.0, 0, 255)
    return _paint_subject(background.astype(np.uint8), area_frac, cx_frac=cx_frac)


def anchored_outflow_frames(
    n: int = N_FRAMES,
    size: int = FRAME_SIZE,
    zoom_per_frame: float = 1.04,
) -> list[np.ndarray]:
    """A clip that should pass all six checks: forward flight *and* good framing.

    Radial outflow as in :func:`radial_outflow_frames`, over a canvas whose
    contrast falls off with radius (detail in the middle, an empty rim), with a
    subject held at the centre of frame — which is what a chase-cam anchor
    actually looks like: the world streams outward past a hero that stays put.
    """
    canvas = _fine_texture(size * 2, size * 2, seed=5)
    ys, xs = np.mgrid[0 : size * 2, 0 : size * 2]
    radius = np.hypot(xs - size, ys - size) / float(size)
    weight = np.clip(1.0 - 1.2 * radius, 0.15, 1.0).astype(np.float32)
    canvas = np.clip(70.0 + (canvas - 128.0) / 127.0 * 40.0 * weight, 0, 255).astype(np.uint8)
    frames: list[np.ndarray] = []
    for i in range(n):
        matrix = cv2.getRotationMatrix2D((size, size), 0, zoom_per_frame**i)
        big = cv2.warpAffine(canvas, matrix, (2 * size, 2 * size), flags=cv2.INTER_LINEAR)
        crop = big[size // 2 : size // 2 + size, size // 2 : size // 2 + size].copy()
        frames.append(_paint_subject(crop, 0.11))
    return frames


def _detail_for(frames: list[np.ndarray]) -> list[dict[str, float]]:
    return [csq.center_detail_stats(f) for f in frames]


def _anchor_for(frames: list[np.ndarray]) -> list[dict[str, float]]:
    return [csq.anchor_stats(f) for f in frames]


# ---------------------------------------------------------------------------
# forward_motion — the card's three acceptance cases
# ---------------------------------------------------------------------------


def test_radial_outflow_is_recognised_as_forward_motion() -> None:
    """Acceptance 1: a radial-outflow clip passes ``forward_motion``.

    Both of §QA's conditions are asserted explicitly, not just the verdict, so
    a future regression says *which* half broke: the radial component must be
    positive, and it must grow with radius (outer ring above inner disc) —
    the signature of ``flow ≈ (s-1)·r``.
    """
    stats = _stats_for(radial_outflow_frames())
    assert len(stats) == N_FRAMES - 1

    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_PASS, f"radial outflow must PASS, got {result.status}: {result.detail}"
    assert result.measured["radial_positive"] is True
    assert result.measured["outer_exceeds_inner"] is True
    assert result.measured["radial_rate"] > csq.DEFAULT_MIN_RADIAL_RATE
    assert result.measured["outer_mean_px"] > result.measured["inner_mean_px"]


def test_static_clip_fails_and_says_static() -> None:
    """Acceptance 2: a locked-off clip FAILs and the message contains 「静态」.

    The wording is load-bearing, not cosmetic: the operator reads one line and
    has to know the clip needs regenerating, not re-encoding.  The advice names
    静态 for every ``forward_motion`` failure by design
    (:data:`~scripts.check_source_quality.FORWARD_FAIL_ADVICE`) because the
    remedy is the same whichever sub-mode was detected.
    """
    result = csq.check_forward_motion(_stats_for(static_frames()))
    assert result.status == csq.STATUS_FAIL, f"static clip must FAIL, got {result.status}: {result.detail}"
    assert "静态" in result.detail + result.advice
    assert result.measured["radial_positive"] is False
    # The card's own description of a locked-off shot: sub-pixel and negative.
    assert abs(result.measured["radial_mean_px"]) <= 1.0


def test_panning_clip_fails_because_radial_component_is_not_positive() -> None:
    """Acceptance 3: a lateral pan FAILs — motion, but not *outward* motion.

    The second assertion is the point of the whole method: the pan's raw flow
    is comparable to (here, larger than) the positive sample's, so a
    magnitude-only test would wave it through.  It is the radial *projection*
    that collapses to ~0.
    """
    result = csq.check_forward_motion(_stats_for(panning_frames()))
    assert result.status == csq.STATUS_FAIL, f"pan must FAIL, got {result.status}: {result.detail}"
    assert result.measured["radial_positive"] is False
    assert result.measured["flow_mean_px"] > 1.0, "the pan fixture must actually contain plenty of motion"
    assert abs(result.measured["radial_rate"]) < csq.DEFAULT_MIN_RADIAL_RATE


def test_the_three_cases_are_separated_by_orders_of_magnitude() -> None:
    """The margin itself is the contract — the thresholds must not sit on a knife edge.

    Asserting only pass/fail would let a future tweak shrink the separation to
    a hair without any test noticing.  This pins the actual daylight: the good
    case clears the threshold by ≥5×, and both bad cases sit ≥5× *below* it.
    """
    good = csq.check_forward_motion(_stats_for(radial_outflow_frames())).measured["radial_rate"]
    still = csq.check_forward_motion(_stats_for(static_frames())).measured["radial_rate"]
    pan = csq.check_forward_motion(_stats_for(panning_frames())).measured["radial_rate"]
    floor = csq.DEFAULT_MIN_RADIAL_RATE

    assert good > floor * 5.0, f"forward-motion margin too thin: {good} vs threshold {floor}"
    assert abs(still) < floor / 5.0, f"static clip too close to the threshold: {still}"
    assert abs(pan) < floor / 5.0, f"pan too close to the threshold: {pan}"


def test_slow_forward_motion_passes_with_a_weak_warning() -> None:
    """Barely-moving-but-moving is a WARN, not a FAIL.

    The distinction is a real operator decision: a slow push still yields
    parallax, so blocking the run would be wrong, but the stereo will be subtle
    and they should know before they spend the GPU hour.
    """
    radius = 100.0
    rate = csq.DEFAULT_MIN_RADIAL_RATE * 1.2  # inside the weak band (< 2x)
    stats = [
        csq.RadialStats(
            radial_mean=rate * radius,
            inner_mean=0.2 * rate * radius,
            outer_mean=1.6 * rate * radius,
            flow_mean=rate * radius,
            radius=radius,
        )
    ] * 3
    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_WARN
    assert result.advice == csq.FORWARD_WEAK_ADVICE


def test_outward_flow_that_does_not_grow_with_radius_fails() -> None:
    """Positive radial mean alone is not enough — condition 2 stands on its own.

    Guards against a future "simplification" that drops the ring comparison:
    forward motion scales the flow with radius, and a field that is uniformly
    outward everywhere is not the geometry of flying into a scene.
    """
    stats = [csq.RadialStats(radial_mean=5.0, inner_mean=6.0, outer_mean=4.0, flow_mean=5.0, radius=100.0)] * 3
    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_FAIL
    assert result.measured["radial_positive"] is True
    assert result.measured["outer_exceeds_inner"] is False


def test_no_usable_pairs_fails_rather_than_silently_passing() -> None:
    """Zero sampled pairs must FAIL: "we measured nothing" is not "it's fine"."""
    result = csq.check_forward_motion([])
    assert result.status == csq.STATUS_FAIL
    assert result.measured["pairs"] == 0


def test_forward_motion_verdict_is_a_median_not_a_mean() -> None:
    """One wild pair (a scene cut) must not flip a clip's verdict.

    A mean over 5 pairs would be dragged negative by a single outlier; the
    median ignores it, which is why the aggregation is specified rather than
    incidental.
    """
    radius = 100.0
    good = csq.RadialStats(radial_mean=3.0, inner_mean=1.0, outer_mean=4.0, flow_mean=3.0, radius=radius)
    cut = csq.RadialStats(radial_mean=-400.0, inner_mean=-400.0, outer_mean=-400.0, flow_mean=400.0, radius=radius)
    assert csq.check_forward_motion([good, good, cut, good, good]).status == csq.STATUS_PASS


def test_radial_rate_is_scale_free() -> None:
    """The same clip at two resolutions yields the same rate.

    This is why the threshold is a fraction of the half-diagonal ``R`` rather
    than a pixel count: the analysis width is an implementation detail
    (``--flow-width``), and a resolution-dependent threshold would quietly mean
    something different on every source.
    """
    small = _stats_for(radial_outflow_frames(size=160))
    large = _stats_for(radial_outflow_frames(size=320))
    rate_small = float(np.median([s.radial_rate for s in small]))
    rate_large = float(np.median([s.radial_rate for s in large]))
    assert rate_small == pytest.approx(rate_large, rel=0.25), (
        f"radial rate must be resolution independent: {rate_small} vs {rate_large}"
    )
    # ...and the raw pixel figures genuinely do differ, so the test above is
    # not passing for the trivial reason that nothing changed.
    px_small = float(np.median([s.radial_mean for s in small]))
    px_large = float(np.median([s.radial_mean for s in large]))
    assert px_large > px_small * 1.5


def test_radial_flow_stats_rejects_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="same-sized grayscale"):
        csq.radial_flow_stats(np.zeros((10, 10), np.uint8), np.zeros((12, 10), np.uint8))


# ---------------------------------------------------------------------------
# edges — letterbox bars and blown highlights
# ---------------------------------------------------------------------------


def _letterboxed_frame(size: int = FRAME_SIZE, bar: int = 20) -> np.ndarray:
    """A frame with pure-black bars top and bottom — the classic 16:9-in-1:1."""
    frame = _texture(size, size)
    frame[:bar, :] = 0
    frame[size - bar :, :] = 0
    return frame


def test_letterbox_bars_fail_and_name_the_top_and_bottom_edges() -> None:
    """Acceptance 4: black bars top/bottom FAIL, and the report says *which* edges.

    Naming the edge is the actionable half: "there is a bar somewhere" leaves
    the operator to hunt, while 上边/下边 tells them what to crop.  The left and
    right edges must stay clean in the same run, otherwise the check is just
    flagging everything.
    """
    frames = [csq.edge_band_stats(_letterboxed_frame()) for _ in range(4)]
    result = csq.check_edges(frames)

    assert result.status == csq.STATUS_FAIL
    assert result.measured["letterboxed_edges"] == ["top", "bottom"]
    assert "上边" in result.detail and "下边" in result.detail
    assert "左边" not in result.detail and "右边" not in result.detail


def test_dark_but_textured_edges_are_not_mistaken_for_letterbox_bars() -> None:
    """A dark *scene* must pass: the variance half of the rule earns its place.

    Night footage has edge bands with a low mean.  Without the variance
    condition this check would fail every dark clip — the fastest way to make
    an operator start ignoring the tool.
    """
    rng = np.random.default_rng(5)
    dark = rng.integers(0, 30, size=(FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    result = csq.check_edges([csq.edge_band_stats(dark) for _ in range(4)])
    assert result.status == csq.STATUS_PASS
    assert result.measured["letterboxed_edges"] == []


def test_blown_out_edge_fails_and_names_that_edge() -> None:
    """A dead-white band on one side FAILs and is attributed to that side."""
    frame = _texture(FRAME_SIZE, FRAME_SIZE)
    frame[:, FRAME_SIZE - 12 :] = 255
    result = csq.check_edges([csq.edge_band_stats(frame) for _ in range(4)])
    assert result.status == csq.STATUS_FAIL
    assert result.measured["overexposed_edges"] == ["right"]
    assert "右边" in result.detail


def test_one_black_frame_does_not_brand_the_clip_as_letterboxed() -> None:
    """A single fade-to-black frame is outvoted by the median across frames.

    Fades are ordinary; a bar is not. The distinction is that a bar is present
    in *every* frame, which is exactly what a median across the sampled frames
    tests for.
    """
    clean = [csq.edge_band_stats(_texture(FRAME_SIZE, FRAME_SIZE)) for _ in range(4)]
    black = csq.edge_band_stats(np.zeros((FRAME_SIZE, FRAME_SIZE), np.uint8))
    assert csq.check_edges([*clean, black]).status == csq.STATUS_PASS


def test_edge_band_stats_reports_every_side() -> None:
    stats = csq.edge_band_stats(_texture(64, 64), band=4)
    assert set(stats) == set(csq.EDGE_LABELS)
    assert all({"mean", "var", "bright_frac"} <= set(v) for v in stats.values())


def test_no_frames_fails_rather_than_silently_passing() -> None:
    assert csq.check_edges([]).status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# detail_ratio — centre 1/3 vs the rim (Red Raion measured 1.58–1.60)
# ---------------------------------------------------------------------------


def test_detail_concentrated_in_the_centre_passes() -> None:
    """Acceptance: dense centre + flat rim PASSes, and not by a whisker.

    This is the composition the reference films use, and the reason to want it
    is ours rather than theirs: the outer ring is where VR180 geometry,
    outward extrapolation and feathering all look worst, so a source with
    nothing out there hides our weakest work for free.
    """
    result = csq.check_detail_ratio(_detail_for([rim_damped_frame(0.18)]))
    assert result.status == csq.STATUS_PASS, f"{result.status}: {result.detail}"
    assert result.measured["ratio"] > csq.DETAIL_RATIO_PASS * 2.0, (
        f"margin too thin to be a real test: {result.measured['ratio']}"
    )
    assert result.measured["center_edge_energy"] > result.measured["periphery_edge_energy"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_evenly_textured_frame_never_passes_detail_ratio(seed: int) -> None:
    """Acceptance: a full-frame even texture must not pass, whatever the seed.

    Parametrised rather than pinned to one lucky seed because the whole claim is
    that the check responds to *composition*, not to which noise field it was
    handed.  Measured across these eight, the ratio spans 0.975–1.035 — every
    one of them below the 1.2 gate, and straddling 1.0, which is why the
    assertion is "not PASS" rather than a single status.
    """
    result = csq.check_detail_ratio(_detail_for([subjectless_frame(seed=seed)]))
    assert result.status in (csq.STATUS_WARN, csq.STATUS_FAIL), result.detail
    assert result.measured["ratio"] < csq.DETAIL_RATIO_PASS


def test_a_barely_denser_centre_warns_and_names_the_rim() -> None:
    """The middle band exists and says the thing the operator has to act on.

    A frame whose rim is only slightly quieter than its middle is not broken —
    it is just not doing the trick that makes the rim invisible — so it warns,
    and the advice carries 「边缘太满」 verbatim.
    """
    result = csq.check_detail_ratio(_detail_for([rim_damped_frame(0.90)]))
    assert result.status == csq.STATUS_WARN, f"{result.status}: {result.detail}"
    assert csq.DETAIL_RATIO_FAIL < result.measured["ratio"] < csq.DETAIL_RATIO_PASS
    assert "边缘太满" in result.advice


def test_a_busier_rim_than_centre_fails_and_gates_the_run() -> None:
    """Acceptance: the inverted composition FAILs, and a FAIL still exits 1.

    Detail out at the rim and none in the middle is the arrangement that walks
    the viewer's eye straight onto the ugliest ring of the projection, so this
    is the one composition verdict allowed to block a run.
    """
    result = csq.check_detail_ratio(_detail_for([empty_centre_frame()]))
    assert result.status == csq.STATUS_FAIL, f"{result.status}: {result.detail}"
    assert result.measured["ratio"] < csq.DETAIL_RATIO_FAIL / 2.0
    assert csq.SourceReport(source="f.png", checks=[result]).exit_code == 1


def test_detail_ratio_verdict_is_a_median_not_a_mean() -> None:
    """One flash frame must not condemn a clip, same rule as every other check."""
    good = {"center": 4.0, "periphery": 2.0, "ratio": 2.0}
    flash = {"center": 0.1, "periphery": 90.0, "ratio": 0.001}
    assert csq.check_detail_ratio([good, good, flash, good, good]).status == csq.STATUS_PASS


def test_detail_ratio_survives_a_resolution_change() -> None:
    """The same picture at 2× must not change verdict.

    The analysis runs at a fixed longest side precisely so that a 2880² render
    and a 1024² keyframe are measured on comparable terms; without that, edge
    energy — which is not scale invariant — would make the threshold mean
    something different for every source.
    """
    small = rim_damped_frame(0.18, size=480)
    large = cv2.resize(small, (960, 960), interpolation=cv2.INTER_CUBIC)
    small_ratio = csq.center_detail_stats(small)["ratio"]
    large_ratio = csq.center_detail_stats(large)["ratio"]
    assert csq.check_detail_ratio([{"center": 0.0, "periphery": 0.0, "ratio": small_ratio}]).status == (
        csq.check_detail_ratio([{"center": 0.0, "periphery": 0.0, "ratio": large_ratio}]).status
    )
    assert small_ratio == pytest.approx(large_ratio, rel=0.25)


def test_detail_ratio_analysis_is_downscaled_not_run_at_source_resolution() -> None:
    """A 2880² frame is measured at :data:`DETAIL_ANALYSIS_MAX_DIM`, not native."""
    fitted = csq._fit_for_analysis(np.zeros((2880, 2880), np.uint8), csq.DETAIL_ANALYSIS_MAX_DIM)
    assert max(fitted.shape) == csq.DETAIL_ANALYSIS_MAX_DIM
    small = np.zeros((320, 320), np.uint8)
    assert csq._fit_for_analysis(small, csq.DETAIL_ANALYSIS_MAX_DIM) is small


def test_detail_ratio_reads_a_colour_frame_as_luminance() -> None:
    """Stills arrive BGR and clips arrive gray; both must measure the same."""
    gray = rim_damped_frame(0.18)
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert csq.center_detail_stats(colour)["ratio"] == pytest.approx(csq.center_detail_stats(gray)["ratio"], rel=1e-6)


def test_detail_ratio_rejects_an_unusable_frame_shape() -> None:
    with pytest.raises(ValueError, match="grayscale or BGR"):
        csq.center_detail_stats(np.zeros((8, 8, 4), np.uint8))


def test_no_frames_fails_detail_ratio_rather_than_passing_it() -> None:
    assert csq.check_detail_ratio([]).status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# anchor — is there a subject up front? (Red Raion: 80 % of frames, 11.3 %, 0.236)
# ---------------------------------------------------------------------------


def test_centred_subject_of_reference_size_passes() -> None:
    """Acceptance: a centred subject at the reference size (~11 %) PASSes.

    Both reported numbers are asserted, not just the verdict, because the
    verdict alone would survive a detector that found the wrong thing and
    happened to land in the band.
    """
    result = csq.check_anchor(_anchor_for([subject_frame(0.11)]))
    assert result.status == csq.STATUS_PASS, f"{result.status}: {result.detail}"
    assert csq.ANCHOR_AREA_MIN <= result.measured["area"] <= csq.ANCHOR_AREA_MAX
    assert result.measured["offset"] < csq.ANCHOR_OFFSET_MAX
    assert result.measured["detected_frac"] == 1.0


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        ("uniform texture", subjectless_frame(seed=4)),
        ("uniform texture, other seed", subjectless_frame(seed=7)),
        ("flat gray + noise", flat_frame()),
        ("empty centre, busy rim", empty_centre_frame()),
    ],
)
def test_subjectless_frames_report_no_anchor(name: str, frame: np.ndarray) -> None:
    """Acceptance: a pure background WARNs and the wording says 「没有锚点」.

    This is the branch that would quietly stop existing if the detector were
    tuned to always find *something*: a threshold that selects a fixed share of
    every frame reports a subject in an empty one too.  The message is asserted
    verbatim because the operator acts on the phrase, not on the number.
    """
    result = csq.check_anchor(_anchor_for([frame]))
    assert result.status == csq.STATUS_WARN, f"{name}: {result.status} — {result.detail}"
    assert "没有锚点" in result.detail + result.advice, f"{name}: {result.detail}"
    assert result.measured["detected_frames"] == 0, name


def test_subject_and_subjectless_frames_are_separated() -> None:
    """The margin is the contract — measured before the threshold was picked.

    Asserting only the verdicts would let a future tweak shrink the gap to a
    hair without any test noticing.  This pins the actual daylight: every frame
    with a subject reports at least 5× :data:`ANCHOR_MIN_AREA`, every frame
    without one at most a fifth of it.
    """
    with_subject = [
        csq.anchor_stats(subject_frame(0.11))["area"],
        csq.anchor_stats(subject_frame(0.40))["area"],
        csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))["area"],
    ]
    without = [csq.anchor_stats(subjectless_frame(seed=s))["area"] for s in range(1, 9)]
    without += [csq.anchor_stats(flat_frame())["area"], csq.anchor_stats(rim_damped_frame(0.18))["area"]]

    assert min(with_subject) > csq.ANCHOR_MIN_AREA * 5.0, f"true subjects too quiet: {with_subject}"
    assert max(without) < csq.ANCHOR_MIN_AREA / 4.0, f"false subjects too loud: {without}"


def test_oversized_subject_warns_and_says_it_is_too_big() -> None:
    """Acceptance: a subject filling 40 % of the frame WARNs, naming 偏大.

    Textured rather than flat-filled on purpose — a flat blob is the one shape
    a percentile threshold measures correctly, so this fixture is what forced
    the implementation onto Otsu.
    """
    result = csq.check_anchor(_anchor_for([subject_frame(0.40)]))
    assert result.status == csq.STATUS_WARN, f"{result.status}: {result.detail}"
    assert "偏大" in result.detail
    assert result.measured["area"] > csq.ANCHOR_AREA_MAX * 2.0
    assert result.measured["offset"] < csq.ANCHOR_OFFSET_MAX, "a centred subject must not also read as off-centre"


def test_off_centre_subject_warns_and_says_it_is_off_centre() -> None:
    """A correctly-sized subject in the wrong place is still a problem.

    The centre prior only breaks ties between competing salient regions; it must
    not drag the *measurement* toward the middle, or this WARN could never fire.
    """
    result = csq.check_anchor(_anchor_for([subject_frame(0.11, cx_frac=0.80)]))
    assert result.status == csq.STATUS_WARN, f"{result.status}: {result.detail}"
    assert "偏离中心" in result.detail
    assert result.measured["offset"] > csq.ANCHOR_OFFSET_MAX
    assert csq.ANCHOR_AREA_MIN <= result.measured["area"] <= csq.ANCHOR_AREA_MAX, (
        "the size reading must not be disturbed by the position"
    )


def test_anchor_never_blocks_a_run_on_content() -> None:
    """Every content verdict is PASS or WARN — no composition FAILs here.

    A landscape fly-through with no hero is a legitimate shot; gating it would
    make the operator start passing ``--skip anchor`` and lose the warning too.
    """
    frames = [subject_frame(0.11), subject_frame(0.40), subject_frame(0.11, cx_frac=0.80), flat_frame()]
    for frame in frames:
        report = csq.SourceReport(source="f.png", checks=[csq.check_anchor(_anchor_for([frame]))])
        assert report.exit_code == 0, csq.format_report(report)


def test_anchor_medians_over_the_frames_that_had_a_subject() -> None:
    """A hero on screen half the time reports an honest rate, not a diluted area.

    Averaging the missing frames in as zero-area would drag every intermittent
    subject into the 偏小 warning; the detection rate is reported separately
    instead, which is also how the reference study states it (80 % of frames).
    """
    present = {"found": True, "area": 0.11, "offset": 0.1, "components": 1}
    absent = {"found": False, "area": 0.0, "offset": 0.0, "components": 0}
    result = csq.check_anchor([present, present, present, absent])

    assert result.status == csq.STATUS_PASS
    assert result.measured["area"] == pytest.approx(0.11)
    assert result.measured["detected_frac"] == pytest.approx(0.75)


def test_a_subject_seen_in_a_minority_of_frames_counts_as_missing() -> None:
    present = {"found": True, "area": 0.11, "offset": 0.1, "components": 1}
    absent = {"found": False, "area": 0.0, "offset": 0.0, "components": 0}
    result = csq.check_anchor([present, absent, absent, absent])
    assert result.status == csq.STATUS_WARN
    assert "没有锚点" in result.detail + result.advice


def test_anchor_reads_a_colour_frame_and_a_gray_one() -> None:
    """Stills come in BGR, sampled clip frames come in gray; both must work."""
    gray = subject_frame(0.11)
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert csq.anchor_stats(colour)["found"] is True
    assert csq.anchor_stats(gray)["found"] is True


def test_anchor_rejects_a_nonsensical_kernel() -> None:
    with pytest.raises(ValueError, match="kernel must be"):
        csq.anchor_stats(subject_frame(), kernel=0)


def test_no_frames_fails_anchor_rather_than_passing_it() -> None:
    assert csq.check_anchor([]).status == csq.STATUS_FAIL


def test_saliency_is_normalised_and_peaks_on_the_subject() -> None:
    """The measure underneath: the subject must be the brightest thing in it."""
    saliency = csq.frequency_tuned_saliency(subject_frame(0.11))
    assert saliency.min() == pytest.approx(0.0, abs=1e-6)
    assert saliency.max() == pytest.approx(1.0, abs=1e-6)
    height, width = saliency.shape
    middle = saliency[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]
    assert middle.mean() > saliency.mean() * 1.5


# ---------------------------------------------------------------------------
# still mode — the cheap gate (G-3a #336)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("keyframe.png", True),
        ("shot.JPG", True),
        ("frame.jpeg", True),
        ("still.webp", True),
        ("plate.tif", True),
        ("clip.mp4", False),
        ("clip.mov", False),
        ("noextension", False),
    ],
)
def test_still_mode_is_chosen_by_extension(name: str, expected: bool) -> None:
    """Extension, not content sniffing — a wrong guess costs the motion check."""
    assert csq.is_still(name) is expected


def _write_still(path: Path, frame: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), frame), f"could not write fixture to {path}"
    return path


def _still_report(tmp_path: Path, monkeypatch, frame: np.ndarray, name: str = "keyframe.png"):
    """``run_checks`` on a real image file, with only ffprobe replaced."""
    still = _write_still(tmp_path / name, frame)
    height, width = frame.shape[:2]
    info = csq.ProbeInfo(width=width, height=height, fps=25.0, duration=0.04, pix_fmt="rgb24", codec="png")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a still must not go through the video frame sampler")

    monkeypatch.setattr(csq, "iter_frame_pairs", fail_if_called)
    return csq.run_checks(still)


def test_a_still_runs_every_check_but_skips_forward_motion(tmp_path: Path, monkeypatch) -> None:
    """Acceptance: a ``.png`` runs to the end, and the motion check is *skipped*.

    ``skipped`` and not ``pass``: one frame is no evidence that the camera
    moves, and a pass here would let a locked-off clip through on its keyframe's
    reputation.  Everything else must produce a real verdict, or vetting the
    keyframe would not be worth doing.
    """
    report = _still_report(tmp_path, monkeypatch, anchored_outflow_frames()[0])

    statuses = {c.name: c.status for c in report.checks}
    assert [c.name for c in report.checks] == list(csq.CHECK_NAMES)
    assert statuses["forward_motion"] == csq.STATUS_SKIP
    assert statuses == {
        "decodable": csq.STATUS_PASS,
        "square": csq.STATUS_PASS,
        "forward_motion": csq.STATUS_SKIP,
        "edges": csq.STATUS_PASS,
        "detail_ratio": csq.STATUS_PASS,
        "anchor": csq.STATUS_PASS,
    }
    assert report.exit_code == 0

    motion = next(c for c in report.checks if c.name == "forward_motion")
    assert "静帧" in motion.detail
    assert motion.advice == csq.STILL_FORWARD_ADVICE


def test_a_still_uses_the_same_exit_code_contract_as_a_clip(tmp_path: Path, monkeypatch) -> None:
    """Acceptance: WARNs still exit 0, a FAIL still exits 1 — identical to video.

    Without this the still path could quietly become advisory-only, and the
    whole point is to *stop* a bad keyframe before ¥50 of generation.
    """
    warned = _still_report(tmp_path, monkeypatch, rim_damped_frame(0.90), name="warn.png")
    assert warned.summary["overall"] == csq.STATUS_WARN
    assert warned.exit_code == 0

    failed = _still_report(tmp_path, monkeypatch, empty_centre_frame(), name="fail.png")
    assert failed.failed is True
    assert failed.exit_code == 1
    assert next(c for c in failed.checks if c.name == "detail_ratio").status == csq.STATUS_FAIL


def test_still_decodable_reports_the_frame_without_inventing_a_duration() -> None:
    """ffmpeg calls a still a 25 fps 0.04 s clip; judging that would be noise.

    The video branch is deliberately left alone — the same metadata *without*
    ``still`` still produces the ordinary verdict.
    """
    info = csq.ProbeInfo(width=1024, height=1024, fps=25.0, duration=0.04, pix_fmt="rgb24", codec="png")
    result = csq.check_decodable(info, still=True)

    assert result.status == csq.STATUS_PASS
    assert result.measured["still"] is True
    assert "静帧" in result.detail
    assert "frame_count_drift" not in result.measured
    assert csq.check_decodable(info).measured.get("still") is None


def test_read_still_round_trips_a_written_image(tmp_path: Path) -> None:
    frame = subject_frame(0.11)
    decoded = csq.read_still(_write_still(tmp_path / "k.png", frame))
    assert decoded.shape == (*frame.shape, 3)
    assert np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), frame)


def test_read_still_raises_on_something_that_is_not_an_image(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not a png")
    with pytest.raises(RuntimeError, match="cannot decode image"):
        csq.read_still(broken)


def test_an_undecodable_still_fails_instead_of_crashing(tmp_path: Path, monkeypatch) -> None:
    """A truncated download must come back as a FAIL row, not a traceback."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not a png")
    info = csq.ProbeInfo(width=1024, height=1024, fps=25.0, duration=0.04, pix_fmt="rgb24", codec="png")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    report = csq.run_checks(broken)
    assert report.exit_code == 1
    assert report.checks[0].name == "decodable"
    assert report.checks[0].status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# square — aspect ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "status"),
    [
        (2880, 2880, csq.STATUS_PASS),
        (1024, 1024, csq.STATUS_PASS),
        (1015, 1000, csq.STATUS_PASS),  # 1.5% off — inside the ±2% tolerance
        (1030, 1000, csq.STATUS_WARN),  # 3% off — outside it
        (1280, 720, csq.STATUS_WARN),
        (1920, 1080, csq.STATUS_WARN),
    ],
)
def test_square_check_warns_but_never_fails_on_shape(width: int, height: int, status: str) -> None:
    """Non-square WARNs and never FAILs — the fisheye and 16:9 routes are legitimate.

    Turning this into a FAIL would block sources the pipeline handles fine
    (§QB), so the exit code must stay 0 for a plain 16:9 clip.
    """
    result = csq.check_square(width, height)
    assert result.status == status
    assert result.measured["width"] == width
    assert result.measured["height"] == height


def test_square_check_fails_only_on_a_nonsensical_size() -> None:
    assert csq.check_square(0, 0).status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# decodable — ffprobe metadata (parsed, never executed)
# ---------------------------------------------------------------------------


def test_self_consistent_metadata_passes_and_reports_pix_fmt() -> None:
    """10-bit is reported, never judged (#292/#298 proved it runs)."""
    info = csq.ProbeInfo(
        width=2880, height=2880, fps=24.0, duration=10.0, nb_frames=240, pix_fmt="yuv420p10le", codec="hevc"
    )
    result = csq.check_decodable(info)
    assert result.status == csq.STATUS_PASS
    assert result.measured["pix_fmt"] == "yuv420p10le"
    assert "yuv420p10le" in result.detail


def test_frame_count_drift_warns_without_blocking_the_run() -> None:
    """Broken timestamps are worth saying out loud, but the file still decodes."""
    info = csq.ProbeInfo(width=1280, height=720, fps=24.0, duration=10.0, nb_frames=120, pix_fmt="yuv420p")
    result = csq.check_decodable(info)
    assert result.status == csq.STATUS_WARN
    assert result.measured["frame_count_drift"] == pytest.approx(0.5)


def test_missing_frame_count_still_passes() -> None:
    """Plenty of containers omit ``nb_frames``; that is not a defect."""
    info = csq.ProbeInfo(width=1280, height=720, fps=24.0, duration=10.0, nb_frames=0, pix_fmt="yuv420p")
    assert csq.check_decodable(info).status == csq.STATUS_PASS


def test_unreadable_dimensions_fail() -> None:
    assert csq.check_decodable(csq.ProbeInfo()).status == csq.STATUS_FAIL


def test_probe_source_parses_captured_ffprobe_json(monkeypatch, tmp_path: Path) -> None:
    """The ffprobe parser is exercised against real captured output, not a live call.

    CI has ffprobe, but a test that decodes a real file would need a real file;
    this pins the JSON shape (including ffprobe's ``"24/1"`` frame-rate
    fractions and string-typed ``nb_frames``) with no subprocess at all.
    """
    payload = {
        "streams": [
            {
                "width": 2880,
                "height": 2880,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
                "nb_frames": "240",
                "pix_fmt": "yuv420p10le",
                "codec_name": "hevc",
                "duration": "10.000000",
            }
        ],
        "format": {"duration": "10.000000"},
    }
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload).encode("utf-8"), b"")

    monkeypatch.setattr(csq.subprocess, "run", fake_run)
    info = csq.probe_source(tmp_path / "clip.mp4")

    assert (info.width, info.height, info.fps, info.nb_frames) == (2880, 2880, 24.0, 240)
    assert info.pix_fmt == "yuv420p10le"
    assert isinstance(captured["cmd"], list) and captured["cmd"][0] == "ffprobe"


def test_probe_source_raises_when_there_is_no_video_stream(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, b'{"streams": []}', b"")

    monkeypatch.setattr(csq.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no video stream"):
        csq.probe_source(tmp_path / "audio.m4a")


# ---------------------------------------------------------------------------
# Report / JSON / exit code
# ---------------------------------------------------------------------------


def _report_with(*checks: csq.CheckResult) -> csq.SourceReport:
    return csq.SourceReport(source="clip.mp4", checks=list(checks))


def test_json_payload_is_loadable_and_carries_name_status_measured() -> None:
    """Acceptance 5a: ``--json`` is machine-readable and complete.

    The card names the three fields a consumer needs on every check; the
    ``measured`` numbers are what let a caller disagree with the verdict
    without re-running the analysis.
    """
    report = _report_with(
        csq.check_square(2880, 2880),
        csq.check_forward_motion(_stats_for(radial_outflow_frames())),
        csq.check_detail_ratio(_detail_for([rim_damped_frame(0.18)])),
        csq.check_anchor(_anchor_for([subject_frame(0.11)])),
    )
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))

    assert {c["name"] for c in payload["checks"]} == {"square", "forward_motion", "detail_ratio", "anchor"}
    for check in payload["checks"]:
        assert {"name", "status", "measured"} <= set(check)
        assert isinstance(check["measured"], dict)
    assert payload["summary"]["overall"] == csq.STATUS_PASS
    assert payload["exit_code"] == 0

    # The composition checks have to ship the numbers their verdict was taken
    # on, or a caller cannot disagree with them without re-running the analysis.
    by_name = {c["name"]: c["measured"] for c in payload["checks"]}
    assert {"ratio", "pass_ratio", "fail_ratio"} <= set(by_name["detail_ratio"])
    assert {"area", "offset", "detected_frac"} <= set(by_name["anchor"])


def test_exit_code_is_one_when_anything_fails_and_zero_for_warn_only() -> None:
    """Acceptance 5b: WARNs never gate a run; FAILs always do.

    A non-square 16:9 source is a WARN and must still exit 0 — otherwise
    wiring this into preflight would block sources the pipeline handles fine.
    """
    failing = _report_with(csq.CheckResult("forward_motion", csq.STATUS_FAIL, "static", {}))
    assert failing.exit_code == 1
    assert failing.summary["overall"] == csq.STATUS_FAIL

    warn_only = _report_with(csq.check_square(1280, 720))
    assert warn_only.exit_code == 0
    assert warn_only.summary["overall"] == csq.STATUS_WARN

    clean = _report_with(csq.check_square(1024, 1024))
    assert clean.exit_code == 0
    assert clean.summary["overall"] == csq.STATUS_PASS


def test_human_report_shows_an_icon_and_the_advice_for_every_check() -> None:
    """The human report has to be readable on its own — icon, numbers, next step."""
    report = _report_with(
        csq.check_square(1280, 720),
        csq.check_forward_motion(_stats_for(static_frames())),
    )
    text = csq.format_report(report)

    assert "❌" in text and "⚠️" in text
    assert "静态" in text
    assert csq.NON_SQUARE_ADVICE in text
    assert "不要进管线" in text


def test_skipping_a_check_marks_it_skipped_instead_of_dropping_it(monkeypatch, tmp_path: Path) -> None:
    """``--skip`` must leave a visible trace, not a silent gap.

    A skipped check that simply vanished from the report would read as "this
    clip has three checks", and nobody would notice the motion test was never
    run.  The frame sampler is monkeypatched away here, which also asserts the
    orchestrator does not decode anything when *all four* frame-hungry checks
    are skipped — the composition pair need frames too, so the decode is only
    avoidable when none of the four is wanted.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=1024, height=1024, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no frames should be decoded when every frame check is skipped")

    monkeypatch.setattr(csq, "iter_frame_pairs", fail_if_called)
    report = csq.run_checks(clip, skip=["forward_motion", "edges", "detail_ratio", "anchor"])

    statuses = {c.name: c.status for c in report.checks}
    assert statuses["forward_motion"] == csq.STATUS_SKIP
    assert statuses["edges"] == csq.STATUS_SKIP
    assert statuses["detail_ratio"] == csq.STATUS_SKIP
    assert statuses["anchor"] == csq.STATUS_SKIP
    assert statuses["square"] == csq.STATUS_PASS
    assert report.exit_code == 0
    assert [c.name for c in report.checks] == list(csq.CHECK_NAMES)


def test_unknown_skip_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown check name"):
        csq.run_checks("clip.mp4", skip=["forwards"])


def test_missing_file_fails_without_touching_ffprobe(monkeypatch, tmp_path: Path) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("probe_source must not run for a missing file")

    monkeypatch.setattr(csq, "probe_source", fail_if_called)
    report = csq.run_checks(tmp_path / "nope.mp4")
    assert report.exit_code == 1
    assert report.checks[0].name == "decodable"


def test_run_checks_wires_the_sampled_pairs_into_every_frame_check(monkeypatch, tmp_path: Path) -> None:
    """End-to-end through ``run_checks`` with the decoder replaced by fixtures.

    This is the only test that exercises the orchestration — sampling, the
    native-resolution edge bands, the downscaled flow analysis, both
    composition measurements and the report assembly — and it does so on
    synthetic frames, so it stays honest on a runner with no video files.

    The fixture is a clip that is good on every axis at once (forward flight,
    detail in the middle, a subject held at the centre), which is the only way
    to assert that *nothing* in the six-check report objects to a clip the
    pipeline should happily take.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=FRAME_SIZE, height=FRAME_SIZE, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    frames = anchored_outflow_frames()
    monkeypatch.setattr(
        csq,
        "iter_frame_pairs",
        lambda *a, **k: iter([(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]),
    )
    report = csq.run_checks(clip)

    statuses = {c.name: c.status for c in report.checks}
    assert statuses == {
        "decodable": csq.STATUS_PASS,
        "square": csq.STATUS_PASS,
        "forward_motion": csq.STATUS_PASS,
        "edges": csq.STATUS_PASS,
        "detail_ratio": csq.STATUS_PASS,
        "anchor": csq.STATUS_PASS,
    }, csq.format_report(report)
    assert report.exit_code == 0


def test_downscale_for_flow_shrinks_wide_frames_and_leaves_small_ones_alone() -> None:
    """A 2880² source is analysed at ``--flow-width``; a 320 px one is untouched."""
    big = np.zeros((2880, 2880), np.uint8)
    assert csq.downscale_for_flow(big, 480).shape[1] == 480
    small = np.zeros((320, 320), np.uint8)
    assert csq.downscale_for_flow(small, 480) is small


def test_radial_basis_is_centred_and_normalised() -> None:
    """The geometry the whole check rests on: unit vectors pointing outward.

    ``R`` is the half-diagonal, so ``r/R`` spans [0, 1] over the frame and the
    corner pixels — the fastest-moving ones under forward motion — are inside
    the outer ring rather than off the end of it.
    """
    ux, uy, r, radius = csq._radial_basis(101, 101)
    assert radius == pytest.approx(0.5 * math.hypot(101, 101))
    assert r[50, 50] == pytest.approx(0.0)
    assert ux[50, 100] == pytest.approx(1.0, abs=1e-3)
    assert uy[100, 50] == pytest.approx(1.0, abs=1e-3)
    assert float(np.hypot(ux, uy)[0, 0]) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# CLI contract + repo discipline
# ---------------------------------------------------------------------------


def test_cli_help_runs_without_pythonpath() -> None:
    """``--help`` exits 0 in a subprocess with PYTHONPATH stripped (K-15 shape).

    argparse only reaches its exit-0 path after the module has fully imported,
    so this catches an import-time crash — the failure mode a pure unit test
    that already has the module loaded cannot see.
    """
    import os

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    for name in csq.CHECK_NAMES:
        assert name in proc.stdout


def test_cli_rejects_an_unknown_skip_name() -> None:
    """``--skip`` is a closed set, so a typo is caught at parse time."""
    with pytest.raises(SystemExit):
        csq.build_parser().parse_args(["clip.mp4", "--skip", "forwards"])


def test_subprocess_calls_are_list_form_without_shell() -> None:
    """Repo discipline, checked at the AST level rather than by convention.

    Every ``subprocess.run`` in the module must take a list literal and must
    never pass ``shell=True``, and no command may be assembled from an f-string
    — the shape that turns a filename with a space (or a quote) into a shell
    injection.
    """
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert calls, "expected at least one subprocess.run in check_source_quality.py"
    for call in calls:
        for kw in call.keywords:
            assert kw.arg != "shell", "shell=True is forbidden"
        assert call.args, "subprocess.run must be given an explicit command"
        command = call.args[0]
        assert isinstance(command, ast.Name | ast.List), f"command must be a list (or a list variable), got {command}"
        assert not isinstance(command, ast.JoinedStr), "command must never be an f-string"


def test_script_never_writes_next_to_the_source() -> None:
    """The tool is read-only on the input: no ``"w"`` opens, no ffmpeg output file.

    A health check that mutates the thing it is checking would be a nasty
    surprise on an operator's only copy of a 4k render.  The one write path in
    the module is the explicit ``--json PATH`` destination.
    """
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.count("write_text") == 1, "the only write should be the explicit --json destination"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            raise AssertionError("check_source_quality must not open files for writing")
