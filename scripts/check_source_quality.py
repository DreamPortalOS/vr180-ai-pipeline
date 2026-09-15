#!/usr/bin/env python3
"""Pre-pipeline source health check — six cheap tests that save a 40-minute run.

Why this exists
---------------
A full VR180 conversion is tens of minutes of GPU time.  Discovering *after* it
finishes that the source was never usable is the most expensive kind of waste
this repo has.  ``docs/RESEARCH_COVERAGE_V2.md`` §QA already lists the checks
the operator was doing by hand (``ffprobe`` for the frame size, eyeballing
whether the camera really flies forward); this script is that checklist turned
into a tool with an exit code.

The one that matters most is :func:`check_forward_motion`.  The whole VR180
route is built on the source *moving forward*: a locked-off camera has no
parallax to convert, so the stereo pass has nothing to work with and the result
is a flat picture pasted on a sphere.  Prompt text saying "camera flies forward"
is a wish, not a guarantee — text-to-video models routinely return a static shot
or a lateral pan instead.  This check measures it.

Stills, and why they are the cheapest gate of all
-------------------------------------------------
Hand an image (``.png``/``.jpg``/…) to this tool and it switches to **still
mode**: everything except ``forward_motion`` runs exactly as it does for a clip,
and ``forward_motion`` is reported as ``skipped`` — one frame carries no
evidence about camera movement, and calling that a pass or a fail would be a
lie either way.  The exit-code contract is unchanged.

This matters commercially, not just tidily.  A 4k/10s generation costs ≈¥50
measured; the operator's ¥100 top-up buys two.  A keyframe costs a fraction of
that, so **vetting the keyframe and refusing to animate a bad one** is the
single biggest saving available here.

The six checks
--------------
``square``
    Is the frame 1:1?  The preferred generation route is a 2880×2880 native
    square (§QA).  A non-square source is **not** fatal — the fisheye route
    (§QB) and 16:9 sources are legitimate — so this is a WARN, never a FAIL.

``forward_motion``
    Dense Farnebäck optical flow on ``--pairs`` adjacent frame pairs spread
    evenly over the clip.  Taking the frame centre as the origin, the *radial
    component* of the flow is ``dot(flow, r̂)``: positive means the picture is
    streaming outward, which is what flying forward looks like.  Two conditions
    must both hold:

    1. the mean radial component is **positive** (above ``--min-radial-rate``),
       and
    2. the outer ring (``r > 0.6R``) exceeds the inner disc (``r < 0.3R``) —
       forward motion scales the flow with radius, ``flow ≈ (s-1)·r``.

    Condition 2 is what makes the test more than a brightness heuristic, and
    condition 1 is what rejects a pan: a lateral move is positive radial on the
    leading half and equally negative on the trailing half, so it averages to
    zero over the frame while its raw flow magnitude is large.  Measured on the
    synthetic fixtures in ``tests/test_check_source_quality.py`` (240×240, 8
    pairs): radial outflow ``radial ≈ +3.6 px`` (outer 4.8 vs inner 1.4),
    static ``≈ -0.0002 px``, pan ``≈ -0.003 px``.  Three orders of magnitude of
    daylight between the good case and both bad ones.

``edges``
    An 8 px band along each of the four sides, checked for (a) a pure-black
    letterbox / pillarbox bar (mean < 8 **and** variance < 2 — dark *content*
    has texture, a bar does not) and (b) blown-out highlights (>30 % of the
    band above 250).  Both are named in §QA's prompt as things the source must
    not have: a bar becomes a hard black seam once warped onto the sphere, and
    a blown edge is a dead zone right where the viewer turns their head.

``decodable``
    ffprobe reads the file, the frame count agrees with ``duration × fps``, and
    the pixel format is printed.  ``pix_fmt`` is **reported, never judged** —
    10-bit HEVC was proven to run through the pipeline fine (#292/#298), so
    flagging it would be a false alarm.  For a still the frame-count/duration
    agreement is meaningless (ffmpeg invents a 25 fps single-frame "clip"), so
    that half is skipped and only the size/format is reported.

``detail_ratio``
    Laplacian edge energy of the central 1/3×1/3 rectangle divided by the same
    energy over the rest of the frame — "is the detail where the viewer is
    looking, or out at the rim?".  Not an aesthetic opinion: the competitor
    teardown (``_research/redraion/ANALYSIS.md`` §3) measured **1.60** across
    34 finished ride films and **1.58** across 613 official stills, two
    independent samples agreeing, with 88 % of frames above 1.  Their outer 6 %
    border carries only **0.77×** the detail of the rest of the picture.  That
    is a deliberate content-side trick and it is worth copying: the rim is
    exactly where VR180/fulldome geometry, outward extrapolation and feathering
    look worst, so a source that is *empty* out there hides our ugliest ring for
    free.  A busy rim advertises it.

``anchor``
    Is there a subject in the forward direction for the eye to hold onto?  The
    same teardown found one in **80 %** of frames, median area **11.3 %** of the
    picture, median eccentricity **0.236** (0 = dead centre), 62 % of them
    inside the central third.  Frequency-tuned saliency (Achanta) **gated by
    local texture** → Otsu → the **densest** salient region, weighted toward the
    centre when several compete, then its area and eccentricity are reported.

    Both of those emphasised words are there because plain Achanta picked the
    sky.  Colour distinctness alone is measured against the frame *mean*, which
    red-brown rock owns, so a bright sky is the most "distinct" thing in a canyon
    frame and a gray aircraft in the middle is one of the least; and scoring
    candidates by total saliency *mass* then hands the verdict to whichever
    region is merely biggest.  The texture gate removes the **flat** impostors
    (blown highlights, sheet water, clear sky); the enclosure gate removes the
    **scenery** (cloud and whitewater, which are textured, survive the first gate
    and run off the edge of the picture); density removes the merely big.
    Measured on the synthetic fixtures, frames with a subject come out at
    11.1–40.2 % of the picture and frames without one at ≤0.27 %, a separation of
    ~41× (it was ~30× before any gating).  A missing anchor is a WARN, not a
    FAIL: not every shot needs one, but the operator should know that without one
    the viewer will go looking at the rim.

Aggregation across the sampled frames/pairs is by **median**, not mean: one
scene cut, one fade-to-black frame or one lens flare must not decide the
verdict for a whole clip.

Usage
-----
::

    python scripts/check_source_quality.py video/src_720p_v2.mp4
    python scripts/check_source_quality.py keyframe.png              # still mode
    python scripts/check_source_quality.py clip.mp4 --pairs 12
    python scripts/check_source_quality.py clip.mp4 --skip forward_motion
    python scripts/check_source_quality.py clip.mp4 --json            # stdout
    python scripts/check_source_quality.py clip.mp4 --json out/q.json

Exit codes (so this can gate a run):

* ``0`` — everything passed, or only WARNs (a non-square source still runs)
* ``1`` — at least one FAIL, or the file could not be analysed at all

The input is opened **read-only**; nothing is ever written next to the source.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults (module constants so tests reference these, not literals)
# ---------------------------------------------------------------------------

#: The six checks, in report order.  ``--skip`` takes any of these names.
CHECK_NAMES: tuple[str, ...] = (
    "decodable",
    "square",
    "forward_motion",
    "edges",
    "detail_ratio",
    "anchor",
)

#: Extensions that put the tool into **still mode**: no adjacent frame pair
#: exists, so ``forward_motion`` is reported ``skipped`` and every other check
#: runs on the one frame.  Extension-driven rather than content-sniffed because
#: the operator's keyframes are named by the generator, and a wrong guess here
#: would silently drop the motion check on a real clip.
STILL_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

#: Adjacent frame pairs sampled evenly across the clip for the flow analysis.
DEFAULT_PAIRS: int = 8

#: Frames are downscaled to this width before Farnebäck.  Flow is a smooth,
#: low-frequency field; the extra pixels of a 2880² source buy nothing and cost
#: ~36× the time.  Every reported pixel figure is at *this* scale, which is why
#: the verdict is taken on the scale-free ``*_rate`` values instead.
DEFAULT_FLOW_WIDTH: int = 480

#: Aspect-ratio slack for the 1:1 check, as a fraction (0.02 = ±2 %).
DEFAULT_ASPECT_TOLERANCE: float = 0.02

#: Width in pixels of the border band sampled on each side.
DEFAULT_EDGE_BAND: int = 8

#: Minimum mean radial flow for "this really moves forward", expressed as a
#: fraction of the frame's half-diagonal ``R`` so it holds at any resolution.
#: For a forward push the flow is ``≈ (s-1)·r``, so this rate is roughly
#: ``0.6·(s-1)``: 0.002 corresponds to a ~0.3 % per-frame zoom — slower than
#: any usable flight, and ~50× above the measured noise floor of a static clip
#: (``|rate| ≲ 4e-5``).
DEFAULT_MIN_RADIAL_RATE: float = 0.002

#: A clip that clears :data:`DEFAULT_MIN_RADIAL_RATE` but stays under this
#: multiple of it passes with a WARN: it *is* moving forward, just slowly, so
#: the stereo effect will be subtle.  The band is deliberately narrow (2×, i.e.
#: 0.002–0.004) so it flags only the genuinely marginal.  Calibrated against the
#: two real clips in the repo: ``src_720p_v2.mp4`` measures 0.0056 and
#: ``googlegemini.mp4`` 0.0112 — both are ordinary forward flights and both must
#: come out as a clean PASS, not as a nag.
WEAK_RADIAL_FACTOR: float = 2.0

#: Ring boundaries as fractions of ``R``, per §QA ("散度随半径增大").
INNER_RADIUS_FRAC: float = 0.3
OUTER_RADIUS_FRAC: float = 0.6

#: Threshold multiplier for a detected circular (fisheye) source.  The
#: rectilinear rate in :data:`DEFAULT_MIN_RADIAL_RATE` is calibrated on
#: ``flow ≈ (s-1)·r`` — displacement grows linearly with radius.  Under the
#: **equidistant** projection a circular fisheye uses (``r_px ∝ θ``), the same
#: forward translation moves a point at polar angle θ at ``∝ sinθ·cosθ`` per
#: pixel — the ``cosθ`` term compresses the rim exactly where the outer ring
#: samples, and integrating over the imaging disc puts the mean radial rate at
#: ≈0.55–0.65× the rectilinear equivalent.  Two empirical anchors fixed the
#: exact value (#322): the lead-verified forward fisheye clip
#: ``gen_1x1_fisheye.mp4`` measures a **0.00185** rate (0.46× of the rectilinear
#: threshold — it must clear the gate, ideally clear the weak band too), while
#: the static noise floor stays at ``|rate| ≲ 4e-5`` (#310), so 0.4 keeps a
#: **20×** margin above noise.  The gate narrows for fisheye, it does not open.
FISHEYE_MIN_RADIAL_FACTOR: float = 0.4

#: A sub-threshold radial rate is called "slow forward" — not a lateral pan —
#: only when the radial component dominates the total flow, carrying at least
#: this share of it.  Measured separation (#322): the slow fisheye clip puts
#: **0.69** of its flow radially while the pan fixture manages **~0.07** — the
#: gate sits between them with headroom both ways, and it is what keeps the
#: 「推进太慢」advice from replacing the 「横移」diagnosis on sideways motion.
RADIAL_SHARE_FOR_SLOW: float = 0.4

#: A band this dark *and* this flat is a letterbox bar, not dark content.
#: Both conditions are needed: a night sky is dark but textured (variance in
#: the hundreds), while a bar is a constant.
LETTERBOX_MEAN_MAX: float = 8.0
LETTERBOX_VAR_MAX: float = 2.0

#: Blown-highlight rule: luminance above :data:`OVEREXPOSED_LEVEL` covering
#: more than :data:`OVEREXPOSED_FRAC_MAX` of a band.
OVEREXPOSED_LEVEL: int = 250
OVEREXPOSED_FRAC_MAX: float = 0.30

#: Frame-count/duration agreement tolerance for the ``decodable`` check.
FRAME_COUNT_TOLERANCE: float = 0.05

# --- Composition thresholds ------------------------------------------------
# Every number in this block is *measured*, not chosen: it comes from the
# teardown of Red Raion's 34 finished ride films (frame-by-frame) and 613
# official gallery stills in ``_research/redraion/ANALYSIS.md`` §3.  The two
# samples were analysed independently and agree, which is why these are treated
# as a reproducible house style rather than one studio's taste.

#: Longest side the ``detail_ratio`` analysis runs at.  The research pipeline
#: measured at this size; edge energy is a scale-dependent quantity, so
#: reproducing its numbers means reproducing its working resolution.
DETAIL_ANALYSIS_MAX_DIM: int = 720

#: Central detail / peripheral detail, above which the composition is doing what
#: the reference films do.  Measured there: **1.60** median across 34 trailers
#: (33 of the 34 above 1.0; p10 1.22) and **1.58** across 613 stills.  The gate
#: is set well below both so that only genuinely rim-heavy material trips it.
DETAIL_RATIO_PASS: float = 1.2

#: Below this the centre carries *less* detail than the rim — the opposite of
#: the reference style, and the arrangement that puts the viewer's eye straight
#: onto the worst ring of the projection.  Between the two constants is the
#: WARN band.
DETAIL_RATIO_FAIL: float = 1.0

#: Longest side the saliency analysis runs at.  Fixed so that an area fraction
#: means the same thing for a 1024² keyframe and a 2880² render, and small
#: enough that the morphology below has a resolution-independent footprint.
ANCHOR_ANALYSIS_MAX_DIM: int = 256

#: Morphological opening then closing, in pixels at
#: :data:`ANCHOR_ANALYSIS_MAX_DIM`.  The **opening is the load-bearing step**:
#: without it a flat or evenly-textured frame's scattered above-threshold
#: speckle gets welded into one frame-filling blob by the closing and reads as a
#: giant subject.  With it, such a frame yields no component above
#: :data:`ANCHOR_MIN_AREA` and is correctly reported as having no anchor.
ANCHOR_MORPH_KERNEL: int = 9

#: Centre prior used only to *choose* between competing salient regions (score =
#: mean of saliency × weight), never to measure them.  A sky band along the top
#: edge and a subject in the middle are both salient; the question this check
#: asks is whether something holds the eye in the *forward* direction, so the
#: middle one wins.  Area and eccentricity are then read off the unweighted
#: mask, which is what keeps an off-centre subject detectable and reportable as
#: off-centre.  Deliberately wide: at σ=0.5 a region on the frame diagonal still
#: keeps ~14 % of its weight, so this breaks ties, it does not veto.
ANCHOR_CENTER_PRIOR_SIGMA: float = 0.5

#: Side, in pixels at :data:`ANCHOR_ANALYSIS_MAX_DIM`, of the box the texture
#: channel averages Laplacian energy over.  ~12 % of the frame: small enough to
#: read a hand-sized subject as textured, large enough that a single hard edge
#: (a cloud rim, a horizon) does not paint a whole flat region as structured.
#:
#: **One of three constants that are a set** — with
#: :data:`ANCHOR_ENCLOSURE_PASSES` and :data:`ANCHOR_ENCLOSURE_REFERENCE`.  Move
#: one and the other two have to be re-measured; see the G-11 table under
#: :data:`ANCHOR_ENCLOSURE_PASSES` for why, and for the scan this value came
#: from.
#:
#: It was 21 from #343 (chosen over 15/31 on the pre-enclosure detector) until
#: G-11 re-scanned 21/25/31/35 against the #361 detector.  31 wins on every axis
#: that was measured and loses on none: the subject-vs-subjectless separation
#: goes from **40.6× to 64.1×** (the loudest subjectless fixture drops from
#: 0.27 % of the frame to 0.17 %), #360's cross-read spread on ``seed_v6.png``
#: falls from **12.2 % to 4.9 %**, and the real-asset centroids do not move —
#: ``seed_v6`` reads (0.535, 0.501) against 21's (0.539, 0.506), ``seed_1x1_drone``
#: (0.506, 0.441) against (0.506, 0.439).  A bigger window finds the *same*
#: subject and is simply quieter about everything that is not one.
#:
#: 35 was scanned too and is worse (44.8×), so this is a peak rather than a
#: "larger is better" gradient.
ANCHOR_TEXTURE_WINDOW: int = 31

#: **Detection floor — "is there anything there", not "is it big enough".**
#: The two questions are separate and this constant only answers the first one.
#: A component below it is speckle and the frame honestly has no anchor; a
#: component above it *exists*, and how well-sized it is then gets judged by
#: :data:`ANCHOR_AREA_MIN`/:data:`ANCHOR_AREA_MAX` (the 8–15 % quality band),
#: which this constant must never be confused with or tuned against.
#:
#: Set from the measured gap between "no subject" and "a real but small
#: subject".  Subjectless fixtures — uniform texture across 8 seeds, flat gray
#: plus sensor noise, rim-damped/empty-centre — top out at **0.27 %** (#341's
#: table quotes 0.35 % as the pre-#343 worst case; the texture channel pushed it
#: to 0.22 % and the #361 enclosure channel put it back to 0.27 %, which is the
#: price of a gate that does not care how bright a region is).  The smallest
#: *real* subject on record is the quadcopter in the owner's ``Gemini_v1.jpg``
#: keyframe at **1.30 %**.  0.8 % sits 2.9× above the loudest false positive and
#: 1.6× below that true one, so both sides keep room.
#:
#: It was 1.5 % until #345, which is what made the check answer 「画面里没有
#: 锚点」 for a frame with a visible drone in it.  That verdict was not merely
#: unhelpful but wrong, and wrong in an expensive direction: it tells the
#: operator to invent a subject from scratch when the actionable truth is that
#: the subject is there and needs to be **bigger** (1.3 % against a target of
#: 8–15 %).  #344 established that no saliency tuning reaches 1.5 % from below
#: without flooding the mask to 65–72 % of the frame, so the floor, not the
#: detector, was the thing that was wrong.
ANCHOR_MIN_AREA: float = 0.008

#: **Quality band — "is it big enough", asked only of a subject that already
#: cleared :data:`ANCHOR_MIN_AREA`.**  The reference band for a subject's share
#: of the picture: Red Raion's median is **11.3 %** (p10 3.3 %, p90 24.9 %), so
#: 8–15 % brackets the median without pretending the tails are wrong — outside
#: it is a WARN with the direction named, never a FAIL.  Untouched by #345:
#: lowering the detection floor changes *which frames get judged*, never the
#: standard they are judged against.
ANCHOR_AREA_MIN: float = 0.08
ANCHOR_AREA_MAX: float = 0.15

#: Eccentricity of the subject's centroid, as a fraction of the half-diagonal
#: (0 = dead centre, 1 = corner).  Red Raion's median is **0.236** and 62 % of
#: their frames put the subject inside the central third.
ANCHOR_OFFSET_MAX: float = 0.30

#: Share of sampled frames that must contain a subject before a clip counts as
#: "anchored".  Red Raion manage **80 %**; a simple majority is a deliberately
#: forgiving gate for a check that only ever WARNs.
ANCHOR_DETECTED_FRAC_MIN: float = 0.5

#: Barrier distance, in CIE-Lab units at :data:`ANCHOR_ANALYSIS_MAX_DIM`, at
#: which :func:`enclosure_gate` saturates — the step you have to climb to get
#: *inside* something that counts as fully enclosed.
#:
#: Absolute rather than self-scaling, and that is the opposite of the choice
#: :func:`texture_gate` makes, for a reason.  A frame's own barrier distribution
#: is not a usable yardstick here: measured at 256², the evenly-textured
#: subjectless fixtures run a *higher* barrier (median 66–72 Lab units) than the
#: owner's canyon stills (29–32), because full-range blob texture is, pixel for
#: pixel, extremely enclosing.  Referencing a percentile of the frame's own
#: distribution therefore promotes exactly the frames that must stay
#: subjectless.  An absolute reference leaves them alone — everything in them
#: sits near the saturation point, the gate reads ≈1 throughout, and the
#: #341 separation is decided by distinctness and texture exactly as before.
#:
#: Set from the measured gap on the real material (mean barrier inside each
#: region, in Lab units, 3 raster passes):
#:
#: * ``seed_v6`` sky                0.7   ← must lose
#: * ``seed_v6`` right wall        10.3   ← must lose
#: * ``seed_v6`` whitewater        41.3   ← must lose
#: * ``seed_1x1_drone`` whitewater 43.6   ← must lose
#: * **reference                   80.0**
#: * ``seed_1x1_drone`` airframe   73.0   ← must win (0.91 of reference)
#: * ``seed_v6`` airframe          84.1   ← must win (saturated)
#:
#: **One of three constants that are a set** — with
#: :data:`ANCHOR_ENCLOSURE_PASSES` and :data:`ANCHOR_TEXTURE_WINDOW`.  Move one
#: and the other two have to be re-measured; the pass count in particular is
#: what the barrier numbers above are denominated in, so a reference quoted
#: without one is meaningless.  See :data:`ANCHOR_ENCLOSURE_PASSES`.
#:
#: The plateau is wide and G-11 re-measured it at the new texture window: over
#: 70–90 every real-asset verdict holds still (``seed_v6`` y within
#: 0.498–0.508, ``seed_1x1_drone`` y within 0.434–0.453, the clip anchored in
#: every frame from 70 to 82).  The #341 separation is the one number that is
#: not flat across it — 64.1× at 78 and 80, 45.5× at 82 — because it is decided
#: by whichever synthetic fixture happens to be loudest, so 80 is kept at the
#: centre of the quiet stretch rather than pushed to an edge.
#: :data:`ANCHOR_ENCLOSURE_EXPONENT` 1.9–2.1 is likewise unmoved.
ANCHOR_ENCLOSURE_REFERENCE: float = 80.0

#: Exponent applied to the enclosure ratio, i.e. how sharply a half-enclosed
#: region is discounted against a fully enclosed one.
#:
#: At 1.0 the whitewater keeps half the aircraft's weight and the two merge into
#: one component; at 2.0 it keeps a quarter and they separate.  Above ~2.5 the
#: airframe itself starts to fragment.  Measured across 78–82 × 1.9–2.1 the
#: answer does not move.
ANCHOR_ENCLOSURE_EXPONENT: float = 2.0

#: Exponent applied to :func:`frequency_tuned_saliency` before the gates.
#:
#: Sub-linear on purpose.  Colour distinctness is evidence of a subject but a
#: badly *scaled* one: measured on ``video/seed_v6.png``, whitewater scores 0.68
#: against the aircraft's 0.30, so at exponent 1.0 the foam still carries 2.3×
#: the aircraft's weight and no achievable enclosure gate closes that.  At 0.7
#: the ratio falls to 1.7× and the enclosure gate's 4× turns the order over.
#: Read it as "twice as distinct is not twice as much of a subject".
ANCHOR_DISTINCTNESS_EXPONENT: float = 0.7

#: Raster sweeps used to approximate the barrier distance.  Each sweep relaxes
#: the whole frame in four directions.
#:
#: Three is a deliberate *under*-relaxation, not convergence: the map is still
#: falling at 25 passes (``seed_v6``'s airframe reads 84.1 / 58.7 / 55.4 / 52.8
#: Lab units at 3 / 6 / 12 / 25).  What matters is that the ordering it produces
#: is already stable — foam below wall below airframe at every pass count — and
#: that three sweeps cost ~0.1 s a frame instead of ~1 s.
#:
#: **These three constants are a set** — this one,
#: :data:`ANCHOR_ENCLOSURE_REFERENCE` and :data:`ANCHOR_TEXTURE_WINDOW` — and
#: changing any one of them obliges you to re-measure the other two.  The
#: pass count and the reference are paired because the reference is an
#: *absolute* Lab distance and the pass count is what the barrier map's absolute
#: scale depends on (halving as above).  The texture window joins the set because
#: the two gates multiply: a wider window removes exactly the finely-textured
#: speckle that the enclosure gate then has to judge, so the reference's
#: operating point moves with it.
#:
#: G-11 (#364) scanned window × passes on the three owner assets, with the
#: reference re-derived per pass count from the airframe ratio above
#: (80 → 56 → 53).  Separation is the #341 subject-vs-subjectless margin;
#: cross-read is #360's spread over ``seed_v6.png`` read at 2048/1024/960/512²
#: in both colour and luminance; the contract is ≤30 %:
#:
#: ====== ====== ========== ============== ================ ==========
#: window passes separation ``seed_v6`` xy ``…_drone`` xy   cross-read
#: ====== ====== ========== ============== ================ ==========
#: 21     3       40.6×     (0.539, 0.506) (0.506, 0.439)      12.2 %
#: 25     3       38.0×     (0.536, 0.503) (0.505, 0.439)       6.0 %
#: **31** **3**  **64.1×**  (0.535, 0.501) (0.506, 0.441)     **4.9 %**
#: 35     3       44.8×     (0.535, 0.501) (0.506, 0.441)       4.3 %
#: 21     6       58.9×     (0.574, 0.574) (0.509, 0.448)      15.6 %
#: 25     6       64.8×     (0.572, 0.569) (0.531, 0.547)      20.9 %
#: 31     6       64.1×     (0.563, 0.547) (0.532, 0.549)      44.2 %
#: 35     6       48.6×     (0.561, 0.542) (0.532, 0.550)      53.9 %
#: 21     12      58.9×     (0.575, 0.573) (0.662, 0.225)      22.7 %
#: 25     12      58.9×     (0.568, 0.557) (0.662, 0.225)      43.4 %
#: 31     12      58.8×     (0.561, 0.543) (0.662, 0.225)      30.3 %
#: 35     12      58.8×     (0.559, 0.540) (0.662, 0.225)      28.3 %
#: ====== ====== ========== ============== ================ ==========
#:
#: The result is the opposite of what the card expected: **more relaxation makes
#: the detector worse**, and three passes survive on merit rather than on cost.
#: Above three, #360's cross-read contract starts failing outright (44 % and
#: 54 % at six passes), and at twelve the ``seed_1x1_drone`` centroid leaves the
#: airframe for the canyon rim at (0.662, 0.225) at *every* window.  A further
#: reference sweep (41–70 in steps of 4, at windows 21/25/31) confirms this is
#: not a mis-chosen reference: at six and twelve passes the canyon centroid flips
#: between the airframe and the rim on a 4-unit reference step and the cross-read
#: spread swings from 1.3 % to 864 %.  A converged barrier map is a *flatter*
#: one — the relaxation only ever lowers distances — so it compresses the very
#: gap between airframe and foam the gate is reading, and the verdict starts
#: turning on noise.  At three passes the same sweep (70–90) leaves every real
#: centroid put.
ANCHOR_ENCLOSURE_PASSES: int = 3

#: Smallest region — as a fraction of the detection floor — that counts as
#: **credible evidence of a subject at all**.  One constant, two uses, and they
#: are the same question asked twice.
#:
#: *Rescue gate (#348).*  The opening that deletes noise specks also erodes a
#: real subject's thin extremities (on the 2048² drone keyframe a 1.38 %-of-frame
#: subject is eaten down to a 0.53 % core, under the floor).  The rescue re-grows
#: such a core to its full pre-morphology extent — but the amplification is
#: largest exactly where the evidence is weakest: on the subjectless fixtures,
#: cores of 0.06–0.22 % restore to 0.3–1.0 %, so rescuing every sub-floor core
#: would fabricate anchors out of specks.
#:
#: *Candidacy gate (#356).*  The same threshold decides who may **enter** the
#: density election in :func:`anchor_stats`.  A region below it can never be
#: reported as an anchor — it is under the detection floor and too small to be
#: rescued — so letting it win the election has exactly one possible effect:
#: vetoing a real subject that was standing right next to it.
#:
#: Re-measured for #356, and the re-measurement is a consequence of the
#: candidacy gate rather than a re-tuning.  #348 set this at 0.4× from a table
#: of noise cores topping out at 0.22 % — but that table could only contain
#: cores that had actually *won* an election, and under the old speck-biased
#: scoring the loudest ones never did.  With the gate in place the empty-centre
#: fixture's 0.358 % rim blob wins its frame, and at 0.4× it was rescued to
#: 0.97 % — a subject conjured out of a textured rim.  The population that
#: matters is "cores that can win", so the floor is set from that one:
#:
#: * loudest subjectless core   0.358 %  (empty centre, busy rim)
#: * floor                      0.44 %   → 1.23× above it
#: * smallest real core         0.531 %  (2048² drone) → 1.21× above the floor
#:
#: Deliberately symmetric: there is no evidence justifying a floor nearer one
#: side than the other, and both neighbours are measured, not assumed.
ANCHOR_RESCUE_CORE_FLOOR: float = 0.55

#: Seconds allowed for one ffmpeg/ffprobe call.
FFMPEG_TIMEOUT: float = 300.0

#: Farnebäck parameters.  Stock OpenCV tutorial values; the three synthetic
#: fixtures separate by three orders of magnitude under them, so there is
#: nothing to tune and no hidden fragility to inherit.
FARNEBACK_KWARGS: dict[str, Any] = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skipped"

_ICON = {STATUS_PASS: "✅", STATUS_WARN: "⚠️", STATUS_FAIL: "❌", STATUS_SKIP: "⏭️"}

#: Edge order and their Chinese labels — the report has to name *which* side is
#: bad, otherwise the operator cannot act on it.
EDGE_LABELS: dict[str, str] = {"top": "上边", "bottom": "下边", "left": "左边", "right": "右边"}

#: Printed verbatim whenever ``forward_motion`` fails.  It always names 静态
#: (locked-off) *and* 横移 (pan) regardless of which one was detected, because
#: the remedy is identical and the operator must not have to decode the
#: sub-classification to know what to do next.
FORWARD_FAIL_ADVICE = (
    "疑似静态机位／横移（不是真前进）。这类素材没有前后视差可转，做出来会是"
    "「贴在球面上的平面画」，整条管线（几十分钟）等于白跑。"
    "请按 1:1 生成卡的运镜要求重生成：镜头持续向前推进、匀速、机位高度不变、"
    "不摇不摆不旋转，画面元素从中心持续向四周散开。"
)

#: Still mode: a single frame carries no evidence either way about camera
#: movement, so the honest report is ``skipped`` — not a pass (which would let a
#: locked-off clip through on a keyframe's say-so) and not a fail (which would
#: block every keyframe there is).
STILL_FORWARD_DETAIL = "静帧模式：只有一帧，无从判断镜头是否向前推进（既不算过也不算不过）"
STILL_FORWARD_ADVICE = (
    "关键帧先把其余各项过掉再花钱生成视频；视频出来之后，对视频本体再跑一次完整体检，"
    "由 forward_motion 确认是不是真前进。"
)

#: Appended when the clip does move forward but only barely.
FORWARD_WEAK_ADVICE = (
    "径向外流为正但偏弱：前进速度慢、或大部分画面是远景（远处视差本来就小）。"
    "可以进管线，但立体感会偏弱；想要更强的沉浸感就把运镜提速或加近景元素。"
)

#: Printed when the radial component IS positive but cannot clear even the
#: (projection-adapted) threshold — a FAIL, not the WARN above.  #322: the old
#: text claimed 「径向分量不为正」 while the same line reported a positive
#: number, sending the operator hunting for a pan that was not there.  This is
#: a different failure from a lateral move and gets different advice.
FORWARD_SLOW_ADVICE = (
    "径向确实外流，只是强度低于可转立体感的下限：推进太慢，或画面以远景为主"
    "（远景视差天然小）。这与横移不同——方向对了、量不够。"
    "请把运镜提速（每帧推进更大）或加入近景元素后重新生成。"
)

#: Printed whenever the frame is flatter in the middle than at the rim.  The
#: fix is a prompt change, not a pipeline change, which is why the wording is
#: phrased as generation guidance.
DETAIL_RATIO_ADVICE = (
    "边缘太满，VR 里最烂的一圈会被看见：VR180／球幕的画面外圈正是几何最假、外扩最勉强、"
    "羽化最明显的地方，边缘越有细节，观众越容易发现它。Red Raion 的 34 条成片与 613 张剧照"
    "都把中央做密（中央/周边细节比 1.58–1.60）、把外 6% 边框做空（只有其余画面的 0.77 倍）、"
    "四角压暗到全画面均值的 0.55。重生成时在提示词里写明：细节集中在画面中央 1/3，"
    "四角自然压暗，边缘只留低对比背景。"
)

#: Printed when no subject could be found at all.  Named 「没有锚点」 verbatim
#: because that is the phrase the operator has to act on.
ANCHOR_MISSING_ADVICE = (
    "正前方没有锚点，观众会盯着边缘瑕疵看。Red Raion 的成片里 80% 的帧都有一个独立主体"
    "（面积中位 11.3%、偏心 0.236，62% 落在中央 1/3 区），三种可直接抄的形态："
    "①载具/角色跟拍 ②轨道/通道引导 ③中央光源当消失点。"
    "重生成时在提示词里指定一个明确的中央主体，并在故事板上把它写成可核对的产出物。"
)

#: Printed when a subject exists but sits outside the reference band.
ANCHOR_OFF_SPEC_ADVICE = (
    "有主体但不在竞品的构图区间内（面积 8–15%、偏心 ≤0.3，实测中位 11.3% / 0.236）。"
    "这不致命——不是每个镜头都要标准锚点——但主体太小抓不住视线、太大会糊住整个画幅、"
    "太偏则观众的视线被带向边缘那圈瑕疵。"
    "注意：主体是找到了的，要改的是它的大小/位置，不是从头加一个主体——"
    "偏小就把它拉近或放大（提示词写明主体占画面 1/3 左右、镜头更贴近），"
    "偏大就退远一点，偏心就把它挪回中央 1/3 区，然后重生成再体检一次。"
)

#: Non-square is legitimate (fisheye / 16:9 routes), so this is guidance, not
#: an error.
NON_SQUARE_ADVICE = (
    "非 1:1 源不致命：鱼眼路线与 16:9 源都走得通，但首选的 1:1 宽视角路线"
    "（2880×2880）要求方图，非方图会被生成端居中裁剪。"
    "若本来就想走 1:1，请重生成为方图；若走鱼眼/16:9 路线，忽略本项。"
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One row of the report.

    ``measured`` carries the raw numbers the verdict was taken on, so a JSON
    consumer never has to re-derive them from prose and a disagreement with the
    verdict is inspectable rather than mysterious.
    """

    name: str
    status: str
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)
    advice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "measured": self.measured,
            "advice": self.advice,
        }


@dataclass
class SourceReport:
    """Everything the CLI prints or serialises."""

    source: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.status == STATUS_FAIL for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == STATUS_WARN for c in self.checks)

    @property
    def exit_code(self) -> int:
        """``1`` if anything FAILed, else ``0`` — WARNs alone never gate."""
        return 1 if self.failed else 0

    @property
    def summary(self) -> dict[str, Any]:
        counts = {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 0, STATUS_SKIP: 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        overall = STATUS_FAIL if self.failed else (STATUS_WARN if self.warned else STATUS_PASS)
        return {**counts, "overall": overall}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "exit_code": self.exit_code,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# 1. decodable — ffprobe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeInfo:
    """The subset of ``ffprobe`` output the checks actually use."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    nb_frames: int = 0
    pix_fmt: str = ""
    codec: str = ""

    @property
    def expected_frames(self) -> float:
        return self.duration * self.fps


def _parse_fps(stream: dict) -> float:
    """``avg_frame_rate`` as a float, tolerating ffprobe's ``"0/0"``."""
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        frac = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(frac) if frac.denominator else 0.0


def probe_source(path: str | Path, ffprobe: str = "ffprobe") -> ProbeInfo:
    """Read stream/format metadata for the first video stream.

    Raises :class:`RuntimeError` when ffprobe fails or the file carries no
    video stream — both are "cannot analyse this at all", which the caller
    surfaces as a FAIL rather than a crash.
    """
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt,codec_name,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise RuntimeError(f"ffprobe failed on {path}: {tail}")
    try:
        info = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned unparseable JSON for {path}: {exc}") from exc
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or (info.get("format") or {}).get("duration") or 0.0)
    try:
        nb_frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nb_frames = 0
    return ProbeInfo(
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=_parse_fps(stream),
        duration=duration,
        nb_frames=nb_frames,
        pix_fmt=str(stream.get("pix_fmt") or ""),
        codec=str(stream.get("codec_name") or ""),
    )


def check_decodable(
    info: ProbeInfo,
    tolerance: float = FRAME_COUNT_TOLERANCE,
    still: bool = False,
) -> CheckResult:
    """ffprobe read it, and the frame count agrees with ``duration × fps``.

    ``pix_fmt`` is reported and never judged: 10-bit HEVC has been proven to
    run through the pipeline (#292/#298), so treating it as suspicious would be
    a false alarm.  A frame-count mismatch is a WARN, not a FAIL — the file is
    still decodable, but broken timestamps make per-frame stage accounting lie.

    ``still=True`` stops after the size/format report.  ffmpeg hands a single
    image back as a 25 fps, 0.04 s "clip" with no ``nb_frames``; running the
    duration agreement against those invented numbers would produce a warning
    about nothing.  The video path below is untouched by this branch.
    """
    measured: dict[str, Any] = {
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 4),
        "duration_s": round(info.duration, 4),
        "nb_frames": info.nb_frames,
        "expected_frames": round(info.expected_frames, 2),
        "pix_fmt": info.pix_fmt,
        "codec": info.codec,
    }
    base = (
        f"{info.width}×{info.height} {info.codec or '?'} {info.pix_fmt or '?'} {info.fps:.4g} fps, {info.duration:.2f}s"
    )
    if info.width <= 0 or info.height <= 0:
        return CheckResult(
            "decodable",
            STATUS_FAIL,
            f"ffprobe 读不到有效画面尺寸（{info.width}×{info.height}）",
            measured,
            "文件可能损坏或不是视频，换一份素材。",
        )
    if still:
        measured["still"] = True
        return CheckResult(
            "decodable",
            STATUS_PASS,
            f"静帧 {info.width}×{info.height} {info.codec or '?'} {info.pix_fmt or '?'}"
            f"（单帧素材：不做帧数/时长自洽核对，pix_fmt 仅报告不判定）",
            measured,
        )
    if info.duration <= 0.0 or info.fps <= 0.0:
        return CheckResult(
            "decodable",
            STATUS_WARN,
            f"{base}；时长或帧率缺失，帧数自洽性无法核对",
            measured,
            "容器元数据不全（常见于流式/裁剪产物）。用 ffmpeg -c copy 重封装一次即可补齐。",
        )
    if info.nb_frames <= 0:
        return CheckResult(
            "decodable",
            STATUS_PASS,
            f"{base}；容器未记录 nb_frames，跳过帧数自洽核对（pix_fmt 仅报告不判定）",
            measured,
        )
    expected = info.expected_frames
    drift = abs(info.nb_frames - expected) / max(expected, 1.0)
    measured["frame_count_drift"] = round(drift, 4)
    if drift > tolerance:
        return CheckResult(
            "decodable",
            STATUS_WARN,
            f"{base}；帧数 {info.nb_frames} 与 时长×帧率 {expected:.1f} 相差 {drift * 100:.1f}%",
            measured,
            "时间戳与帧数对不上（VFR 或截断）。用 ffmpeg -vsync cfr 重编一次再进管线，否则分段/进度会算错。",
        )
    return CheckResult(
        "decodable",
        STATUS_PASS,
        f"{base}，{info.nb_frames} 帧（与时长自洽，pix_fmt 仅报告不判定）",
        measured,
    )


# ---------------------------------------------------------------------------
# 2. square — aspect ratio
# ---------------------------------------------------------------------------


def check_square(width: int, height: int, tolerance: float = DEFAULT_ASPECT_TOLERANCE) -> CheckResult:
    """1:1 within ``tolerance``.  Non-square WARNs; it never blocks a run."""
    if width <= 0 or height <= 0:
        return CheckResult(
            "square",
            STATUS_FAIL,
            f"画面尺寸无效：{width}×{height}",
            {"width": width, "height": height},
            "尺寸读不出来，先解决 decodable。",
        )
    ratio = width / height
    deviation = abs(ratio - 1.0)
    measured = {
        "width": width,
        "height": height,
        "aspect_ratio": round(ratio, 6),
        "deviation": round(deviation, 6),
        "tolerance": tolerance,
    }
    if deviation <= tolerance:
        return CheckResult(
            "square",
            STATUS_PASS,
            f"{width}×{height}，宽高比 {ratio:.4f}（1:1，偏差 {deviation * 100:.2f}% ≤ {tolerance * 100:g}%）",
            measured,
        )
    return CheckResult(
        "square",
        STATUS_WARN,
        f"{width}×{height}，宽高比 {ratio:.4f}，偏离 1:1 达 {deviation * 100:.1f}%（容差 {tolerance * 100:g}%）",
        measured,
        NON_SQUARE_ADVICE,
    )


# ---------------------------------------------------------------------------
# 3. forward_motion — radial divergence of the optical flow
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _radial_basis(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(ûx, ûy, r, R)`` for a frame of this size, centred on the frame centre.

    ``R`` is the half-diagonal, so ``r/R`` spans ``[0, 1]`` over the whole
    frame and every pixel — corners included — lands inside some ring.
    Cached because it depends only on the shape and is rebuilt for every pair
    otherwise.  The returned arrays are treated as read-only.
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    r = np.hypot(dx, dy)
    inv = 1.0 / np.maximum(r, 1e-6)
    return dx * inv, dy * inv, r, 0.5 * math.hypot(width, height)


@dataclass(frozen=True)
class RadialStats:
    """Radial flow summary for one adjacent frame pair, in analysis pixels."""

    radial_mean: float
    inner_mean: float
    outer_mean: float
    flow_mean: float
    radius: float

    def _rate(self, value: float) -> float:
        return value / self.radius if self.radius > 0.0 else 0.0

    @property
    def radial_rate(self) -> float:
        """Mean radial flow as a fraction of ``R`` — resolution independent."""
        return self._rate(self.radial_mean)

    @property
    def inner_rate(self) -> float:
        return self._rate(self.inner_mean)

    @property
    def outer_rate(self) -> float:
        return self._rate(self.outer_mean)

    @property
    def flow_rate(self) -> float:
        return self._rate(self.flow_mean)


def estimate_imaging_circle(frame: np.ndarray) -> float | None:
    """Radius of the imaging circle (fraction of the half-diagonal), or None.

    A circular fisheye source paints pure black outside its inscribed imaging
    circle — the four frame corners are black *and flat* (compression noise
    aside), while real dark content (night sky, shadow) is dark *and textured*.
    #322: those black corners are zero-flow pixels that dragged the outer-ring
    statistics toward zero and failed genuinely-forward fisheye clips.

    The estimate is a radial profile of the dark-pixel fraction: inside the
    circle it is whatever the content is (≈0 after binning), outside it is ≈1.
    The circle radius is where the profile crosses ½, validated by requiring a
    genuine step (dark inside the transition band ≲25 %, dark outside ≳75 %) —
    patchy darkness does not produce a step and returns ``None``, so a
    rectilinear source is never masked.  The accepted range ``[0.45, 0.85]``
    brackets the inscribed circle of a square frame (``1/√2 ≈ 0.707``) with
    room for slight over/underscan.
    """
    gray = _as_gray(frame)
    small = _fit_for_analysis(gray, 256)
    height, width = small.shape
    _, _, r, radius = _radial_basis(height, width)
    r_norm = (r / radius).ravel()
    dark = (small < 16).ravel()

    corner = max(4, int(0.05 * min(height, width)))
    patches = [
        small[:corner, :corner],
        small[:corner, -corner:],
        small[-corner:, :corner],
        small[-corner:, -corner:],
    ]
    corners_black = all(p.mean() < 16.0 and float(p.std()) < 6.0 for p in patches)
    centre = small[
        height // 2 - corner : height // 2 + corner,
        width // 2 - corner : width // 2 + corner,
    ]
    if not corners_black or centre.mean() < 24.0:
        return None

    n_bins = 32
    bin_idx = np.clip((r_norm * n_bins).astype(int), 0, n_bins - 1)
    frac = np.bincount(bin_idx, weights=dark, minlength=n_bins) / np.maximum(np.bincount(bin_idx, minlength=n_bins), 1)
    crossing = np.argmax(frac >= 0.5) if (frac >= 0.5).any() else -1
    if crossing <= 0:
        return None
    # The crossing bin *and* its inner neighbour straddle the circle edge
    # (part content, part black), so both are excluded from the validation
    # bands — only bins fully inside / fully outside speak to the step.
    inside_band = frac[max(0, crossing - 4) : max(0, crossing - 1)]
    outside_band = frac[crossing + 1 : min(n_bins, crossing + 4)]
    circle = (crossing + 0.5) / n_bins
    if inside_band.size == 0 or outside_band.size == 0:
        return None
    if inside_band.max() > 0.25 or outside_band.min() < 0.75:
        return None
    if not 0.45 <= circle <= 0.85:
        return None
    return float(circle)


def radial_flow_stats(
    prev: np.ndarray,
    nxt: np.ndarray,
    inner_frac: float = INNER_RADIUS_FRAC,
    outer_frac: float = OUTER_RADIUS_FRAC,
    circle_frac: float | None = None,
) -> RadialStats:
    """Dense flow ``prev → nxt``, projected onto the radial direction.

    The sign convention is the one §QA states: **positive means outward**, i.e.
    the picture streaming away from the centre, which is what flying forward
    looks like.  A pan projects to ``+|v|`` ahead of the centre and ``-|v|``
    behind it, so it cancels to ~0 here while ``flow_mean`` stays large — that
    asymmetry between the two numbers is what separates the two failure modes.

    ``circle_frac`` masks the statistics to inside a detected imaging circle
    (:func:`estimate_imaging_circle`); ``None`` (the default, and every
    rectilinear source) samples the whole frame unchanged (#322: the black
    corners of a circular fisheye are zero-flow pixels that poisoned the
    outer-ring mean).
    """
    if prev.shape != nxt.shape or prev.ndim != 2:
        raise ValueError(f"expected two same-sized grayscale frames, got {prev.shape} and {nxt.shape}")
    a = prev if prev.dtype == np.uint8 else prev.astype(np.uint8)
    b = nxt if nxt.dtype == np.uint8 else nxt.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK_KWARGS)
    height, width = a.shape
    ux, uy, r, radius = _radial_basis(height, width)
    radial = flow[..., 0] * ux + flow[..., 1] * uy
    valid = r <= circle_frac * radius if circle_frac is not None else np.ones(r.shape, dtype=bool)
    inner = valid & (r < inner_frac * radius)
    outer = valid & (r > outer_frac * radius)
    return RadialStats(
        radial_mean=float(radial[valid].mean()),
        inner_mean=float(radial[inner].mean()) if inner.any() else 0.0,
        outer_mean=float(radial[outer].mean()) if outer.any() else 0.0,
        flow_mean=float(np.hypot(flow[..., 0], flow[..., 1])[valid].mean()),
        radius=radius,
    )


def check_forward_motion(
    stats: Sequence[RadialStats],
    min_radial_rate: float = DEFAULT_MIN_RADIAL_RATE,
    circle_frac: float | None = None,
) -> CheckResult:
    """Verdict over the sampled pairs: does this clip really move forward?

    Aggregation is the **median** across pairs — a single scene cut or a flash
    frame produces a wild flow field, and a mean would let it decide the whole
    clip's fate.

    Both of §QA's conditions must hold: the radial component is positive
    (beyond ``min_radial_rate``) *and* it grows with radius (outer ring above
    inner disc).  When it fails, the detail names which failure mode it looks
    like — locked-off, inward, too-slow, lateral, or "outward but not
    radius-scaled" — with distinct advice where the operator's next step
    differs.  #322: "positive" here means *clears the threshold*; the failure
    text must never claim the sign is wrong when the numbers printed beside it
    are positive.

    ``circle_frac`` records a detected imaging circle in ``measured`` and adds
    it to the detail line so the projection adaptation is visible, not silent.
    """
    if not stats:
        return CheckResult(
            "forward_motion",
            STATUS_FAIL,
            "没有取到任何可用的相邻帧对（片子太短或解码失败）",
            {"pairs": 0},
            "先确认文件能正常解码（见 decodable），或把 --pairs 调小。",
        )

    radial_rate = float(np.median([s.radial_rate for s in stats]))
    inner_rate = float(np.median([s.inner_rate for s in stats]))
    outer_rate = float(np.median([s.outer_rate for s in stats]))
    flow_rate = float(np.median([s.flow_rate for s in stats]))
    radius = float(np.median([s.radius for s in stats]))
    positive = radial_rate >= min_radial_rate
    grows = outer_rate > inner_rate

    measured: dict[str, Any] = {
        "pairs": len(stats),
        "analysis_radius_px": round(radius, 2),
        "radial_rate": round(radial_rate, 6),
        "inner_rate": round(inner_rate, 6),
        "outer_rate": round(outer_rate, 6),
        "flow_rate": round(flow_rate, 6),
        "radial_mean_px": round(float(np.median([s.radial_mean for s in stats])), 4),
        "inner_mean_px": round(float(np.median([s.inner_mean for s in stats])), 4),
        "outer_mean_px": round(float(np.median([s.outer_mean for s in stats])), 4),
        "flow_mean_px": round(float(np.median([s.flow_mean for s in stats])), 4),
        "min_radial_rate": min_radial_rate,
        "weak_radial_rate": round(min_radial_rate * WEAK_RADIAL_FACTOR, 6),
        "radial_positive": positive,
        "outer_exceeds_inner": grows,
    }
    if circle_frac is not None:
        measured["imaging_circle_frac"] = round(circle_frac, 3)
    numbers = (
        f"径向 {measured['radial_mean_px']:+.3f}px（内圈 {measured['inner_mean_px']:+.3f} → "
        f"外圈 {measured['outer_mean_px']:+.3f}），总光流 {measured['flow_mean_px']:.3f}px，"
        f"{len(stats)} 对相邻帧中位数"
    )
    if circle_frac is not None:
        numbers += f"（圆形鱼眼：成像圆 {circle_frac:.2f}R，已排除圆外黑区，阈值×{FISHEYE_MIN_RADIAL_FACTOR:g}）"

    if positive and grows:
        weak = radial_rate < min_radial_rate * WEAK_RADIAL_FACTOR
        return CheckResult(
            "forward_motion",
            STATUS_WARN if weak else STATUS_PASS,
            f"真前进：径向散度为正且随半径递增。{numbers}",
            measured,
            FORWARD_WEAK_ADVICE if weak else "",
        )

    # Classify by whether the radial component *dominates* the flow field
    # first, then by its sign (#322): a pan's radial projection cancels to a
    # whisper that can land on either side of zero, and only a radial-
    # dominated field says anything about forward vs backward.
    radial_dominant = abs(radial_rate) >= RADIAL_SHARE_FOR_SLOW * flow_rate
    if flow_rate < min_radial_rate:
        reason = "疑似静态机位：整帧光流几乎为零，画面基本不动"
        advice = FORWARD_FAIL_ADVICE
    elif not radial_dominant:
        reason = "径向分量不显著：运动几乎不向外，疑似横移／摇镜（pan），不是向前推进"
        advice = FORWARD_FAIL_ADVICE
    elif radial_rate < 0.0:
        reason = "径向分量为负：画面向中心收缩，这是后撤／拉远，不是前进"
        advice = FORWARD_FAIL_ADVICE
    elif not positive:
        reason = (
            "径向分量为正但强度低于阈值：确有外流，只是推进太慢"
            "（或画面以远景为主），未达到可转立体感的下限——这与横移不同，方向是对的、量不够"
        )
        advice = FORWARD_SLOW_ADVICE
    else:
        reason = "径向分量为正但不随半径递增：外圈没有比内圈流得更快，不符合前进的几何"
        advice = FORWARD_FAIL_ADVICE
    return CheckResult("forward_motion", STATUS_FAIL, f"{reason}。{numbers}", measured, advice)


# ---------------------------------------------------------------------------
# 4. edges — letterbox bars and blown highlights
# ---------------------------------------------------------------------------


def edge_band_stats(frame: np.ndarray, band: int = DEFAULT_EDGE_BAND) -> dict[str, dict[str, float]]:
    """Mean / variance / blown-pixel fraction for each side's border band."""
    if frame.ndim != 2:
        raise ValueError(f"expected a grayscale frame, got shape {frame.shape}")
    height, width = frame.shape
    if band < 1:
        raise ValueError(f"band must be >= 1, got {band}")
    band_h = min(band, height)
    band_w = min(band, width)
    bands = {
        "top": frame[:band_h, :],
        "bottom": frame[height - band_h :, :],
        "left": frame[:, :band_w],
        "right": frame[:, width - band_w :],
    }
    out: dict[str, dict[str, float]] = {}
    for name, region in bands.items():
        values = region.astype(np.float64)
        out[name] = {
            "mean": float(values.mean()),
            "var": float(values.var()),
            "bright_frac": float((values > OVEREXPOSED_LEVEL).mean()),
        }
    return out


def check_edges(
    per_frame: Sequence[dict[str, dict[str, float]]],
    mean_max: float = LETTERBOX_MEAN_MAX,
    var_max: float = LETTERBOX_VAR_MAX,
    bright_frac_max: float = OVEREXPOSED_FRAC_MAX,
    band: int = DEFAULT_EDGE_BAND,
) -> CheckResult:
    """Verdict over the sampled frames' border bands.

    Per-edge aggregation is the median across frames, so a single fade-to-black
    frame cannot brand a clip as letterboxed, while a real bar — present in
    every frame — still trips it.
    """
    if not per_frame:
        return CheckResult(
            "edges",
            STATUS_FAIL,
            "没有取到任何可分析的帧",
            {"frames": 0},
            "先确认文件能正常解码（见 decodable）。",
        )

    measured: dict[str, Any] = {"frames": len(per_frame), "band_px": band}
    letterboxed: list[str] = []
    overexposed: list[str] = []
    for edge in EDGE_LABELS:
        mean = float(np.median([f[edge]["mean"] for f in per_frame]))
        var = float(np.median([f[edge]["var"] for f in per_frame]))
        bright = float(np.median([f[edge]["bright_frac"] for f in per_frame]))
        measured[edge] = {"mean": round(mean, 3), "var": round(var, 3), "bright_frac": round(bright, 4)}
        if mean < mean_max and var < var_max:
            letterboxed.append(edge)
        if bright > bright_frac_max:
            overexposed.append(edge)

    measured["letterboxed_edges"] = letterboxed
    measured["overexposed_edges"] = overexposed
    measured["thresholds"] = {
        "letterbox_mean_max": mean_max,
        "letterbox_var_max": var_max,
        "overexposed_level": OVEREXPOSED_LEVEL,
        "overexposed_frac_max": bright_frac_max,
    }

    problems: list[str] = []
    advice: list[str] = []
    if letterboxed:
        names = "、".join(EDGE_LABELS[e] for e in letterboxed)
        problems.append(f"{names}有纯黑信箱条／黑边（{band}px 边带亮度均值 < {mean_max:g} 且方差 < {var_max:g}）")
        advice.append(
            f"先裁掉黑边再进管线（ffmpeg cropdetect + crop），否则黑条会被一起投到球面上，"
            f"在头显里变成一道硬黑缝。当前判定的边：{names}。"
        )
    if overexposed:
        names = "、".join(EDGE_LABELS[e] for e in overexposed)
        problems.append(f"{names}大面积过曝（亮度 > {OVEREXPOSED_LEVEL} 的像素占比 > {bright_frac_max * 100:g}%）")
        advice.append(f"{names}是死白区，转头看过去就是一片没有细节的墙。重生成时要求「四边不要大面积纯白过曝」。")

    if problems:
        return CheckResult("edges", STATUS_FAIL, "；".join(problems), measured, " ".join(advice))
    detail = "、".join(
        f"{EDGE_LABELS[e]} {measured[e]['mean']:.1f}/{measured[e]['bright_frac'] * 100:.0f}%" for e in EDGE_LABELS
    )
    return CheckResult(
        "edges",
        STATUS_PASS,
        f"四边干净，无信箱条、无大面积过曝（{band}px 边带 亮度均值/过曝占比：{detail}）",
        measured,
    )


# ---------------------------------------------------------------------------
# 5. detail_ratio — is the detail in the middle, or out at the rim?
# ---------------------------------------------------------------------------


def _as_gray(frame: np.ndarray) -> np.ndarray:
    """Luminance view of a frame that may arrive gray (video) or BGR (still)."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected a grayscale or BGR frame, got shape {frame.shape}")


def _fit_for_analysis(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Shrink so the longest side is ``max_dim`` (no-op if already smaller)."""
    height, width = frame.shape[:2]
    scale = max_dim / float(max(height, width))
    if scale >= 1.0:
        return frame
    size = (max(2, round(width * scale)), max(2, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def center_detail_stats(frame: np.ndarray, max_dim: int = DETAIL_ANALYSIS_MAX_DIM) -> dict[str, float]:
    """Laplacian edge energy of the central 1/3×1/3 rectangle vs. the rest.

    The blur before the Laplacian is what makes this a *detail* measure rather
    than a noise measure: without it, sensor grain and compression mosquito
    noise contribute as much energy as real structure, and a grainy rim would
    read as a detailed one.

    The analysis is done at a fixed longest side because edge energy is not
    scale invariant — the same picture at 2880 px and at 720 px gives different
    absolute energies, and the reference numbers (1.58/1.60) were measured at
    720.  The *ratio* is much more stable than either term, which is why the
    verdict is taken on it.
    """
    gray = _fit_for_analysis(_as_gray(frame), max_dim)
    height, width = gray.shape
    energy = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F))
    half_h, half_w = max(1, height // 6), max(1, width // 6)
    top, bottom = height // 2 - half_h, height // 2 + half_h
    left, right = width // 2 - half_w, width // 2 + half_w
    center = energy[top:bottom, left:right]
    outside = np.ones(energy.shape, dtype=bool)
    outside[top:bottom, left:right] = False
    center_mean = float(center.mean())
    periphery_mean = float(energy[outside].mean()) if outside.any() else 0.0
    return {
        "center": center_mean,
        "periphery": periphery_mean,
        "ratio": center_mean / (periphery_mean + 1e-6),
    }


def check_detail_ratio(
    per_frame: Sequence[dict[str, float]],
    pass_ratio: float = DETAIL_RATIO_PASS,
    fail_ratio: float = DETAIL_RATIO_FAIL,
) -> CheckResult:
    """Verdict over the sampled frames' central-vs-peripheral detail.

    Three bands, all of them taken from the measured reference rather than from
    taste: at or above ``pass_ratio`` the frame is composed the way the films
    that sell tickets are composed; between the two constants the rim is as busy
    as the middle (WARN); below ``fail_ratio`` the rim is *busier*, which points
    the viewer straight at the part of the projection we cannot make look good.
    """
    if not per_frame:
        return CheckResult(
            "detail_ratio",
            STATUS_FAIL,
            "没有取到任何可分析的帧",
            {"frames": 0},
            "先确认文件能正常解码（见 decodable）。",
        )

    ratio = float(np.median([f["ratio"] for f in per_frame]))
    measured: dict[str, Any] = {
        "frames": len(per_frame),
        "ratio": round(ratio, 4),
        "center_edge_energy": round(float(np.median([f["center"] for f in per_frame])), 3),
        "periphery_edge_energy": round(float(np.median([f["periphery"] for f in per_frame])), 3),
        "pass_ratio": pass_ratio,
        "fail_ratio": fail_ratio,
        "reference_ratio": 1.60,
        "analysis_max_dim": DETAIL_ANALYSIS_MAX_DIM,
    }
    numbers = (
        f"中央 1/3 区边缘能量 {measured['center_edge_energy']:.2f} / 周边 "
        f"{measured['periphery_edge_energy']:.2f} = {ratio:.2f}"
        f"（{len(per_frame)} 帧中位数；竞品实测 1.58–1.60，门槛 {pass_ratio:g}）"
    )

    if ratio >= pass_ratio:
        return CheckResult(
            "detail_ratio",
            STATUS_PASS,
            f"细节集中在中央、边缘做得空：{numbers}",
            measured,
        )
    if ratio >= fail_ratio:
        return CheckResult(
            "detail_ratio",
            STATUS_WARN,
            f"中央只是勉强比周边更密：{numbers}",
            measured,
            DETAIL_RATIO_ADVICE,
        )
    return CheckResult(
        "detail_ratio",
        STATUS_FAIL,
        f"周边比中央还密，与竞品的构图规律相反：{numbers}",
        measured,
        DETAIL_RATIO_ADVICE,
    )


# ---------------------------------------------------------------------------
# 6. anchor — is there a subject in the forward direction?
# ---------------------------------------------------------------------------


def frequency_tuned_saliency(frame: np.ndarray, max_dim: int = ANCHOR_ANALYSIS_MAX_DIM) -> np.ndarray:
    """Achanta frequency-tuned saliency, normalised to ``[0, 1]``.

    Distance in CIE Lab between each (lightly blurred) pixel and the frame's
    mean colour: "how unlike the rest of this picture is this spot?".  Cheap,
    dependency-free and, unlike a gradient-based measure, it responds to the
    *inside* of a large flat subject rather than only to its outline — which is
    what lets the 40 %-of-frame case below be measured as 40 % instead of as a
    thin ring.

    A grayscale frame (what the video sampler decodes) is promoted to BGR, so
    the measure degrades to a luminance-distinctness one rather than failing.
    """
    small = _fit_for_analysis(frame, max_dim)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    saliency = np.linalg.norm(lab - mean_lab, axis=2)
    return (saliency - saliency.min()) / (float(np.ptp(saliency)) + 1e-9)


def texture_gate(
    frame: np.ndarray,
    window: int = ANCHOR_TEXTURE_WINDOW,
    max_dim: int = ANCHOR_ANALYSIS_MAX_DIM,
) -> np.ndarray:
    """How *structured* each spot is, as a ``[0, 1]`` gate on saliency.

    Local mean of the **Laplacian magnitude** over a ``window`` box, divided by
    the frame's median and clipped at 1.  Read it as a yes/no question with a
    soft edge: "does this spot carry at least as much fine detail as a typical
    spot in this frame?".  Flat sky, blown highlights and open water answer no;
    rock, foliage, foam and any manufactured object answer yes and saturate.

    Three properties matter and none is negotiable:

    * **High-frequency only.**  The obvious alternative, local *standard
      deviation*, counts a soft gradient as structure — and a cloud is exactly
      that: a large, smooth light-to-dark swing.  Measured on the owner's
      keyframe, stddev leaves the cloud band at 0.53–0.66 of the reference while
      the Laplacian puts it at 0.33, against 1.0 for the aircraft.  That is the
      difference between a gate that trims the sky and one that removes it, and
      "bright but *soft*" is what a sky, a water sheet and a blown highlight all
      have in common.
    * **Self-scaling.** The reference is the frame's own median, so a hazy
      low-contrast frame and a punchy one are judged on the same terms and no
      absolute contrast constant has to be invented.
    * **Saturating.** A ranking measure (divide by the 99th percentile, or take
      the percentile rank) is dominated by object *outlines* — the boundary of a
      bright subject on a dark field is the strongest edge in the picture, so the
      subject's own interior scores low and Otsu then cuts the subject in half.
      Clipping at the median makes the whole interior of anything textured read
      as 1.0, which leaves the mask, and therefore the reported area, where the
      ungated saliency put it.

    Grayscale in, grayscale out — the video sampler decodes ``-pix_fmt gray`` and
    this costs nothing extra there.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    gray = _fit_for_analysis(_as_gray(frame), max_dim).astype(np.float32)
    energy = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (window, window))
    reference = float(np.median(energy)) + 1e-6
    return np.clip(energy / reference, 0.0, 1.0)


def _barrier_sweep(channel: np.ndarray, passes: int) -> np.ndarray:
    """Minimum-barrier distance from the frame border for one image channel.

    The barrier of a path is ``max(path) - min(path)``; the distance of a pixel
    is the smallest barrier over all paths reaching it from any border pixel.
    Read it as "how big a colour step do you have to climb over to get here from
    outside the picture?" — zero along the edge, zero throughout any region that
    joins the edge smoothly, and large inside anything the edge cannot reach
    without crossing a boundary.

    Computed by the standard raster relaxation: each pass sweeps the frame
    top-down, left-right, bottom-up and right-left, propagating the running
    ``(max, min)`` of the best path found so far.  Each sweep is vectorised
    across the axis it is *not* walking, so the Python loop runs once per row or
    column rather than once per pixel — 256 iterations a sweep at
    :data:`ANCHOR_ANALYSIS_MAX_DIM`, ~0.1 s for a whole frame.

    The relaxation only ever *lowers* the distance, so a small ``passes`` yields
    an over-estimate rather than a wrong ordering; see
    :data:`ANCHOR_ENCLOSURE_PASSES` for why three is the operating point.
    """
    height, width = channel.shape
    distance = np.full((height, width), np.inf, np.float32)
    high = channel.copy()
    low = channel.copy()
    distance[0, :] = distance[-1, :] = distance[:, 0] = distance[:, -1] = 0.0

    def relax(target: Any, source: Any) -> None:
        candidate_high = np.maximum(high[source], channel[target])
        candidate_low = np.minimum(low[source], channel[target])
        candidate = candidate_high - candidate_low
        better = candidate < distance[target]
        high[target][better] = candidate_high[better]
        low[target][better] = candidate_low[better]
        distance[target][better] = candidate[better]

    every = slice(None)
    for _ in range(passes):
        for row in range(1, height):
            relax((row, every), (row - 1, every))
        for column in range(1, width):
            relax((every, column), (every, column - 1))
        for row in range(height - 2, -1, -1):
            relax((row, every), (row + 1, every))
        for column in range(width - 2, -1, -1):
            relax((every, column), (every, column + 1))
    return distance


def enclosure_barrier(
    frame: np.ndarray,
    max_dim: int = ANCHOR_ANALYSIS_MAX_DIM,
    passes: int = ANCHOR_ENCLOSURE_PASSES,
) -> np.ndarray:
    """How enclosed each spot is, in CIE-Lab units: the colour step from outside.

    Whatever a picture is *of*, the strip around its edge is mostly not it.  Sky
    runs off the top, a canyon wall runs off the side, a river runs off the
    bottom — each reachable from the border without ever crossing a colour
    boundary, so each scores low here.  A subject is by construction the thing
    the frame is wrapped around, so getting to it from outside means climbing
    over its outline, and it scores high.

    This is **not** a centre prior, and the distinction is what makes it
    admissible in this check at all: it asks whether a region reaches the frame's
    edge without crossing a boundary, never where in the frame it sits.  A
    subject parked in a corner is still enclosed and still scores high, so
    「主体偏离中心」 stays a reachable verdict — which a centre prior would
    quietly make impossible.

    Reduced over L, a and b by **maximum**, not mean: "the largest step in any
    one channel".  The mean makes the measure depend on how the frame arrived —
    a grayscale frame has no a/b barrier at all, so its mean is divided by three
    for no physical reason, and ``seed_v6``'s airframe then reads 31.9 in colour
    against 27.3 in luminance (17 % apart).  Under the maximum the same airframe
    reads 84.1 and 81.9 (2.7 % apart), because the luminance channel carries the
    step in both cases.  #360's contract is that the still path and the video
    sampler's ``-pix_fmt gray`` path measure the same picture; this reduction is
    what makes that true of this channel rather than lucky.

    Returned in raw Lab units, never min-max normalised: a barrier of 40 units is
    the same visible step in a hazy frame and a punchy one, whereas normalising
    would stretch a frame containing nothing but noise until its noise looked
    like a subject.
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    small = _fit_for_analysis(frame, max_dim)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    barrier = np.zeros(lab.shape[:2], np.float32)
    for index in range(3):
        np.maximum(barrier, _barrier_sweep(np.ascontiguousarray(lab[..., index]), passes), out=barrier)
    return barrier


def enclosure_gate(
    frame: np.ndarray,
    reference: float = ANCHOR_ENCLOSURE_REFERENCE,
    exponent: float = ANCHOR_ENCLOSURE_EXPONENT,
) -> np.ndarray:
    """How *enclosed* each spot is, as a ``[0, 1]`` gate on saliency.

    :func:`enclosure_barrier` divided by :data:`ANCHOR_ENCLOSURE_REFERENCE`,
    clipped at 1 and raised to :data:`ANCHOR_ENCLOSURE_EXPONENT`.  Read it, like
    :func:`texture_gate`, as a yes/no question with a soft edge: "is this spot
    something the picture is wrapped around, or is it the backdrop?".

    Unlike :func:`texture_gate` the reference is **absolute**, not the frame's
    own median — see :data:`ANCHOR_ENCLOSURE_REFERENCE` for the measurement that
    forces that choice.  The short version: an evenly-textured frame is, by this
    measure, enclosing everywhere, so a self-scaling reference would manufacture
    subjects out of exactly the fixtures that must have none.
    """
    if reference <= 0:
        raise ValueError(f"reference must be > 0, got {reference}")
    return np.clip(enclosure_barrier(frame) / reference, 0.0, 1.0) ** exponent


def structured_saliency(frame: np.ndarray, window: int = ANCHOR_TEXTURE_WINDOW) -> np.ndarray:
    """Distinctness × structure × enclosure, normalised to ``[0, 1]``.

    Three questions, each a gate, and an object is the only thing that answers
    all three: is this spot *unlike* the rest of the picture
    (:func:`frequency_tuned_saliency`), does it carry *structure*
    (:func:`texture_gate`), and is it something the frame is *wrapped around*
    (:func:`enclosure_gate`)?

    **Distinctness** alone has a systematically wrong answer on the footage this
    pipeline exists for: in a canyon frame the red-brown rock owns the mean, so
    the *sky* is the most distinct thing in the picture and a gray aircraft in
    the middle is among the least.  **Texture** (#343) removes the flat bright
    region — ablated on a fixture with a blown-out sky band over textured ground,
    the ungated detector returns the sky (21.7 % of the frame, centroid y = 0.11)
    and the gated one the subject (10.7 %, y = 0.51).

    **Enclosure** (#361) is what removes the bright *textured* backdrop, which
    neither of the other two can.  Real whitewater and real cloud are distinct
    and structured, so they pass both earlier gates — and on
    ``video/seed_v6.png`` the aircraft's combined distinctness × texture score is
    0.26 against the foam's 0.56, with Otsu cutting at 0.38.  That is the whole
    of G-10: the subject was not merely losing the election, it was **not on the
    ballot** — no component of the mask covered it, so no amount of re-scoring or
    vetoing candidates could have reached it.  The enclosure gate multiplies the
    foam down (barrier 41 Lab units against the airframe's 84, squared against a
    reference of 80: 0.26 against 1.0) and puts the aircraft in the mask.

    Distinctness is measured on **luminance**, not colour, and the exponent is
    sub-linear; see :data:`ANCHOR_DISTINCTNESS_EXPONENT`.  Dropping the a/b
    channels does not change *what* is found — ``seed_v6`` answers (0.54, 0.49)
    in colour and (0.54, 0.50) in luminance — only *how much of it*: in colour
    the extra chroma evidence pulls in more of the airframe (3.4 % of the frame
    against 2.2 %), so the same still measured two legal ways disagreed by 75 %
    and #360's ≤30 % contract failed.  Feeding this channel luminance makes the
    still path and the video sampler's gray path agree by construction; the
    spread is 12 %.
    """
    combined = (
        frequency_tuned_saliency(_as_gray(frame)) ** ANCHOR_DISTINCTNESS_EXPONENT
        * texture_gate(frame, window)
        * enclosure_gate(frame)
    )
    return (combined - combined.min()) / (float(np.ptp(combined)) + 1e-9)


def anchor_stats(
    frame: np.ndarray,
    kernel: int = ANCHOR_MORPH_KERNEL,
    sigma: float = ANCHOR_CENTER_PRIOR_SIGMA,
    min_area: float = ANCHOR_MIN_AREA,
    rescue: bool = True,
) -> dict[str, float]:
    """Locate the frame's anchor and report its area share and eccentricity.

    The pipeline is: :func:`structured_saliency` → **Otsu** mask → open (drop
    speckle) → close (heal a subject broken up by a highlight) → connected
    components → pick the one with the **densest** centre-weighted saliency.

    Density (mean per pixel), not total mass, and this is the half of issue #343
    that actually moved the owner's frames.  Mass is area × density, so it is won
    by whatever is *biggest*: a sky band or a stretch of whitewater beats a
    subject that is more distinct per pixel but a fifth of the size, every time,
    and **no amount of texture gating fixes an ordering dominated by the area
    term** — real cloud and real foam are textured, so they pass the gate and
    then win on sheer size.  Measured on the owner's keyframes, with mass the
    detector returns the cloud band (centroid y = 0.06) or the whitewater
    (y = 0.79) whether or not the texture gate is applied; with density it
    returns the aircraft (y = 0.47 / 0.36).

    Asking which region is most strongly "subject" per pixel is the question the
    check actually poses, and it is scale-free — the same object read at 1024²
    and at 2880² scores the same.

    Density is nonetheless only a way of *ranking the ballot*, and #361 is the
    reminder of what that cannot do.  On ``video/seed_v6.png`` the aircraft was
    not a low-scoring candidate, it was not a candidate: its distinctness ×
    texture score is 0.26 where Otsu cuts at 0.38, so no component of the mask
    covered it and every re-ranking, veto and background prior applied to the
    candidate list was arguing about the wrong set.  The fix belongs in
    :func:`structured_saliency` — the enclosure gate — not here.

    Density has one failure the #343 work did not close, and #356 is it: the
    measure is *biased towards small regions*.  A region's mean converges on its
    own peak as it shrinks, so an eight-pixel specular glint scores higher than
    any real object can, wins the election outright, and — being far below the
    detection floor itself — makes the frame answer 「没有锚点」 while an 11 %
    subject sits untouched beside it.  Worse, it does so *intermittently*: a
    speck that big is at the mercy of resampling and compression, so it survives
    the opening as its own component in some renderings of a frame and not
    others.  Measured on ``video/seed_v6.png``, the same picture answered
    **10.7 %** at 2048², 1024² and 256² and **0.12 %** at 960² and 512², and
    ``video/gen_1x1_720p_v6.mp4`` produced 0.12 / 0.27 / 1.45 / 1.11 / 1.48 /
    0.82 / 0.15 / 0.63 % across eight frames of one continuous shot — a
    bimodal reading, not a resolution effect.

    The fix is candidacy, not scoring: a component under
    :data:`ANCHOR_RESCUE_CORE_FLOOR` × ``min_area`` cannot be reported as an
    anchor under any circumstance — it is below the detection floor and too
    small to be rescued — so it has no business deciding which region is.  Only
    components at or above that floor stand in the election; if a frame has
    none (a genuinely subjectless one), every component stands and the answer is
    "no anchor" exactly as before.  Note what this does **not** claim to fix: a
    *credible-sized* small region that is more distinct per pixel than a larger
    one still wins, which is the intended behaviour — at that size it is a
    subject, and the quality band is what judges it.

    Otsu rather than a fixed percentile, and the difference is not cosmetic.  A
    percentile decides *in advance* how much of the frame is salient, so it can
    only return the true area of a subject whose saliency is perfectly uniform:
    measured on the fixtures, a top-decile mask reads a 40 %-of-frame subject as
    38 % when it is flat-filled and as **0.2 %** once it carries any internal
    texture, because the cut then falls inside the subject's own distribution
    and the opening sweeps up the crumbs.  Otsu puts the cut *between* the two
    modes instead and reads the same subject as 39 % either way.

    The opening is what makes "no anchor" a reachable answer at all.  Otsu
    always returns a threshold, so a frame with no subject still yields a
    dusting of unrelated specks; closing alone would merge them into one
    frame-filling region and report a giant, perfectly-centred subject.  Opening
    first deletes them and leaves nothing above ``min_area``.

    But opening is also a measurement error on real subjects (#348): it erodes
    thin extremities — a drone's arms — as effectively as it erodes specks, and
    on the 2048² keyframe that alone pushed a 1.38 %-of-frame subject under the
    detection floor (its opened core measures 0.53 %).  The fix is a **rescue**,
    applied only when it is needed and only where it is safe.  If the winning
    core already clears the detection floor, the subject is established and the
    conservative core measurement stands — every fixture's numbers are
    unchanged.  If the core is too small to be credible evidence of a subject —
    under :data:`ANCHOR_RESCUE_CORE_FLOOR` × ``min_area`` — restoration would
    amplify speck noise four- to five-fold, so it is withheld.  Only a core in
    between (credible, yet sub-floor) is re-grown to its full pre-morphology
    extent: the union of the core and every raw-mask component continuous with
    it — open-by-reconstruction for exactly one region.  Pass ``rescue=False``
    to measure the bare core — the pre-#348 behaviour — which exists so the
    regression proving the rescue is load-bearing can disable exactly this
    step.

    ``found`` is the answer to "is there an anchor"; ``area``/``offset`` are only
    meaningful when it is true.  ``centroid_x``/``centroid_y`` are the picked
    region's centre in ``[0, 1]`` frame coordinates and exist so that a
    regression can assert *which* thing was found, not merely that something
    was: the bug this replaced reported a perfectly plausible area while
    pointing at the sky.
    """
    if kernel < 1:
        raise ValueError(f"kernel must be >= 1, got {kernel}")
    saliency = structured_saliency(frame)
    height, width = saliency.shape
    _, raw_mask = cv2.threshold(
        np.clip(saliency * 255.0, 0, 255).astype(np.uint8),
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    element = np.ones((kernel, kernel), np.uint8)
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, element)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    empty = {
        "found": False,
        "area": 0.0,
        "offset": 0.0,
        "components": max(0, count - 1),
        "centroid_x": 0.5,
        "centroid_y": 0.5,
    }
    if count <= 1:
        return empty

    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    half_diagonal = 0.5 * math.hypot(width, height)
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.hypot(xs - cx, ys - cy) / half_diagonal
    weight = np.exp(-(radius**2) / (2.0 * sigma**2)).astype(np.float32)
    weighted = saliency * weight

    # #356: only regions that could actually *be* an anchor may stand in the
    # election.  Mean density is a biased score — as a region shrinks its mean
    # converges on its own peak — so an 8-pixel specular glint outscores any
    # real subject, and the winner-takes-all pick then hands the frame's verdict
    # to a speck that is itself far too small to be reported.  Whether such a
    # speck survives the opening as a separate component turns on resampling
    # noise, which is what made the same frame answer 10.7 % at 2048²/1024²/256²
    # and 0.12 % at 960²/512², and made 4 of 8 frames of one clip "anchorless".
    credible = ANCHOR_RESCUE_CORE_FLOOR * min_area * height * width
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= credible]
    if not candidates:  # nothing credible in frame — judge the specks as before
        candidates = list(range(1, count))

    best_index, best_density = candidates[0], -1.0
    for index in candidates:
        region = labels == index
        density = float(weighted[region].mean())
        if density > best_density:
            best_index, best_density = index, density

    pixels = int(stats[best_index, cv2.CC_STAT_AREA])
    area = pixels / float(height * width)
    blob_x, blob_y = centroids[best_index]

    # #348: rescue a credible-but-sub-floor core by re-growing it to its full
    # pre-morphology extent.  The opening that deletes noise specks also eats
    # a real subject's thin extremities — on the 2048² drone keyframe it alone
    # pushed a 1.38 %-of-frame subject under the 0.8 % detection floor, leaving
    # a 0.53 % core.  Every raw-mask component continuous with the core is part
    # of the same subject; unioning them measures the subject, not its eroded
    # skeleton.  Gated on ANCHOR_RESCUE_CORE_FLOOR so specks are not amplified.
    if rescue and ANCHOR_RESCUE_CORE_FLOOR * min_area <= area < min_area:
        _count_raw, labels_raw, _stats_raw, _cents_raw = cv2.connectedComponentsWithStats(raw_mask, 8)
        under_core = labels_raw[labels == best_index]
        frag_ids = np.unique(under_core[under_core > 0])
        if frag_ids.size:
            region = labels == best_index
            for r in frag_ids:
                region |= labels_raw == r
            area = float(region.sum()) / float(height * width)
            ys_r, xs_r = np.nonzero(region)
            blob_x, blob_y = float(xs_r.mean()), float(ys_r.mean())

    offset = math.hypot(blob_x - cx, blob_y - cy) / half_diagonal
    return {
        "found": area >= min_area,
        "area": area,
        "offset": offset,
        "components": count - 1,
        "centroid_x": blob_x / max(1.0, width - 1.0),
        "centroid_y": blob_y / max(1.0, height - 1.0),
    }


def check_anchor(
    per_frame: Sequence[dict[str, float]],
    area_min: float = ANCHOR_AREA_MIN,
    area_max: float = ANCHOR_AREA_MAX,
    offset_max: float = ANCHOR_OFFSET_MAX,
    detected_frac_min: float = ANCHOR_DETECTED_FRAC_MIN,
) -> CheckResult:
    """Verdict over the sampled frames: is something holding the eye up front?

    Never a FAIL.  Plenty of legitimate shots — a pure landscape fly-through, an
    abstract tunnel — have no subject, so blocking a run over it would be wrong;
    what the operator needs is to be *told*, because the alternative to looking
    at an anchor is looking at the rim, and the rim is where our geometry is
    weakest.

    Aggregation is the median over the frames that *had* a subject, with a
    separate detection rate, so a clip that only shows its hero half the time
    reports an honest 50 % rather than an area averaged with zeros.
    """
    if not per_frame:
        return CheckResult(
            "anchor",
            STATUS_FAIL,
            "没有取到任何可分析的帧",
            {"frames": 0},
            "先确认文件能正常解码（见 decodable）。",
        )

    found = [f for f in per_frame if f["found"]]
    detected_frac = len(found) / len(per_frame)
    measured: dict[str, Any] = {
        "frames": len(per_frame),
        "detected_frames": len(found),
        "detected_frac": round(detected_frac, 4),
        "detected_frac_min": detected_frac_min,
        "area_min": area_min,
        "area_max": area_max,
        "offset_max": offset_max,
        "reference_area": 0.113,
        "reference_offset": 0.236,
    }

    if detected_frac < detected_frac_min:
        measured["area"] = 0.0
        measured["offset"] = 0.0
        return CheckResult(
            "anchor",
            STATUS_WARN,
            f"检不出锚点主体：{len(found)}/{len(per_frame)} 帧有主体（低于 {detected_frac_min * 100:g}%），"
            f"画面里没有锚点",
            measured,
            ANCHOR_MISSING_ADVICE,
        )

    area = float(np.median([f["area"] for f in found]))
    offset = float(np.median([f["offset"] for f in found]))
    measured["area"] = round(area, 4)
    measured["offset"] = round(offset, 4)
    numbers = (
        f"主体面积 {area * 100:.1f}%、偏心 {offset:.2f}"
        f"（{len(found)}/{len(per_frame)} 帧检出；竞品实测中位 11.3% / 0.236）"
    )

    problems: list[str] = []
    if area > area_max:
        problems.append(f"面积偏大：{area * 100:.1f}% > {area_max * 100:g}%，主体糊住了画幅")
    elif area < area_min:
        problems.append(f"面积偏小：{area * 100:.1f}% < {area_min * 100:g}%，抓不住视线")
    if offset > offset_max:
        problems.append(f"偏离中心：偏心 {offset:.2f} > {offset_max:g}，锚点不在正前方")

    if not problems:
        return CheckResult("anchor", STATUS_PASS, f"正前方有锚点主体：{numbers}", measured)
    return CheckResult(
        "anchor",
        STATUS_WARN,
        f"{'；'.join(problems)}。{numbers}",
        measured,
        ANCHOR_OFF_SPEC_ADVICE,
    )


# ---------------------------------------------------------------------------
# Frame sampling — always list-form subprocess, never shell=True
# ---------------------------------------------------------------------------


def _even(value: float) -> int:
    """Round to a positive even int (ffmpeg's rawvideo/scale prefer even)."""
    return max(2, round(value / 2.0) * 2)


def _grab_pair(
    path: Path,
    timestamp: float,
    width: int,
    height: int,
    ffmpeg: str = "ffmpeg",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Two *consecutive* grayscale frames starting at ``timestamp``.

    ``-ss`` before ``-i`` is the fast (keyframe) seek; the two frames that come
    out are still adjacent to each other, which is all the flow needs.
    Grayscale raw video keeps this to a third of rgb24's bytes and every check
    here is luminance-only anyway.  Returns ``None`` when the clip ends before
    a second frame could be read.
    """
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "2",
        "-vf",
        "format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
    stride = width * height
    if proc.returncode != 0 or len(proc.stdout) < 2 * stride:
        return None
    first = np.frombuffer(proc.stdout[:stride], dtype=np.uint8).reshape(height, width)
    second = np.frombuffer(proc.stdout[stride : 2 * stride], dtype=np.uint8).reshape(height, width)
    return first, second


def iter_frame_pairs(
    path: str | Path,
    info: ProbeInfo,
    pairs: int = DEFAULT_PAIRS,
    ffmpeg: str = "ffmpeg",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``pairs`` adjacent native-resolution grayscale frame pairs.

    Sampled at the midpoints of ``pairs`` equal slices, so the whole clip is
    represented and neither end is over-weighted.  A clip with no usable
    duration falls back to a single pair at t=0.
    """
    if pairs < 1:
        raise ValueError(f"--pairs must be >= 1, got {pairs}")
    src = Path(path)
    stamps = [0.0] if info.duration <= 0.0 else [info.duration * (i + 0.5) / pairs for i in range(pairs)]
    for stamp in stamps:
        pair = _grab_pair(src, stamp, info.width, info.height, ffmpeg=ffmpeg)
        if pair is not None:
            yield pair


def is_still(path: str | Path) -> bool:
    """Does this path name an image rather than a clip?  Extension only."""
    return Path(path).suffix.lower() in STILL_SUFFIXES


def read_still(path: str | Path) -> np.ndarray:
    """Decode a still to BGR, read-only, without ffmpeg.

    ``np.fromfile`` + ``cv2.imdecode`` rather than ``cv2.imread`` because
    ``imread`` goes through a non-Unicode path API on Windows and returns
    ``None`` for any path with a non-ASCII character in it — the operator's
    ``video/`` names are ASCII today, but a silent "file unreadable" on a
    perfectly good keyframe is a bad way to find that out.
    """
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def downscale_for_flow(frame: np.ndarray, target_width: int = DEFAULT_FLOW_WIDTH) -> np.ndarray:
    """Shrink to ``target_width`` for Farnebäck (no-op if already narrower)."""
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    new_w = _even(target_width)
    new_h = _even(height * new_w / width)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_checks(
    path: str | Path,
    pairs: int = DEFAULT_PAIRS,
    flow_width: int = DEFAULT_FLOW_WIDTH,
    tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    band: int = DEFAULT_EDGE_BAND,
    min_radial_rate: float = DEFAULT_MIN_RADIAL_RATE,
    skip: Sequence[str] = (),
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> SourceReport:
    """Run every non-skipped check against ``path``.  Read-only, always.

    The clip is decoded **once**: each sampled pair feeds the edge bands and the
    composition checks at native resolution (an 8 px bar must not be blurred
    away by a downscale; the composition pair do their own fixed-size
    downscale) and the flow analysis at :data:`DEFAULT_FLOW_WIDTH`.

    A still takes the same route with one frame and no pair, which is why
    ``forward_motion`` comes back ``skipped`` rather than judged.  Everything
    else — including both composition checks, the ones that make vetting a
    keyframe worth doing — runs identically to the video path.
    """
    report = SourceReport(source=str(path))
    skipped = set(skip)
    unknown = skipped - set(CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown check name(s) in --skip: {sorted(unknown)}; valid: {list(CHECK_NAMES)}")

    src = Path(path)
    if not src.is_file():
        report.checks.append(
            CheckResult("decodable", STATUS_FAIL, f"文件不存在：{path}", {"exists": False}, "检查路径。")
        )
        return report

    still = is_still(src)
    try:
        info = probe_source(src, ffprobe=ffprobe)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        report.checks.append(CheckResult("decodable", STATUS_FAIL, str(exc), {}, "文件可能损坏或不是视频。"))
        return report

    frame_checks = {"forward_motion", "edges", "detail_ratio", "anchor"}
    needs_frames = not frame_checks <= skipped
    flow_stats: list[RadialStats] = []
    band_stats: list[dict[str, dict[str, float]]] = []
    detail_stats: list[dict[str, float]] = []
    subject_stats: list[dict[str, float]] = []
    circle_frac: float | None = None

    def measure(frame: np.ndarray) -> None:
        """Every per-frame measurement, taken on the one frame we decoded."""
        if "edges" not in skipped:
            band_stats.append(edge_band_stats(_as_gray(frame), band=band))
        if "detail_ratio" not in skipped:
            detail_stats.append(center_detail_stats(frame))
        if "anchor" not in skipped:
            subject_stats.append(anchor_stats(frame))

    if needs_frames and still:
        try:
            measure(read_still(src))
        except (RuntimeError, OSError) as exc:
            report.checks.append(CheckResult("decodable", STATUS_FAIL, str(exc), {}, "图片损坏或格式不支持。"))
            return report
    elif needs_frames:
        for first, second in iter_frame_pairs(src, info, pairs=pairs, ffmpeg=ffmpeg):
            measure(first)
            if "forward_motion" not in skipped:
                small_first = downscale_for_flow(first, flow_width)
                if circle_frac is None:
                    # Detect once from the first sampled frame; the imaging
                    # circle does not move between frames of one clip (#322).
                    circle_frac = estimate_imaging_circle(small_first)
                flow_stats.append(
                    radial_flow_stats(
                        small_first,
                        downscale_for_flow(second, flow_width),
                        circle_frac=circle_frac,
                    )
                )

    def forward_motion_result() -> CheckResult:
        if still:
            return CheckResult(
                "forward_motion",
                STATUS_SKIP,
                STILL_FORWARD_DETAIL,
                {"still": True, "pairs": 0},
                STILL_FORWARD_ADVICE,
            )
        rate = min_radial_rate
        if circle_frac is not None:
            rate *= FISHEYE_MIN_RADIAL_FACTOR
        return check_forward_motion(flow_stats, min_radial_rate=rate, circle_frac=circle_frac)

    builders = {
        "decodable": lambda: check_decodable(info, still=still),
        "square": lambda: check_square(info.width, info.height, tolerance=tolerance),
        "forward_motion": forward_motion_result,
        "edges": lambda: check_edges(band_stats, band=band),
        "detail_ratio": lambda: check_detail_ratio(detail_stats),
        "anchor": lambda: check_anchor(subject_stats),
    }
    for name in CHECK_NAMES:
        if name in skipped:
            report.checks.append(CheckResult(name, STATUS_SKIP, "已通过 --skip 跳过", {}))
        else:
            report.checks.append(builders[name]())
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: SourceReport) -> str:
    """The human-readable per-check report with a verdict line."""
    lines = [f"素材体检 — {report.source}", "=" * 68]
    for check in report.checks:
        lines.append(f"{_ICON.get(check.status, '?')} {check.name}: {check.detail}")
        if check.advice:
            lines.append(f"   → {check.advice}")
    counts = report.summary
    lines.append("=" * 68)
    lines.append(
        f"合计: {counts[STATUS_PASS]} pass / {counts[STATUS_WARN]} warn / "
        f"{counts[STATUS_FAIL]} fail / {counts[STATUS_SKIP]} skipped"
    )
    if report.failed:
        lines.append("结论: ❌ 不要进管线——先按上面的建议处理或重生成。")
    elif report.warned:
        lines.append("结论: ⚠️ 可以进管线，但请先读一遍上面的 warn。")
    else:
        lines.append("结论: ✅ 体检全过，可以进管线。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_source_quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "素材进管线前的六项体检：decodable / square / forward_motion / edges / "
            "detail_ratio / anchor（只读，不改源文件）。传图片则自动进入静帧模式。"
        ),
        epilog=(
            "退出码: 0 = 全过或仅 WARN；1 = 有 FAIL（可直接用于 preflight 门禁）。\n"
            "最关键的一项是 forward_motion：静态机位或横移的素材没有前后视差，\n"
            "跑完整条管线只会得到「贴在球面上的平面画」，几十分钟白费。\n"
            "传 .png/.jpg 等图片时进入静帧模式：forward_motion 标 skipped，其余照跑——\n"
            "先验关键帧再决定要不要花钱生成视频，是最省钱的一步。\n"
            "detail_ratio / anchor 的阈值来自 Red Raion 34 条成片 + 613 张剧照的实测\n"
            "（中央/周边细节比 1.58–1.60；80% 的帧有主体，面积中位 11.3%、偏心 0.236）。\n"
        ),
    )
    parser.add_argument("source", help="要体检的视频或静帧图片（只读）")
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help=f"抽取的相邻帧对数量，均匀分布在整片上（默认 {DEFAULT_PAIRS}）",
    )
    parser.add_argument(
        "--flow-width",
        type=int,
        default=DEFAULT_FLOW_WIDTH,
        help=f"光流分析前缩放到的宽度（默认 {DEFAULT_FLOW_WIDTH}）",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_ASPECT_TOLERANCE,
        help=f"1:1 宽高比容差，小数（默认 {DEFAULT_ASPECT_TOLERANCE} 即 ±2%%）",
    )
    parser.add_argument(
        "--edge-band",
        type=int,
        default=DEFAULT_EDGE_BAND,
        help=f"四边各取多宽的边带做黑边/过曝检测，像素（默认 {DEFAULT_EDGE_BAND}）",
    )
    parser.add_argument(
        "--min-radial-rate",
        type=float,
        default=DEFAULT_MIN_RADIAL_RATE,
        help=(
            f"判定「真前进」所需的最小平均径向光流，单位是半对角线 R 的比例"
            f"（默认 {DEFAULT_MIN_RADIAL_RATE}，与分辨率无关）"
        ),
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=CHECK_NAMES,
        metavar="NAME",
        help=f"跳过某一项检查，可重复。可选：{', '.join(CHECK_NAMES)}",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="输出结构化 JSON；裸 --json 打到 stdout，--json PATH 写文件（同时仍打印人类报告）",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件（默认 ffmpeg）")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe 可执行文件（默认 ffprobe）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    to_stdout = args.json == "-"
    try:
        report = run_checks(
            args.source,
            pairs=args.pairs,
            flow_width=args.flow_width,
            tolerance=args.tolerance,
            band=args.edge_band,
            min_radial_rate=args.min_radial_rate,
            skip=args.skip,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if to_stdout:
        print(payload)
    else:
        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        print(format_report(report))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
