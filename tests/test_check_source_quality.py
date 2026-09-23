r"""Tests for the pre-pipeline source health check (scripts/check_source_quality.py, W-1 #305).

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
with :data:`~scripts.check_source_quality.ANCHOR_MIN_AREA` sitting between them
— so "no anchor" is a verdict the check can actually reach, not a branch that
never fires.  ``test_subject_and_subjectless_frames_are_separated`` pins that
margin directly; the texture channel added in #343 widened it to ~50×, the
enclosure channel added in #361 gave back some of that (~41×, because a gate
that refuses to look at brightness also refuses to punish bright speckle), and
G-11 (#364) recovered it and more by widening
:data:`~scripts.check_source_quality.ANCHOR_TEXTURE_WINDOW` from 21 to 31 —
**64.1×**, with the loudest subjectless fixture down to 0.17 % of the frame.
``test_the_texture_channel_widens_the_subject_separation`` pins the 55× floor
that none of them may cross.

Two thresholds, two questions (#345)
------------------------------------
``ANCHOR_MIN_AREA`` (0.8 %) is a **detection floor** — *is anything there* — and
``ANCHOR_AREA_MIN``/``ANCHOR_AREA_MAX`` (8–15 %) are a **quality band** — *is it
big enough*.  Conflating them is what #345 fixed: at a 1.5 % floor the owner's
``Gemini_v1.jpg`` keyframe, whose quadcopter measured 1.30 %, was reported as
「画面里没有锚点」 when the true and actionable answer was 「主体面积 1.3%，远小于
8%」.  The first sends the operator off to invent a subject; the second tells him
to enlarge the one he has.

That keyframe no longer exists (#366).  The owner deleted it, and re-running the
prompt cannot reproduce a 1.30 % reading pixel for pixel, so the numbers it
carried had to be **re-measured** rather than carried over.  The two canyon
stills under ``video/`` hold exactly the property the card was about — a small,
roughly centred aircraft over water — and the #345 pins now run on them:
``seed_v6.png`` at **2.41 %** and ``seed_1x1_drone.png`` at **1.93 %**, both far
under the 8 % band and comfortably over the 0.8 % floor.

The floor moved and the band did not, so the tests are split the same way:
``test_a_small_real_subject_is_small_not_absent`` pins the new verdict on
``seed_v6.png``,
``test_the_detection_floor_sits_between_noise_and_the_smallest_real_subject``
pins the headroom on both sides (loudest subjectless fixture 0.17 %, floor
0.8 %, smallest real subject 1.93 %), and the subjectless fixtures above are
deliberately **unchanged** — a lower floor that started finding anchors in an
empty frame would be a worse bug than the one it replaced.

And the sky is not the subject (#343), nor the river (#361)
-----------------------------------------------------------
The anchor detector's first field trial failed: on the owner's canyon keyframes
it reported the *cloud band* and the *whitewater* as the subject, because a
red-rock frame's mean colour is owned by the rock, which leaves bright sky as
the most colour-"distinct" thing in the picture.  #343 answered the *flat* half
of that with a texture channel; #361 answered the rest, because real whitewater
is bright **and** finely textured and passes a structure test honestly.  What
separates foam from airframe is neither tone nor position but **enclosure**: the
river runs off the edge of the picture and the aircraft does not.

Each fix owns a fixture and a matched mutation, so no acceptance test can
quietly start passing for the wrong reason:

============================  ==============================  ====================
composition                   acceptance                      mutation
============================  ==============================  ====================
``flat_sky_frame`` (#343)     subject, not the sky band       blind both structure
                                                              channels → y = 0.11
``gorge_frame`` (#361)        subject at y = 0.26, not the    blind enclosure
                              river at y = 0.75               → y = 0.75
``thin_limbed_frame`` (#348)  3.97 %, not a 0.73 % skeleton   ``rescue=False``
                                                              → 「没有锚点」
============================  ==============================  ====================

The #343 mutation needs *both* channels blinded because a sky band that runs off
the top edge is unenclosed as well as flat, so either channel alone now disposes
of it; the texture channel's own necessity is pinned instead on the #341 margin,
which collapses from 64.1× to 7.6× without it.

G-11 (#364) extended that pattern one step further, and the table above records
the current state rather than the historical one.  With the texture window at 31
the foam on ``seed_v6.png`` is held off by the texture gate alone, so blinding
*only* the enclosure channel no longer lands the anchor in the water there —
it lands on the canyon wall at x = 0.11 instead.  The real-still mutation
therefore asserts "off the airframe", and the enclosure channel's own necessity
is pinned where it is still unshadowed: :func:`gorge_frame`, in CI.

Almost everything here runs on synthetic arrays: no test decodes video or shells
out to ffmpeg.  The exceptions are the regressions against the footage the bugs
were actually reported on (``CANYON_STILL``, ``SEED_STILL``, ``SEED_CLIP``),
which is read **read-only** and ``skip``\ ped when absent — ``video/`` is
git-ignored, so none of it ships with the repo and none of it exists in CI or in
a **git worktree**.  Every one of those skips names the missing file, the
environment variable that re-points it and where the main checkout keeps it
(:func:`_missing_assets`), because a skip nobody can act on is how the fourth
constant that used to live here spent four cards pointing at a file the owner
had deleted without anyone noticing (#366).
The ffprobe/ffmpeg layer is covered by parsing captured ffprobe
JSON and by an AST sweep (``test_subprocess_calls_are_list_form_without_shell``)
that holds every ``subprocess.run`` in the module to list form with no
``shell=True``.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import check_source_quality as csq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_source_quality.py"

#: The canyon still issue #343 was reported against, and the one #361 reproduced
#: the whitewater failure on.  It does not ship with the repo — ``video/`` is
#: git-ignored — so its regressions **skip** wherever the footage is absent
#: rather than fail.  CI keeps them honest a different way: :func:`flat_sky_frame`
#: and :func:`gorge_frame` reproduce the same failures synthetically and run
#: everywhere.
#:
#: #343's *first* frame, ``C:\Users\musof\Downloads\Gemini_v1.jpg``, used to be
#: wired up beside it as ``OWNER_KEYFRAME``.  The owner deleted that file, which
#: left four tests skipping on every machine on earth — a test that can never
#: run is worse than no test, because it still counts (#366).  Its readings
#: (1.30 % at (0.50, 0.52)) cannot be reproduced by re-generating the prompt, so
#: the pins it carried were re-measured on the two stills below instead.
CANYON_STILL = Path(os.environ.get("VR180_CANYON_STILL", str(REPO_ROOT / "video" / "seed_1x1_drone.png")))

#: The #356 pair: a 2048² keyframe and the 960² clip generated from it.  Same
#: gorge, same aircraft, two orders of magnitude apart in the old reading — the
#: evidence the card was filed on.  Git-ignored like the one above, so these
#: regressions skip wherever ``video/`` is not present.
SEED_STILL = Path(os.environ.get("VR180_SEED_STILL", str(REPO_ROOT / "video" / "seed_v6.png")))
SEED_CLIP = Path(os.environ.get("VR180_SEED_CLIP", str(REPO_ROOT / "video" / "gen_1x1_720p_v6.mp4")))

#: Which environment variable re-points which asset, for the skip messages.
_ASSET_ENV = {
    CANYON_STILL: "VR180_CANYON_STILL",
    SEED_STILL: "VR180_SEED_STILL",
    SEED_CLIP: "VR180_SEED_CLIP",
}


def _missing_assets(*assets: Path) -> str:
    """Why a real-asset test did not run, in enough detail to act on.

    ``skip`` is how this file survives not shipping its footage, and a skip whose
    reason is a bare path is indistinguishable from a test that has quietly
    stopped testing anything.  That is precisely what happened to
    ``OWNER_KEYFRAME``: it named a deleted file, every run said so, and four
    cards went by before anyone read it as 「这条从来没跑过」 (#366).  So each
    reason names the file, the variable that re-points it, and where a checkout
    that has the footage keeps it — enough for a reader to tell 「本来就不该跑」
    from 「该跑却没跑」 without opening this file.
    """
    return "; ".join(
        f"{path.name} not found at {path} — set {_ASSET_ENV[path]} to it "
        f"(a checkout with footage keeps it at <repo>/video/{path.name}; "
        f"video/ is git-ignored, so it is absent in CI and in every git worktree)"
        for path in assets
        if not path.is_file()
    )


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


def flat_sky_frame(area_frac: float = 0.11, size: int = FRAME_SIZE, sky_frac: float = 0.22) -> np.ndarray:
    """The #343 composition: a blown-out flat sky band over textured ground.

    A deliberately unfair frame for a colour-distinctness detector, built to the
    shape of the real failure.  The textured mid-tone ground owns most of the
    picture and therefore owns the frame's mean colour, which leaves the
    **sky** — flat, blown out, nothing to look at — as the most colour-distinct
    thing in it.  The actual subject is small, dead centre and *textured*, and
    its tone sits much closer to the mean, so on colour alone it loses.

    The numbers are chosen so the sky wins on colour by a clear margin rather
    than a hair: measured with the texture channel disabled this frame reports a
    21.7 %-of-picture "subject" whose centroid is at y = 0.11, i.e. the sky band
    itself.  Nothing here is flat-filled — the ground and the subject are both
    real texture, so the only thing separating them from the sky is structure.
    """
    rng = np.random.default_rng(5)
    frame = 105.0 + (_fine_texture(size, size, seed=9) - 128.0) / 127.0 * 28.0
    band = int(size * sky_frac)
    frame[:band, :] = 244.0 + rng.normal(0.0, 0.8, (band, size))
    ground = np.clip(frame, 0, 255).astype(np.uint8)
    return _paint_subject(ground, area_frac, level=70.0, amplitude=35.0, seed=13)


def gorge_frame(
    size: int = FRAME_SIZE,
    rock: float = 110.0,
    water: float = 250.0,
    subject: float = 60.0,
    area_frac: float = 0.10,
    cy_frac: float = 0.26,
) -> np.ndarray:
    """The #361 composition: a dark subject *high* in the frame, water *low*.

    :func:`flat_sky_frame` cannot catch G-10, because the region that must lose
    there is **flat** and the texture channel alone disposes of it.  The region
    that must lose here is bright, finely textured, four times more
    colour-distinct than the subject and — the only thing that separates them —
    it *runs off the bottom edge of the picture*, the way a river does and the
    way an aircraft does not.

    The geometry is the point.  ``video/seed_1x1_drone.png``, the still the
    previous regression used, puts its whitewater within ±0.12 of the airframe's
    own centroid, so ``centroid_y < 0.60`` passed whichever of the two the
    detector had picked and the test proved nothing.  Here the subject sits at
    y = 0.26 and the water's centre of mass at y = 0.75 — **0.49 apart**, four
    times the tolerance — so the assertion can only pass by naming the right one.

    Measured: the detector answers 9.26 % at (0.498, 0.260); with the enclosure
    channel disabled it answers 5.10 % at (0.502, 0.751).
    """
    frame = rock + (_fine_texture(size, size, seed=11) - 128.0) / 127.0 * 15.0
    foam = water + (_fine_texture(size, size, seed=23) - 128.0) / 127.0 * 6.0
    band = np.zeros((size, size), np.uint8)
    cv2.rectangle(band, (int(0.30 * size), int(0.60 * size)), (int(0.70 * size), size), 1, -1)
    frame[band == 1] = foam[band == 1]
    body = subject + (_fine_texture(size, size, seed=29) - 128.0) / 127.0 * 20.0
    disc = np.zeros((size, size), np.uint8)
    radius = round(math.sqrt(area_frac * size * size / math.pi))
    cv2.circle(disc, (size // 2, int(cy_frac * size)), radius, 1, -1)
    frame[disc == 1] = body[disc == 1]
    return np.clip(frame, 0, 255).astype(np.uint8)


def thin_limbed_frame(
    size: int = FRAME_SIZE,
    hub: float = 0.09,
    limb: float = 0.021,
    reach: float = 0.30,
    level: float = 215.0,
) -> np.ndarray:
    """A textured hub with four arms thinner than the morphological kernel.

    The #348 composition in synthetic form: the opening that deletes speckle
    deletes a quadcopter's arms just as efficiently, leaving a core far too small
    to clear the detection floor for a subject that is plainly there.  ``limb``
    is 5 px at the 256² analysis size against a 9 px kernel, so the arms cannot
    survive the erosion and the hub can.

    Measured: the opened core reads 0.73 % — inside the rescue window
    ``[0.44 %, 0.80 %)`` — and the rescued extent 3.97 %, a 5.5× difference.
    """
    field = np.clip(45.0 + (_fine_texture(size, size, seed=3) - 128.0) / 127.0 * 30.0, 0, 255)
    body = level + (_fine_texture(size, size, seed=17) - 128.0) / 127.0 * 26.0
    mask = np.zeros((size, size), np.uint8)
    centre = size // 2
    cv2.circle(mask, (centre, centre), round(hub * size / 2), 1, -1)
    half = max(1, round(limb * size / 2))
    span = round(reach * size)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(mask, (centre, centre), (centre + dx * span, centre + dy * span), 1, 2 * half)
    out = field.copy()
    out[mask == 1] = body[mask == 1]
    return np.clip(out, 0, 255).astype(np.uint8)


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


def _loudest_subjectless() -> float:
    """Largest "subject" the detector can be provoked into finding in an empty frame.

    The false-positive half of the #341 margin, in one number: eight seeds of
    uniform texture plus the flat and rim-damped compositions, which is the same
    population ``test_subject_and_subjectless_frames_are_separated`` measures.
    """
    return max(
        [csq.anchor_stats(subjectless_frame(seed=s))["area"] for s in range(1, 9)]
        + [csq.anchor_stats(flat_frame())["area"], csq.anchor_stats(rim_damped_frame(0.18))["area"]]
    )


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


def _fisheye_outflow_frames(n: int = N_FRAMES, size: int = FRAME_SIZE, zoom: float = 1.04) -> list[np.ndarray]:
    """Radial-outflow frames masked to an inscribed circle (outside is pure black).

    A circular fisheye at rest paints its corners black, so the synthetic
    fisheye fixture blacks out everything outside the inscribed circle.  The
    #322 bug was that those black corners are zero-flow pixels that dragged
    the outer-ring mean toward zero on genuinely forward clips.
    """
    frames = radial_outflow_frames(n, size, zoom)
    cy = cx = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    inside = np.hypot(xs - cx, ys - cy) <= size / 2.0
    return [np.where(inside, f, 0).astype(np.uint8) for f in frames]


def _fisheye_static_frames(n: int = N_FRAMES, size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Static frames masked to an inscribed circle."""
    frames = static_frames(n, size)
    cy = cx = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    inside = np.hypot(xs - cx, ys - cy) <= size / 2.0
    return [np.where(inside, f, 0).astype(np.uint8) for f in frames]


def test_circular_fisheye_outflow_passes_forward_motion() -> None:
    """Acceptance (#322): a circular-fisheye forward clip must PASS.

    Without the imaging-circle detection and threshold adaptation the black
    corners made the outer-ring mean collapse, and the clip failed despite
    being obviously forward-moving.
    """
    frames = _fisheye_outflow_frames()
    circle = csq.estimate_imaging_circle(frames[0])
    assert circle is not None, "the fisheye fixture must trigger circle detection"
    stats = [csq.radial_flow_stats(frames[i], frames[i + 1], circle_frac=circle) for i in range(len(frames) - 1)]
    rate = csq.DEFAULT_MIN_RADIAL_RATE * csq.FISHEYE_MIN_RADIAL_FACTOR
    result = csq.check_forward_motion(stats, min_radial_rate=rate, circle_frac=circle)
    assert result.status == csq.STATUS_PASS, f"fisheye outflow must PASS: {result.detail}"
    assert result.measured["imaging_circle_frac"] is not None


def test_circular_fisheye_static_still_fails() -> None:
    """Acceptance (#322): a circular-fisheye locked-off clip must still FAIL.

    The gate narrows for fisheye; it does not open.  The static fisheye
    fixture must not be waved through by the adapted threshold.
    """
    frames = _fisheye_static_frames()
    circle = csq.estimate_imaging_circle(frames[0])
    assert circle is not None
    stats = [csq.radial_flow_stats(frames[i], frames[i + 1], circle_frac=circle) for i in range(len(frames) - 1)]
    rate = csq.DEFAULT_MIN_RADIAL_RATE * csq.FISHEYE_MIN_RADIAL_FACTOR
    result = csq.check_forward_motion(stats, min_radial_rate=rate, circle_frac=circle)
    assert result.status == csq.STATUS_FAIL, f"fisheye static must FAIL: {result.detail}"
    assert "静态" in result.detail + result.advice


def test_rectangular_source_has_no_imaging_circle() -> None:
    """Acceptance (#322): a rectangular source must not be masked.

    Patchy darkness or vignetting must not produce a false circle — the
    detection requires a genuine step, so a rectilinear source returns None
    and the existing behaviour is preserved exactly.
    """
    assert csq.estimate_imaging_circle(radial_outflow_frames()[0]) is None
    assert csq.estimate_imaging_circle(static_frames()[0]) is None
    assert csq.estimate_imaging_circle(panning_frames()[0]) is None


def test_rectangular_verdicts_are_unchanged_by_fisheye_logic() -> None:
    """Acceptance (#322): the three rectilinear verdicts are byte-identical."""
    # Outflow still PASS, static still FAIL, pan still FAIL — and no imaging
    # circle key appears in measured, confirming the fisheye path was not taken.
    good = csq.check_forward_motion(_stats_for(radial_outflow_frames()))
    assert good.status == csq.STATUS_PASS
    assert "imaging_circle_frac" not in good.measured

    still = csq.check_forward_motion(_stats_for(static_frames()))
    assert still.status == csq.STATUS_FAIL
    assert "imaging_circle_frac" not in still.measured

    pan = csq.check_forward_motion(_stats_for(panning_frames()))
    assert pan.status == csq.STATUS_FAIL
    assert "imaging_circle_frac" not in pan.measured


def test_slow_forward_does_not_say_not_positive() -> None:
    """Acceptance (#322): a positive-but-slow radial rate must not say 「不为正」.

    The old text said "径向分量不为正" while printing a positive number beside
    it — the failure must name the real cause (too slow), and the advice must
    differ from a lateral pan's.
    """
    radius = 100.0
    rate = csq.DEFAULT_MIN_RADIAL_RATE * 0.7  # positive but below threshold
    stats = [
        csq.RadialStats(
            radial_mean=rate * radius,
            inner_mean=0.3 * rate * radius,
            outer_mean=1.5 * rate * radius,
            flow_mean=rate * radius * 1.5,  # above static gate; radial dominates → slow, not pan
            radius=radius,
        )
    ] * 3
    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_FAIL
    assert "不为正" not in result.detail
    assert result.advice == csq.FORWARD_SLOW_ADVICE
    assert result.advice != csq.FORWARD_FAIL_ADVICE


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
    hair without any test noticing.  This pins the actual daylight between the
    two populations — the property #341 bought — in **absolute** measured area
    rather than as a multiple of :data:`ANCHOR_MIN_AREA`, because #345 moved that
    threshold and a margin expressed in units of the thing being moved would have
    silently rescaled with it instead of catching the change.

    Measured today: true subjects 11.1 / 40.2 / 11.3 %, false ones at most
    0.17 %, i.e. ~64× apart.  The bounds below are deliberately slacker than the
    measurements so that ordinary detector noise does not flake the suite.
    """
    with_subject = [
        csq.anchor_stats(subject_frame(0.11))["area"],
        csq.anchor_stats(subject_frame(0.40))["area"],
        csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))["area"],
    ]
    without = [csq.anchor_stats(subjectless_frame(seed=s))["area"] for s in range(1, 9)]
    without += [csq.anchor_stats(flat_frame())["area"], csq.anchor_stats(rim_damped_frame(0.18))["area"]]

    assert min(with_subject) > 0.05, f"true subjects too quiet: {with_subject}"
    assert max(without) < 0.005, f"false subjects too loud: {without}"
    assert min(with_subject) / max(max(without), 1e-9) > 15.0, (
        f"separation collapsed: {min(with_subject)} vs {max(without)}"
    )


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
# anchor — the sky is not the subject (G-4 #343)
# ---------------------------------------------------------------------------


def _blind_texture_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the texture channel: every pixel claims to be equally structured.

    This is the mutation the fix is measured against.  It leaves the rest of the
    pipeline — Achanta, Otsu, the morphology, the centre prior, the density
    scoring — completely untouched, so anything it breaks is attributable to the
    texture channel and to nothing else.
    """
    monkeypatch.setattr(
        csq,
        "texture_gate",
        lambda frame, *_a, **_kw: np.ones_like(
            csq._fit_for_analysis(csq._as_gray(frame), csq.ANCHOR_ANALYSIS_MAX_DIM), dtype=np.float32
        ),
    )


def _blind_enclosure_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the enclosure channel: every pixel claims to be equally enclosed.

    The #361 mutation, built to the same rule as :func:`_blind_texture_gate` —
    Achanta, Otsu, the morphology, the centre prior, the density scoring and the
    texture channel are all left exactly as they are, so whatever this breaks is
    attributable to the enclosure channel and to nothing else.
    """
    monkeypatch.setattr(
        csq,
        "enclosure_gate",
        lambda frame, *_a, **_kw: np.ones_like(
            csq._fit_for_analysis(csq._as_gray(frame), csq.ANCHOR_ANALYSIS_MAX_DIM), dtype=np.float32
        ),
    )


def test_a_flat_bright_sky_does_not_pose_as_the_subject() -> None:
    """Acceptance (#343): the anchor of a sky-over-ground frame is the subject.

    The synthetic stand-in for the owner's canyon keyframes, and the one that
    runs in CI.  A blown-out sky band is the most colour-distinct region in the
    picture and it is *four times the subject's size*, so both of the old
    detector's instincts — "most distinct" and "biggest" — point at it.  The
    detector must nonetheless come back with the small textured thing in the
    middle.
    """
    stats = csq.anchor_stats(flat_sky_frame())

    assert stats["found"] is True, stats
    assert stats["centroid_x"] == pytest.approx(0.5, abs=0.06), f"not the centred subject: {stats}"
    assert stats["centroid_y"] == pytest.approx(0.5, abs=0.06), f"looking at the sky band: {stats}"
    assert stats["area"] == pytest.approx(0.11, abs=0.05), f"the sky leaked into the region: {stats}"


def test_removing_both_structure_channels_puts_the_sky_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation check (#343, restated for #361): the sky fix is load-bearing.

    It used to be enough to blind :func:`~scripts.check_source_quality.texture_gate`
    on its own, and #361 is why that stopped being true: a blown-out sky band
    that runs off the top edge is not merely structureless, it is also
    **unenclosed**, so the channel added in #361 declines to promote it for a
    second and independent reason.  Measured on this fixture, blinding either
    channel alone still returns the subject (y = 0.526 with texture blind,
    y = 0.498 with enclosure blind).

    Asserting "blind one channel and it breaks" would therefore now fail for a
    *good* reason, and weakening it to "blind one channel and something changes"
    would assert nothing.  Blinding both restores the original bug exactly — the
    21.7 %-of-frame sky band at y = 0.11 that :func:`flat_sky_frame`'s docstring
    was written around — which keeps
    ``test_a_flat_bright_sky_does_not_pose_as_the_subject`` from quietly passing
    for the wrong reason.  Each channel's *own* necessity is pinned separately:
    texture by ``test_removing_the_texture_channel_collapses_the_separation``,
    enclosure by ``test_removing_the_enclosure_channel_puts_the_anchor_in_the_water``.
    """
    frame = flat_sky_frame()
    gated = csq.anchor_stats(frame)

    _blind_texture_gate(monkeypatch)
    _blind_enclosure_gate(monkeypatch)
    ungated = csq.anchor_stats(frame)

    assert gated["centroid_y"] == pytest.approx(0.5, abs=0.06)
    assert ungated["centroid_y"] < 0.25, (
        f"the structure channels changed nothing — the mutation must move the anchor onto the sky: {ungated}"
    )
    assert ungated["area"] > gated["area"] * 1.5, (
        f"the ungated region must swell to swallow the sky: {ungated} vs {gated}"
    )


def test_removing_the_texture_channel_collapses_the_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation check (#343): what the texture channel is still uniquely for.

    Since #361 the sky band is held off by two channels, so the texture gate's
    necessity has to be demonstrated where it is *not* shadowed: the #341 margin
    between a frame with a subject and a frame without one.  An evenly-textured
    frame is full of speckle that is both distinct and — being blob texture —
    thoroughly enclosed; structure is the only one of the three questions that
    answers "no" to it.  Measured, blinding this channel takes the loudest
    subjectless fixture from 0.17 % to 1.46 % and the margin from 64.1× to 7.6×,
    i.e. straight through :data:`~scripts.check_source_quality.ANCHOR_MIN_AREA`.
    """
    quietest_true = min(
        csq.anchor_stats(subject_frame(0.11))["area"],
        csq.anchor_stats(subject_frame(0.40))["area"],
        csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))["area"],
    )
    gated = _loudest_subjectless()

    _blind_texture_gate(monkeypatch)
    ungated = _loudest_subjectless()

    assert gated < csq.ANCHOR_MIN_AREA, f"the gated detector must find nothing in an empty frame: {gated}"
    assert ungated > csq.ANCHOR_MIN_AREA, (
        f"the texture channel changed nothing — without it an empty frame must grow an anchor: {ungated}"
    )
    assert quietest_true / ungated < 15.0, (
        f"the #341 margin must collapse without the texture channel: {quietest_true} vs {ungated}"
    )


def test_the_texture_channel_widens_the_subject_separation() -> None:
    """The gate must not buy the sky fix by blurring the #341 margin.

    #341's contract is that a frame with a subject and a frame without one are
    separated by a wide margin rather than a hair.  Gating widened it (≈50×
    against the ungated ≈32×) because speckle in an evenly-textured frame is
    exactly what a structure measure declines to promote; #361's enclosure gate
    gave part of that back (≈41×) because it judges scenery by where it leaks to
    and not by how bright it is; G-11's wider window took it to **64.1×**, which
    is the whole reason that card moved the constant.

    **55× is the floor, and it is the mutation check for G-11's retune.**  The
    number is chosen to sit above the 40.6× the detector measured at the old
    window of 21 and below today's 64.1×, so reverting
    :data:`~scripts.check_source_quality.ANCHOR_TEXTURE_WINDOW` turns this test
    red rather than leaving it quietly passing on a worse constant.  Every
    window in G-11's scan that clears it does so by a margin (21 → 40.6×,
    25 → 38.0×, 31 → 64.1×, 35 → 44.8×): 31 is the only one that passes.
    """
    with_subject = min(
        csq.anchor_stats(subject_frame(0.11))["area"],
        csq.anchor_stats(subject_frame(0.40))["area"],
        csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))["area"],
    )
    without = _loudest_subjectless()
    assert with_subject / max(without, 1e-9) > 55.0, f"separation regressed: {with_subject} vs {without}"


def test_texture_gate_reads_flat_as_flat_and_textured_as_textured() -> None:
    """The channel itself: bright-but-flat scores low, textured saturates at 1.

    Asserted on the *sky band* rather than on a synthetic constant patch because
    "flat" in real footage still carries sensor noise, and a gate that only
    recognises mathematically-zero variance would be useless on a photograph.
    """
    gate = csq.texture_gate(flat_sky_frame())
    height = gate.shape[0]
    sky = gate[: int(height * 0.18), :]
    ground = gate[int(height * 0.75) :, :]

    assert gate.min() >= 0.0
    assert gate.max() <= 1.0
    assert gate.max() == pytest.approx(1.0), "a textured frame must saturate the gate somewhere"
    assert sky.mean() < ground.mean() * 0.75, f"flat sky {sky.mean():.3f} vs textured ground {ground.mean():.3f}"


def test_texture_gate_rejects_a_nonsensical_window() -> None:
    with pytest.raises(ValueError, match="window must be"):
        csq.texture_gate(subject_frame(), window=0)


def test_anchor_reports_where_it_found_the_subject() -> None:
    """``centroid_*`` exists so a regression can assert *what* was found.

    The #343 bug reported a perfectly plausible area and eccentricity while
    pointing at the clouds; without a position in the payload no test could tell
    that apart from a correct answer.
    """
    centred = csq.anchor_stats(subject_frame(0.11))
    off_centre = csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))

    assert centred["centroid_x"] == pytest.approx(0.5, abs=0.05)
    assert centred["centroid_y"] == pytest.approx(0.5, abs=0.05)
    assert off_centre["centroid_x"] == pytest.approx(0.8, abs=0.05)
    assert csq.anchor_stats(flat_frame())["centroid_x"] == pytest.approx(0.5)


@pytest.mark.skipif(not CANYON_STILL.is_file(), reason=_missing_assets(CANYON_STILL))
def test_the_canyon_still_anchors_on_the_aircraft_not_the_backdrop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance (#343/#361) on a real photograph — re-pointed by #366.

    This slot held ``Gemini_v1.jpg``, the 1024² frame #343 was filed on, whose
    cloud band along the top edge the old detector reported as the subject at
    (0.49, 0.06).  The owner deleted that file, so the pin was re-measured on
    ``video/seed_1x1_drone.png``: the same composition at 2048² — overcast sky
    along the top edge, red-rock walls, whitewater down the middle, a gray
    quadcopter at **(0.506, 0.441)** measuring 1.93 %.

    Two things are asserted and the second is why this is not decoration.

    The centroid has to clear the sky *and* the river.  Only the lower bound is
    new: ``test_canyon_still_finds_the_aircraft_not_the_whitewater`` bounds this
    same frame from above (y < 0.60) and nothing bounds it from below, so a
    detector that went back to answering the cloud band at y = 0.06 would pass
    every other assertion in this file that touches a real photograph.

    Then the frame is held to a mutation, because a surviving margin is worth
    more than a number: blind the enclosure channel and the anchor drops into the
    rapids at (0.527, 0.758), 10.80 % of the picture — the #361 bug reproduced on
    the asset rather than on a fixture.  That is coverage ``seed_v6.png`` can no
    longer give, since G-11 (#364) widened the texture window far enough that
    blinding a channel there relocates the anchor to the canyon wall instead (see
    ``test_removing_the_enclosure_channel_takes_the_seed_still_off_the_airframe``).

    Neither surviving still can reproduce the *cloud* half of #343 under any
    mutation — blinding both structure channels gives (0.510, 0.373) here and
    (0.115, 0.435) on ``seed_v6.png``, both well clear of the sky — so that half
    stays pinned where it is demonstrable, on :func:`flat_sky_frame`, in CI.
    """
    frame = csq.read_still(CANYON_STILL)
    gated = csq.anchor_stats(frame)

    assert gated["found"], f"the aircraft must be found at all: {gated}"
    assert gated["centroid_x"] == pytest.approx(0.51, abs=0.12), f"not the aircraft: {gated}"
    assert 0.20 <= gated["centroid_y"] <= 0.60, f"on the sky band or in the river: {gated}"

    _blind_enclosure_gate(monkeypatch)
    ungated = csq.anchor_stats(frame)

    assert ungated["centroid_y"] > 0.60, (
        f"the enclosure channel changed nothing — the mutation must land in the rapids: {ungated}"
    )


@pytest.mark.skipif(not SEED_STILL.is_file(), reason=_missing_assets(SEED_STILL))
def test_a_small_real_subject_is_small_not_absent() -> None:
    """Acceptance (#345): the drone reads as *small*, never as *absent*.

    This is the whole card in one assertion pair.  The quadcopter measures
    **2.41 %** of ``video/seed_v6.png`` — comfortably a subject, nowhere near the
    8–15 % reference band — so the only correct report is 「面积偏小」 with the
    number in it.  Before the floor moved, a frame like this produced 「画面里
    没有锚点」, which is a different and much more expensive instruction to hand
    an operator.

    Re-pointed by #366: the card was measured on ``Gemini_v1.jpg`` at 1.30 % and
    that file is gone.  ``seed_v6.png`` was chosen over ``seed_1x1_drone.png``
    because it is the better-centred of the two (offset 0.049 against 0.083),
    which keeps the verdict a pure statement about *size* — 「面积偏小」 and
    nothing else — exactly as it was on the frame the card was filed on.

    Both halves are asserted: the phrase that must appear *and* the phrase that
    must not, because a detector that started reporting some other blob at a
    plausible size would satisfy the first on its own.
    """
    result = csq.check_anchor(_anchor_for([csq.read_still(SEED_STILL)]))

    assert result.status == csq.STATUS_WARN, f"{result.status}: {result.detail}"
    assert result.measured["detected_frames"] == 1, f"the drone must be found at all: {result.measured}"
    assert "面积偏小" in result.detail, result.detail
    assert "没有锚点" not in result.detail + result.advice, f"still claiming the frame is empty: {result.detail}"
    assert csq.ANCHOR_MIN_AREA < result.measured["area"] < csq.ANCHOR_AREA_MIN, (
        f"the drone must land between the detection floor and the quality band: {result.measured}"
    )
    assert result.measured["area"] == pytest.approx(0.024, abs=0.004), result.measured


@pytest.mark.skipif(
    not (SEED_STILL.is_file() and CANYON_STILL.is_file()),
    reason=_missing_assets(SEED_STILL, CANYON_STILL),
)
def test_the_detection_floor_sits_between_noise_and_the_smallest_real_subject() -> None:
    """The floor has to clear the loudest false positive and duck the quietest true one.

    #345 lowered it, so the question "did it go too far" needs an answer that is
    measured rather than asserted.  Headroom on both sides, across every real
    still this repo can reach and the subjectless fixtures #341 established:

    * loudest subjectless fixture  0.17 %   → floor is ~4.7× above it
    * floor                        0.80 %
    * smallest real subject        1.93 %   → ~2.4× above the floor

    Written as ratios so the margin, not the constant, is what is pinned.  The
    lower figure has moved twice: 0.22 % before #361, 0.27 % after it (the
    enclosure channel costs headroom on that side because it does not care how
    bright a speck is), and 0.17 % since G-11 widened
    :data:`~scripts.check_source_quality.ANCHOR_TEXTURE_WINDOW`, which is the
    quietest it has ever been.

    The upper figure moved for a different reason.  #345 measured it on
    ``Gemini_v1.jpg`` at 1.30 % and that file is gone (#366), so "smallest real
    subject" is now taken as the **minimum over both surviving stills** —
    ``seed_1x1_drone.png`` at 1.93 % against ``seed_v6.png`` at 2.41 % — rather
    than over one frame.  Adding a third still can only tighten this, never
    loosen it, which is the direction a headroom claim should be able to move.
    """
    noise = _loudest_subjectless()
    smallest_real = min(
        csq.anchor_stats(csq.read_still(SEED_STILL))["area"],
        csq.anchor_stats(csq.read_still(CANYON_STILL))["area"],
    )

    assert noise * 2.5 < csq.ANCHOR_MIN_AREA, f"floor too close to the noise: {noise}"
    assert smallest_real > csq.ANCHOR_MIN_AREA * 1.25, f"floor too close to the real subject: {smallest_real}"
    assert csq.ANCHOR_MIN_AREA < csq.ANCHOR_AREA_MIN, "the detection floor must stay below the quality band"


@pytest.mark.skipif(not CANYON_STILL.is_file(), reason=_missing_assets(CANYON_STILL))
def test_canyon_still_finds_the_aircraft_not_the_whitewater() -> None:
    """Acceptance (#343), second frame: the bright foam must not win either.

    ``seed_1x1_drone.png`` is 2048² with the aircraft at about (0.50, 0.48) and a
    stretch of whitewater filling the bottom of the gorge.  The old detector
    returned the foam at (0.51, 0.79); anything below y≈0.6 in this frame is
    river, not subject.  It now reads 1.93 % at (0.506, 0.441).

    **This test is weak on its own and #361 is why**: the detector's answer
    before that card was (0.510, 0.371) — the bright bend of the river directly
    above the fuselage — and both assertions below passed on it, because in
    *this* composition the water that must lose sits within ±0.12 of the
    airframe that must win.  A regression that can be satisfied by either
    candidate is a smoke test, not an acceptance test.  The discriminating
    version is ``test_the_gorge_fixture_finds_the_subject_not_the_water``, whose
    two candidates are 0.49 apart.  Since #366 this frame also carries
    ``test_the_canyon_still_anchors_on_the_aircraft_not_the_backdrop``, which
    bounds the centroid from *below* (the sky band this test would happily
    accept) and blinds the enclosure channel to prove the water is still
    reachable on the photograph itself.
    """
    stats = csq.anchor_stats(csq.read_still(CANYON_STILL))

    assert stats["centroid_x"] == pytest.approx(0.50, abs=0.12), f"not the aircraft: {stats}"
    assert stats["centroid_y"] < 0.60, f"still on the whitewater: {stats}"


def test_a_sub_floor_core_is_rescued_to_its_full_extent() -> None:
    """Acceptance (#348): a thin-limbed subject measures itself, not its skeleton.

    The opening eats a quadcopter's arms exactly as it eats specks, leaving a
    core under the detection floor for a subject that is plainly there.  The
    rescue re-grows the winning core to its pre-morphology extent; these
    assertions pin the rescued verdict (found, area above the floor but below
    the quality band, centroid unmoved) and, via the ``rescue=False`` pair, that
    the rescue is load-bearing rather than decorative: switch it off and the
    frame is back to 「没有锚点」.

    Asserted on :func:`thin_limbed_frame` rather than on ``CANYON_STILL``, which
    is where #348 measured it.  The gap the rescue was built to close is a
    property of the *morphology*, not of that photograph, and #361 stopped that
    photograph from demonstrating it: with the enclosure channel the detector
    resolves the airframe well enough that its opened core clears the floor on
    its own (1.93 %), so the real still can no longer show the rescue doing
    anything.  A synthetic composition can, always, and in CI.
    """
    frame = thin_limbed_frame()
    rescued = csq.anchor_stats(frame)
    bare = csq.anchor_stats(frame, rescue=False)

    assert rescued["found"], f"the subject must be found at all: {rescued}"
    assert csq.ANCHOR_MIN_AREA < rescued["area"] < csq.ANCHOR_AREA_MIN, (
        f"the subject must land between the detection floor and the quality band: {rescued}"
    )
    assert rescued["area"] >= 2.0 * bare["area"], (
        f"the rescue must more than double the eroded skeleton's area: {rescued} vs {bare}"
    )
    assert csq.ANCHOR_RESCUE_CORE_FLOOR * csq.ANCHOR_MIN_AREA <= bare["area"] < csq.ANCHOR_MIN_AREA, (
        f"the fixture must place its core inside the rescue window or it proves nothing: {bare}"
    )
    assert rescued["centroid_x"] == pytest.approx(0.5, abs=0.06), f"off the subject: {rescued}"
    assert rescued["centroid_y"] == pytest.approx(0.5, abs=0.06), f"off the subject: {rescued}"
    assert not bare["found"], f"disabling the rescue must restore the #348 bug: {bare}"


@pytest.mark.skipif(not CANYON_STILL.is_file(), reason=_missing_assets(CANYON_STILL))
def test_the_canyon_still_no_longer_needs_the_rescue() -> None:
    """The #348 gap, re-measured on the frame it was found on, after #361.

    Kept as a *record* rather than deleted: #348's numbers (a 0.53 % core for a
    1.38 % subject) are quoted throughout this file and the next reader needs to
    know why they no longer reproduce.  The enclosure channel resolves the
    airframe directly, so the core and the rescued extent are now the same
    region and the rescue is a no-op here.
    """
    frame = csq.read_still(CANYON_STILL)

    assert csq.anchor_stats(frame) == csq.anchor_stats(frame, rescue=False), (
        "the canyon still is expected to clear the floor without the rescue since #361"
    )


@pytest.mark.skipif(not CANYON_STILL.is_file(), reason=_missing_assets(CANYON_STILL))
def test_the_rescue_is_size_invariant() -> None:
    """Acceptance (#348): the same still at 1024²/2048²/2880² measures the same.

    All anchor analysis runs at :data:`check_source_quality.ANCHOR_ANALYSIS_MAX_DIM`,
    so the rescued area must be essentially identical regardless of source
    resolution — the card's bound is a 30 % relative spread, measured <1 %.
    """
    frame = csq.read_still(CANYON_STILL)
    areas = []
    for size in (1024, 2048, 2880):
        img = frame if size == frame.shape[0] else cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        stats = csq.anchor_stats(img)
        assert stats["found"], f"{size}² lost the subject: {stats}"
        areas.append(stats["area"])

    spread = (max(areas) - min(areas)) / min(areas)
    assert spread <= 0.30, f"size-dependent measurement: {areas} (spread {spread:.1%})"


@pytest.mark.skipif(not SEED_STILL.is_file(), reason=_missing_assets(SEED_STILL))
def test_the_rescue_leaves_established_and_noise_cores_alone() -> None:
    """The rescue is gated on both sides and must stay that way.

    A core that already clears the floor (``video/seed_v6.png`` at 2.41 %) must
    measure *identically* with and without the rescue — the #345 pins depend on
    it.  And the rescue must not fabricate anchors out of specks: every
    subjectless fixture's sub-floor core stays sub-floor, because cores under
    ``ANCHOR_RESCUE_CORE_FLOOR`` × the detection floor are not re-grown.

    Re-pointed by #366, which read the owner's deleted keyframe at 1.30 %.
    Deliberately a *different* still from
    ``test_the_canyon_still_no_longer_needs_the_rescue``, which makes the same
    no-op claim on ``seed_1x1_drone.png`` at 1.93 %: the gate is an inequality
    against the floor, so pinning it at one measured area proves less than
    pinning it at two.
    """
    frame = csq.read_still(SEED_STILL)
    assert csq.anchor_stats(frame) == csq.anchor_stats(frame, rescue=False), (
        "the rescue must not touch a core that already clears the floor"
    )

    for seed in range(1, 9):
        stats = csq.anchor_stats(subjectless_frame(seed=seed))
        assert not stats["found"], f"seed {seed} grew an anchor out of specks: {stats}"


# ---------------------------------------------------------------------------
# anchor — a speck must not veto the subject (G-9 #356)
# ---------------------------------------------------------------------------


def glinted_frame(
    area_frac: float = 0.11,
    glint_radius: int = 7,
    glint_cx: float = 0.25,
    glint_cy: float = 0.42,
    size: int = FRAME_SIZE,
) -> np.ndarray:
    """:func:`subject_frame` with one tiny, maximally distinct specular glint.

    The #356 composition, reduced to its essentials.  The glint is a 7 px disc —
    **0.20 % of the frame**, a quarter of the detection floor — of blown-out,
    finely textured white sitting on the dark field beside the subject.  Nothing
    about it is reportable: it is far too small to be an anchor and too small
    even to be rescued.  But it is *pure peak*, and a mean-density score
    converges on a region's peak as the region shrinks, so it out-scores an
    11 %-of-frame subject by roughly 2:1 and — under winner-takes-all — takes
    the frame's verdict with it.

    Real footage is full of these: a rotor highlight, a sun glitter on water, a
    compression ring around a blown highlight.  What makes them poisonous is not
    that they exist but that their *survival of the opening* is a coin flip
    decided by resampling, so the same picture answers two different things at
    two different resolutions (see
    ``test_an_unreportable_speck_does_not_move_the_answer_at_any_size``).
    """
    frame = subject_frame(area_frac).astype(np.float32)
    height, width = frame.shape
    hot = 255.0 + (_fine_texture(height, width, seed=23) - 128.0) / 127.0 * 45.0
    glint = np.zeros((height, width), np.uint8)
    cv2.circle(glint, (int(width * glint_cx), int(height * glint_cy)), glint_radius, 1, -1)
    frame[glint == 1] = hot[glint == 1]
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_a_speck_too_small_to_report_cannot_veto_the_subject() -> None:
    """Acceptance (#356): the headline failure, in one frame.

    Painting a 0.20 % glint next to an 11 % subject used to delete the subject
    from the report altogether — ``found=False, area=0.20 %`` — because the
    glint won the density election outright and was then, inevitably, judged
    too small to be an anchor.  A region that cannot be *reported* as the
    anchor has no business *choosing* it.

    Asserted against the same frame without the glint rather than against a
    constant, so what is pinned is "the speck changed nothing", which is the
    actual claim.
    """
    clean = csq.anchor_stats(subject_frame(0.11))
    glinted = csq.anchor_stats(glinted_frame(0.11))

    assert clean["found"], f"the fixture itself lost its subject: {clean}"
    assert glinted["found"], f"a 0.2 % speck deleted an 11 % subject: {glinted}"
    assert glinted["area"] == pytest.approx(clean["area"], rel=0.10), (
        f"the speck moved the measured area: {glinted} vs {clean}"
    )
    assert glinted["centroid_x"] == pytest.approx(0.5, abs=0.06), f"looking at the glint: {glinted}"
    assert glinted["centroid_y"] == pytest.approx(0.5, abs=0.06), f"looking at the glint: {glinted}"


@pytest.mark.parametrize("size", [240, 480, 960, 1920])
def test_an_unreportable_speck_does_not_move_the_answer_at_any_size(size: int) -> None:
    """Acceptance (#356): the *instability*, which is the bug's real shape.

    The card was filed as "same scene, two resolutions, 8× apart", and the
    tempting reading is that the morphology is resolution-dependent.  It is not
    — every anchor measurement is taken at
    :data:`check_source_quality.ANCHOR_ANALYSIS_MAX_DIM`, so the kernel sees the
    same number of pixels either way.  What changes with resolution is whether
    the resampler leaves a speck big enough to survive the opening *as its own
    component*, and that is a coin flip: measured on ``video/seed_v6.png``, the
    old detector answered 10.7 % at 2048², 1024² and 256² and **0.12 %** at 960²
    and 512².  Bimodal, not graded.

    So the regression renders one frame at four sizes and demands one answer.
    """
    frame = cv2.resize(glinted_frame(0.11), (size, size), interpolation=cv2.INTER_AREA)
    reference = csq.anchor_stats(subject_frame(0.11))
    stats = csq.anchor_stats(frame)

    assert stats["found"], f"{size}² lost the subject to the glint: {stats}"
    assert stats["area"] == pytest.approx(reference["area"], rel=0.30), (
        f"{size}² disagrees with the un-glinted frame: {stats} vs {reference}"
    )


def test_the_candidacy_gate_does_not_manufacture_an_anchor() -> None:
    """The other side of #356: gating the election must not invent subjects.

    Restricting the election to credible-sized regions is only safe if a frame
    with *no* credible region still answers 「没有锚点」.  Two things are checked
    on every subjectless fixture: that the fallback branch is the one running
    (no component reaches the credibility floor at all, so the gate cannot be
    what decides these frames), and that the verdict is still "nothing here".

    ``empty_centre_frame`` is the one that matters and the reason
    :data:`check_source_quality.ANCHOR_RESCUE_CORE_FLOOR` moved: its busiest rim
    blob is 0.358 %, the loudest subjectless core on record, and at the old 0.4×
    gate the rescue grew it to 0.97 % — an anchor conjured out of a textured
    rim.  Before #356 the frame was saved only by accident, because a smaller,
    denser speck happened to win the election and fall under the floor.
    """
    floor = csq.ANCHOR_RESCUE_CORE_FLOOR * csq.ANCHOR_MIN_AREA
    frames = {f"subjectless seed={s}": subjectless_frame(seed=s) for s in range(1, 9)}
    frames["flat gray + noise"] = flat_frame()
    frames["empty centre, busy rim"] = empty_centre_frame()
    frames["rim damped"] = rim_damped_frame(0.18)

    for name, frame in frames.items():
        assert not csq.anchor_stats(frame)["found"], f"{name} grew an anchor: {csq.anchor_stats(frame)}"
        saliency = csq.structured_saliency(frame)
        _, raw = cv2.threshold(
            np.clip(saliency * 255.0, 0, 255).astype(np.uint8),
            0,
            1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        element = np.ones((csq.ANCHOR_MORPH_KERNEL, csq.ANCHOR_MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, element)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        biggest = max(
            (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)),
            default=0,
        ) / float(mask.size)
        assert biggest < floor, f"{name} now has a credible candidate ({biggest:.4%} >= {floor:.4%})"


@pytest.mark.skipif(not SEED_STILL.is_file(), reason=_missing_assets(SEED_STILL))
def test_the_seed_still_measures_the_same_at_every_resolution() -> None:
    """Acceptance (#356): ``video/seed_v6.png`` reads one number, not two.

    The card's evidence, reproduced without needing the clip: the same 2048²
    still resampled to 1024²/960²/512² and read both in colour (the still path)
    and in luminance (what the video sampler hands over, ``-pix_fmt gray``).
    Before the candidacy gate those readings were 0.12 % or ~11 % with nothing
    in between — an 88× spread.  They are now 2.35–2.46 %, a **4.9 %** spread,
    against the card's 30 % bound.  (The absolute share fell from ~11 % to ~2 %
    when #361 moved the anchor off the rapids and onto the airframe, which is a
    much smaller thing; G-11's wider texture window then took the spread itself
    from 12.2 % to 4.9 %.)
    """
    still = csq.read_still(SEED_STILL)
    areas = {}
    for size in (2048, 1024, 960, 512):
        img = still if size == still.shape[0] else cv2.resize(still, (size, size), interpolation=cv2.INTER_AREA)
        for tag, frame in (("bgr", img), ("gray", csq._as_gray(img))):
            stats = csq.anchor_stats(frame)
            assert stats["found"], f"{size}² {tag} lost the subject: {stats}"
            areas[f"{size}{tag}"] = stats["area"]

    spread = (max(areas.values()) - min(areas.values())) / min(areas.values())
    assert spread <= 0.30, f"resolution- or colour-path-dependent measurement: {areas} (spread {spread:.1%})"


@pytest.mark.skipif(
    not (SEED_STILL.is_file() and SEED_CLIP.is_file()),
    reason=_missing_assets(SEED_STILL, SEED_CLIP),
)
def test_the_clip_agrees_with_its_own_seed_frame() -> None:
    """Acceptance (#356): the 2048² still and the 960² clip stop disagreeing.

    Compared **frame to frame**, which is the only comparison that isolates the
    bug.  ``gen_1x1_720p_v6.mp4`` is a ten-second flight down the gorge and the
    whitewater's share of the picture genuinely shrinks as the drone descends,
    so the clip's median over eight sampled frames is not a measurement of the
    same picture as the seed still and never was.  Its *first* sampled frame is.

    Before: still 10.7 %, first frame 0.12 % (86× apart) and four of the eight
    frames reporting no anchor at all.  After: 2.41 % against 2.60 %, 8 %
    apart, with all eight frames anchored.
    """
    info = csq.probe_source(SEED_CLIP)
    frames = [first for first, _second in csq.iter_frame_pairs(SEED_CLIP, info, pairs=8)]
    assert len(frames) == 8, f"expected eight sampled frames, got {len(frames)}"

    still = csq.anchor_stats(csq.read_still(SEED_STILL))
    per_frame = [csq.anchor_stats(frame) for frame in frames]
    opening = per_frame[0]

    assert still["found"] and opening["found"], f"still={still} opening={opening}"
    spread = abs(still["area"] - opening["area"]) / min(still["area"], opening["area"])
    assert spread <= 0.30, f"still {still['area']:.4%} vs first frame {opening['area']:.4%} ({spread:.1%} apart)"

    detected = sum(1 for stats in per_frame if stats["found"])
    assert detected >= 6, f"the clip is still losing its subject: {[s['area'] for s in per_frame]}"


# ---------------------------------------------------------------------------
# anchor — the water is not the subject either (G-10 #361)
# ---------------------------------------------------------------------------
#
# #343 taught the detector that a *flat* bright region is not a subject.  G-10 is
# the same sentence with the adjective removed: real whitewater is bright,
# finely textured and four times more colour-distinct than a gray airframe, so
# it passes both of the older channels and wins on density.  On
# ``video/seed_v6.png`` the aircraft scored 0.26 against the foam's 0.56 with
# Otsu cutting at 0.38 — the subject was not losing the election, it was not on
# the ballot, and no candidate-level veto could have reached it.
#
# What separates them is neither brightness (white fuselages, snow and lit rock
# would all be casualties) nor position (a centre prior would make 「主体偏离
# 中心」 unreachable) but **enclosure**: the water runs off the edge of the
# picture and the aircraft does not.


def test_the_gorge_fixture_finds_the_subject_not_the_water() -> None:
    """Acceptance (#361): the bright textured backdrop must not win.

    The synthetic half of the card, and the one that runs in CI.  Subject at
    y = 0.26, water's centre of mass at y = 0.75; see :func:`gorge_frame` for why
    the gap has to be that wide before the assertion means anything.
    """
    stats = csq.anchor_stats(gorge_frame())

    assert stats["found"] is True, stats
    assert stats["centroid_x"] == pytest.approx(0.5, abs=0.08), f"not the centred subject: {stats}"
    assert stats["centroid_y"] == pytest.approx(0.26, abs=0.12), f"looking at the river: {stats}"
    assert stats["area"] == pytest.approx(0.10, abs=0.05), f"the water leaked into the region: {stats}"


def test_removing_the_enclosure_channel_puts_the_anchor_in_the_water(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation check (#361): the enclosure channel is load-bearing, not decoration.

    Blind it and this frame's anchor drops to the river — y = 0.751 instead of
    0.260, on the far side of a 0.49 gap.  If a future change makes
    :func:`~scripts.check_source_quality.enclosure_gate` a no-op, or normalises
    it into uselessness, this fails and the test above stops meaning anything.
    """
    frame = gorge_frame()
    gated = csq.anchor_stats(frame)

    _blind_enclosure_gate(monkeypatch)
    ungated = csq.anchor_stats(frame)

    assert gated["centroid_y"] == pytest.approx(0.26, abs=0.12)
    assert ungated["centroid_y"] > 0.60, (
        f"the enclosure channel changed nothing — the mutation must move the anchor into the river: {ungated}"
    )


def test_the_enclosure_channel_does_not_narrow_the_subject_separation() -> None:
    """The new gate must not buy the water fix by blurring the #341 margin.

    Same contract as ``test_the_texture_channel_widens_the_subject_separation``
    and the same reason it is asserted separately: an absolute-referenced gate
    could have quietly promoted speckle in an evenly-textured frame, since blob
    texture is — by the barrier measure — thoroughly enclosing.  It does not,
    because the reference is set above where those frames saturate; measured
    64.1× against G-11's floor of 55×.
    """
    quietest_true = min(
        csq.anchor_stats(subject_frame(0.11))["area"],
        csq.anchor_stats(subject_frame(0.40))["area"],
        csq.anchor_stats(subject_frame(0.11, cx_frac=0.80))["area"],
    )
    assert quietest_true / max(_loudest_subjectless(), 1e-9) > 55.0, "separation regressed"


def test_enclosure_gate_reads_the_border_as_open_and_the_subject_as_closed() -> None:
    """The channel itself: zero where the picture leaks, high where it encloses.

    Asserted on the three regions of :func:`gorge_frame` because that is what the
    gate claims to distinguish — and note the water, which is *not* on the border
    and is the brightest thing in the frame, reads 0.002: what demotes it is that
    it is continuous with the bottom edge, not that it is bright.
    """
    frame = gorge_frame()
    gate = csq.enclosure_gate(frame)
    height, width = gate.shape
    border = np.concatenate([gate[0], gate[-1], gate[:, 0], gate[:, -1]])
    subject = gate[int(0.20 * height) : int(0.33 * height), int(0.43 * width) : int(0.57 * width)]
    water = gate[int(0.70 * height) :, int(0.35 * width) : int(0.65 * width)]

    assert gate.min() >= 0.0
    assert gate.max() <= 1.0
    assert border.max() == pytest.approx(0.0, abs=1e-6), "the frame's own edge is by definition not enclosed"
    assert water.mean() < 0.05, f"water continuous with the bottom edge must read as open: {water.mean():.4f}"
    assert subject.mean() > water.mean() * 10.0, (
        f"the enclosed subject must outscore the open water: {subject.mean():.4f} vs {water.mean():.4f}"
    )


def test_enclosure_gate_rejects_nonsensical_settings() -> None:
    with pytest.raises(ValueError, match="reference must be"):
        csq.enclosure_gate(gorge_frame(), reference=0.0)
    with pytest.raises(ValueError, match="passes must be"):
        csq.enclosure_barrier(gorge_frame(), passes=0)


def test_enclosure_barrier_reads_a_colour_frame_and_a_gray_one_alike() -> None:
    """#360's contract, at the level of the channel that could most easily break it.

    The still path hands this function BGR and the video sampler hands it
    ``-pix_fmt gray``; reducing the per-channel barrier by *mean* would divide
    the grayscale answer by three for no physical reason.  The maximum keeps the
    two within a few percent, which is what lets the same picture measure the
    same area down both paths.
    """
    frame = cv2.cvtColor(gorge_frame(), cv2.COLOR_GRAY2BGR)
    colour = csq.enclosure_barrier(frame)
    gray = csq.enclosure_barrier(csq._as_gray(frame))

    assert colour.shape == gray.shape
    assert float(np.abs(colour - gray).max()) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.skipif(not SEED_STILL.is_file(), reason=_missing_assets(SEED_STILL))
def test_the_seed_still_anchors_on_the_aircraft_not_the_rapids() -> None:
    """Acceptance (#361), on the still the card was filed against.

    ``video/seed_v6.png`` is 2048² with the quadcopter across the middle of the
    frame and a stretch of whitewater filling the bottom of the gorge.  The
    detector answered **10.65 % at (0.396, 0.787)** — the rapids — and the card's
    own bound is that the reported centroid has to land in the airframe band,
    y ∈ [0.28, 0.55].  It now answers **2.41 % at (0.535, 0.501)**.

    Asserted as a band rather than a point because the reported region is the
    fuselage and gimbal rather than the full rotor span, and where its centre of
    mass falls inside the airframe is not something this card fixed.
    """
    stats = csq.anchor_stats(csq.read_still(SEED_STILL))

    assert stats["found"], f"the aircraft must be found at all: {stats}"
    assert 0.28 <= stats["centroid_y"] <= 0.55, f"still on the rapids: {stats}"
    assert stats["centroid_x"] == pytest.approx(0.50, abs=0.12), f"not the aircraft: {stats}"


@pytest.mark.skipif(not SEED_STILL.is_file(), reason=_missing_assets(SEED_STILL))
def test_removing_the_enclosure_channel_takes_the_seed_still_off_the_airframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation check (#361) on the real frame, re-derived by G-11 (#364).

    **What this asserted until #364, and why it had to change.**  At a texture
    window of 21, blinding this channel put ``seed_v6.png`` back at 13.27 % and
    y = 0.792 — the reported bug reproduced to within a centroid's width of the
    card's own number (10.65 % at y = 0.787) — so the test asserted "back on the
    whitewater".  G-11 widened the window to 31 and that stopped being true: a
    31-pixel box averages the foam's fine structure down far enough that the
    *texture* gate alone now keeps the rapids out of the running on this
    photograph.  Blinding the enclosure channel moves the anchor to the left
    canyon wall instead, at (0.113, 0.489), and blinding **both** structure
    channels does not reach the water either (it gives (0.115, 0.435)).  There
    is no longer any mutation that reproduces the rapids on this asset.

    That is the same shadowing #361 itself created for #343's sky mutation, one
    step further along, and it is handled the same way: assert what the mutation
    still demonstrably does — takes the anchor **off the airframe** — and leave
    the enclosure channel's own necessity pinned where nothing shadows it, on
    :func:`gorge_frame` in ``test_removing_the_enclosure_channel_puts_the_anchor_in_the_water``,
    which runs in CI and still moves y from 0.26 to 0.76.

    Asserted on ``centroid_x`` because that is where the daylight is: 0.535
    gated against 0.113 ungated, a gap of 0.42, and the ungated answer is
    identical to three decimals at 2048²/1024²/960²/512² down both the colour
    and the luminance path.  This is a stable relocation, not a wobble.
    """
    frame = csq.read_still(SEED_STILL)
    gated = csq.anchor_stats(frame)

    _blind_enclosure_gate(monkeypatch)
    ungated = csq.anchor_stats(frame)

    assert 0.28 <= gated["centroid_y"] <= 0.55, gated
    assert gated["centroid_x"] == pytest.approx(0.50, abs=0.12), gated
    assert ungated["centroid_x"] < 0.30, f"the mutation must take the anchor off the airframe: {ungated}"


@pytest.mark.skipif(not SEED_CLIP.is_file(), reason=_missing_assets(SEED_CLIP))
def test_the_clip_anchors_on_the_aircraft_frame_by_frame() -> None:
    """Acceptance (#361), video path: the sampler's gray frames agree with the still.

    The still path reads colour and the clip path reads ``-pix_fmt gray``, and
    the enclosure channel is the one part of the detector that could plausibly
    have behaved differently down the two — see
    ``test_enclosure_barrier_reads_a_colour_frame_and_a_gray_one_alike``.  What
    this pins is the end-to-end consequence: over eight sampled frames of the
    gorge flight the median winning centroid is y = 0.468, inside the airframe
    band, with every frame anchored.
    """
    info = csq.probe_source(SEED_CLIP)
    frames = [first for first, _second in csq.iter_frame_pairs(SEED_CLIP, info, pairs=8)]
    per_frame = [csq.anchor_stats(frame) for frame in frames]
    ys = [stats["centroid_y"] for stats in per_frame]

    assert 0.28 <= float(np.median(ys)) <= 0.55, f"the clip is anchored on the water: {[round(y, 3) for y in ys]}"
    assert sum(1 for stats in per_frame if stats["found"]) >= 6, (
        f"the clip is losing its subject: {[round(s['area'], 4) for s in per_frame]}"
    )


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
# anchor — angular verdict on the fisheye route (G-12 #372, #398)
# ---------------------------------------------------------------------------
#
# The area band (8–15 %) was measured on Red Raion's *plane* stills.  The VR180
# route maps a flat source onto a 180° sphere via an equidistant fisheye, under
# which 8 % of the frame is ≈51° of the viewer's field — what the owner reported
# as "still too big, it blocks the forward view" across three Quest rounds,
# while the tool was calling it 偏小.  #372 is the card; #398 lands it.  The
# fisheye route now judges by angular size (6–12°) and FAILs both sides with a
# suggested rescale; the rect route keeps the area band unchanged.


def test_anchor_angle_is_the_closed_form() -> None:
    """The conversion itself, asserted against the #372 table.

    ``deg = √area × fov`` (equivalent-square width × angular span), so 8 %
    area is ≈28 % width is ≈51° at fov 180 — the three numbers #372's table
    carries and that the owner's Quest feedback calibrated against.  The rect
    route swaps ``fov`` for ``--src-hfov`` on the same linear rule.
    """
    assert csq.anchor_angle(0.08, "fisheye", fisheye_fov=180.0) == pytest.approx(math.sqrt(0.08) * 180.0, rel=1e-9)
    # 8 % area -> ~51° at fisheye-fov 180 (the acceptance number).
    assert csq.anchor_angle(0.08, "fisheye") == pytest.approx(50.91, abs=0.5)
    # width 1/20 -> area 1/400 -> 9° at fov 180 (the other acceptance number).
    assert csq.anchor_angle((1 / 20) ** 2, "fisheye") == pytest.approx(9.0, abs=0.01)
    # rectilinear uses src_hfov on the same linear rule.
    assert csq.anchor_angle(0.25, "rect", src_hfov=90.0) == pytest.approx(math.sqrt(0.25) * 90.0, rel=1e-9)
    assert csq.anchor_angle(0.0, "fisheye") == 0.0


def test_synthetic_anchor_at_8_percent_area_reports_about_51_deg_and_fails_too_big() -> None:
    """Acceptance (#398): 8 % area under fisheye-180 reports ~51° and FAILs 太大.

    The same anchor the rect route calls a clean PASS (it is inside the 8–15 %
    band) reads ≈51° on the fisheye route — well above the 12° ceiling — so it
    FAILs with 偏大 and a shrink factor, which is the advice the owner's three
    Quest rounds were asking for and not getting.  The area is read off the real
    detector (``anchor_stats`` on :func:`subject_frame`), so the verdict is taken
    on the same number the tool actually reports, not on a hand-set dict.
    """
    stats = csq.anchor_stats(subject_frame(0.08))
    assert stats["found"], "the 8 % fixture must clear the detection floor"

    result = csq.check_anchor([stats], projection="fisheye", fisheye_fov=180.0)

    assert result.status == csq.STATUS_FAIL, f"{result.status}: {result.detail}"
    assert result.measured["anchor_verdict"] == "too_big"
    assert result.measured["anchor_deg"] == pytest.approx(51, abs=2.0), (
        f"8 % area must read ~51° at fov 180, got {result.measured['anchor_deg']}"
    )
    assert "偏大" in result.detail
    assert result.measured["suggested_scale"] == pytest.approx(
        csq.ANCHOR_MAX_DEG / result.measured["anchor_deg"], rel=1e-3
    )
    assert result.measured["suggested_scale"] < 1.0, "too-big must suggest a shrink (< 1)"


def test_synthetic_anchor_at_one_twentieth_width_reports_about_9_deg_and_passes() -> None:
    """Acceptance (#398): width 1/20 -> ~9° -> PASS on the fisheye route.

    Asserted on a synthetic ``found`` dict rather than on :func:`subject_frame`
    because the detector's own 0.8 % area floor corresponds to ≈16° at fov 180
    — directly above the whole 6–12° pass band — so no anchor the real detector
    can find at 180° is small enough to land in the band.  A 9° subject would
    need a sub-floor detection that the rescue path turns into false anchors
    (see :func:`test_fisheye_floor_keeps_subjectless_frames_anchorless`); the
    verdict logic itself is what this test pins, on a frame the caller has
    vouched for.
    """
    frame = {"found": True, "area": (1 / 20) ** 2, "offset": 0.1, "components": 1}
    result = csq.check_anchor([frame], projection="fisheye", fisheye_fov=180.0)

    assert result.status == csq.STATUS_PASS, f"{result.status}: {result.detail}"
    assert result.measured["anchor_verdict"] == "ok"
    assert result.measured["anchor_deg"] == pytest.approx(9.0, abs=0.01)
    assert result.measured["suggested_scale"] is None


def test_a_too_small_anchor_fails_and_suggests_an_enlarge_factor() -> None:
    """Below :data:`ANCHOR_MIN_DEG` the fisheye route FAILs 太小 with a grow factor.

    The mirror of the too-big case: a vouched-for 4° anchor is real but too
    small to hold the eye, so it FAILs (not WARNs) and the suggested factor is a
    number > 1 that would enlarge it past the 6° floor.
    """
    area = (4.0 / 180.0) ** 2  # 4° at fov 180
    frame = {"found": True, "area": area, "offset": 0.1, "components": 1}
    result = csq.check_anchor([frame], projection="fisheye", fisheye_fov=180.0)

    assert result.status == csq.STATUS_FAIL, f"{result.status}: {result.detail}"
    assert result.measured["anchor_verdict"] == "too_small"
    assert result.measured["anchor_deg"] == pytest.approx(4.0, abs=0.01)
    assert "偏小" in result.detail
    assert result.measured["suggested_scale"] == pytest.approx(6.0 / 4.0, rel=1e-3)
    assert result.measured["suggested_scale"] > 1.0, "too-small must suggest an enlarge (> 1)"


def test_fisheye_failure_gates_the_run_but_rect_warn_does_not() -> None:
    """The whole point of making the fisheye verdict a FAIL: it can stop a run.

    A too-big anchor on the rect route is a WARN (exit 0, the shot is merely not
    to spec); the same anchor on the fisheye route is a FAIL (exit 1, the subject
    is the wrong size and the generation must be redone).  #372's complaint was
    that the tool's advice pointed the wrong way — this is the gate that makes
    the right advice actually bite.
    """
    stats = csq.anchor_stats(subject_frame(0.08))
    rect = csq.SourceReport(source="f.png", checks=[csq.check_anchor([stats])])
    fish = csq.SourceReport(source="f.png", checks=[csq.check_anchor([stats], projection="fisheye", fisheye_fov=180.0)])
    assert rect.exit_code == 0
    assert fish.exit_code == 1
    assert fish.failed is True


def test_the_rect_route_reports_the_angle_but_still_judges_by_area() -> None:
    """Backward compatibility: rect keeps the area band, and ``anchor_deg`` is context only.

    The 8 % anchor is inside the rect band, so it PASSes on area — and the
    angle is reported (≈51° at the default fov would be nonsensical for a dome,
    so here it is reported at the default ``src_hfov`` 70°), but moving the
    angle never moves the verdict.  This is the seam that lets the rect suite
    stay unchanged while the fisheye route gets the new logic.
    """
    stats = csq.anchor_stats(subject_frame(0.08))
    result = csq.check_anchor([stats])  # default projection="rect"

    assert result.measured["projection"] == "rect"
    assert "anchor_deg" in result.measured
    assert result.measured["anchor_deg"] == pytest.approx(math.sqrt(stats["area"]) * csq.DEFAULT_SRC_HFOV, rel=1e-3)
    # The verdict key exists and is set, but the rect route's pass/fail is the
    # area band's, not the angle band's.
    assert result.measured["anchor_verdict"] == "ok"


def test_fisheye_floor_keeps_subjectless_frames_anchorless(monkeypatch, tmp_path: Path) -> None:
    """The lowered fisheye floor must not fabricate anchors out of speck.

    Lowering the area floor to let a 9° anchor through is only safe if empty
    frames stay empty.  A low area floor plus the rescue path amplifies sub-floor
    speck four- to five-fold (measured: 9 of 11 subjectless fixtures report a
    found anchor at area-floor 0.003), which is why :func:`run_checks` gates the
    rescue off on the fisheye route.  This runs the subjectless fixtures through
    the full ``run_checks`` fisheye path and demands every one comes back
    ``missing`` — the end-to-end proof that the floor + no-rescue combination
    holds, not just the unit-level one.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=FRAME_SIZE, height=FRAME_SIZE, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    subjectless = [subjectless_frame(seed=s) for s in range(1, 9)] + [
        flat_frame(),
        empty_centre_frame(),
        rim_damped_frame(0.18),
    ]

    def feeder_of(f):
        # Bind ``f`` now (default-arg), not at call time — the loop rebinds the
        # name each iteration, so a bare closure would hand every run the last
        # frame.
        return lambda *a, _f=f, **k: iter([(_f, _f)])

    for frame in subjectless:
        monkeypatch.setattr(csq, "iter_frame_pairs", feeder_of(frame))
        report = csq.run_checks(clip, input_projection="fisheye", fisheye_fov=180.0)
        anchor = next(c for c in report.checks if c.name == "anchor")
        assert anchor.status == csq.STATUS_WARN, (
            f"a subjectless frame fabricated an anchor under the fisheye floor: {anchor.detail}"
        )
        assert anchor.measured["anchor_verdict"] == "missing"
        assert anchor.measured["detected_frames"] == 0


def test_run_checks_fisheye_fails_an_oversized_real_anchor() -> None:
    """End-to-end on the fisheye route: a real 11 % subject reads ~60° and FAILs.

    The :func:`anchored_outflow_frames` fixture's 0.11-area hero is a clean PASS
    on the rect route (inside the band); on the fisheye route it subtends ≈60°,
    well past the 12° ceiling, so the anchor check FAILs too-big with a shrink
    factor.  This is the case the owner kept hitting: a subject the area band
    called fine that was in fact too big once mapped onto the sphere.
    """
    present = {"found": True, "area": 0.11, "offset": 0.1, "components": 1}
    result = csq.check_anchor([present], projection="fisheye", fisheye_fov=180.0)
    assert result.status == csq.STATUS_FAIL
    assert result.measured["anchor_deg"] == pytest.approx(math.sqrt(0.11) * 180.0, rel=1e-3)
    assert result.measured["anchor_verdict"] == "too_big"


def test_fisheye_json_carries_the_new_fields() -> None:
    """The machine-readable payload ships ``anchor_deg`` / ``anchor_verdict`` / ``suggested_scale``.

    A consumer that disagrees with the verdict has to be able to read the angle
    and the rescale factor without re-running the analysis, the same contract
    the rect route's ``area``/``offset`` already meet.
    """
    stats = csq.anchor_stats(subject_frame(0.08))
    report = csq.SourceReport(
        source="f.png",
        checks=[csq.check_anchor([stats], projection="fisheye", fisheye_fov=180.0)],
    )
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    anchor = payload["checks"][0]["measured"]
    assert {"anchor_deg", "anchor_verdict", "suggested_scale"} <= set(anchor)
    assert anchor["anchor_verdict"] == "too_big"
    assert anchor["suggested_scale"] is not None


def test_reverting_to_the_area_verdict_turns_the_too_big_fail_red() -> None:
    """Mutation check (#372): without the angle verdict the too-big FAIL disappears.

    Force the rect (area) route on the 8 % anchor and the FAIL becomes a PASS/WARN
    on area — i.e. the angle logic is what produces the owner-requested 太大
    verdict, not something incidental.  This is the card's "把换算去掉退回纯
    面积判定，上面断言必须变红" applied at the verdict level.
    """
    stats = csq.anchor_stats(subject_frame(0.08))
    angle = csq.check_anchor([stats], projection="fisheye", fisheye_fov=180.0)
    area = csq.check_anchor([stats])  # default rect route = the pre-#398 behaviour

    assert angle.status == csq.STATUS_FAIL
    assert area.status != csq.STATUS_FAIL, "the area route must not FAIL an 8 % anchor — that is the bug #372 filed"


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


def test_small_real_anchor_survives_the_fisheye_contrast_gate(monkeypatch, tmp_path: Path) -> None:
    """A real ~8° subject (below the rect floor) is still found on the fisheye route.

    The contrast gate (:data:`csq.ANCHOR_SMALL_MIN_CONTRAST`) rejects speck, but a
    subject that stands out from its surroundings must pass it — otherwise the
    lowered floor would buy nothing and every 6–12° anchor would read missing.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=FRAME_SIZE, height=FRAME_SIZE, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)
    frame = subject_frame(0.0025)
    monkeypatch.setattr(csq, "iter_frame_pairs", lambda *a, **k: iter([(frame, frame)]))

    report = csq.run_checks(clip, input_projection="fisheye", fisheye_fov=180.0)
    anchor = next(c for c in report.checks if c.name == "anchor")

    assert anchor.measured["detected_frames"] == 1, anchor.detail
    assert anchor.measured["anchor_verdict"] == "ok", anchor.detail
    assert 6.0 <= anchor.measured["anchor_deg"] <= 12.0


def test_anchor_local_contrast_separates_speck_from_subjects() -> None:
    """The measured split the gate relies on: speck < threshold < real subjects."""
    floor = (csq.ANCHOR_MIN_DEG_FLOOR / 180.0) ** 2
    speck = [subjectless_frame(seed=s) for s in range(1, 9)] + [empty_centre_frame()]
    for frame in speck:
        stats = csq.anchor_stats(frame, min_area=floor, rescue=False)
        if stats["found"]:
            assert csq.anchor_local_contrast(frame, stats) < csq.ANCHOR_SMALL_MIN_CONTRAST
    for area in (0.002, 0.0025, 0.004):
        frame = subject_frame(area)
        stats = csq.anchor_stats(frame, min_area=floor, rescue=False)
        assert stats["found"]
        assert csq.anchor_local_contrast(frame, stats) > csq.ANCHOR_SMALL_MIN_CONTRAST
    assert csq.anchor_local_contrast(flat_frame(), {"found": False, "area": 0.0}) == 0.0
