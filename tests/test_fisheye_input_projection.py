"""C-2 (#294): equidistant fisheye source -> hequirect.

The numbers asserted here were **measured first**, by rendering the synthetic
512² target below through the real ffmpeg ``v360`` filter under four different
configurations, and only then written down:

===========================  ================================================
config                       measured centre-row landing of the 45° ring
===========================  ================================================
``ih_fov=iv_fov=180``        x = 127.5 / 383.5   (expected 128 / 384)  ✔
``ih_fov=iv_fov=150``        x = 149   / 362     (matches θ=37.5° model) ✔
``id_fov=180``               x = 165   / 346     (~37 px off — wrong)   ✘
``+ h_fov=v_fov=100``        bit-identical to the first config           →
===========================  ================================================

which is why the filter-string tests below ban ``id_fov`` and ``h_fov``/
``v_fov`` outright rather than merely preferring ``ih_fov``/``iv_fov``.

The ffmpeg path is exercised only when ffmpeg + v360 are present (so CI stays
green on a minimal runner); the OpenCV fallback is always exercised, and the
geometry assertions run against **both** so the two cannot drift apart.

K-27 (#302) added the non-square half of that story.  ``ih_fov``/``iv_fov`` are
per-axis angles, so writing the *same* number into both — as this module did
until #302 — describes an ellipse the moment ``width != height``, and no lens
forms one.  ``TestNonSquareVerticalFovConvention`` measures a 16:9 source and
holds the vertical landing to 2 px; the tables in the class docstring are, as
above, measurements taken before the assertions were written.

K-26 (#301) finished the last row of that first table.  The rectilinear branch
still emitted ``h_fov=180:v_fov=180`` long after the fisheye branch had been
shown they do nothing, so the code contradicted its own comment.  It now emits
neither, and ``test_dropping_output_h_fov_and_v_fov_changes_no_pixel`` renders
both branches with and without the pair and compares the raw output bytes, so
the "inert" claim is checked rather than remembered.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess

import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg v360 unavailable")

_S = 512  # synthetic source *and* output edge, so 1 output px = 180/512 deg
_R = _S / 2.0
_GREY = (128, 128, 128)  # the fill OUTSIDE the image circle


def _synthetic_fisheye(circle_fov_deg: float = 180.0) -> np.ndarray:
    """A 512² equidistant fisheye target: concentric rings + radial spokes.

    The inscribed image circle spans *circle_fov_deg*, so with the default 180
    the frame edge is the 90° rim.  Every marker sits at a known polar angle,
    which is what makes the ±2 px assertions meaningful:

    * centre    -> red     (polar 0°)
    * 45° ring  -> green
    * rim       -> blue    (polar ``circle_fov_deg`` / 2)
    * spokes    -> yellow  (azimuth 45/135/225/315°)
    * inside    -> white
    * outside the image circle -> grey 128

    Nothing in the frame is black, so — exactly as in the #254/#255 fixture —
    any black pixel downstream can only have come from the out-of-FOV fill.
    """
    yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float64)
    dx, dy = (xx + 0.5) - _R, (yy + 0.5) - _R
    r = np.hypot(dx, dy)
    az = np.arctan2(dy, dx)
    inside = r <= _R

    img = np.full((_S, _S, 3), _GREY, dtype=np.uint8)
    img[inside] = (255, 255, 255)
    polar = r / _R * (circle_fov_deg / 2.0)  # equidistant: r is linear in theta
    for spoke in (45, 135, 225, 315):
        delta = np.abs(np.arctan2(np.sin(az - np.radians(spoke)), np.cos(az - np.radians(spoke))))
        img[inside & (delta < np.radians(1.5))] = (255, 255, 0)
    img[inside & (np.abs(polar - 45.0) < 0.6)] = (0, 255, 0)
    img[inside & (polar > circle_fov_deg / 2.0 - 1.2)] = (0, 0, 255)
    img[r <= 6] = (255, 0, 0)
    return img


def test_synthetic_target_has_no_black_pixels():
    """Guard the guard: a black marker would make every "hole is black"
    assertion below silently vacuous (the #254 trap).
    """
    assert not np.any(_synthetic_fisheye().sum(axis=2) == 0)


def _mapper(use_ffmpeg: bool, fov: float = 180.0) -> EquirectangularMapper:
    return EquirectangularMapper(
        output_width=_S,
        output_height=_S,
        use_ffmpeg=use_ffmpeg,
        input_projection="fisheye",
        fisheye_fov=fov,
    )


def _v360_args(flt: str) -> dict[str, str]:
    """Parse the ``v360=k=v:k=v:...`` head of a filter chain into a dict.

    Parsing beats substring matching: ``"h_fov" in flt`` is *always* true
    because ``ih_fov`` contains it, so a naive "no h_fov" assertion could
    never pass and a naive "has ih_fov" one could never fail.
    """
    head = flt.split(",")[0]
    assert head.startswith("v360="), head
    return dict(part.split("=", 1) for part in head[len("v360=") :].split(":"))


def _black_fraction(rgb: np.ndarray) -> float:
    return float((rgb[:, :, :3].sum(axis=2) == 0).mean())


def _green_run_centres(row: np.ndarray) -> list[float]:
    """Centres of the green-dominant runs along an RGB row."""
    r, g, b = row[:, 0].astype(int), row[:, 1].astype(int), row[:, 2].astype(int)
    hit = (g > 140) & (g > r + 60) & (g > b + 60)
    centres: list[float] = []
    run: list[int] = []
    for x, on in enumerate(hit):
        if on:
            run.append(x)
        elif run:
            centres.append(sum(run) / len(run))
            run = []
    if run:
        centres.append(sum(run) / len(run))
    return centres


# --------------------------------------------------------------------------- #
# The v360 head must say exactly what the geometry needs — and nothing else
# --------------------------------------------------------------------------- #


class TestFisheyeFilterString:
    def test_uses_equidistant_fisheye_input_and_hequirect_output(self):
        args = _v360_args(_mapper(use_ffmpeg=True)._v360_filter(_S, _S))
        assert args["input"] == "fisheye"
        assert args["output"] == "hequirect"
        assert args["alpha_mask"] == "1"

    @pytest.mark.parametrize("fov,expected", [(180.0, "180"), (150.0, "150"), (220.5, "220.5")])
    def test_square_source_carries_the_fisheye_fov_on_both_axes(self, fov, expected):
        """Only the **square** case puts ``fisheye_fov`` on both axes: since
        #302 a non-square source derives ``iv_fov`` from the aspect, which
        ``TestNonSquareVerticalFovConvention`` covers.
        """
        args = _v360_args(_mapper(use_ffmpeg=True, fov=fov)._v360_filter(_S, _S))
        assert args["ih_fov"] == expected
        assert args["iv_fov"] == expected

    def test_never_uses_id_fov(self):
        """``id_fov`` is the *diagonal* angle: on a square frame ``id_fov=180``
        works out to only 127.3° per axis, which measurably puts the 45° ring
        ~37 px off and lets the corners outside the image circle leak in.
        """
        assert "id_fov" not in _v360_args(_mapper(use_ffmpeg=True)._v360_filter(_S, _S))

    def test_never_passes_output_h_fov_or_v_fov(self):
        """``hequirect`` output ignores ``h_fov``/``v_fov`` — measured: adding
        ``h_fov=100:v_fov=100`` leaves the output bit-identical — so emitting
        them would advertise a control that does not exist.
        """
        args = _v360_args(_mapper(use_ffmpeg=True)._v360_filter(_S, _S))
        assert "h_fov" not in args
        assert "v_fov" not in args

    def test_reuses_the_shared_black_composite_tail(self):
        """#258's composite must not be re-implemented for the fisheye branch."""
        flt = _mapper(use_ffmpeg=True)._v360_filter(_S, _S)
        assert EquirectangularMapper._BLACK_COMPOSITE in flt
        assert flt.endswith("format=rgb24")

    def test_rectilinear_geometry_is_untouched(self):
        """The fisheye branch must not have moved the pinhole one: same
        ``input=flat``, same ``ih_fov``, same aspect-solved ``iv_fov``.
        """
        args = _v360_args(EquirectangularMapper(64, 64, src_hfov=90.0)._v360_filter(64, 32))
        assert args["input"] == "flat"
        assert args["output"] == "hequirect"
        assert args["ih_fov"] == "90.0"
        assert args["iv_fov"] == "53.13"  # 2*atan(tan(45°) * 32/64)


# --------------------------------------------------------------------------- #
# K-26 (#301): neither branch may advertise an output FOV hequirect ignores
# --------------------------------------------------------------------------- #


def _projection_mapper(projection: str) -> EquirectangularMapper:
    return EquirectangularMapper(
        output_width=_S,
        output_height=_S,
        src_hfov=90.0,
        input_projection=projection,
        fisheye_fov=180.0,
    )


@pytest.mark.parametrize("projection", ["rectilinear", "fisheye"])
def test_neither_branch_passes_output_h_fov_or_v_fov(projection):
    """``output=hequirect`` fixes its span at ±90° and ignores ``h_fov``/
    ``v_fov``, so emitting them advertises a control that does not exist —
    which is how the "white edge" hunt once burned time in this very chain.

    The *source* spans (``ih_fov``/``iv_fov``) are the real knobs and must
    survive; note ``_v360_args`` parses keys, so ``ih_fov`` cannot satisfy a
    ``h_fov`` check by substring.
    """
    args = _v360_args(_projection_mapper(projection)._v360_filter(_S, _S))
    assert args["output"] == "hequirect"
    assert "h_fov" not in args
    assert "v_fov" not in args
    assert "ih_fov" in args and "iv_fov" in args


def _render_rgb24(vfilter: str, src: np.ndarray) -> bytes:
    """Run one frame through ``vfilter`` and return the raw rgb24 output bytes."""
    h, w = src.shape[:2]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-i",
        "pipe:0",
        "-vf",
        vfilter,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    return subprocess.run(cmd, input=src.tobytes(), capture_output=True, timeout=60, check=True).stdout


@_FFMPEG
@pytest.mark.parametrize("projection", ["rectilinear", "fisheye"])
def test_dropping_output_h_fov_and_v_fov_changes_no_pixel(projection):
    """The pair was not merely unused, it was **inert** (K-26 #301).

    Renders the shipped chain and a copy with ``h_fov=180:v_fov=180`` spliced
    back in, and compares SHA-256 over the raw rgb24 output.  If ffmpeg ever
    gives ``hequirect`` a real output-FOV control this goes red — the signal
    to revisit the removal deliberately, not to restore it by reflex.
    """
    src = _smooth_fisheye()
    shipped = _projection_mapper(projection)._v360_filter(_S, _S)
    assert "h_fov" not in _v360_args(shipped)  # key-parsed: ``ih_fov`` must not count
    with_inert = shipped.replace("output=hequirect:", "output=hequirect:h_fov=180:v_fov=180:", 1)
    assert _v360_args(with_inert)["h_fov"] == "180"

    a, b = _render_rgb24(shipped, src), _render_rgb24(with_inert, src)
    assert len(a) == _S * _S * 3, f"short render: {len(a)} bytes"
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- #
# Geometry — both mapper paths, so ffmpeg and the OpenCV fallback cannot drift
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])
class TestFisheyeGeometry:
    """At ``fisheye_fov=180`` the hemisphere exactly fills the hequirect frame."""

    def test_image_circle_centre_maps_to_output_centre(self, use_ffmpeg):
        out = _mapper(use_ffmpeg).map_single(_synthetic_fisheye())
        px = out[_S // 2, _S // 2].astype(int)
        assert px[0] > 150 and px[1] < 110 and px[2] < 110, f"expected the red centre dot, got {tuple(px)}"

    def test_45_degree_ring_lands_a_quarter_in_within_2px(self, use_ffmpeg):
        out = _mapper(use_ffmpeg).map_single(_synthetic_fisheye())
        centres = _green_run_centres(out[_S // 2])
        assert len(centres) == 2, f"expected two 45° crossings on the centre row, got {centres}"
        # hequirect spans +/-90 deg across the width, so polar 45 deg sits at 1/4 and 3/4.
        assert abs(centres[0] - _S * 0.25) <= 2.0, centres
        assert abs(centres[1] - _S * 0.75) <= 2.0, centres

    def test_45_degree_ring_is_symmetric_on_the_centre_column(self, use_ffmpeg):
        """Same angle vertically: ``iv_fov`` must equal ``ih_fov``, so a
        horizontal-only scale error cannot hide.
        """
        out = _mapper(use_ffmpeg).map_single(_synthetic_fisheye())
        centres = _green_run_centres(out[:, _S // 2])
        assert len(centres) == 2, centres
        assert abs(centres[0] - _S * 0.25) <= 2.0, centres
        assert abs(centres[1] - _S * 0.75) <= 2.0, centres

    def test_90_degree_rim_maps_to_the_left_and_right_edges(self, use_ffmpeg):
        out = _mapper(use_ffmpeg).map_single(_synthetic_fisheye())
        row = out[_S // 2]
        for x in (0, _S - 1):
            px = row[x].astype(int)
            assert px[2] > 140 and px[0] < 120, f"expected the blue rim at x={x}, got {tuple(px)}"

    def test_outside_the_image_circle_never_reaches_a_180_degree_output(self, use_ffmpeg):
        """Every direction a 180° hequirect can address is inside the image
        circle, so the grey corner wedges must be culled, not smeared in.
        """
        out = _mapper(use_ffmpeg).map_single(_synthetic_fisheye()).astype(int)
        spread = out.max(axis=2) - out.min(axis=2)
        grey = (spread < 18) & (np.abs(out[:, :, 0] - 128) < 18)
        assert grey.mean() < 0.005, f"grey (outside-circle) leak fraction {grey.mean():.4f}"


@pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])
class TestSubHemisphereFisheyeGoesBlack:
    """Below 180° the source cannot fill a 180° output, so #258's contract
    applies to the uncovered rim: pure black RGB, not v360's edge smear.

    At exactly 180° there is *no* uncovered region — the output corners are the
    poles, which sit on the 90° rim — so the black-corner check only has
    meaning here.  Measured at ``fisheye_fov=150``: the rim lands at x≈43/468
    and everything outside it is black.
    """

    def test_corners_are_pure_black(self, use_ffmpeg):
        out = _mapper(use_ffmpeg, fov=150.0).map_single(_synthetic_fisheye())
        for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert tuple(int(v) for v in out[r, c]) == (0, 0, 0)

    def test_black_fill_starts_exactly_where_the_source_stops(self, use_ffmpeg):
        """The uncovered band's *width* is the assertion, not merely that some
        black exists — that is what makes this non-vacuous.

        A 150° source reaches polar 75°, and hequirect spans ±90° across the
        width, so coverage begins at ``256 - 256 * 75/90 = 42.7``.  Measured:
        x=43 on both paths, and the first covered pixel is the blue rim.
        """
        out = _mapper(use_ffmpeg, fov=150.0).map_single(_synthetic_fisheye())
        row = out[_S // 2]
        covered = np.where(row.sum(axis=1) != 0)[0]
        assert abs(int(covered.min()) - 43) <= 2, f"left edge at x={covered.min()}, expected ~43"
        assert abs(int(covered.max()) - 468) <= 2, f"right edge at x={covered.max()}, expected ~468"
        # Everything outside that band is pure black, ...
        assert _black_fraction(row[: covered.min()].reshape(1, -1, 3)) == 1.0
        # ... and the first covered pixel is the rim itself, not a smear.
        assert int(row[covered.min()][2]) > 140, tuple(int(v) for v in row[covered.min()])

    def test_centre_is_not_blackened(self, use_ffmpeg):
        out = _mapper(use_ffmpeg, fov=150.0).map_single(_synthetic_fisheye())
        assert _black_fraction(out[_S // 2 - 8 : _S // 2 + 8, _S // 2 - 8 : _S // 2 + 8]) == 0.0

    def test_sub_180_source_must_already_be_black_outside_its_circle(self, use_ffmpeg):
        """Pins the caveat in ``_v360_fisheye_head``'s docstring.

        v360's visibility test is the source **frame**, not the image circle.
        Below 180° the output rim samples the frame corners *outside* the
        circle, so whatever lives there is passed through rather than culled.
        Measured with a grey-128 surround: 0.00 leak at 180°, 0.12 at 150°,
        0.10 at 120°.  Operators feeding a sub-180° fisheye must therefore
        pre-black the surround; ``alpha_mask`` will not do it for them.
        """
        out = _mapper(use_ffmpeg, fov=150.0).map_single(_synthetic_fisheye()).astype(int)
        spread = out.max(axis=2) - out.min(axis=2)
        grey = (spread < 18) & (np.abs(out[:, :, 0] - 128) < 18)
        assert grey.mean() > 0.02, (
            "expected the outside-circle surround to leak in below 180°; if this now "
            "culls to the image circle, v360's visibility rule changed and the "
            "_v360_fisheye_head docstring needs updating"
        )


def _smooth_fisheye() -> np.ndarray:
    """A ramp-only fisheye: radius in R, azimuth in G/B, black outside.

    The marker target above is deliberately full of hard colour edges, where
    two Lanczos implementations ring differently — that noise would swamp a
    path-parity comparison.  With no hard edges, any remaining disagreement is
    a genuine geometry disagreement.
    """
    yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float64)
    dx, dy = (xx + 0.5) - _R, (yy + 0.5) - _R
    r = np.hypot(dx, dy)
    az = np.arctan2(dy, dx)
    inside = r <= _R
    img = np.zeros((_S, _S, 3), np.uint8)
    img[..., 0] = np.where(inside, r / _R * 200 + 30, 0)
    img[..., 1] = np.where(inside, (np.sin(az) * 0.5 + 0.5) * 200 + 30, 0)
    img[..., 2] = np.where(inside, (np.cos(az) * 0.5 + 0.5) * 200 + 30, 0)
    return img


@_FFMPEG
@pytest.mark.parametrize("fov", [180.0, 150.0])
def test_ffmpeg_and_opencv_paths_agree(fov):
    """Both paths must implement the *same* equidistant model.

    Measured on the smooth source: mean |diff| 1.47 / p99 20 at 180°, and
    0.86 / 11 at 150°.  A per-axis scale slip (e.g. ``id_fov``) moves the 45°
    ring ~37 px and would blow straight through these bounds.
    """
    src = _smooth_fisheye()
    a = _mapper(use_ffmpeg=True, fov=fov).map_single(src).astype(int)
    b = _mapper(use_ffmpeg=False, fov=fov).map_single(src).astype(int)
    diff = np.abs(a - b)
    assert float(diff.mean()) < 4.0, f"mean |diff| = {diff.mean()}"
    assert float(np.percentile(diff, 99)) < 30.0, f"p99 |diff| = {np.percentile(diff, 99)}"


# --------------------------------------------------------------------------- #
# K-27 (#302): a non-square source reads its vertical axis from the aspect
# --------------------------------------------------------------------------- #

#: A 16:9 analysis frame (the calibrator's own default operating point) and the
#: horizontal span its image circle is built at.
_NSW, _NSH, _NS_FOV = 640, 360, 100.0

#: Polar angle of the ring that has to land correctly on **both** axes.  The
#: vertical half-span of the frame above is only ``100/2 * 360/640`` = 28.1°,
#: so 20° is inside it with room to spare, while the 40° ring below exists to
#: prove the horizontal axis was not quietly traded away for the vertical one.
_NS_INNER, _NS_OUTER = 20.0, 40.0


def _non_square_fisheye(width: int = _NSW, height: int = _NSH, hfov: float = _NS_FOV) -> np.ndarray:
    """An *isotropic* equidistant fisheye on a non-square frame — a real lens.

    Built in numpy, not by ffmpeg, so the ring radii are ground truth rather
    than a second opinion: an image circle has the same degrees per pixel on
    both axes, so ``r = k*θ`` with ``k = (width/2) / (hfov/2)``.  The frame is a
    *crop* of that circle — which is what a full-frame fisheye on a 16:9 sensor
    produces, and the case ``ih_fov = iv_fov`` cannot describe.
    """
    k = (width / 2.0) / (hfov / 2.0)  # px per degree, same on both axes
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = (xx + 0.5) - width / 2.0, (yy + 0.5) - height / 2.0
    polar = np.hypot(dx, dy) / k
    img = np.full((height, width, 3), _GREY, dtype=np.uint8)
    img[np.abs(polar - _NS_INNER) < 0.35] = (0, 255, 0)
    img[np.abs(polar - _NS_OUTER) < 0.35] = (255, 0, 255)
    img[np.hypot(dx, dy) <= 4] = (255, 0, 0)
    return img


def _magenta_run_centres(row: np.ndarray) -> list[float]:
    r, g, b = row[:, 0].astype(int), row[:, 1].astype(int), row[:, 2].astype(int)
    hit = (r > 140) & (b > 140) & (g < 110)
    centres: list[float] = []
    run: list[int] = []
    for x, on in enumerate(hit):
        if on:
            run.append(x)
        elif run:
            centres.append(sum(run) / len(run))
            run = []
    if run:
        centres.append(sum(run) / len(run))
    return centres


def _hequirect_landing(polar_deg: float) -> tuple[float, float]:
    """Where a marker at ``polar_deg`` must land on the centre row / column.

    A 180° hequirect spans ±90° across its width and its height, so an angle
    ``p`` off the forward axis sits ``p/180`` of the frame either side of the
    centre.  This is pure projection arithmetic — no ffmpeg opinion in it.
    """
    return _S * (0.5 - polar_deg / 180.0), _S * (0.5 + polar_deg / 180.0)


class TestNonSquareVerticalFovConvention:
    """The mapper must read ``--fisheye-fov`` the way ``calibrate_hfov`` writes it.

    Before #302 the fisheye head emitted ``iv_fov = ih_fov`` for every aspect,
    stretching the image circle into an ellipse whenever ``width != height``.
    Measured on the 640x360 / 100° source above, mapped to 512²:

    ==================================  ===================  ==================
    reading                             20° ring, centre row  ... centre column
    ==================================  ===================  ==================
    theory                              198.5 / 312.9        198.5 / 312.9
    ``iv_fov = ih_fov`` (pre-#302)      198.5 / 312.5   ✔    154.0 / 357.0  ✘
    ``iv_fov = ih_fov * h/w`` (#302)    198.5 / 312.5   ✔    198.5 / 312.5  ✔
    ==================================  ===================  ==================

    45 px on a 512-px hemisphere is ~16° of angular misregistration — and the
    horizontal axis was right either way, which is exactly why nobody saw it.
    """

    def test_vertical_fov_scales_with_the_aspect(self):
        m = _mapper(use_ffmpeg=True, fov=_NS_FOV)
        assert m._fisheye_vertical_fov(_NSW, _NSH) == pytest.approx(_NS_FOV * _NSH / _NSW)
        assert m._fisheye_vertical_fov(_NSW, _NSH) == pytest.approx(56.25)

    def test_vertical_fov_of_a_square_frame_is_the_fisheye_fov_itself(self):
        """The square case must not merely be close — existing behaviour is
        pinned bit-exactly, since ``h / w`` is exactly 1.0 there."""
        for fov in (180.0, 150.0, 220.5):
            m = _mapper(use_ffmpeg=True, fov=fov)
            assert m._fisheye_vertical_fov(_S, _S) == fov

    def test_filter_string_carries_the_derived_vertical_fov(self):
        args = _v360_args(_mapper(use_ffmpeg=True, fov=_NS_FOV)._v360_filter(_NSW, _NSH))
        assert args["ih_fov"] == "100"
        assert args["iv_fov"] == "56.25"

    def test_filter_string_is_unchanged_for_a_square_source(self):
        """Guards the #294 contract: square sources keep ``ih_fov == iv_fov``."""
        args = _v360_args(_mapper(use_ffmpeg=True, fov=150.0)._v360_filter(_S, _S))
        assert args["ih_fov"] == args["iv_fov"] == "150"

    def test_rejects_a_degenerate_source_size(self):
        with pytest.raises(ValueError, match="source size"):
            _mapper(use_ffmpeg=True)._fisheye_vertical_fov(0, 360)

    @pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])
    def test_ring_lands_within_2px_on_the_vertical_axis(self, use_ffmpeg):
        """**The** regression: 45 px off before #302, ~0.6 px after."""
        out = _mapper(use_ffmpeg, fov=_NS_FOV).map_single(_non_square_fisheye())
        lo, hi = _hequirect_landing(_NS_INNER)
        centres = _green_run_centres(out[:, _S // 2])
        assert len(centres) == 2, f"expected two 20° crossings on the centre column, got {centres}"
        assert abs(centres[0] - lo) <= 2.0, centres
        assert abs(centres[1] - hi) <= 2.0, centres

    @pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])
    def test_ring_still_lands_within_2px_on_the_horizontal_axis(self, use_ffmpeg):
        """The axis that was already right must stay right."""
        out = _mapper(use_ffmpeg, fov=_NS_FOV).map_single(_non_square_fisheye())
        row = out[_S // 2]
        for polar, centres in ((_NS_INNER, _green_run_centres(row)), (_NS_OUTER, _magenta_run_centres(row))):
            lo, hi = _hequirect_landing(polar)
            assert len(centres) == 2, f"{polar}°: expected two crossings, got {centres}"
            assert abs(centres[0] - lo) <= 2.0, (polar, centres)
            assert abs(centres[1] - hi) <= 2.0, (polar, centres)

    @pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])
    def test_a_portrait_source_is_handled_the_same_way(self, use_ffmpeg):
        """9:16 is the mirror error case; ``iv_fov`` grows past ``ih_fov`` here
        (100 * 640/360 = 177.8), so a hard-coded equality cannot fake it.
        """
        src = _non_square_fisheye(width=_NSH, height=_NSW)
        out = _mapper(use_ffmpeg, fov=_NS_FOV).map_single(src)
        lo, hi = _hequirect_landing(_NS_INNER)
        for centres in (_green_run_centres(out[_S // 2]), _green_run_centres(out[:, _S // 2])):
            assert len(centres) == 2, centres
            assert abs(centres[0] - lo) <= 2.0, centres
            assert abs(centres[1] - hi) <= 2.0, centres


@_FFMPEG
def test_ffmpeg_and_opencv_paths_agree_on_a_non_square_source():
    """Both branches must derive the vertical span identically, not just the
    horizontal one — otherwise #302 could be fixed on one path alone.
    """
    src = _non_square_fisheye()
    a = _mapper(use_ffmpeg=True, fov=_NS_FOV).map_single(src).astype(int)
    b = _mapper(use_ffmpeg=False, fov=_NS_FOV).map_single(src).astype(int)
    diff = np.abs(a - b)
    assert float(diff.mean()) < 4.0, f"mean |diff| = {diff.mean()}"
    assert float(np.percentile(diff, 99)) < 30.0, f"p99 |diff| = {np.percentile(diff, 99)}"


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #


class TestFisheyeConstructorValidation:
    def test_rejects_unknown_projection(self):
        with pytest.raises(ValueError, match="input_projection"):
            EquirectangularMapper(64, 64, input_projection="stereographic")

    @pytest.mark.parametrize("bad", [0.0, -10.0, 361.0, float("nan")])
    def test_rejects_out_of_range_fisheye_fov(self, bad):
        with pytest.raises(ValueError, match="fisheye_fov"):
            EquirectangularMapper(64, 64, input_projection="fisheye", fisheye_fov=bad)

    def test_defaults_stay_rectilinear(self):
        m = EquirectangularMapper(64, 64)
        assert m.input_projection == "rectilinear"
        assert m.fisheye_fov == 180.0
