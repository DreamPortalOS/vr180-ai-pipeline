"""Fulldome (球幕) mapper — single-pass ffmpeg v360 fisheye domemaster renderer.

No depth/stereo/spherical metadata — pure mono fisheye projection via ffmpeg.

D-1 (#334) fixed the four defects that kept the output from being a legal
domemaster.  A domemaster is a **square canvas holding an inscribed image
circle**; everything outside that circle must be pure black, and the circle is
the whole deliverable.  Before #334 the renderer emitted neither:

===  =======================================  =====================================
 #   defect                                    fix
===  =======================================  =====================================
 1   no ``alpha_mask``, no black composite,    ``alpha_mask=1`` +
     no image circle — the corners carried     :attr:`~FulldomeMapper._BLACK_COMPOSITE`
     v360's edge-clamp smear (measured         (the *same* constant the VR180 side
     RGB(45,39,34), 0.00003% pure black) and   uses, #258) + an inscribed-circle
     "content" ran out to 1.414 R              multiply.  Measured: corners 100%
                                               pure black, content stops at 1.000 R.
 2   ``-an`` was hard-coded, so every dome     ``-map 0:a:0? -c:a copy`` — the same
     master came out silent                    lossless passthrough
                                               ``pipeline.audio_mux`` does for VR180.
 3   ``iv_fov`` was ``ih_fov * height/width``  delegates to
     — a *linear* ratio.  A 16:9 source got    :meth:`EquirectangularMapper._calc_vertical_fov`
     67.5° where the pinhole geometry needs    (``2*atan(tan(hfov/2)*h/w)`` → 88.51°).
     88.5°, squashing the frame by 21°.
 4   ``interp`` unset → ffmpeg's default       ``interp=lanczos``, matching the
     bilinear (``line``)                       VR180 side.
===  =======================================  =====================================

D-2 (#371) adds the two knobs the dome route was missing: which projection the
source frame is in (:attr:`~FulldomeMapper.INPUT_PROJECTIONS`) and how the
dome is oriented relative to it (``pitch``/``yaw``/``roll``).  Both reuse the
VR180 side's implementations rather than restating them — see
:meth:`FulldomeMapper._vr180_geometry`.  At the defaults (``rectilinear``, all
three angles 0) the emitted filtergraph is byte-for-byte the pre-#371 one.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import ClassVar

from pipeline.equirectangular_mapper import EquirectangularMapper

log = logging.getLogger("fulldome-mapper")


class FulldomeMapper:
    """Map a flat 2D video to a fisheye domemaster for fulldome projection.

    Uses a single ffmpeg v360 pass over the whole video (not per-frame).
    The resulting output is a square canvas holding an inscribed fisheye image
    circle on a **pure black** background, with the source's audio track carried
    through untouched — suitable for dome projection systems.

    Parameters
    ----------
    dome_fov : float
        Fisheye field-of-view of the output domemaster in degrees (default 180,
        up to 220 for some dome systems). This is the angle spanned by the
        **inscribed circle**, which is also where the circle mask cuts.
    coverage_h_fov : float
        How many degrees of horizontal FOV the source video covers on the
        input sphere (default 120). Lower values = screen-like patch; higher
        values = fuller dome but more geometric stretch. The sphere outside that
        patch is genuinely black (not smeared) since #334.  For
        ``input_projection="fisheye"`` this is the equidistant span across the
        frame **width** — the same quantity ``EquirectangularMapper.fisheye_fov``
        names (#302) and ``scripts/calibrate_hfov.py`` reports.  Not consulted
        for ``"equirect"``, whose source *is* the whole sphere (measured: v360
        renders ``input=equirect`` byte-identically with and without
        ``ih_fov=360:iv_fov=180``, its defaults).
    coverage_v_fov : float | None
        Vertical coverage FOV. If None, auto-computed from the source aspect
        ratio — the **pinhole** relation (``2*atan(tan(hfov/2)*h/w)``, #334) for
        a rectilinear source, the **equidistant** one (``h_fov*h/w``, #303) for
        a fisheye source.
    input_projection : str
        How to read the source frame, one of :attr:`INPUT_PROJECTIONS`
        (D-2, #371).  ``"rectilinear"`` (default) is a pinhole patch — the
        pre-#371 behaviour; ``"fisheye"`` an equidistant circular fisheye;
        ``"equirect"`` a full 360°×180° sphere.
    pitch, yaw, roll : float
        Dome orientation in degrees (D-2, #371), all 0 by default.  The
        convention is v360's, measured (#324): ``pitch`` is
        ``90° − (the elevation the source's optical axis takes on the dome)``,
        i.e. ``--dome-pitch 20`` drops the source centre from the zenith to 70°
        of elevation.  ``yaw`` swings it round the dome's vertical axis and
        ``roll`` spins the domemaster about its own centre.
    output_size : int
        Width and height of the square output domemaster in pixels (default 4096).
        Must be even.
    codec : str
        Video codec for output (default "h264").
    crf : int
        Constant rate factor for encoding quality (default 18).
    """

    #: Accepted ``input_projection`` values (D-2, #371).  ``rectilinear`` and
    #: ``fisheye`` mean exactly what they mean on the VR180 side
    #: (:attr:`EquirectangularMapper.INPUT_PROJECTIONS`); ``equirect`` is extra
    #: here because a dome master *can* take a full sphere as its source, while
    #: the VR180 path joins equirect eyes without ever projecting them (#286).
    INPUT_PROJECTIONS: tuple[str, ...] = ("rectilinear", "fisheye", "equirect")

    #: ``input_projection`` → the ``v360`` ``input=`` token it maps to.
    _V360_INPUT: ClassVar[dict[str, str]] = {
        "rectilinear": "flat",
        "fisheye": "fisheye",
        "equirect": "equirect",
    }

    def __init__(
        self,
        dome_fov: float = 180.0,
        coverage_h_fov: float = 120.0,
        coverage_v_fov: float | None = None,
        output_size: int = 4096,
        codec: str = "h264",
        crf: int = 18,
        input_projection: str = "rectilinear",
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        if input_projection not in self.INPUT_PROJECTIONS:
            raise ValueError(f"input_projection must be one of {self.INPUT_PROJECTIONS}, got {input_projection!r}")
        if output_size % 2 != 0:
            output_size += 1  # ffmpeg requires even dimensions
        self.dome_fov = dome_fov
        self.coverage_h_fov = coverage_h_fov
        self.coverage_v_fov = coverage_v_fov
        self.output_size = output_size
        self.codec = codec
        self.crf = crf
        self.input_projection = input_projection
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)
        # Build the delegate once, here, so an out-of-range angle fails at
        # construction (EquirectangularMapper owns v360's own ±180° bound,
        # #323) instead of surfacing as an ffmpeg option error halfway through
        # a 4096² render.
        self._vr180_geometry()

    def convert(self, input_path: str, output_path: str) -> str:
        """Run single ffmpeg v360 pass over the whole video.

        Parameters
        ----------
        input_path : str
            Path to the source flat video.
        output_path : str
            Path for the output fisheye domemaster video.

        Returns
        -------
        str
            The output path on success.
        """
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Auto-compute coverage_v_fov from source aspect ratio if not given.
        # An ``equirect`` source spans the whole sphere and emits no
        # ``ih_fov``/``iv_fov`` at all (#371), so there is nothing to probe for
        # — skipping the ffprobe keeps the log honest rather than announcing a
        # coverage angle the filtergraph never uses.
        coverage_v_fov = self.coverage_v_fov
        if coverage_v_fov is None:
            coverage_v_fov = 0.0 if self.input_projection == "equirect" else self._probe_coverage_v_fov(input_path)

        codec_map = {"h264": "libx264", "h265": "libx265"}
        encoder = codec_map.get(self.codec, f"libx{self.codec}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter_complex",
            self._filter_complex(coverage_v_fov),
            "-map",
            f"[{self.OUT_LABEL}]",
            # D-1 (#334) defect 2: this used to be a hard-coded ``-an``, so every
            # dome master shipped silent.  ``?`` makes the audio optional, so a
            # source without a track still converts; ``copy`` keeps the same
            # lossless passthrough guarantee ``pipeline.audio_mux`` gives VR180.
            "-map",
            "0:a:0?",
            "-c:a",
            "copy",
            "-c:v",
            encoder,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        coverage = (
            "coverage=whole-sphere "
            if self.input_projection == "equirect"
            else f"coverage=({self.coverage_h_fov:g}×{coverage_v_fov:g})° "
        )
        log.info(
            "Running fulldome conversion: "
            f"dome_fov={self.dome_fov}° "
            f"input={self.input_projection} "
            f"{coverage}"
            f"orientation=(yaw {self.yaw:g}°, pitch {self.pitch:g}°, roll {self.roll:g}°) "
            f"output={self.output_size}×{self.output_size} "
            f"codec={self.codec} crf={self.crf} "
            f"circle_mask=on audio=copy"
        )
        log.debug(f"ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg v360 conversion failed (exit {result.returncode}):\nstderr:\n{result.stderr[:2000]}"
            )

        log.info(f"✅ Fulldome domemaster written to {output_path}")
        return output_path

    #: Composite the ``alpha_mask=1`` v360 frame onto black — reused *verbatim*
    #: from the VR180 side (#258) instead of restating it here, so the two paths
    #: cannot drift.  ``alpha_mask=1`` on its own only zeroes the **alpha**
    #: plane; the RGB planes still hold v360's edge-clamped smear and alpha is
    #: thrown away the instant the frame is encoded to ``yuv420p`` (#255).  The
    #: hole therefore has to become black *RGB* before the frame leaves ffmpeg.
    _BLACK_COMPOSITE = EquirectangularMapper._BLACK_COMPOSITE

    #: Filtergraph labels.  ``_BLACK_COMPOSITE`` already owns ``_fg``/``_bgsrc``/
    #: ``_bg``, so these three must not collide with those.
    DOME_LABEL = "_dome"
    CIRCLE_LABEL = "_circle"
    OUT_LABEL = "_domemaster"

    def _v360_filter(self, coverage_v_fov: float) -> str:
        """The ``v360`` term: flat source → square fisheye domemaster.

        ``interp=lanczos`` (D-1 #334 defect 4) matches the VR180 side; ffmpeg's
        default is ``line`` (bilinear), which visibly softens a 4096² master
        that is only ever upsampled from a 2880² source.

        ``alpha_mask=1`` marks every output pixel whose sample falls outside the
        source frame; :attr:`_BLACK_COMPOSITE` then turns those into real black.
        For ``output=fisheye`` the inscribed circle spans ``dome_fov``, so with
        the default 120° flat coverage the honest content stops at 0.75 R and
        the annulus out to the rim is exactly what used to be brown smear.

        D-2 (#371) made the ``input=`` token and the orientation tail
        configurable.  Both additions are inert at the defaults: ``rectilinear``
        maps back to ``input=flat`` and
        :meth:`_orientation_terms` is empty at 0/0/0, so the emitted string is
        character-for-character the pre-#371 one.
        """
        return (
            f"v360=input={self._V360_INPUT[self.input_projection]}:output=fisheye"
            f"{self._input_fov_terms(coverage_v_fov)}"
            f":h_fov={self.dome_fov:g}"
            f":v_fov={self.dome_fov:g}"
            f":w={self.output_size}"
            f":h={self.output_size}"
            f":interp=lanczos"
            f":alpha_mask=1"
            f"{self._orientation_terms()}"
        )

    def _input_fov_terms(self, coverage_v_fov: float) -> str:
        """The ``:ih_fov=..:iv_fov=..`` pair describing the *source* span.

        Empty for ``equirect``: v360's equirect input is the full sphere and
        those two options are its 360/180 defaults.  Measured — the rendered
        frame is byte-identical with and without ``ih_fov=360:iv_fov=180``, and
        passing anything *else* (say 120×60) does not crop the source, it
        rescales the sphere and wraps the content round the dome several times
        over.  Writing them would therefore only advertise a control the dome
        route does not offer, the same reasoning that keeps ``h_fov``/``v_fov``
        off the VR180 ``hequirect`` heads (#294/#301).
        """
        if self.input_projection == "equirect":
            return ""
        return f":ih_fov={self.coverage_h_fov:g}:iv_fov={coverage_v_fov:g}"

    def _orientation_terms(self) -> str:
        """The ``:yaw=..:pitch=..:roll=..`` tail, or ``""`` at 0/0/0 (D-2, #371).

        Delegated to :meth:`EquirectangularMapper._v360_orientation_terms` — the
        same reuse-don't-restate rule :meth:`_pinhole_vertical_fov` follows.
        That matters twice over: the rotation convention is ``rorder=ypr``
        recovered by measurement (#324, six orderings fitted, only this one to
        0.003), and the "emit nothing at all when every angle is 0" rule is the
        zero-regression contract both routes are pinned to.
        """
        return self._vr180_geometry()._v360_orientation_terms()

    def _circle_mask_filter(self) -> str:
        """A one-frame black/white **inscribed-circle** mask, built inside ffmpeg.

        A domemaster's image circle is the inscribed circle of the square canvas
        and the corners outside it must be pure black (IMERSA); ``alpha_mask``
        alone does not guarantee that.  It happens to for ``input=flat`` — the
        corners are at polar angle > 90°, i.e. behind the pinhole — but for the
        equirect/fisheye sources D-2 is about to add, those same corners are
        perfectly valid below-the-horizon content and would leak straight out to
        1.414 R.  The mask makes the guarantee unconditional.

        Implementation notes:

        * The mask is generated by a ``color`` **source filter inside the
          filtergraph** (``r=1:d=1`` → exactly one frame), so there is no second
          ``-i``, no temp PNG to write and clean up, and nothing to cache: the
          expensive part (``geq``) runs once for the whole render, not per frame.
          Measured at 4096²/24 frames: 5.07 s with the mask vs 4.75 s without.
        * ``blend=all_mode=multiply`` computes ``A*B/255``, so a 255 mask pixel
          is a bit-exact identity (verified: zero differing bytes inside the
          circle) and a 0 mask pixel is exactly black.  ``repeatlast=1``
          (blend's default) holds that single mask frame against every frame of
          the video — verified on a 72-frame clip, last frame included.
        * The radius test uses pixel **centres** (``X + 0.5 - size/2``), which
          is why the expression subtracts ``(size - 1) / 2``.
        """
        centre = (self.output_size - 1) / 2.0
        radius = self.output_size / 2.0
        inside = f"if(lte(hypot(X-{centre:g},Y-{centre:g}),{radius:g}),255,0)"
        return (
            f"color=c=black:s={self.output_size}x{self.output_size}:r=1:d=1"
            f",format=gbrp"
            f",geq=r='{inside}':g='{inside}':b='{inside}'"
        )

    def _filter_complex(self, coverage_v_fov: float) -> str:
        """The whole graph: v360 → black composite → circle mask → yuv420p.

        Kept separate from :meth:`convert` so the geometry can be rendered (and
        deliberately mutated) by tests without going through the encoder wiring.
        """
        return (
            f"[0:v]{self._v360_filter(coverage_v_fov)},{self._BLACK_COMPOSITE},format=gbrp[{self.DOME_LABEL}];"
            f"{self._circle_mask_filter()}[{self.CIRCLE_LABEL}];"
            f"[{self.DOME_LABEL}][{self.CIRCLE_LABEL}]"
            f"blend=all_mode=multiply:repeatlast=1,format=yuv420p[{self.OUT_LABEL}]"
        )

    def _probe_coverage_v_fov(self, input_path: str) -> float:
        """Probe source video dimensions and derive vertical coverage FOV.

        D-1 (#334) defect 3: this used to return ``coverage_h_fov * height /
        width`` — a *linear* ratio.  ``input=flat`` is a pinhole camera, where
        the vertical span is ``2*atan(tan(hfov/2) * h/w)``; the two agree only
        on a square frame (``h/w == 1``), which is exactly why the bug survived
        — the house 1:1 sources hid it.  On a 16:9 source the linear rule gives
        67.5° where the geometry needs 88.51°, so the frame was squashed by 21°
        of vertical field.

        The correct formula is **not restated here**: it is
        :meth:`EquirectangularMapper._calc_vertical_fov`, and this calls that
        one implementation so the dome and VR180 sides cannot drift apart.
        """
        import json

        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            input_path,
        ]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
            if probe_result.returncode != 0:
                log.warning(f"ffprobe failed, falling back to default iv_fov=90: {probe_result.stderr}")
                return 90.0
            info = json.loads(probe_result.stdout)
            streams = info.get("streams", [])
            if not streams:
                return 90.0
            w = int(streams[0].get("width", 1920))
            h = int(streams[0].get("height", 1080))
        except Exception as exc:
            log.warning(f"Could not probe source dimensions: {exc}")
            return 90.0

        if h <= 0 or w <= 0:
            return 90.0

        if self.input_projection == "fisheye":
            computed = self._fisheye_vertical_fov(w, h)
            rule = f"equidistant: {self.coverage_h_fov:g}°·{h}/{w}"
        else:
            computed = self._pinhole_vertical_fov(w, h)
            rule = f"pinhole: 2·atan(tan({self.coverage_h_fov:g}°/2)·{h}/{w})"
        log.info(f"Source {w}×{h} → auto iv_fov = {computed:.2f}° ({rule})")
        return computed

    def _pinhole_vertical_fov(self, src_width: int, src_height: int) -> float:
        """Pinhole vertical FOV for *coverage_h_fov* — the VR180 implementation.

        Deliberately a one-line delegation rather than a copy of the ``atan``:
        a second copy of the formula is a second thing to keep in sync, and the
        drift is silent (a square source makes both formulas agree).
        """
        return self._vr180_geometry()._calc_vertical_fov(src_width, src_height)

    def _fisheye_vertical_fov(self, src_width: int, src_height: int) -> float:
        """Equidistant vertical FOV for *coverage_h_fov* (D-2, #371).

        The other half of the same delegation: an equidistant fisheye puts the
        same degrees per pixel on both axes, so the vertical span is the
        aspect-scaled horizontal one with no ``tan`` anywhere — the #303
        convention, owned by
        :meth:`EquirectangularMapper._fisheye_vertical_fov`.  Feeding a fisheye
        source through :meth:`_pinhole_vertical_fov` instead is exactly the
        silent-on-square-sources failure #334 found on the other branch.
        """
        return self._vr180_geometry()._fisheye_vertical_fov(src_width, src_height)

    def _vr180_geometry(self) -> EquirectangularMapper:
        """A VR180 mapper configured like this dome, for geometry delegation.

        The dome route owns no geometry formulas of its own: the pinhole solve
        (#334), the equidistant solve (#303) and the ``rorder=ypr`` orientation
        tail (#323/#324) all live on :class:`EquirectangularMapper` and are
        reached through here.  ``use_ffmpeg=False`` keeps the delegate inert —
        its constructor allocates nothing but a handful of attributes and never
        spawns a worker — and it doubles as the validator for the angles and
        for ``coverage_h_fov``, which it bounds to v360's own legal ranges.

        ``coverage_h_fov`` feeds *both* ``src_hfov`` (pinhole) and
        ``fisheye_fov`` (equidistant) because on the dome route it is the one
        "how much of the sphere does the source cover" knob
        (``--dome-coverage-h``); which of the two the delegate actually reads
        is decided by the caller, not by a second flag.
        """
        return EquirectangularMapper(
            src_hfov=self.coverage_h_fov,
            fisheye_fov=self.coverage_h_fov,
            use_ffmpeg=False,
            pitch=self.pitch,
            yaw=self.yaw,
            roll=self.roll,
        )
