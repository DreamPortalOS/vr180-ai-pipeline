"""Tests for issue #267 (F-4) — ``--outpaint`` really applied on the streaming path.

#243 forwarded ``--outpaint`` (and the mask sub-params) into
:class:`StreamingPipeline` but only *stored* them: the streaming module had
no call site, so ``--quality standard --outpaint gradient`` (the production
route) changed nothing — silently, because the swallowed-arg detector was now
satisfied.  This card applies the fill per frame between the equirect map and
the #261 feather, reusing ``pipeline.outpainter``'s gradient filler and the
alpha plane #266 already derives.

Covered here (the #266 harness: fake depth/stereo backends, the real OpenCV
mapper at 192²/eye, cv2/ffmpeg mocked, no model):

  - ``none`` (default) ⇒ the bytes handed to the encoder are exactly the
    mapper's output, frame for frame, and no alpha plane is derived;
  - ``gradient`` ⇒ the FOV hole (black on the mapper's output) gets non-zero
    pixels, the content is untouched, and the frame equals the batch
    ``Outpainter`` result (MAE < 2 per the card; in fact bit-exact) — on both
    fuse-loop branches;
  - ``gradient`` + feather 165→180 ⇒ fill first, then a fade anchored at the
    physical 180° rim (the #259 rule) that is monotone down to 0 — and equal
    to the batch path;
  - the alpha plane is derived once and shared by the mask and the feather;
  - ``ai`` ⇒ exactly one degrade WARNING and output identical to ``none``;
  - an unknown mode raises ValueError (same rule as ``Outpainter``).
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.outpainter import Outpainter  # noqa: E402
from tests.test_streaming_feather import (  # noqa: E402
    _EYE,
    _alpha_calls,
    _FakeWholeClipStereo,
    _make_pipeline,
    _reference,
    _run,
    _src_rgb,
)

_STREAM_LOGGER = "vr180-streaming"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _batch(mode: str, **feather) -> np.ndarray:
    """The non-streaming result (Stage 3.5, ``run_outpaint_stage``) for the same frame and alpha."""
    sbs, alpha = _reference()
    return Outpainter(mode=mode, **feather).process([sbs], alpha=alpha)[0]


def _hole_changed(frame: np.ndarray) -> tuple[int, int]:
    """(pixels changed inside the FOV hole, hole size) relative to the mapper's output."""
    sbs, alpha = _reference()
    hole = alpha == 0
    return int(np.count_nonzero(np.any(frame != sbs, axis=2) & hole)), int(np.count_nonzero(hole))


def _equator_ray_to_rim(frame: np.ndarray) -> np.ndarray:
    """Brightness (channel 0) along the left eye's equator, centre → frame edge (the 180° rim)."""
    return frame[_EYE // 2, _EYE // 2 : _EYE, 0].astype(int)


def _content_edge_on_equator() -> int:
    """Index (into the ray above) of the last content pixel before the hole."""
    _, alpha = _reference()
    return int(np.flatnonzero(alpha[_EYE // 2, _EYE // 2 : _EYE] > 0)[-1])


def _degrade_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == _STREAM_LOGGER and "outpaint" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
#  Regression: none (default) ⇒ pixel-identical to the pre-#267 stream
# ---------------------------------------------------------------------------


class TestNoneIsByteIdentical:
    def test_default_mode_is_none(self):
        p = _make_pipeline()
        assert p.outpaint == "none"
        assert p._fill_mode == "none"
        assert not p._stage35_enabled

    @pytest.mark.parametrize("kwargs", [{}, {"outpaint": "none"}])
    def test_encoder_receives_exactly_the_mapper_output(self, kwargs):
        p = _make_pipeline(**kwargs)
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=2)
        sbs, _ = _reference()
        assert len(frames) == 2
        for f in frames:
            assert np.array_equal(f, sbs)
        # No alpha plane is derived when there is nothing to fill or feather — no extra mapper work at all.
        assert _alpha_calls(p.eq_mapper) == []

    def test_mask_sub_params_are_still_stored(self):
        """#243 parity: the sub-params keep reaching the instance (they drive the batch fallback mask)."""
        p = _make_pipeline(
            outpaint="gradient", outpaint_mask_threshold=20, outpaint_mask_top_ratio=0.3, outpaint_mask_bottom_ratio=0.4
        )
        assert (p.outpaint_mask_threshold, p.outpaint_mask_top_ratio, p.outpaint_mask_bottom_ratio) == (20, 0.3, 0.4)


# ---------------------------------------------------------------------------
#  gradient: the hole is really filled, per frame, like the batch path
# ---------------------------------------------------------------------------


class TestGradientFillsTheHole:
    def test_hole_pixels_become_nonzero_and_content_is_untouched(self):
        p = _make_pipeline(outpaint="gradient")
        assert p._fill_mode == "gradient" and p._stage35_enabled
        out = _run(p, num_frames=1)[0]
        sbs, alpha = _reference()
        hole = alpha == 0
        assert hole.any() and np.all(sbs[hole] == 0), "precondition: the 126° source leaves a black hole"
        changed, hole_px = _hole_changed(out)
        assert changed > 0, "the stream must change pixels outside the FOV (the #267 defect: it changed none)"
        assert changed > 0.5 * hole_px, f"only {changed}/{hole_px} hole pixels filled — smear did not reach the hole"
        assert np.array_equal(out[~hole], sbs[~hole]), "the fill never touches content pixels"
        # (No "both eyes alike" check: the batch filler works on the whole SBS frame and splits the
        # interior run between the eyes at its midpoint, so the two inner side holes can differ by
        # one column of smear band — the stream reproduces that bit-exactly, see the next test.)

    def test_matches_the_batch_path_outpainter(self):
        out = _run(_make_pipeline(outpaint="gradient"), num_frames=1)[0]
        batch = _batch("gradient")
        mae = float(np.abs(out.astype(int) - batch.astype(int)).mean())
        assert mae < 2.0, f"MAE {mae}"
        assert np.array_equal(out, batch), "same filler on the same alpha mask ⇒ bit-exact"

    def test_mask_is_built_once_for_the_whole_stream(self):
        p = _make_pipeline(outpaint="gradient")
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=3)
        assert len(_alpha_calls(p.eq_mapper)) == 1, "the mask is geometry-only: alpha derived once, not per frame"
        assert len(frames) == 3
        assert all(np.array_equal(f, frames[0]) for f in frames)
        assert p._boundary_plan is not None and p._boundary_plan.fill_mask is not None
        assert p._boundary_plan.feather is None

    def test_wholeclip_stereo_branch_is_filled_too(self, tmp_path):
        """The precomputed-L/R fuse loop (StereoCrafter) goes through the same Stage 3 as the per-frame loop."""
        src = _src_rgb()
        p = _make_pipeline(outpaint="gradient", stereo_renderer=_FakeWholeClipStereo(), temp_dir=str(tmp_path))
        left = [src.copy(), src.copy()]
        right = [src.copy(), src.copy()]
        with patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]):
            frames = _run(p, num_frames=2)
        batch = _batch("gradient")
        assert len(frames) == 2
        for f in frames:
            assert np.array_equal(f, batch)

    def test_equirect_stage_timing_still_reported(self):
        p = _make_pipeline(outpaint="gradient")
        _run(p, num_frames=2)
        assert p.stage_timings["equirect"] > 0.0


# ---------------------------------------------------------------------------
#  gradient + feather: fill first, then fade to black at the physical rim
# ---------------------------------------------------------------------------


class TestGradientThenFeather:
    def test_fill_precedes_feather_and_the_fade_is_monotone_to_zero_at_the_rim(self):
        p = _make_pipeline(outpaint="gradient", edge_feather_start=165, edge_feather_end=180)
        out = _run(p, num_frames=1)[0]
        sbs, _ = _reference()
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        ray = _equator_ray_to_rim(out)
        sbs_ray = _equator_ray_to_rim(sbs)
        edge = _content_edge_on_equator()
        assert ray[0] == orig, "centre untouched"
        # The content — including the mapper's anti-aliased last pixel before the hole — is
        # byte-exact: the filler restores it and a 165°→180° feather anchored at the rim never
        # reaches the 126° edge.  (The alpha-anchored feather of ``--outpaint none`` blacks that
        # edge pixel out — see the negative control below.)
        assert ray[: edge + 1].tolist() == sbs_ray[: edge + 1].tolist(), "content untouched up to the FOV edge"
        assert ray[edge] > 0, "the content edge is not blacked out: the feather anchors at the rim, not the alpha edge"
        # Fill → feather: the first former-hole pixel carries the smeared content edge at (near)
        # full strength.  With the opposite order the alpha-anchored feather would black out the
        # content edge first and the filler would then smear black — the hole would stay black.
        assert ray[edge + 1] >= 0.5 * ray[edge], (
            "the pixel just outside the FOV is painted (fill ran before the feather)"
        )
        assert ray[-1] == 0, "fully black at the physical 180° rim"
        hole_ray = ray[edge + 1 :]
        assert np.all(np.diff(hole_ray) <= 0), "monotone from the first hole pixel to the rim"
        assert (-np.diff(hole_ray)).max() <= 0.5 * hole_ray[0], "no step — a hard edge would be a full drop to 0"
        changed, _ = _hole_changed(out)
        assert changed > 0

    def test_matches_the_batch_path_outpainter(self):
        p = _make_pipeline(outpaint="gradient", edge_feather_start=165, edge_feather_end=180)
        out = _run(p, num_frames=1)[0]
        batch = _batch("gradient", edge_feather_start=165, edge_feather_end=180)
        mae = float(np.abs(out.astype(int) - batch.astype(int)).mean())
        assert mae < 2.0, f"MAE {mae}"
        assert np.array_equal(out, batch), "same filler + same rim-anchored feather ⇒ bit-exact"

    def test_alpha_is_derived_once_and_shared_by_mask_and_feather(self):
        p = _make_pipeline(outpaint="gradient", edge_feather_start=165)
        p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
        frames = _run(p, num_frames=3)
        assert len(_alpha_calls(p.eq_mapper)) == 1, (
            "one map_single(with_alpha=True) feeds both the mask and the weights"
        )
        assert len(frames) == 3 and all(np.array_equal(f, frames[0]) for f in frames)
        plan = p._boundary_plan
        assert plan is not None and plan.fill_mask is not None and plan.feather is not None

    def test_feather_alone_is_unchanged_from_261(self):
        """Negative control: with no fill the feather still anchors at the alpha edge (#266 behaviour)."""
        out = _run(_make_pipeline(edge_feather_start=165, edge_feather_end=180), num_frames=1)[0]
        assert np.array_equal(out, _batch("none", edge_feather_start=165, edge_feather_end=180))
        assert _equator_ray_to_rim(out)[_content_edge_on_equator()] == 0, "alpha-anchored: content edge is black"


# ---------------------------------------------------------------------------
#  ai: not available on the stream ⇒ one loud WARNING, then behave as none
# ---------------------------------------------------------------------------


class TestAiDegradesToNone:
    def test_warns_exactly_once_and_output_equals_none(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_STREAM_LOGGER):
            p = _make_pipeline(outpaint="ai")
            p.eq_mapper.map_single = MagicMock(wraps=p.eq_mapper.map_single)
            frames = _run(p, num_frames=2)
        warnings = _degrade_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one degrade warning, got {warnings}"
        assert "degrading" in warnings[0] and "none" in warnings[0]
        assert p.outpaint == "ai", "the requested mode is preserved (#243 passthrough contract)"
        assert p._fill_mode == "none"
        sbs, _ = _reference()
        assert len(frames) == 2
        for f in frames:
            assert np.array_equal(f, sbs), "ai degrades to none: bytes identical to the mapper's output"
        assert _alpha_calls(p.eq_mapper) == []

    def test_ai_with_feather_keeps_the_alpha_anchored_feather(self):
        """Degrading to none means the feather anchors at the alpha edge, exactly like ``--outpaint none``."""
        out = _run(_make_pipeline(outpaint="ai", edge_feather_start=165, edge_feather_end=180), num_frames=1)[0]
        assert np.array_equal(out, _batch("none", edge_feather_start=165, edge_feather_end=180))

    def test_unknown_mode_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown outpaint mode"):
            _make_pipeline(outpaint="blur")
