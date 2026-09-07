"""Tests for pipeline/sphere_accumulator.py — F-2 temporal spherical accumulation (#262).

Synthetic frames only (concentric-ring pattern flowing radially outward on a
square hemisphere canvas); no model, no ffmpeg.
"""

import cv2
import numpy as np
import pytest

import pipeline.outpainter as outpainter_mod
import pipeline.sphere_accumulator as accumulator_mod
from pipeline.outpainter import Outpainter, _clear_feather_caches, apply_edge_feather, compute_edge_feather_weights
from pipeline.sphere_accumulator import (
    DEFAULT_SEAM_FEATHER_DEG,
    SphereAccumulator,
    radial_warp_maps,
    seam_weights,
    warp_radial,
)

SIZE = 256  # per-eye canvas (square hemisphere)
SCALE = 1.05  # per-frame radial expansion while flying forward
N_FRAMES = 10
SRC_HFOV = 126.0  # production source FOV → the outer ~27° of the canvas is a hole

# ---------------------------------------------------------------------------
#  Synthetic geometry / frames
# ---------------------------------------------------------------------------


def _polar(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Pixel radius / azimuth about the canvas centre ``(size - 1) / 2``."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.hypot(xx - c, yy - c), np.arctan2(yy - c, xx - c)


def _fov_mask(size: int = SIZE, hfov: float = SRC_HFOV) -> np.ndarray:
    """Covered area of the source: pixel disk whose equator radius is *hfov* on the 0–180 scale."""
    r_px, _ = _polar(size)
    return r_px <= (hfov / 180.0) * (size / 2.0)


def _ring_pattern(k: int, size: int = SIZE, scale: float = SCALE) -> np.ndarray:
    """Whole-canvas RGB pattern at frame *k*: concentric rings whose radii grow by ``scale`` per frame.

    Radial sinusoid (period ``size / 8`` px at frame 0) plus azimuth-only terms
    that are invariant under the radial flow — so ``pattern(k)`` is exactly
    ``pattern(k - 1)`` expanded by *scale* about the centre.
    """
    r_px, phi = _polar(size)
    r = r_px / scale**k
    period = size / 8.0
    ch_r = 128 + 100 * np.sin(2 * np.pi * r / period)
    ch_g = 128 + 100 * np.cos(2 * np.pi * r / period) * np.cos(2 * phi)
    ch_b = 128 + 100 * np.sin(3 * phi)
    return np.clip(np.rint(np.dstack([ch_r, ch_g, ch_b])), 0, 255).astype(np.uint8)


def _flow_frame(k: int, size: int = SIZE) -> np.ndarray:
    """RGBA frame *k* of the forward-flight sequence: pattern inside the FOV, alpha 0 outside."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = _ring_pattern(k, size)
    rgba[:, :, 3] = _fov_mask(size) * 255
    return rgba


def _uniform_frame(value: int = 200, size: int = SIZE) -> np.ndarray:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = value
    rgba[:, :, 3] = _fov_mask(size) * 255
    return rgba


def _equator_ray(rgb: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Channel-0 brightness along the equator, centre → right rim."""
    return rgb[size // 2, size // 2 :, 0].astype(int)


def _ray_index_at_fov(fov_deg: float, size: int = SIZE) -> int:
    """Index into :func:`_equator_ray` of the pixel centred at *fov_deg* (0–180 scale)."""
    return round((fov_deg / 360.0 + 0.5) * size - 0.5) - size // 2


def _run(acc: SphereAccumulator, scale: float, n: int = N_FRAMES, frames=_flow_frame) -> np.ndarray:
    out = None
    for k in range(n):
        out = acc.update(frames(k), scale)
    assert out is not None
    return out


def _hole_stats(acc: SphereAccumulator, out: np.ndarray) -> tuple[float, float]:
    """(buffer-alpha coverage, non-zero output fraction) over the hole outside the source FOV."""
    hole = ~_fov_mask()
    return float(np.mean(acc.alpha[hole] > 0)), float(np.mean(out[hole].max(axis=1) > 0))


# ---------------------------------------------------------------------------
#  Pure functions
# ---------------------------------------------------------------------------


class TestPureFunctions:
    def test_radial_warp_maps_scale_about_the_centre(self):
        map_x, map_y = radial_warp_maps((129, 129), 2.0)
        assert map_x.shape == map_y.shape == (129, 129) and map_x.dtype == np.float32
        assert (map_x[64, 64], map_y[64, 64]) == (64.0, 64.0), "centre is a fixed point"
        assert (map_x[64, 104], map_y[64, 104]) == (84.0, 64.0), "output at +40 samples input at +20"
        assert (map_x[24, 64], map_y[24, 64]) == (64.0, 44.0)

    def test_warp_radial_moves_content_outward(self):
        buf = np.zeros((129, 129, 4), dtype=np.float32)
        buf[64, 84] = 1.0  # 20 px right of the centre
        out = warp_radial(buf, 2.0)
        assert out.shape == buf.shape and out.dtype == np.float32
        assert out[64, 104, 3] == pytest.approx(1.0), "lands 40 px right of the centre"
        assert out[64, 84, 3] == pytest.approx(0.0), "nothing left at the old spot"
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_warp_radial_expands_a_disk_and_fills_the_outside_with_transparent(self):
        r_px, _ = _polar(129)
        buf = np.zeros((129, 129, 4), dtype=np.float32)
        buf[r_px <= 40] = 1.0
        out = warp_radial(buf, 1.5)
        edge = np.flatnonzero(out[64, 64:, 3] > 0.5)[-1]
        assert abs(edge - 60) <= 1, f"disk radius 40 → 60 after ×1.5, got {edge}"
        assert np.all(out[r_px > 62] == 0), "outside the expanded disk is transparent"

    def test_seam_weights_ramp_inside_the_content_edge(self):
        covered = _fov_mask()
        w = seam_weights(covered, feather_px=6.0)
        assert w.shape == covered.shape and w.dtype == np.float32
        assert np.all(w[~covered] == 0), "hole gets no current-frame weight"
        ray = w[SIZE // 2, SIZE // 2 :]
        content = np.flatnonzero(covered[SIZE // 2, SIZE // 2 :])
        assert ray[content[-1]] == 0.0, "last content pixel before the hole is fully blended"
        assert ray[0] == 1.0 and ray[content[-1] - 10] == 1.0, "deep inside → 1"
        assert np.all(np.diff(ray[: content[-1] + 1]) <= 0), "monotone from centre to the edge"
        assert 0 < ray[content[-1] - 3] < 1, "the ramp lives inside the content"

    def test_seam_weights_degenerate_masks(self):
        full = np.ones((32, 32), dtype=bool)
        assert np.all(seam_weights(full, 6.0) == 1.0), "no hole → no seam, canvas border is the rim"
        assert np.all(seam_weights(np.zeros((32, 32), dtype=bool), 6.0) == 0.0)
        disk = _fov_mask(32)
        assert np.array_equal(seam_weights(disk, 0.0), disk.astype(np.float32)), "0 px feather → binary"


# ---------------------------------------------------------------------------
#  Cold start: frame 0 is the input plus the fade (regression baseline)
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_single_frame_is_input_plus_fade(self):
        frame = _flow_frame(0)
        out = SphereAccumulator(SIZE).update(frame, SCALE)
        masked = frame[:, :, :3] * _fov_mask()[:, :, None]
        expected = apply_edge_feather(masked, compute_edge_feather_weights(frame[:, :, 3], 165.0, 180.0))
        assert np.array_equal(out, expected), "frame 0 == input on black × 165→180 fade, byte-exact"
        assert np.all(out[~_fov_mask()] == 0), "the hole stays black — nothing to remember yet"

    def test_single_frame_matches_the_outpainter_feather(self):
        """Same bytes as the existing Stage 3.5 feather (mode=none, 165→180) on an SBS of this eye."""
        frame = _flow_frame(0)
        masked = frame[:, :, :3] * _fov_mask()[:, :, None]
        sbs = np.concatenate([masked, masked], axis=1)
        alpha_sbs = np.concatenate([frame[:, :, 3], frame[:, :, 3]], axis=1)
        ref = Outpainter(mode="none", edge_feather_start=165, edge_feather_end=180).process([sbs], alpha=alpha_sbs)[0]
        out = SphereAccumulator(SIZE, 165.0, 180.0).update(frame, SCALE)
        assert np.array_equal(out, ref[:, :SIZE])

    def test_fade_off_single_frame_is_input_on_black(self):
        frame = _flow_frame(0)
        out = SphereAccumulator(SIZE, None, None).update(frame, 1.0)
        assert np.array_equal(out, frame[:, :, :3] * _fov_mask()[:, :, None])

    def test_output_shape_dtype_and_introspection(self):
        acc = SphereAccumulator(SIZE)
        assert acc.frames_seen == 0 and acc.fade == (165.0, 180.0)
        assert acc.seam_feather_deg == DEFAULT_SEAM_FEATHER_DEG
        assert acc.alpha.shape == (SIZE, SIZE) and acc.alpha.dtype == np.float32 and not acc.alpha.any()
        out = acc.update(_flow_frame(0), SCALE)
        assert out.shape == (SIZE, SIZE, 3) and out.dtype == np.uint8
        assert acc.frames_seen == 1
        assert np.array_equal(acc.alpha > 0, _fov_mask()), "coverage after frame 0 is exactly the source FOV"
        assert acc.alpha.min() >= 0.0 and acc.alpha.max() <= 1.0


# ---------------------------------------------------------------------------
#  Forward flight: the outer ring is filled from history
# ---------------------------------------------------------------------------


class TestForwardFlow:
    def test_outer_ring_is_filled_after_ten_frames(self):
        acc = SphereAccumulator(SIZE)
        out0 = acc.update(_flow_frame(0), SCALE)
        cov0, nz0 = _hole_stats(acc, out0)
        assert cov0 == 0.0 and nz0 == 0.0, "frame 0: the hole is empty"

        out = None
        for k in range(1, N_FRAMES):
            out = acc.update(_flow_frame(k), SCALE)
        assert out is not None
        cov, nz = _hole_stats(acc, out)
        # Geometry: FOV radius 0.7·128 = 89.6 px grows by 1.05⁹ → 139 px, past the rim on the
        # axes but short of the corners (181 px) → ~87 % of the hole is remembered content.
        assert cov > 0.8, f"buffer coverage of the hole after {N_FRAMES} frames: {cov:.3f}"
        assert nz > 0.7, f"non-zero output in the hole after the fade: {nz:.3f}"
        at_150 = _ray_index_at_fov(150)
        assert _equator_ray(out0)[at_150] == 0 and out[SIZE // 2, SIZE // 2 + at_150].max() > 0, (
            "150° on the equator: black at frame 0, real content at frame 10"
        )

    def test_outer_ring_matches_the_true_content_mae_below_15(self):
        """What the ring shows must be what was really there — compared against
        (a) the previous frame's content moved to its current position and
        (b) the analytic whole-canvas pattern of the current frame."""
        acc = SphereAccumulator(SIZE, None, None)  # fade off: measure the raw accumulation
        out = _run(acc, SCALE)
        hole = ~_fov_mask()
        solid = cv2.erode((acc.alpha > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        region = hole & solid
        assert region.sum() > 0.5 * hole.sum(), "most of the hole is solid remembered content"

        prev_moved = warp_radial(_ring_pattern(N_FRAMES - 2).astype(np.float32), SCALE)
        mae_prev = np.abs(out[region].astype(np.float32) - prev_moved[region]).mean()
        assert mae_prev < 15.0, f"MAE vs previous frame moved outward: {mae_prev:.2f}"

        truth = _ring_pattern(N_FRAMES - 1)
        mae_true = np.abs(out[region].astype(np.float32) - truth[region]).mean()
        assert mae_true < 15.0, f"MAE vs the true content at frame {N_FRAMES - 1}: {mae_true:.2f}"

    def test_default_fade_leaves_the_inner_ring_content_intact(self):
        raw = _run(SphereAccumulator(SIZE, None, None), SCALE)
        acc = SphereAccumulator(SIZE)
        out = _run(acc, SCALE)
        weights = compute_edge_feather_weights((acc.alpha > 0).astype(np.uint8) * 255, 165.0, 180.0)
        keep = (weights >= 0.999) & ~_fov_mask()
        assert keep.sum() > 0.4 * (~_fov_mask()).sum()
        assert np.array_equal(out[keep], raw[keep]), "inside the fade start the ring is the raw accumulation"
        assert np.all(out[SIZE // 2, -1] == 0) and np.all(out[SIZE // 2, 0] == 0), "rim is black"

    def test_current_frame_wins_inside_its_fov(self):
        acc = SphereAccumulator(SIZE, None, None)
        out = _run(acc, SCALE)
        frame = _flow_frame(N_FRAMES - 1)
        interior = cv2.erode(_fov_mask().astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        assert np.array_equal(out[interior], frame[:, :, :3][interior]), "deep inside the FOV: the live frame"


# ---------------------------------------------------------------------------
#  radial_scale <= 1: composite only, never expand
# ---------------------------------------------------------------------------


class TestStaticAndBackward:
    @pytest.mark.parametrize("scale", [1.0, 0.9])
    def test_no_expansion_keeps_the_hole_black(self, scale):
        acc = SphereAccumulator(SIZE)
        out = _run(acc, scale)
        hole = ~_fov_mask()
        assert not out[hole].any(), "outer ring stays black without forward motion"
        assert not acc.alpha[hole].any(), "and the buffer never grows"
        assert np.array_equal(acc.alpha > 0, _fov_mask())

    def test_static_uniform_sequence_is_stable(self):
        acc = SphereAccumulator(SIZE)
        first = acc.update(_uniform_frame(), 1.0)
        for _ in range(N_FRAMES - 1):
            assert np.array_equal(acc.update(_uniform_frame(), 1.0), first), "no drift, no cumulative darkening"

    def test_backward_step_does_not_shrink_what_was_remembered(self):
        acc = SphereAccumulator(SIZE)
        _run(acc, SCALE, n=4)
        remembered = acc.alpha
        assert remembered[~_fov_mask()].any()
        acc.update(_flow_frame(4), 0.9)
        assert np.array_equal(acc.alpha > 0, remembered > 0), "a backward frame only composites"


# ---------------------------------------------------------------------------
#  Fade: the outermost 165→180 still goes monotonically to black after accumulation
# ---------------------------------------------------------------------------


class TestFade:
    def test_fade_monotone_to_zero_after_accumulation(self):
        acc = SphereAccumulator(SIZE)
        ray0 = _equator_ray(acc.update(_uniform_frame(), SCALE))
        assert ray0[_ray_index_at_fov(150)] == 0, "150° is in the hole at frame 0"
        out = _run(acc, SCALE, frames=lambda _k: _uniform_frame())
        ray = _equator_ray(out)
        assert ray[0] == 200, "centre untouched"
        assert ray[-1] == 0, "rim fully black"
        assert np.all(np.diff(ray) <= 0), "non-increasing from centre to rim"
        assert (-np.diff(ray)).max() <= 40, f"largest step {(-np.diff(ray)).max()} looks like a hard edge"
        assert ray[_ray_index_at_fov(150)] == 200, "the ring is real content at full brightness now"
        assert ray[_ray_index_at_fov(160)] == 200, "fade starts at 165°"
        assert 0 < ray[_ray_index_at_fov(172)] < 200, "172° sits inside the ramp"

    def test_fade_anchors_at_the_accumulated_edge_while_the_ring_is_still_growing(self):
        acc = SphereAccumulator(SIZE)
        out = _run(acc, SCALE, n=3, frames=lambda _k: _uniform_frame())  # edge at 89.6·1.05² ≈ 99 px
        ray = _equator_ray(out)
        edge = np.flatnonzero(acc.alpha[SIZE // 2, SIZE // 2 :] > 0)[-1]
        assert 96 <= edge <= 100
        assert ray[0] == 200 and ray[_ray_index_at_fov(110)] == 200
        assert ray[edge] == 0 and np.all(ray[edge + 1 :] == 0), "black from the accumulated edge outward"
        assert np.all(np.diff(ray[: edge + 1]) <= 0)
        assert 0 < ray[edge - 3] < 200, "ramp lives inside the remembered content, not at the 180° rim"

    def test_fade_is_output_only_and_never_darkens_the_buffer(self):
        acc = SphereAccumulator(SIZE)
        out = _run(acc, SCALE, frames=lambda _k: _uniform_frame())
        inner = _polar()[0] <= _ray_index_at_fov(150)
        assert np.all(out[inner] == 200), "everything inside 150° — FOV and remembered ring alike — is exact"


# ---------------------------------------------------------------------------
#  Validation / reset
# ---------------------------------------------------------------------------


class TestValidation:
    def test_frame_shape_and_dtype(self):
        acc = SphereAccumulator(SIZE)
        with pytest.raises(ValueError, match="shape"):
            acc.update(np.zeros((SIZE, SIZE, 3), np.uint8), 1.0)
        with pytest.raises(ValueError, match="shape"):
            acc.update(np.zeros((SIZE, SIZE // 2, 4), np.uint8), 1.0)
        with pytest.raises(ValueError, match="shape"):
            acc.update(np.zeros((SIZE // 2, SIZE // 2, 4), np.uint8), 1.0)
        with pytest.raises(TypeError, match="uint8"):
            acc.update(np.zeros((SIZE, SIZE, 4), np.float32), 1.0)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_radial_scale_must_be_finite_positive(self, scale):
        with pytest.raises(ValueError, match="radial_scale"):
            SphereAccumulator(SIZE).update(_flow_frame(0), scale)

    def test_constructor_arguments(self):
        with pytest.raises(ValueError):
            SphereAccumulator(1)
        with pytest.raises(ValueError):
            SphereAccumulator(SIZE, 170.0, 165.0)  # start > end
        with pytest.raises(ValueError):
            SphereAccumulator(SIZE, 165.0, 190.0)
        with pytest.raises(ValueError):
            SphereAccumulator(SIZE, seam_feather_deg=-1.0)
        assert SphereAccumulator(SIZE, None, None).fade is None
        assert SphereAccumulator(SIZE, 150.0, None).fade == (150.0, 180.0), "one bound → other at its default"

    def test_reset_forgets_everything(self):
        acc = SphereAccumulator(SIZE)
        _run(acc, SCALE, n=3)
        assert acc.frames_seen == 3 and acc.alpha[~_fov_mask()].any()
        acc.reset()
        assert acc.frames_seen == 0 and not acc.alpha.any()
        assert np.array_equal(acc.update(_flow_frame(0), SCALE), SphereAccumulator(SIZE).update(_flow_frame(0), SCALE))


# ---------------------------------------------------------------------------
#  Issue #276 — the fade-weights memo in front of the outpainter's cache.
#
#  The card proposed dropping ``_fade_weights``' whole-canvas ``array_equal``
#  memo and relying on the outpainter's mask cache alone.  Measured at 2880²
#  (numbers in PR #276's description): the memo costs 2 ms/frame while the mask
#  grows, but once the mask is static (the steady state after the ring has
#  filled, and any ``radial_scale <= 1`` sequence) it *saves* ~11 ms/frame —
#  delegating would pay uint8 conversion + digest + exact confirm + a 33 MB
#  copy on every hit.  The memo therefore stays.  These spies pin what it must
#  do: compute an identical mask once, never serve stale weights, forget on
#  reset — and its hit must be byte-identical to a fresh outpainter call.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_feather_caches():
    _clear_feather_caches()
    yield
    _clear_feather_caches()


def _spy(monkeypatch, module, name, counter, key):
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        counter[key] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


class TestFadeWeightsMemo:
    def test_same_mask_on_consecutive_frames_computes_the_weights_once(self, monkeypatch, fresh_feather_caches):
        """Two frames with an identical coverage mask: one call into the outpainter,
        one ray trace inside it; the second frame is served by the memo before
        the outpainter's own (digest + exact-confirm + copy) hit path runs."""
        calls = {"weights": 0, "trace": 0}
        _spy(monkeypatch, accumulator_mod, "compute_edge_feather_weights", calls, "weights")
        _spy(monkeypatch, outpainter_mod, "_edge_from_covered", calls, "trace")

        acc = SphereAccumulator(SIZE)
        first = acc.update(_uniform_frame(), 1.0)
        second = acc.update(_uniform_frame(), 1.0)  # static: identical coverage mask
        assert calls == {"weights": 1, "trace": 1}
        assert outpainter_mod._WEIGHTS_CACHE.hits == 0 and outpainter_mod._WEIGHTS_CACHE.misses == 1
        assert np.array_equal(first, second)

    def test_memo_hit_is_byte_identical_to_a_fresh_outpainter_call(self, fresh_feather_caches):
        acc = SphereAccumulator(SIZE)
        acc.update(_uniform_frame(), 1.0)
        out = acc.update(_uniform_frame(), 1.0)  # memo hit
        _clear_feather_caches()  # force the outpainter to recompute from scratch
        weights = compute_edge_feather_weights((acc.alpha > 0).astype(np.uint8) * 255, 165.0, 180.0)
        expected = apply_edge_feather(_uniform_frame()[:, :, :3] * _fov_mask()[:, :, None], weights)
        assert np.array_equal(out, expected)

    def test_a_changed_mask_is_recomputed_every_frame(self, monkeypatch, fresh_feather_caches):
        """While the accumulated coverage grows no frame may reuse stale weights."""
        calls = {"weights": 0, "trace": 0}
        _spy(monkeypatch, accumulator_mod, "compute_edge_feather_weights", calls, "weights")
        _spy(monkeypatch, outpainter_mod, "_edge_from_covered", calls, "trace")
        acc = SphereAccumulator(SIZE)
        _run(acc, SCALE, n=3)  # radial expansion: the coverage mask grows every frame
        assert calls == {"weights": 3, "trace": 3}
        assert outpainter_mod._WEIGHTS_CACHE.hits == 0

    def test_reset_forgets_the_memo(self, monkeypatch, fresh_feather_caches):
        calls = {"weights": 0}
        _spy(monkeypatch, accumulator_mod, "compute_edge_feather_weights", calls, "weights")
        acc = SphereAccumulator(SIZE)
        acc.update(_uniform_frame(), 1.0)
        acc.reset()
        acc.update(_uniform_frame(), 1.0)
        assert calls["weights"] == 2, "after reset the first frame goes back to the outpainter"

    def test_fade_off_never_touches_the_outpainter(self, monkeypatch, fresh_feather_caches):
        calls = {"weights": 0}
        _spy(monkeypatch, accumulator_mod, "compute_edge_feather_weights", calls, "weights")
        _run(SphereAccumulator(SIZE, None, None), SCALE, n=3)
        assert calls["weights"] == 0 and len(outpainter_mod._WEIGHTS_CACHE) == 0
