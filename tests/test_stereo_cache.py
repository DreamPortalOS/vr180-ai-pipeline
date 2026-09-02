"""Tests for the StereoCrafter content-keyed product cache (issue #183, I-8b).

The cache sits in front of the pluggable backend in
``StereoCrafterRenderer.render_video``: a hit serves the cached L/R videos and
never spawns the (CUDA) subprocess/injected backend; a miss runs the backend
and stores its product.  The key is *content-only* for the input video — same
source at a different path still hits — folded with the stereo backend's
output-affecting params AND the upstream depth side's cache key (issue #159
lead decision: a changed depth model must invalidate stereo, so stereo
computed from the wrong depths is never silently served).

All tests use mock backends and plain files under ``tmp_path``-based
``cache_dir`` roots — no CUDA, no real model, no repo working-tree pollution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.stereo_crafter import (
    StereoCrafterBackend,
    StereoCrafterRenderer,
    _depth_dir_digest,
    _load_stereo_cache,
    _save_stereo_cache,
    _stereo_cache_key,
    _stereo_cache_params,
)

# ---------------------------------------------------------------------------
# Mock backends — pluggable, like the production ones
# ---------------------------------------------------------------------------


class CountingStereoBackend(StereoCrafterBackend):
    """Mock backend that records every call and returns deterministic L/R paths.

    Mirrors the real backend's attribute surface (``max_resolution``,
    ``max_disp``, and the Stage-2 throughput knobs from issue #217) so the
    cache key has something to fold in.  ``call_count`` is the assertion hook
    for "subprocess not spawned".
    """

    def __init__(
        self,
        max_resolution: int = 512,
        max_disp: float = 20.0,
        frames_chunk: int | None = None,
        overlap: int | None = None,
        tile_num: int | None = None,
    ) -> None:
        self.max_resolution = max_resolution
        self.max_disp = max_disp
        self.frames_chunk = frames_chunk
        self.overlap = overlap
        self.tile_num = tile_num
        self.call_count = 0
        self.last_outputs: tuple[str, str] | None = None

    def render_video(
        self,
        input_path: str,
        depth_dir: str,
        output_left: str,
        output_right: str,
    ) -> tuple[str, str]:
        self.call_count += 1
        self.last_outputs = (output_left, output_right)
        # Materialize the L/R outputs the cache will later stage.
        for p in (output_left, output_right):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(b"stereo-mock")
        return output_left, output_right


class BareStereoBackend(StereoCrafterBackend):
    """Backend with NONE of the optional cache params (issue #182 round 1 mirror).

    Has no ``max_resolution`` / ``max_disp`` / throughput knobs — proves the
    safe ``getattr(..., None)`` reads never raise ``AttributeError`` and the
    cache path still works (hit and miss).
    """

    def render_video(
        self,
        input_path: str,
        depth_dir: str,
        output_left: str,
        output_right: str,
    ) -> tuple[str, str]:
        self.call_count = getattr(self, "call_count", 0) + 1
        for p in (output_left, output_right):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(b"stereo-bare")
        return output_left, output_right


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_fake_video(path: Path, content: bytes = b"fake video content") -> Path:
    """Write a plain file standing in for an input clip (NOT a NamedTemporaryFile).

    A plain path is used instead of ``tempfile.NamedTemporaryFile`` because
    on Windows the latter holds an exclusive lock that prevents the content
    digest's ``open(path, 'rb')`` from reading it.  Using a plain file means
    the cache path is exercised for real.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_depth_dir(path: Path, num_frames: int = 3, seed: int = 0) -> Path:
    """Populate *path* with ``depth_*.npy`` maps and return it.

    Distinct ``seed`` values yield byte-distinct files, so two depth dirs can
    stand in for "different depth products" (e.g. produced by different depth
    models).
    """
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(num_frames):
        np.save(path / f"depth_{i:06d}.npy", rng.random((4, 4)).astype(np.float32))
    return path


def _make_renderer(backend: StereoCrafterBackend, cache_dir: Path) -> StereoCrafterRenderer:
    """Build a renderer whose cache lives under *cache_dir* (never the repo)."""
    with patch("pipeline.stereo_crafter._assert_cuda"):  # bypass CUDA check in CI
        return StereoCrafterRenderer(backend=backend, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Key primitives: _stereo_cache_key / _stereo_cache_params / _depth_dir_digest
# ---------------------------------------------------------------------------


def test_cache_key_folds_in_depth_key(tmp_path: Path) -> None:
    """Same video + same stereo params, different upstream depth key -> different key.

    This is the CORE acceptance criterion of issue #183: changing the depth
    model (which changes the depth cache key) must NOT hit a stereo entry
    computed from the old depths.
    """
    clip = _write_fake_video(tmp_path / "clip.mp4")
    backend = CountingStereoBackend()
    k1 = _stereo_cache_key(str(clip), backend, depth_cache_key="depth-key-alpha")
    k2 = _stereo_cache_key(str(clip), backend, depth_cache_key="depth-key-beta")
    assert k1 != k2


def test_cache_key_identical_when_depth_key_and_params_same(tmp_path: Path) -> None:
    """Same video + same params + same depth key -> identical key (a hit)."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    backend = CountingStereoBackend()
    k1 = _stereo_cache_key(str(clip), backend, depth_cache_key="same-depth-key")
    k2 = _stereo_cache_key(str(clip), backend, depth_cache_key="same-depth-key")
    assert k1 == k2


def test_cache_key_changes_when_stereo_max_disp_changes(tmp_path: Path) -> None:
    """Changing a stereo key param (max_disp) -> different key -> no stale hit."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    k1 = _stereo_cache_key(str(clip), CountingStereoBackend(max_disp=20.0), depth_cache_key="dkey")
    k2 = _stereo_cache_key(str(clip), CountingStereoBackend(max_disp=40.0), depth_cache_key="dkey")
    assert k1 != k2


def test_cache_key_changes_when_max_res_changes(tmp_path: Path) -> None:
    """A backend with a different max_resolution yields a different key."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    k1 = _stereo_cache_key(str(clip), CountingStereoBackend(max_resolution=512), depth_cache_key="dkey")
    k2 = _stereo_cache_key(str(clip), CountingStereoBackend(max_resolution=768), depth_cache_key="dkey")
    assert k1 != k2


def test_cache_key_changes_when_frames_chunk_changes(tmp_path: Path) -> None:
    """A Stage-2 throughput knob (frames_chunk) also invalidates the key (issue #217)."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    k1 = _stereo_cache_key(str(clip), CountingStereoBackend(frames_chunk=None), depth_cache_key="dkey")
    k2 = _stereo_cache_key(str(clip), CountingStereoBackend(frames_chunk=16), depth_cache_key="dkey")
    assert k1 != k2


def test_cache_key_same_content_different_path_still_matches(tmp_path: Path) -> None:
    """Content-only video digest: same bytes at a new path + name still matches."""
    backend = CountingStereoBackend()
    a = _write_fake_video(tmp_path / "originals" / "clip.mp4", b"identical bytes")
    b = _write_fake_video(tmp_path / "elsewhere" / "renamed.mp4", b"identical bytes")
    assert _stereo_cache_key(str(a), backend, "dkey") == _stereo_cache_key(str(b), backend, "dkey")


def test_depth_dir_digest_fallback_distinguishes_depth_products(tmp_path: Path) -> None:
    """Different depth dirs (no depth cache key) -> different fallback digest."""
    d1 = _make_depth_dir(tmp_path / "depth_a", seed=1)
    d2 = _make_depth_dir(tmp_path / "depth_b", seed=2)
    assert _depth_dir_digest(str(d1)) != _depth_dir_digest(str(d2))


def test_depth_dir_digest_fallback_is_content_only(tmp_path: Path) -> None:
    """Same depth content moved to a different dir -> same fallback digest."""
    d1 = _make_depth_dir(tmp_path / "depth_a", seed=7)
    # Copy the same files to a different directory (preserving byte content).
    d2 = tmp_path / "depth_b"
    d2.mkdir(parents=True, exist_ok=True)
    for f in d1.iterdir():
        (d2 / f.name).write_bytes(f.read_bytes())
    assert _depth_dir_digest(str(d1)) == _depth_dir_digest(str(d2))


def test_cache_key_fallback_to_depth_dir_when_no_depth_key(tmp_path: Path) -> None:
    """With depth_cache_key=None, the key falls back to the depth dir digest
    and still distinguishes different depth products."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    backend = CountingStereoBackend()
    d1 = _make_depth_dir(tmp_path / "depth_a", seed=1)
    d2 = _make_depth_dir(tmp_path / "depth_b", seed=2)
    k1 = _stereo_cache_key(str(clip), backend, depth_cache_key=None, depth_dir=str(d1))
    k2 = _stereo_cache_key(str(clip), backend, depth_cache_key=None, depth_dir=str(d2))
    assert k1 != k2


def test_cache_key_raises_when_neither_depth_key_nor_dir(tmp_path: Path) -> None:
    """A real call always provides one; an impossible/buggy state is loud."""
    import pytest

    clip = _write_fake_video(tmp_path / "clip.mp4")
    backend = CountingStereoBackend()
    with pytest.raises(ValueError, match="requires either a depth_cache_key"):
        _stereo_cache_key(str(clip), backend, depth_cache_key=None, depth_dir=None)


def test_stereo_cache_params_safe_on_bare_backend() -> None:
    """A backend without any cache attrs contributes None, never raises."""
    params = _stereo_cache_params(BareStereoBackend())
    assert params["max_res"] is None
    assert params["max_disp"] is None
    assert params["frames_chunk"] is None
    assert params["overlap"] is None
    assert params["tile_num"] is None


# ---------------------------------------------------------------------------
# Hit / miss / reuse behaviour via StereoCrafterRenderer
# ---------------------------------------------------------------------------


def test_second_call_hits_cache_and_skips_backend(tmp_path: Path) -> None:
    """Same input + same params + same depth key twice -> second hits, backend NOT called."""
    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")

    first = renderer.render_video(
        input_path=str(clip),
        depth_dir=str(depth_dir),
        depth_cache_key="depth-key-1",
    )
    second = renderer.render_video(
        input_path=str(clip),
        depth_dir=str(depth_dir),
        depth_cache_key="depth-key-1",
    )

    assert backend.call_count == 1, "backend must not be invoked on a cache hit"
    # A miss returns the resolved temp L/R paths; a hit returns the staged
    # cache L/R files — they differ (cache stores copies), but both exist and
    # carry the backend's output content, which is the contract that matters.
    assert Path(first[0]).is_file() and Path(first[1]).is_file()
    assert Path(second[0]).is_file() and Path(second[1]).is_file()
    assert Path(first[0]).read_bytes() == b"stereo-mock"
    assert Path(second[0]).read_bytes() == b"stereo-mock"
    # The hit result is the cached copy, not the temp path.
    assert "cache" in str(Path(second[0]).parent)


def test_hit_logs_key_prefix(tmp_path: Path, caplog) -> None:
    """A cache hit logs ``[cache] hit <key前8位>``."""
    import logging

    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")
    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="k")  # seed

    with caplog.at_level(logging.INFO, logger="pipeline.stereo_crafter"):
        renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="k")  # hit

    assert any("[cache] hit" in r.message for r in caplog.records)
    hit_rec = next(r for r in caplog.records if "[cache] hit" in r.message)
    assert any(c.isalnum() for c in hit_rec.message)


def test_changed_depth_key_misses_and_recomputes(tmp_path: Path) -> None:
    """The CORE case: same video + same stereo params, but the upstream depth
    key changed (depth model swapped) -> MISS -> backend called again.

    Without folding the depth key in, this would wrongly hit and serve stereo
    computed from the wrong depths — exactly the bug issue #159/#183 closes.
    """
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")
    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")

    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="depth-key-v1")
    assert backend.call_count == 1

    # Same everything except the depth key (a different depth model / product).
    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="depth-key-v2")
    assert backend.call_count == 2, "changed depth key must not hit the old stereo entry"


def test_changed_stereo_max_disp_misses_and_recomputes(tmp_path: Path) -> None:
    """Changing a stereo key param (max_disp) -> miss -> backend called again."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")

    backend_a = CountingStereoBackend(max_disp=20.0)
    render_a = _make_renderer(backend_a, cache_dir=tmp_path / "cache")
    render_a.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend_a.call_count == 1

    backend_b = CountingStereoBackend(max_disp=40.0)
    render_b = _make_renderer(backend_b, cache_dir=tmp_path / "cache")
    render_b.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend_b.call_count == 1, "changed max_disp must not hit the old entry"


def test_use_cache_false_forces_recompute(tmp_path: Path) -> None:
    """``use_cache=False`` bypasses the cache entirely — even after a seed."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")
    cache_dir = tmp_path / "cache"
    backend = CountingStereoBackend()

    with patch("pipeline.stereo_crafter._assert_cuda"):
        render_cache = StereoCrafterRenderer(backend=backend, cache_dir=cache_dir)
        render_nocache = StereoCrafterRenderer(backend=backend, cache_dir=cache_dir, use_cache=False)

    render_cache.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend.call_count == 1

    render_nocache.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend.call_count == 2, "use_cache=False must force a recompute"


def test_same_content_different_path_still_hits(tmp_path: Path) -> None:
    """Content-only video digest: copy the clip elsewhere -> still a hit."""
    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    depth_dir = _make_depth_dir(tmp_path / "depth")

    original = _write_fake_video(tmp_path / "originals" / "clip.mp4", b"identical bytes")
    renderer.render_video(str(original), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend.call_count == 1

    copy = _write_fake_video(tmp_path / "elsewhere" / "renamed.mp4", b"identical bytes")
    renderer.render_video(str(copy), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend.call_count == 1, "same content at a new path must hit the cache"


def test_existing_call_sites_work_unchanged(tmp_path: Path) -> None:
    """Backward-compat: a call with NO depth_cache_key still works (fallback path).

    Existing call sites pass neither use_cache overrides nor depth_cache_key;
    the new params default, the key falls back to the depth dir digest, and
    the cache path still functions (seed + hit).
    """
    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")

    renderer.render_video(str(clip), depth_dir=str(depth_dir))  # no depth_cache_key
    assert backend.call_count == 1
    renderer.render_video(str(clip), depth_dir=str(depth_dir))  # same -> hit
    assert backend.call_count == 1, "existing call shape must still hit on the second call"


# ---------------------------------------------------------------------------
# Bare backend (no optional params) — round 1 regression mirror
# ---------------------------------------------------------------------------


def test_bare_backend_caches_hit(tmp_path: Path) -> None:
    """A backend lacking max_resolution/max_disp still hits on the second call."""
    backend = BareStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")

    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")
    assert backend.call_count == 1, "bare backend must hit on the second call"


# ---------------------------------------------------------------------------
# Cache entry contents / provenance (reuses the #121 shape via _save_stereo_cache)
# ---------------------------------------------------------------------------


def test_save_and_load_stereo_cache_roundtrip(tmp_path: Path) -> None:
    """_save_stereo_cache writes left/right + meta.json; _load_stereo_cache reads them back."""
    entry = tmp_path / "cache" / "abcdef0123456789"
    left = tmp_path / "out" / "left.mp4"
    right = tmp_path / "out" / "right.mp4"
    left.parent.mkdir(parents=True, exist_ok=True)
    left.write_bytes(b"left-data")
    right.write_bytes(b"right-data")

    backend = CountingStereoBackend(max_resolution=768, max_disp=30.0)
    _save_stereo_cache(
        entry_dir=entry,
        left_path=str(left),
        right_path=str(right),
        sbs_path=None,
        backend=backend,
        depth_cache_key="dkey",
        depth_dir=str(tmp_path / "depth"),
    )

    loaded = _load_stereo_cache(entry)
    assert loaded is not None
    assert Path(loaded["left"]).read_bytes() == b"left-data"
    assert Path(loaded["right"]).read_bytes() == b"right-data"
    meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    assert meta["stereo_model"] == type(backend).__name__
    assert meta["max_res"] == 768
    assert meta["max_disp"] == 30.0
    assert meta["depth_cache_key"] == "dkey"
    assert "timestamp" in meta


def test_load_stereo_cache_returns_none_when_absent(tmp_path: Path) -> None:
    """Missing entry dir or missing L/R files -> None (miss -> recompute)."""
    assert _load_stereo_cache(tmp_path / "nope") is None
    partial = tmp_path / "partial"
    partial.mkdir(parents=True)
    (partial / "left.mp4").write_bytes(b"x")  # right.mp4 absent
    assert _load_stereo_cache(partial) is None


# ---------------------------------------------------------------------------
# meta.json provenance mirrors the depth side (#121 / I-8a shape)
# ---------------------------------------------------------------------------


def test_cache_entry_has_meta_json_with_provenance(tmp_path: Path) -> None:
    """A stored entry carries a meta.json with stereo_model + params + depth key."""
    backend = CountingStereoBackend(max_resolution=768, max_disp=30.0, frames_chunk=16)
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")
    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dkey-1")

    entries = [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]
    assert len(entries) == 1
    meta = json.loads((entries[0] / "meta.json").read_text(encoding="utf-8"))
    assert meta["stereo_model"] == type(backend).__name__
    assert meta["max_res"] == 768
    assert meta["max_disp"] == 30.0
    assert meta["frames_chunk"] == 16
    assert meta["depth_cache_key"] == "dkey-1"
    assert meta["timestamp"]


# ---------------------------------------------------------------------------
# No working-tree pollution — tests never leave cache files in the repo
# ---------------------------------------------------------------------------


def test_cache_uses_only_the_given_cache_dir(tmp_path: Path) -> None:
    """No cache files appear outside the tmp_path-based cache_dir."""
    backend = CountingStereoBackend()
    renderer = _make_renderer(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    depth_dir = _make_depth_dir(tmp_path / "depth")
    renderer.render_video(str(clip), depth_dir=str(depth_dir), depth_cache_key="dk")

    assert (tmp_path / "cache").is_dir()
    entries = [p for p in (tmp_path / "cache").iterdir()]
    assert len(entries) == 1
    files = {p.name for p in entries[0].iterdir()}
    assert "meta.json" in files
    assert "left.mp4" in files
    assert "right.mp4" in files
