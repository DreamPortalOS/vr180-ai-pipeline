"""Tests for the depth-product content cache (I-8a, issue #182).

All tests are mock-based — no CUDA, no real model, no subprocess.  The cache
is keyed by **file content** (size + first/last 4 MB sha256), the model name
and output-affecting params — so the same bytes at a different path must hit,
and changing a key param (``max_res`` etc.) must miss.

The estimator's cache layer sits *in front of* the pluggable backend, so the
tests inject a counting ``MockBackend`` whose ``estimate_video`` records each
call and returns deterministic depth maps.  A cache **hit** must invoke that
backend zero times on the second run; a **miss** must invoke it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.depth_crafter import (
    DepthCrafterBackend,
    DepthCrafterEstimator,
    _backend_cache_params,
    _cache_dir_for,
    _compute_cache_key,
)

# ---------------------------------------------------------------------------
# Counting mock backend — records every estimate_video() invocation.
# ---------------------------------------------------------------------------


class CountingMockBackend(DepthCrafterBackend):
    """Mock backend that counts calls and returns deterministic depth maps.

    Deliberately has NO ``max_resolution`` attribute (issue #182 round-1
    rejection): the cache must key safely off any pluggable backend, and a
    missing knob must contribute a placeholder, never raise ``AttributeError``.
    """

    def __init__(self, num_frames: int = 3, h: int = 8, w: int = 12, max_resolution=None) -> None:
        self.num_frames = num_frames
        self.h = h
        self.w = w
        # Only set max_resolution when explicitly provided, so the
        # ``has no max_resolution`` test path exercises the getattr default.
        if max_resolution is not None:
            self.max_resolution = max_resolution
        self.call_count = 0
        self.last_target_size: tuple[int, int] | None = None

    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        self.call_count += 1
        self.last_target_size = target_size
        # Deterministic output keyed only on (h, w, num_frames) so two runs of
        # the same input produce identical depth maps (cache equivalence check).
        rng = np.random.default_rng(seed=hash((self.h, self.w, self.num_frames)) & 0xFFFFFFFF)
        return [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]


def _make_estimator(backend: CountingMockBackend, cache_dir: Path, *, use_cache: bool = True) -> DepthCrafterEstimator:
    """Build an estimator whose cache points at *cache_dir* (CUDA bypassed)."""
    with patch("pipeline.depth_crafter._assert_cuda"):
        return DepthCrafterEstimator(
            backend=backend,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )


def _write_video(path: Path, content: bytes = b"fake video content for hashing") -> Path:
    """Write a small stand-in video file under tmp_path (no NamedTemporaryFile)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Cache key unit tests
# ---------------------------------------------------------------------------


def test_backend_cache_params_missing_attr_is_none(tmp_path: Path) -> None:
    """A backend without ``max_resolution`` contributes ``None``, not an error."""
    backend = CountingMockBackend()  # no max_resolution attribute set
    assert not hasattr(backend, "max_resolution")
    params = _backend_cache_params(backend)
    assert params == {
        "max_res": None,
        "process_length": None,
        "target_fps": None,
    }


def test_backend_cache_params_present_attrs_captured() -> None:
    """When the backend exposes the knobs, they are pulled into the key params."""

    class WithKnobs:
        max_resolution = 512
        process_length = 64
        target_fps = 24

    params = _backend_cache_params(WithKnobs())
    assert params == {"max_res": 512, "process_length": 64, "target_fps": 24}


def test_compute_cache_key_is_deterministic(tmp_path: Path) -> None:
    """Same file + same backend → same key."""
    vid = _write_video(tmp_path / "a.mp4", b"identical bytes")
    backend = CountingMockBackend(max_resolution=512)
    k1 = _compute_cache_key(str(vid), backend)
    k2 = _compute_cache_key(str(vid), backend)
    assert k1 is not None
    assert k1 == k2


def test_compute_cache_key_differs_on_max_res(tmp_path: Path) -> None:
    """A different ``max_res`` produces a different key."""
    vid = _write_video(tmp_path / "a.mp4", b"identical bytes")
    k1 = _compute_cache_key(str(vid), CountingMockBackend(max_resolution=512))
    k2 = _compute_cache_key(str(vid), CountingMockBackend(max_resolution=768))
    assert k1 != k2


def test_compute_cache_key_ignores_path_same_content(tmp_path: Path) -> None:
    """Same bytes at a different path → same key (content-only contract)."""
    content = b"identical bytes across two paths"
    vid1 = _write_video(tmp_path / "a.mp4", content)
    vid2 = _write_video(tmp_path / "subdir" / "b.mp4", content)
    backend = CountingMockBackend(max_resolution=512)
    k1 = _compute_cache_key(str(vid1), backend)
    k2 = _compute_cache_key(str(vid2), backend)
    assert k1 is not None
    assert k1 == k2


def test_compute_cache_key_differs_on_content(tmp_path: Path) -> None:
    """Different file content → different key (even same size edge case)."""
    # Make them the same length so the size component is identical — only the
    # sampled bytes differ, exercising the head/tail hashing not just the size.
    vid1 = _write_video(tmp_path / "a.mp4", b"AAAAAAAAAAAAAAAA")
    vid2 = _write_video(tmp_path / "b.mp4", b"BBBBBBBBBBBBBBBB")
    backend = CountingMockBackend(max_resolution=512)
    assert _compute_cache_key(str(vid1), backend) != _compute_cache_key(str(vid2), backend)


def test_compute_cache_key_unreadable_returns_none(tmp_path: Path) -> None:
    """A non-existent file yields ``None`` (miss, not an exception)."""
    assert _compute_cache_key(str(tmp_path / "nope.mp4"), CountingMockBackend()) is None


def test_cache_dir_for_uses_cache_dir(tmp_path: Path) -> None:
    """Explicit cache_dir is honoured; the key becomes the leaf subdir."""
    d = _cache_dir_for("deadbeef", tmp_path / "mycache")
    assert d == tmp_path / "mycache" / "deadbeef"


# ---------------------------------------------------------------------------
# Hit / miss behaviour through the estimator
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_hit_skips_backend_on_second_run(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """Same input + same params twice: 2nd run hits cache, backend call_count==0."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=3)
    est = _make_estimator(backend, tmp_path / "cache")

    d1 = est.estimate_video(str(vid))
    assert backend.call_count == 1
    assert len(d1) == 3

    d2 = est.estimate_video(str(vid))
    assert backend.call_count == 1, "2nd run must NOT invoke the backend (cache hit)"
    assert len(d2) == 3
    # Cached maps equal the first run's maps (byte-for-byte content).
    for a, b in zip(d1, d2, strict=True):
        assert np.array_equal(a, b)


@patch("pipeline.depth_crafter._assert_cuda")
def test_hit_writes_meta_json_reusing_121_schema(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """A stored entry carries meta.json with the #121 field names (no new schema)."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    est = _make_estimator(backend, tmp_path / "cache")
    est.estimate_video(str(vid))

    key = _compute_cache_key(str(vid), backend)
    entry = _cache_dir_for(key, tmp_path / "cache")
    meta_path = entry / "meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # #121 schema fields — reusing them, not inventing new ones.
    assert meta["depth_model"] == "depthcrafter"
    assert meta["num_frames"] == 2
    assert meta["max_res"] == 512
    assert "timestamp" in meta
    # depth_*.npy sequence persisted.
    npys = sorted((entry).glob("depth_*.npy"))
    assert len(npys) == 2


@patch("pipeline.depth_crafter._assert_cuda")
def test_param_change_misses_and_recomputes(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """Changing ``max_res`` → miss → recompute → second entry stored."""
    vid = _write_video(tmp_path / "clip.mp4")

    backend_lo = CountingMockBackend(num_frames=2, max_resolution=512)
    est_lo = _make_estimator(backend_lo, tmp_path / "cache")
    est_lo.estimate_video(str(vid))
    assert backend_lo.call_count == 1

    # Different max_res on a different backend instance → different key → miss.
    backend_hi = CountingMockBackend(num_frames=2, max_resolution=768)
    est_hi = _make_estimator(backend_hi, tmp_path / "cache")
    est_hi.estimate_video(str(vid))
    assert backend_hi.call_count == 1, "different max_res must miss → recompute"
    # Two distinct entries now live in the cache (keyed by the param).
    entries = [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]
    assert len(entries) == 2


@patch("pipeline.depth_crafter._assert_cuda")
def test_use_cache_false_forces_recompute(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """``use_cache=False`` skips both lookup and persist → backend runs every time."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    est_cache = _make_estimator(backend, tmp_path / "cache", use_cache=True)
    # Warm the cache.
    est_cache.estimate_video(str(vid))
    assert backend.call_count == 1

    # Now disable the cache — must recompute even though a valid entry exists.
    est_no_cache = _make_estimator(backend, tmp_path / "cache", use_cache=False)
    est_no_cache.estimate_video(str(vid))
    assert backend.call_count == 2, "use_cache=False must bypass the cache and recompute"


@patch("pipeline.depth_crafter._assert_cuda")
def test_same_content_different_path_hits(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """The 'only-by-content' acceptance: copy the file to a new path → cache hit."""
    content = b"same bytes, different path: must still hit"
    vid1 = _write_video(tmp_path / "alpha" / "clip.mp4", content)
    vid2 = _write_video(tmp_path / "beta" / "other.mp4", content)
    # Sanity: the two paths are genuinely different.
    assert str(vid1) != str(vid2)

    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    est = _make_estimator(backend, tmp_path / "cache")

    d1 = est.estimate_video(str(vid1))
    assert backend.call_count == 1

    d2 = est.estimate_video(str(vid2))
    assert backend.call_count == 1, "same content at a new path must hit the cache"
    for a, b in zip(d1, d2, strict=True):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Backend-without-max_resolution (issue #182 round-1 regression guard)
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_backend_without_max_resolution_caches_and_hits(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """A backend with NO ``max_resolution`` attr must still cache and hit."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=2)  # no max_resolution set
    assert not hasattr(backend, "max_resolution"), "test precondition: attr absent"

    est = _make_estimator(backend, tmp_path / "cache")

    d1 = est.estimate_video(str(vid))
    assert backend.call_count == 1
    assert len(d1) == 2

    # Second run hits the cache — the missing attr never raised.
    est.estimate_video(str(vid))
    assert backend.call_count == 1, "backend without max_resolution must still hit"


@patch("pipeline.depth_crafter._assert_cuda")
def test_backend_without_max_resolution_misses_on_content_change(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """Missing ``max_resolution`` backend still misses when content differs."""
    vid1 = _write_video(tmp_path / "a.mp4", b"content one" * 10)
    vid2 = _write_video(tmp_path / "b.mp4", b"content two" * 10)
    backend = CountingMockBackend(num_frames=2)
    assert not hasattr(backend, "max_resolution")
    est = _make_estimator(backend, tmp_path / "cache")

    est.estimate_video(str(vid1))
    assert backend.call_count == 1
    est.estimate_video(str(vid2))
    assert backend.call_count == 2, "different content must miss even without max_resolution"


# ---------------------------------------------------------------------------
# Stale / partial entry → treated as a miss (not a silent reuse)
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_partial_entry_recomputes(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """A cache entry whose .npy count disagrees with meta.num_frames → miss."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=3, max_resolution=512)
    est = _make_estimator(backend, tmp_path / "cache")
    est.estimate_video(str(vid))
    assert backend.call_count == 1

    # Corrupt the entry: drop one .npy so the count no longer matches meta.
    key = _compute_cache_key(str(vid), backend)
    entry = _cache_dir_for(key, tmp_path / "cache")
    npys = sorted(entry.glob("depth_*.npy"))
    npys[0].unlink()
    assert len(list(entry.glob("depth_*.npy"))) == 2  # meta says 3

    est.estimate_video(str(vid))
    assert backend.call_count == 2, "partial entry must miss → recompute"


@patch("pipeline.depth_crafter._assert_cuda")
def test_meta_without_npy_recomputes(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """meta.json present but no .npy maps → miss (no silent empty reuse)."""
    vid = _write_video(tmp_path / "clip.mp4")
    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    est = _make_estimator(backend, tmp_path / "cache")

    # Hand-seed a meta.json with no accompanying .npy (tampered/aborted entry).
    key = _compute_cache_key(str(vid), backend)
    entry = _cache_dir_for(key, tmp_path / "cache")
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "meta.json").write_text(
        json.dumps({"depth_model": "depthcrafter", "num_frames": 2, "max_res": 512}),
        encoding="utf-8",
    )

    est.estimate_video(str(vid))
    assert backend.call_count == 1, "meta-without-npy must miss → recompute"


# ---------------------------------------------------------------------------
# File > 4 MB exercises the tail-read branch of the fingerprint
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_large_file_uses_head_and_tail(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """A >4 MB file: head+tail fingerprint; same content different path hits.

    Two copies of a >4 MB file at different paths must still produce the same
    cache key and hit, proving the tail-read branch is content-stable too.
    """
    payload = os.urandom(5 * 1024 * 1024)  # 5 MB > 4 MB chunk
    vid1 = _write_video(tmp_path / "big" / "a.mp4", payload)
    vid2 = _write_video(tmp_path / "big2" / "b.mp4", payload)
    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    est = _make_estimator(backend, tmp_path / "cache")

    est.estimate_video(str(vid1))
    assert backend.call_count == 1
    est.estimate_video(str(vid2))
    assert backend.call_count == 1, "large-file same content diff path must hit"


@patch("pipeline.depth_crafter._assert_cuda")
def test_large_file_tail_change_misses(mock_cuda: MagicMock, tmp_path: Path) -> None:
    """Editing only the last 4 MB of a >4 MB file changes the key (tail read)."""
    head = b"\x01" * (5 * 1024 * 1024)  # 5 MB of head
    vid1 = _write_video(tmp_path / "a.mp4", head + b"TAIL-ONE")
    vid2 = _write_video(tmp_path / "b.mp4", head + b"TAIL-TWO")
    backend = CountingMockBackend(num_frames=2, max_resolution=512)
    # Same size, same head, only the tail bytes differ — the tail read must
    # distinguish them (otherwise two different videos would wrongly collide).
    assert _compute_cache_key(str(vid1), backend) != _compute_cache_key(str(vid2), backend)
