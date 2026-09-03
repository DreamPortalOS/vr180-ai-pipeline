"""Tests for the DepthCrafter content-keyed product cache (issue #182, I-8a).

The cache sits in front of the pluggable backend in
``DepthCrafterEstimator.estimate_video``: a hit serves the cached depth maps
and never spawns the (CUDA) subprocess/injected backend; a miss runs the
backend and stores its product.  The key is *content-only* — same source at a
different path still hits — folded with the backend's output-affecting params
(``target_size`` included).

All tests use mock backends and plain files under ``tmp_path``-based
``cache_dir`` roots — no CUDA, no real model, no repo working-tree pollution.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.depth_crafter import (
    DepthCrafterBackend,
    DepthCrafterEstimator,
    _backend_cache_params,
    _depth_cache_key,
    _video_content_digest,
)

# ---------------------------------------------------------------------------
# Mock backends — pluggable, like the production ones
# ---------------------------------------------------------------------------


class CountingBackend(DepthCrafterBackend):
    """Mock backend that records every call and returns deterministic depths.

    Mirrors the real backend's attribute surface (``max_resolution``,
    ``process_length``, ``target_fps``) so the cache key has something to fold
    in.  ``call_count`` is the assertion hook for "subprocess not spawned".
    """

    def __init__(
        self,
        num_frames: int = 3,
        h: int = 64,
        w: int = 96,
        max_resolution: int = 512,
        process_length: int | None = None,
        target_fps: int | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.max_resolution = max_resolution
        self.process_length = process_length
        self.target_fps = target_fps
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
        rng = np.random.default_rng(42)
        return [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]


class BareBackend(DepthCrafterBackend):
    """Backend with NONE of the optional cache params (issue #182 round 1).

    Has no ``max_resolution`` / ``process_length`` / ``target_fps`` — proves
    the safe ``getattr(..., None)`` reads never raise ``AttributeError`` and
    the cache path still works (hit and miss).
    """

    def __init__(self, num_frames: int = 2, h: int = 32, w: int = 48) -> None:
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.call_count = 0

    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        self.call_count += 1
        rng = np.random.default_rng(7)
        return [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]


class FileWritingBackend(DepthCrafterBackend):
    """Mock backend that writes ``depth_*.npy`` into *output_dir* (miss path).

    ``CountingBackend`` / ``BareBackend`` never touch disk, so they can't
    exercise the miss path's file-freshness contract.  The real CLIBackend
    writes the depth maps into the caller's output dir as a subprocess side
    effect; this mock does the same in-process so the miss-path mtime tests
    (issue #235) run without CUDA.  Mirrors ``CountingBackend``'s param
    surface so ``_backend_cache_params`` has something to fold in.
    """

    def __init__(
        self,
        num_frames: int = 2,
        h: int = 32,
        w: int = 48,
        max_resolution: int = 512,
        process_length: int | None = None,
        target_fps: int | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.max_resolution = max_resolution
        self.process_length = process_length
        self.target_fps = target_fps
        self.call_count = 0

    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        self.call_count += 1
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        depths = [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]
        for i, d in enumerate(depths):
            np.save(str(out / f"depth_{i:06d}.npy"), d)
        return depths


def _write_fake_video(path: Path, content: bytes = b"fake video content") -> Path:
    """Write a plain file standing in for an input clip (NOT a NamedTemporaryFile).

    A plain path is used instead of ``tempfile.NamedTemporaryFile`` because
    on Windows the latter holds an exclusive lock that prevents the content
    digest's ``open(path, 'rb')`` from reading it — the exact gotcha that
    forces ``estimate_video`` to disable caching for locked inputs.  Using a
    plain file means the cache path is exercised for real.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Content digest + cache key primitives
# ---------------------------------------------------------------------------


def test_video_content_digest_is_path_independent(tmp_path: Path) -> None:
    """Same bytes at a different path → same digest (content-only, not path)."""
    a = _write_fake_video(tmp_path / "clip_a.mp4", b"hello world video")
    b = _write_fake_video(tmp_path / "nested" / "clip_b.mp4", b"hello world video")
    assert _video_content_digest(a) == _video_content_digest(b)


def test_video_content_digest_differs_on_content(tmp_path: Path) -> None:
    """Different bytes → different digest."""
    a = _write_fake_video(tmp_path / "a.mp4", b"content one")
    b = _write_fake_video(tmp_path / "b.mp4", b"content two")
    assert _video_content_digest(a) != _video_content_digest(b)


def test_video_content_digest_differs_on_size_only(tmp_path: Path) -> None:
    """Same prefix, different length → different digest (size term)."""
    a = _write_fake_video(tmp_path / "a.mp4", b"abc")
    b = _write_fake_video(tmp_path / "b.mp4", b"abcd")
    assert _video_content_digest(a) != _video_content_digest(b)


def test_cache_key_includes_target_size(tmp_path: Path) -> None:
    """A different target_size MUST yield a different key (round 2 regression)."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    backend = CountingBackend()
    k1 = _depth_cache_key(str(clip), backend, target_size=(720, 1280))
    k2 = _depth_cache_key(str(clip), backend, target_size=(360, 640))
    assert k1 != k2


def test_cache_key_changes_when_max_res_changes(tmp_path: Path) -> None:
    """A backend with a different max_resolution yields a different key."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    k1 = _depth_cache_key(str(clip), CountingBackend(max_resolution=512), target_size=None)
    k2 = _depth_cache_key(str(clip), CountingBackend(max_resolution=768), target_size=None)
    assert k1 != k2


def test_backend_cache_params_safe_on_bare_backend() -> None:
    """A backend without max_resolution etc. contributes None, never raises."""
    params = _backend_cache_params(BareBackend(), target_size=None)
    assert params["max_res"] is None
    assert params["process_length"] is None
    assert params["target_fps"] is None
    assert params["target_size"] is None


# ---------------------------------------------------------------------------
# Hit / miss / reuse behaviour via DepthCrafterEstimator
# ---------------------------------------------------------------------------


def _make_estimator(backend: DepthCrafterBackend, cache_dir: Path) -> DepthCrafterEstimator:
    """Build an estimator whose cache lives under *cache_dir* (never the repo)."""
    with patch("pipeline.depth_crafter._assert_cuda"):  # bypass CUDA check in CI
        return DepthCrafterEstimator(backend=backend, cache_dir=cache_dir)


def test_second_call_hits_cache_and_skips_backend(tmp_path: Path) -> None:
    """Same input + same params twice → second call hits, backend NOT called again."""
    backend = CountingBackend(num_frames=3)
    estimator = _make_estimator(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")

    first = estimator.estimate_video(str(clip))
    second = estimator.estimate_video(str(clip))

    assert backend.call_count == 1, "backend must not be invoked on a cache hit"
    assert len(first) == 3 and len(second) == 3
    # Byte-exact reuse: the cached maps are the very same arrays returned the first time.
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_hit_logs_key_prefix(tmp_path: Path, caplog) -> None:
    """A cache hit logs ``[cache] hit <key前8位>``."""
    import logging

    backend = CountingBackend()
    estimator = _make_estimator(backend, cache_dir=tmp_path / "cache")
    clip = _write_fake_video(tmp_path / "clip.mp4")
    estimator.estimate_video(str(clip))  # seed

    with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
        estimator.estimate_video(str(clip))  # hit

    assert any("[cache] hit" in r.message for r in caplog.records)
    # The 8-char key prefix is surfaced.
    hit_rec = next(r for r in caplog.records if "[cache] hit" in r.message)
    assert any(c.isalnum() for c in hit_rec.message)


def test_changed_max_res_misses_and_recomputes(tmp_path: Path) -> None:
    """Changing a key param (max_resolution) → miss → backend called again."""
    clip = _write_fake_video(tmp_path / "clip.mp4")

    backend_a = CountingBackend(max_resolution=512)
    est_a = _make_estimator(backend_a, cache_dir=tmp_path / "cache")
    est_a.estimate_video(str(clip))
    assert backend_a.call_count == 1

    # Same clip, different max_res → different key → miss.
    backend_b = CountingBackend(max_resolution=768)
    est_b = _make_estimator(backend_b, cache_dir=tmp_path / "cache")
    est_b.estimate_video(str(clip))
    assert backend_b.call_count == 1, "changed max_res must not hit the old entry"


def test_changed_target_size_misses_and_recomputes(tmp_path: Path) -> None:
    """Changing ONLY target_size (everything else identical) → miss → re-invoke."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    cache_dir = tmp_path / "cache"

    backend = CountingBackend()
    est = _make_estimator(backend, cache_dir=cache_dir)

    est.estimate_video(str(clip), target_size=(720, 1280))
    assert backend.call_count == 1
    assert backend.last_target_size == (720, 1280)

    est.estimate_video(str(clip), target_size=(360, 640))
    assert backend.call_count == 2, "different target_size must re-invoke the backend"
    assert backend.last_target_size == (360, 640)

    # And the same target_size as before now hits (it was cached on its miss).
    est.estimate_video(str(clip), target_size=(720, 1280))
    assert backend.call_count == 2, "returning to a previously-cached target_size must hit"


def test_use_cache_false_forces_recompute(tmp_path: Path) -> None:
    """``use_cache=False`` bypasses the cache entirely — even after a seed."""
    clip = _write_fake_video(tmp_path / "clip.mp4")
    cache_dir = tmp_path / "cache"
    backend = CountingBackend()

    with patch("pipeline.depth_crafter._assert_cuda"):
        est_cache = DepthCrafterEstimator(backend=backend, cache_dir=cache_dir)
        est_nocache = DepthCrafterEstimator(backend=backend, cache_dir=cache_dir, use_cache=False)

    est_cache.estimate_video(str(clip))  # seed the cache (backend called once)
    assert backend.call_count == 1

    est_nocache.estimate_video(str(clip))  # use_cache=False → must NOT hit
    assert backend.call_count == 2, "use_cache=False must force a recompute"


def test_same_content_different_path_still_hits(tmp_path: Path) -> None:
    """The 'only by content' requirement: copy the clip elsewhere → still a hit."""
    cache_dir = tmp_path / "cache"
    backend = CountingBackend()
    est = _make_estimator(backend, cache_dir=cache_dir)

    original = _write_fake_video(tmp_path / "originals" / "clip.mp4", b"identical bytes")
    est.estimate_video(str(original))  # seed
    assert backend.call_count == 1

    # Copy the SAME bytes to a different location + name.  Content digest is
    # path-independent, so the cache must hit and the backend stays at 1 call.
    copy = _write_fake_video(tmp_path / "elsewhere" / "renamed_clip.mp4", b"identical bytes")
    est.estimate_video(str(copy))
    assert backend.call_count == 1, "same content at a new path must hit the cache"


def test_different_content_misses(tmp_path: Path) -> None:
    """Genuinely different content → different digest → miss → recompute."""
    cache_dir = tmp_path / "cache"
    backend = CountingBackend()
    est = _make_estimator(backend, cache_dir=cache_dir)

    a = _write_fake_video(tmp_path / "a.mp4", b"clip content A")
    b = _write_fake_video(tmp_path / "b.mp4", b"clip content B")

    est.estimate_video(str(a))
    est.estimate_video(str(b))
    assert backend.call_count == 2, "different content must not hit"


# ---------------------------------------------------------------------------
# Round 1 regression: backend without the optional params
# ---------------------------------------------------------------------------


def test_bare_backend_caches_hit(tmp_path: Path) -> None:
    """A backend lacking max_resolution/process_length/target_fps still hits."""
    cache_dir = tmp_path / "cache"
    backend = BareBackend(num_frames=2)
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")

    est.estimate_video(str(clip))
    est.estimate_video(str(clip))
    assert backend.call_count == 1, "bare backend must hit on the second call"


def test_bare_backend_with_target_size_caches(tmp_path: Path) -> None:
    """Bare backend + a target_size still caches, and a new target_size misses."""
    cache_dir = tmp_path / "cache"
    backend = BareBackend()
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")

    est.estimate_video(str(clip), target_size=(100, 200))
    est.estimate_video(str(clip), target_size=(100, 200))  # same → hit
    assert backend.call_count == 1

    est.estimate_video(str(clip), target_size=(300, 400))  # different → miss
    assert backend.call_count == 2


# ---------------------------------------------------------------------------
# meta.json provenance (reuses the #121 structure)
# ---------------------------------------------------------------------------


def test_cache_entry_has_meta_json_with_provenance(tmp_path: Path) -> None:
    """A stored entry carries a #121-style meta.json (model + params + timestamp)."""
    cache_dir = tmp_path / "cache"
    backend = CountingBackend(max_resolution=640, num_frames=4)
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")
    est.estimate_video(str(clip), target_size=(720, 1280))

    # Exactly one entry dir under the cache root.
    entries = [p for p in cache_dir.iterdir() if p.is_dir()]
    assert len(entries) == 1
    entry = entries[0]

    meta_path = entry / "meta.json"
    assert meta_path.is_file(), "cache entry must write a meta.json (#121 shape)"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # #121 provenance fields.
    assert meta["depth_model"] == type(backend).__name__
    assert meta["num_frames"] == 4
    assert meta["max_res"] == 640
    assert meta["target_size"] == [720, 1280]  # JSON list, not tuple
    assert "timestamp" in meta
    assert meta["timestamp"]


# ---------------------------------------------------------------------------
# No working-tree pollution — tests never leave cache files in the repo
# ---------------------------------------------------------------------------


def test_cache_uses_only_the_given_cache_dir(tmp_path: Path) -> None:
    """No cache files appear outside the tmp_path-based cache_dir."""
    cache_dir = tmp_path / "cache"
    backend = CountingBackend()
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")
    est.estimate_video(str(clip))

    # The only thing under cache_dir is the one entry dir.
    assert cache_dir.is_dir()
    entries = [p for p in cache_dir.iterdir()]
    assert len(entries) == 1
    # And the entry holds npy + meta.json only.
    files = {p.name for p in entries[0].iterdir()}
    assert "meta.json" in files
    assert any(name.startswith("depth_") and name.endswith(".npy") for name in files)


# ---------------------------------------------------------------------------
# Round 3 regression (issue #231, K-23a): a cache hit must *also* write the
# depth_*.npy files into the caller's depth_dir so the streaming path's
# K-16 metric probe (which globs for depth_*.npy in <temp_dir>/depth) never
# sees an empty dir on a repeat run.  The miss path already did this; only
# the hit path was broken.
# ---------------------------------------------------------------------------


def _depth_npy_files(depth_dir: Path) -> list[Path]:
    """Sorted list of ``depth_*.npy`` files in *depth_dir*."""
    return sorted(Path(depth_dir).glob("depth_*.npy"))


def _seed_and_hit(tmp_path: Path):
    """Set up a cache entry and a fresh depth_dir, returning objects for a hit.

    ``CountingBackend`` (like any in-memory mock) does NOT write depth_*.npy
    files on the miss path — only the real CLIBackend subprocess does that as
    a side effect.  To exercise the *hit* path in isolation (which is exactly
    the K-23a bug), we seed the cache entry dir directly, then clear depth_dir
    so the hit has an empty dir to refill.  Returns
    ``(estimator, backend, depth_dir, clip, entry_dir, cached_depths)``.
    """
    from pipeline.depth_crafter import _save_depth_cache

    cache_dir = tmp_path / "cache"
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    backend = CountingBackend(num_frames=3)
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")

    # Seed the cache entry directly (the miss path's _save_depth_cache step).
    rng = np.random.default_rng(42)
    cached_depths = [rng.random((64, 96)).astype(np.float32) for _ in range(3)]
    key = _depth_cache_key(str(clip), backend, target_size=None)
    entry_dir = cache_dir / key
    _save_depth_cache(entry_dir, cached_depths, backend, target_size=None)
    assert entry_dir.is_dir()

    return est, backend, depth_dir, clip, entry_dir, cached_depths


def test_hit_materializes_npy_into_depth_dir(tmp_path: Path) -> None:
    """Hit path with a depth_dir must write depth_*.npy there (the K-23a bug)."""
    est, backend, depth_dir, clip, _entry_dir, _cached = _seed_and_hit(tmp_path)

    # depth_dir exists but is empty — the exact situation the streaming
    # pipeline creates on a repeat run.
    assert depth_dir.is_dir()
    assert len(_depth_npy_files(depth_dir)) == 0

    depths = est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 0, "cache hit must not invoke the backend"
    assert len(depths) == 3
    assert len(_depth_npy_files(depth_dir)) == 3, "hit path must materialize npy into depth_dir"


def test_hit_materialized_files_equal_cache_files(tmp_path: Path) -> None:
    """Files materialized by the hit path must be byte-equal to the cached ones."""
    est, backend, depth_dir, clip, _entry_dir, cached_depths = _seed_and_hit(tmp_path)

    depths = est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 0
    for materialized, cached in zip(depths, cached_depths, strict=True):
        np.testing.assert_array_equal(materialized, cached)
    for i, p in enumerate(_depth_npy_files(depth_dir), start=0):
        on_disk = np.load(str(p))
        np.testing.assert_array_equal(on_disk, cached_depths[i])


def test_hit_with_none_depth_dir_is_memory_only(tmp_path: Path) -> None:
    """Hit + depth_dir=None must NOT write any files and still returns arrays."""
    est, backend, depth_dir, clip, _entry_dir, _cached = _seed_and_hit(tmp_path)

    depths = est.estimate_video(input_path=str(clip), output_dir=None)

    assert backend.call_count == 0, "must be a cache hit"
    assert len(depths) == 3
    assert len(_depth_npy_files(depth_dir)) == 0, "depth_dir=None must not create any files"


def test_hit_materialize_logs_materialized_message(tmp_path: Path, caplog) -> None:
    """Hit + depth_dir logs ``[cache] hit <key> → materialized N maps``."""
    import logging

    est, _backend, depth_dir, clip, _entry_dir, _cached = _seed_and_hit(tmp_path)

    with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
        est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    materialized = [r for r in caplog.records if "materialized" in r.message]
    assert len(materialized) == 1
    assert "materialized 3 maps" in materialized[0].message
    assert str(depth_dir) in materialized[0].message


def test_hit_materialize_link_fallback_to_copy(tmp_path: Path, monkeypatch) -> None:
    """When os.link raises OSError, fall back to shutil.copy2 and still succeed."""
    import shutil

    est, backend, depth_dir, clip, _entry_dir, _cached = _seed_and_hit(tmp_path)

    calls = {"link": 0, "copy": 0}

    def fake_link(src, dst):
        calls["link"] += 1
        raise OSError("simulated cross-device link failure")

    def fake_copy2(src, dst, *a, **kw):
        calls["copy"] += 1
        with open(src, "rb") as f_in, open(dst, "wb") as f_out:
            f_out.write(f_in.read())

    monkeypatch.setattr("pipeline.depth_crafter.os.link", fake_link)
    monkeypatch.setattr(shutil, "copy2", fake_copy2)

    depths = est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 0
    assert len(depths) == 3
    assert len(_depth_npy_files(depth_dir)) == 3, "copy fallback must still populate depth_dir"
    assert calls["link"] >= 1, "should have attempted os.link"
    assert calls["copy"] == 3, "should have fallen back to copy2 for each npy"


# ---------------------------------------------------------------------------
# Round 4 regression (issue #235, K-24): the K-16 freshness gate in
# make_comparison.default_depth_dir_resolver rejects any depth dir whose
# newest depth_*.npy mtime predates render_started - 1s.  A cache hit hard-
# links the cached npy into the caller's depth dir, and a hard link shares the
# cache file's (old) mtime — so on every repeat run the materialized files
# looked stale and the metrics stayed "—" even though the dir was populated.
# Fix: stamp "now" onto every materialized file (and rely on the natural "now"
# of the miss path), verified below.
# ---------------------------------------------------------------------------

#: An mtime unambiguously older than any wall-clock "now" the test can observe.
_STALE_MTIME = 946684800  # 2000-01-01 00:00:00 UTC


def test_hit_materialized_files_get_fresh_mtime(tmp_path: Path) -> None:
    """Cache hit must stamp fresh mtimes even when the cached npy is old.

    This is the K-24 bug directly: age the cache entry's npy to 2000, take a
    hit, and assert every materialized file in depth_dir is no older than the
    call's start time.  Also re-checks byte-identity (stamping mtime must not
    change content).
    """
    est, backend, depth_dir, clip, entry_dir, cached_depths = _seed_and_hit(tmp_path)

    # Age every cached npy to 2000 — the "18h-old leftover" state the
    # freshness gate is built to reject.
    for src in entry_dir.glob("depth_*.npy"):
        os.utime(src, (_STALE_MTIME, _STALE_MTIME))

    start = time.time()
    est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 0, "cache hit must not invoke the backend"
    materialized = _depth_npy_files(depth_dir)
    assert len(materialized) == len(cached_depths)
    for i, p in enumerate(materialized):
        assert p.stat().st_mtime >= start, f"{p.name} mtime is stale: {p.stat().st_mtime} < {start}"
        # Regression: stamping mtime must not touch the bytes.
        np.testing.assert_array_equal(np.load(str(p)), cached_depths[i])


def test_hit_materialize_copy_fallback_sets_fresh_mtime(tmp_path: Path, monkeypatch) -> None:
    """Copy fallback must stamp fresh mtimes too (copy2 preserves the old one)."""
    est, backend, depth_dir, clip, entry_dir, cached_depths = _seed_and_hit(tmp_path)

    for src in entry_dir.glob("depth_*.npy"):
        os.utime(src, (_STALE_MTIME, _STALE_MTIME))

    # Force the copy2 fallback for every file.  shutil.copy2 is left real so
    # the test exercises the genuine "copy2 preserves old mtime → os.utime
    # overrides it" path, not a hand-rolled stub.
    def fake_link(src, dst):
        raise OSError("simulated cross-device link failure")

    monkeypatch.setattr("pipeline.depth_crafter.os.link", fake_link)

    start = time.time()
    est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 0
    materialized = _depth_npy_files(depth_dir)
    assert len(materialized) == len(cached_depths)
    for i, p in enumerate(materialized):
        assert p.stat().st_mtime >= start, f"{p.name} mtime is stale after copy fallback"
        np.testing.assert_array_equal(np.load(str(p)), cached_depths[i])


def test_miss_path_files_get_fresh_mtime(tmp_path: Path) -> None:
    """Miss-path (backend-written) npy files have fresh mtimes by nature.

    Pins the contract so a future change can't silently regress it: the real
    CLIBackend writes depth_*.npy as it runs, so their mtime is necessarily
    "now"; a FileWritingBackend stands in for that side effect here.
    """
    cache_dir = tmp_path / "cache"
    depth_dir = tmp_path / "depth"
    backend = FileWritingBackend(num_frames=2)
    est = _make_estimator(backend, cache_dir=cache_dir)
    clip = _write_fake_video(tmp_path / "clip.mp4")

    start = time.time()
    est.estimate_video(input_path=str(clip), output_dir=str(depth_dir))

    assert backend.call_count == 1, "fresh cache_dir → must be a miss"
    files = _depth_npy_files(depth_dir)
    assert len(files) == 2
    for p in files:
        assert p.stat().st_mtime >= start, f"{p.name} mtime is stale on the miss path"
