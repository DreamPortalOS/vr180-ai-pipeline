"""Tests for I-8a (#182): depth-product cache reuse.

Verifies the content-keyed cache for DepthCrafterEstimator:
- Same input + same params twice => second run is a cache hit; the
  subprocess runner is never invoked on the hit.
- Changing a key param (max_res / process_length / target_fps) => cache miss.
- use_cache=False forces a recompute even when a matching entry exists.
- Path-independence: the same bytes at a different path must hit.

All heavy work is stubbed — CI runs without GPU / models / ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.depth_crafter import (
    DepthCrafterEstimator,
    _cache_params,
    compute_cache_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _backend(**overrides) -> MagicMock:
    """Build a mock backend with the key params DepthCrafterEstimator reads.

    The cache key is derived from the backend's ``max_resolution``,
    ``process_length`` and ``target_fps``; every other attribute is
    irrelevant and the mock stubs ``estimate_video`` so no subprocess runs.
    """
    b = MagicMock(spec=["max_resolution", "process_length", "target_fps", "estimate_video"])
    b.max_resolution = 512
    b.process_length = None
    b.target_fps = None
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


def _video_bytes(tmp_path: Path) -> bytes:
    """Deterministic fake video payload."""
    return b"HEAD" + b"x" * 1000 + b"TAIL"


def _write_video(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(_video_bytes(tmp_path))
    return p


class FakeNumpyDepthProduct:
    """Write a tiny .npy sequence into a dir so the cache loader finds product."""

    @staticmethod
    def write(depth_dir: Path, num_frames: int = 3, h: int = 4, w: int = 4) -> None:
        depth_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i in range(num_frames):
            np.save(str(depth_dir / f"depth_{i:06d}.npy"), rng.random((h, w)).astype(np.float32))


# ---------------------------------------------------------------------------
# _cache_params / compute_cache_key
# ---------------------------------------------------------------------------


class TestCacheParams:
    def test_only_output_affecting_params_are_included(self) -> None:
        b = _backend()
        params = _cache_params(b)
        assert params == {"max_res": 512}

    def test_process_length_and_target_fps_included_when_set(self) -> None:
        b = _backend(process_length=64, target_fps=24)
        params = _cache_params(b)
        assert params == {"max_res": 512, "process_length": 64, "target_fps": 24}

    def test_path_fields_not_in_params(self) -> None:
        b = _backend()
        # The real CLIBackend carries repo_dir/python_exe/model_dir — a mock
        # cannot prove they're excluded by _cache_params (it reads only the
        # three output-affecting fields).  The invariant is enforced at the
        # _cache_params implementation boundary: nothing path-like is ever
        # read off the backend object.
        params = _cache_params(b)
        for forbidden in ("repo_dir", "python_exe", "model_dir", "checkpoint_dir"):
            assert forbidden not in params


class TestCacheKey:
    def test_same_content_same_key(self, tmp_path) -> None:
        b = _backend()
        a = _write_video(tmp_path, "a.mp4")
        c = _write_video(tmp_path, "c.mp4")
        assert compute_cache_key(str(a), b) == compute_cache_key(str(c), b)

    def test_different_content_different_key(self, tmp_path) -> None:
        b = _backend()
        a = _write_video(tmp_path, "a.mp4")
        other = tmp_path / "b.mp4"
        other.write_bytes(b"DIFF" + b"y" * 1000 + b"TAIL")
        assert compute_cache_key(str(a), b) != compute_cache_key(str(other), b)

    def test_path_independence(self, tmp_path) -> None:
        """Same bytes at two different paths must hash identically (lead decision)."""
        b = _backend()
        src = tmp_path / "src" / "clip.mp4"
        src.parent.mkdir()
        src.write_bytes(_video_bytes(tmp_path))
        dst = tmp_path / "elsewhere" / "renamed_clip_v2.mp4"
        dst.parent.mkdir()
        dst.write_bytes(_video_bytes(tmp_path))
        assert compute_cache_key(str(src), b) == compute_cache_key(str(dst), b)

    def test_max_res_changes_key(self, tmp_path) -> None:
        a = _write_video(tmp_path, "a.mp4")
        b1 = _backend(max_resolution=512)
        b2 = _backend(max_resolution=1024)
        assert compute_cache_key(str(a), b1) != compute_cache_key(str(a), b2)

    def test_process_length_changes_key(self, tmp_path) -> None:
        a = _write_video(tmp_path, "a.mp4")
        b1 = _backend(process_length=None)
        b2 = _backend(process_length=64)
        assert compute_cache_key(str(a), b1) != compute_cache_key(str(a), b2)

    def test_target_fps_changes_key(self, tmp_path) -> None:
        a = _write_video(tmp_path, "a.mp4")
        b1 = _backend(target_fps=None)
        b2 = _backend(target_fps=30)
        assert compute_cache_key(str(a), b1) != compute_cache_key(str(a), b2)

    def test_fingerprint_is_content_not_path(self, tmp_path) -> None:
        """Same bytes, arbitrarily-named paths => identical key."""
        b = _backend()
        a = tmp_path / "videos" / "0001_src_720p.mp4"
        a.parent.mkdir(parents=True)
        a.write_bytes(_video_bytes(tmp_path))
        b_file = tmp_path / "uploads" / "user_upload_final.mp4"
        b_file.parent.mkdir(parents=True)
        b_file.write_bytes(_video_bytes(tmp_path))
        assert compute_cache_key(str(a), b) == compute_cache_key(str(b_file), b)

    def test_file_size_differs_key_differs(self, tmp_path) -> None:
        """Truncated copy of the same content must not collide with the full clip."""
        b = _backend()
        full = tmp_path / "full.mp4"
        full.write_bytes(b"x" * 1000)
        truncated = tmp_path / "trunc.mp4"
        truncated.write_bytes(b"x" * 500)
        assert compute_cache_key(str(full), b) != compute_cache_key(str(truncated), b)


# ---------------------------------------------------------------------------
# DepthCrafterEstimator cache integration
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_backend(tmp_path):
    """A mock backend that writes a deterministic .npy depth product on each call.

    Each invocation records the input_path/output_dir it saw and produces
    3 float32 depth maps.  The products are written from inside
    ``estimate_video`` (the real CLIBackend does the same via subprocess),
    so a cache hit can prove the runner was never invoked a second time.
    """
    backend = _backend()
    produced: list[list[np.ndarray]] = []

    def _fake_estimate(input_path: str, output_dir: str, target_size=None):
        frames = [np.random.default_rng(i).random((4, 4)).astype(np.float32) for i in range(3)]
        produced.append(frames)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            np.save(str(out / f"depth_{i:06d}.npy"), f)
        return frames

    backend.estimate_video.side_effect = _fake_estimate
    backend.produced = produced
    return backend


def _make_estimator(tmp_path, **overrides):
    """Build a DepthCrafterEstimator that skips the CUDA gate + uses a tmp cache dir.

    Defaults to a bare mock backend; pass ``backend=...`` in *overrides* to
    supply a specific one (e.g. the :func:`fake_backend` fixture).
    """
    cache_root = tmp_path / ".cache" / "depth"
    kw = dict(use_cache=True, cache_dir=cache_root, backend=_backend())
    kw.update(overrides)
    with patch("pipeline.depth_crafter._assert_cuda"):
        return DepthCrafterEstimator(**kw)


class TestCacheHitMiss:
    def test_second_run_is_hit_and_runner_not_invoked(self, tmp_path, fake_backend) -> None:
        """Same input + same params twice => the second call never touches the runner."""
        video = _write_video(tmp_path, "src.mp4")

        estimator = _make_estimator(tmp_path, backend=fake_backend)

        out1 = tmp_path / "out1"
        depths1 = estimator.estimate_video(str(video), output_dir=str(out1))
        assert len(depths1) == 3
        assert fake_backend.estimate_video.call_count == 1

        out2 = tmp_path / "out2"
        depths2 = estimator.estimate_video(str(video), output_dir=str(out2))
        assert len(depths2) == 3
        # Critical assertion: the runner was NOT called on the second (hit) run.
        assert fake_backend.estimate_video.call_count == 1

    def test_hit_returns_bytes_identical_depths(self, tmp_path, fake_backend) -> None:
        """Cache-hit depths must be byte-identical to the producer's output."""
        video = _write_video(tmp_path, "src.mp4")
        estimator = _make_estimator(tmp_path, backend=fake_backend)

        out1 = tmp_path / "out1"
        depths1 = estimator.estimate_video(str(video), output_dir=str(out1))
        out2 = tmp_path / "out2"
        depths2 = estimator.estimate_video(str(video), output_dir=str(out2))

        assert len(depths1) == len(depths2)
        for d1, d2 in zip(depths1, depths2, strict=True):
            assert np.array_equal(d1, d2)

    def test_cache_hit_log_line(self, tmp_path, fake_backend, caplog) -> None:
        import logging

        video = _write_video(tmp_path, "src.mp4")
        estimator = _make_estimator(tmp_path, backend=fake_backend)
        estimator.estimate_video(str(video), output_dir=str(tmp_path / "out1"))
        with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
            estimator.estimate_video(str(video), output_dir=str(tmp_path / "out2"))
        assert any("[cache] hit" in r.message for r in caplog.records)

    def test_meta_json_written_to_output_and_cache(self, tmp_path, fake_backend) -> None:
        """meta.json must be co-located with the depth maps in both the cache and the run output."""
        video = _write_video(tmp_path, "src.mp4")
        estimator = _make_estimator(tmp_path, backend=fake_backend)
        out = tmp_path / "out"
        estimator.estimate_video(str(video), output_dir=str(out))

        # Resolve the cache dir the estimator used.
        key = compute_cache_key(str(video), fake_backend)
        cache_dir = tmp_path / ".cache" / "depth" / key

        cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        out_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert cache_meta["depth_model"] == "depthcrafter"
        assert cache_meta["num_frames"] == 3
        assert cache_meta["max_res"] == 512
        # The output dir's meta.json is a faithful copy of the cache's.
        assert out_meta["depth_model"] == cache_meta["depth_model"]
        assert out_meta["num_frames"] == cache_meta["num_frames"]
        assert out_meta["max_res"] == cache_meta["max_res"]


class TestCacheMiss:
    def _writes_products(self, tmp_path):
        """Return an estimate_video side_effect that writes a .npy product + returns depths."""

        def _side_effect(input_path: str, output_dir: str, target_size=None):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(str(out / "depth_000000.npy"), np.zeros((4, 4), dtype=np.float32))
            return [np.zeros((4, 4), dtype=np.float32)]

        return _side_effect

    def test_changed_max_res_misses(self, tmp_path) -> None:
        video = _write_video(tmp_path, "src.mp4")
        backend1 = _backend(max_resolution=512)
        backend1.estimate_video.side_effect = self._writes_products(tmp_path)

        backend2 = _backend(max_resolution=1024)
        backend2.estimate_video.side_effect = self._writes_products(tmp_path)

        with patch("pipeline.depth_crafter._assert_cuda"):
            est1 = DepthCrafterEstimator(backend=backend1, cache_dir=tmp_path / ".cache" / "depth")
            est2 = DepthCrafterEstimator(backend=backend2, cache_dir=tmp_path / ".cache" / "depth")

        est1.estimate_video(str(video), output_dir=str(tmp_path / "out1"))
        est2.estimate_video(str(video), output_dir=str(tmp_path / "out2"))

        # Both backends must have run — different max_res => distinct cache keys.
        assert backend1.estimate_video.call_count == 1
        assert backend2.estimate_video.call_count == 1

    def test_changed_process_length_misses(self, tmp_path) -> None:
        video = _write_video(tmp_path, "src.mp4")
        b1 = _backend(process_length=None)
        b1.estimate_video.side_effect = self._writes_products(tmp_path)
        b2 = _backend(process_length=64)
        b2.estimate_video.side_effect = self._writes_products(tmp_path)

        with patch("pipeline.depth_crafter._assert_cuda"):
            e1 = DepthCrafterEstimator(backend=b1, cache_dir=tmp_path / ".cache" / "depth")
            e2 = DepthCrafterEstimator(backend=b2, cache_dir=tmp_path / ".cache" / "depth")

        e1.estimate_video(str(video), output_dir=str(tmp_path / "out1"))
        e2.estimate_video(str(video), output_dir=str(tmp_path / "out2"))
        assert b1.estimate_video.call_count == 1
        assert b2.estimate_video.call_count == 1


class TestUseCacheFalse:
    def test_use_cache_false_forces_recompute(self, tmp_path, fake_backend) -> None:
        """use_cache=False must recompute every call, even over an existing cache entry."""
        video = _write_video(tmp_path, "src.mp4")
        estimator = _make_estimator(tmp_path, backend=fake_backend, use_cache=False)

        for out_name in ("out_a", "out_b"):
            estimator.estimate_video(str(video), output_dir=str(tmp_path / out_name))

        # Two calls, two backend invocations — the cache was never consulted.
        assert fake_backend.estimate_video.call_count == 2
        # And no cache entry should have been written.
        cache_root = tmp_path / ".cache" / "depth"
        assert not cache_root.exists() or not any(cache_root.iterdir())


class TestPathIndependence:
    def test_same_content_different_path_is_hit(self, tmp_path, fake_backend, caplog) -> None:
        """Copy the video to another path and the second call must hit the cache."""
        import logging

        video_a = tmp_path / "videos" / "clip.mp4"
        video_a.parent.mkdir(parents=True)
        video_a.write_bytes(_video_bytes(tmp_path))
        video_b = tmp_path / "uploads" / "renamed_final.mp4"
        video_b.parent.mkdir(parents=True)
        video_b.write_bytes(_video_bytes(tmp_path))

        estimator = _make_estimator(tmp_path, backend=fake_backend)
        estimator.estimate_video(str(video_a), output_dir=str(tmp_path / "out_a"))
        with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
            estimator.estimate_video(str(video_b), output_dir=str(tmp_path / "out_b"))

        assert fake_backend.estimate_video.call_count == 1
        assert any("[cache] hit" in r.message for r in caplog.records)


class TestCacheIntegrity:
    def test_partial_cache_dir_is_not_treated_as_hit(self, tmp_path, fake_backend) -> None:
        """An empty dir or a dir missing meta.json is treated as a miss."""
        video = _write_video(tmp_path, "src.mp4")
        estimator = _make_estimator(tmp_path, backend=fake_backend)

        key = compute_cache_key(str(video), fake_backend)
        fake_cache = tmp_path / ".cache" / "depth" / key
        fake_cache.mkdir(parents=True)
        # Stub a product WITHOUT meta.json — the estimator must NOT call this a hit.
        FakeNumpyDepthProduct.write(fake_cache)

        estimator.estimate_video(str(video), output_dir=str(tmp_path / "out"))
        # The runner was invoked despite the pre-existing (invalid) dir.
        assert fake_backend.estimate_video.call_count == 1
        # And the run overwrote the dir with a valid entry.
        assert (fake_cache / "meta.json").is_file()

    def test_stale_cache_dir_is_wiped_on_miss(self, tmp_path) -> None:
        """A miss wipes any leftover files from a crashed prior run before persisting."""
        video = _write_video(tmp_path, "src.mp4")
        backend = _backend()

        def _fake_estimate(input_path: str, output_dir: str, target_size=None):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(str(out / "depth_000000.npy"), np.zeros((4, 4), dtype=np.float32))
            return [np.zeros((4, 4), dtype=np.float32)]

        backend.estimate_video.side_effect = _fake_estimate

        with patch("pipeline.depth_crafter._assert_cuda"):
            estimator = DepthCrafterEstimator(backend=backend, cache_dir=tmp_path / ".cache" / "depth")

        key = compute_cache_key(str(video), backend)
        stale_dir = tmp_path / ".cache" / "depth" / key
        stale_dir.mkdir(parents=True)
        (stale_dir / "zombie.npy").write_bytes(b"do-not-keep")

        estimator.estimate_video(str(video), output_dir=str(tmp_path / "out"))
        assert not (stale_dir / "zombie.npy").exists()
        # Fresh products + meta.json are present.
        assert (stale_dir / "meta.json").is_file()
        assert any(stale_dir.glob("depth_*.npy"))

    def test_default_cache_dir_is_models_dot_cache_depth(self, tmp_path) -> None:
        """When cache_dir is None, the default is models/.cache/depth relative to the repo."""
        with patch("pipeline.depth_crafter._assert_cuda"):
            estimator = DepthCrafterEstimator(backend=_backend(), cache_dir=None)
        assert estimator.cache_dir.name == "depth"
        assert estimator.cache_dir.parent.name == ".cache"
        assert estimator.cache_dir.parents[1].name == "models"

    def test_use_cache_defaults_to_true(self, tmp_path) -> None:
        """Existing call sites pass no new args — use_cache must default True so they still cache."""
        with patch("pipeline.depth_crafter._assert_cuda"):
            estimator = DepthCrafterEstimator(backend=_backend())
        assert estimator.use_cache is True
