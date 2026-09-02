"""Tests for K-21 (#224) — streaming pipeline honours --temp-dir.

The streaming path (``StreamingPipeline.process_stream``) previously ignored
any caller-supplied work directory and wrote its depth products into a
``tempfile.mkdtemp`` it never let the caller see — so
``make_comparison``'s depth-dir resolver could never find the ``depth_*.npy``
files the K-16 depth-stability metrics need, and the metric cells stayed ``—``
in every real render.

These tests pin the K-21 fix:

  - With ``temp_dir`` set, depth products land under ``<temp_dir>/depth/`` and
    stereo intermediates under ``<temp_dir>/stereo/``, and they **survive** the
    run (the caller owns the lifetime).
  - The on-disk layout is exactly what
    ``scripts.make_comparison.default_depth_dir_resolver`` globs — the test
    imports that resolver directly and asserts it resolves the dir.
  - Without ``temp_dir`` the streaming path leaves nothing under the repo or
    ``video/`` tree (the #163/#164 no-pollution contract).  Its mkdtemp lives
    in the system temp, never in the working tree.

All tests are CPU-only: cv2 capture / ffmpeg / model stages are mocked or
faked.  No real model download, no real inference, no real ffmpeg.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# K-15 (#205): let this script run directly without PYTHONPATH set.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.make_comparison as mc  # noqa: E402

from pipeline.streaming_pipeline import StreamingPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Fake whole-clip backends (mirrors tests/test_streaming_backends.py — no CUDA,
# no real inference).  Kept local so this file stays self-contained.
# ---------------------------------------------------------------------------


class _FakeWholeClipStereo:
    """A whole-clip stereo backend (exposes ``render_video``) that records the
    depth_dir it was handed and returns canned L/R frame lists."""

    def __init__(self, num_frames=2):
        self.num_frames = num_frames
        self.calls = []

    def render_video(self, input_path, depth_dir, output_left, output_right):
        self.calls.append((input_path, depth_dir, output_left, output_right))
        # The actual L/R frames are loaded back via a patched _load_video_frames
        # in _run, so this backend only needs to record the call + return paths.
        return output_left, output_right


def _make_pipeline(temp_dir=None, **kwargs):
    """Build a StreamingPipeline with the heavy eq_mapper mocked out and the
    whole-clip stereo backend + per-frame depth estimator wired so the
    ``_emit_perframe_depths`` path (the one that persists ``depth_*.npy``)
    actually runs."""
    with patch("pipeline.streaming_pipeline.EquirectangularMapper"):
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("output_width", 100)
        kwargs.setdefault("output_height", 50)
        if temp_dir is not None:
            kwargs["temp_dir"] = temp_dir
        stereo = _FakeWholeClipStereo(num_frames=kwargs.pop("_num_frames", 2))
        kwargs.setdefault("stereo_renderer", stereo)
        kwargs.setdefault("stereo_backend_name", "stereocrafter")
        p = StreamingPipeline(**kwargs)
        # Per-frame depth estimator stand-in (Depth-Anything).  Spec it so the
        # pipeline does NOT mistake it for a whole-clip depth backend.
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.full((8, 8), 0.5, dtype=np.float32)
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)
        return p, stereo


def _fake_cap(num_frames=2, w=8, h=8):
    """A cv2.VideoCapture stand-in yielding *num_frames* fake frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: w,  # CAP_PROP_FRAME_WIDTH
        4: h,  # CAP_PROP_FRAME_HEIGHT
        7: float(num_frames),  # CAP_PROP_FRAME_COUNT
        5: 30.0,  # CAP_PROP_FPS
    }.get(prop, 0.0)
    reads = [(True, np.zeros((h, w, 3), dtype=np.uint8)) for _ in range(num_frames)]
    reads.append((False, None))
    cap.read.side_effect = reads
    return cap


def _run(p, num_frames=2):
    """Drive ``process_stream`` to completion with cv2/Popen mocked."""
    cap = _fake_cap(num_frames=num_frames)
    proc = MagicMock()
    proc.returncode = 0
    with (
        patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
        patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
        patch("pipeline.streaming_pipeline._load_video_frames", return_value=([], [])),
    ):
        p.process_stream("in.mp4", "out.mp4")


# ---------------------------------------------------------------------------
# temp_dir honoured: products land under <temp_dir>/{depth,stereo} and survive
# ---------------------------------------------------------------------------


class TestTempDirHonoured:
    """With ``temp_dir`` set the streaming path writes its products into the
    caller-owned tree and leaves them there."""

    def test_depth_npy_lands_under_temp_dir_depth_and_survives(self, tmp_path):
        temp_dir = tmp_path / "work"
        p, _stereo = _make_pipeline(temp_dir=str(temp_dir), _num_frames=3)
        _run(p, num_frames=3)

        depth_dir = temp_dir / "depth"
        # The npy files the per-frame depth stage wrote are still on disk.
        npy = sorted(depth_dir.glob("depth_*.npy"))
        assert npy, f"no depth_*.npy under {depth_dir}"
        assert len(npy) == 3
        # Sanity: they really are numpy arrays.
        loaded = np.load(str(npy[0]))
        assert loaded.shape == (8, 8)

    def test_stereo_intermediates_under_temp_dir_stereo(self, tmp_path):
        temp_dir = tmp_path / "work"
        p, _stereo = _make_pipeline(temp_dir=str(temp_dir), _num_frames=2)
        _run(p, num_frames=2)

        stereo_dir = temp_dir / "stereo"
        # The stereo work dir must be created under <temp_dir>/stereo/ so the
        # caller (and any downstream stage) can find the L/R intermediates the
        # real backend would write there.  The fake backend doesn't persist mp4s
        # (tests stay CPU-only, no real ffmpeg), so we assert the dir exists and
        # that the pipeline proposed paths inside it — not that mp4s are present.
        assert stereo_dir.is_dir()
        # The backend was handed L/R paths inside the caller's stereo dir.
        _left, _depth_dir, left_out, right_out = _stereo.calls[0]
        assert Path(left_out).is_relative_to(stereo_dir)
        assert Path(right_out).is_relative_to(stereo_dir)

    def test_default_depth_dir_resolver_finds_it(self, tmp_path):
        """The on-disk layout must be exactly what the comparison resolver
        globs — import it and assert it resolves (the decisive K-16 gate)."""
        temp_dir = tmp_path / "work"
        # render_started = before the run, so the freshness gate (npy mtime must
        # be after render_started - 1s) passes for files written during _run.
        render_started = time.time()
        p, _stereo = _make_pipeline(temp_dir=str(temp_dir), _num_frames=2)
        _run(p, num_frames=2)

        result = mc.RecipeResult(
            recipe="temporal",
            temp_dir=str(temp_dir),
            render_started=render_started,
        )
        resolved = mc.default_depth_dir_resolver(
            "src.mp4",
            mc.Recipe(name="temporal", args=[]),
            result,
        )
        assert resolved == str(temp_dir / "depth")

    def test_depth_dir_handed_to_stereo_is_the_temp_dir_one(self, tmp_path):
        """The stereo backend must receive the depth dir under temp_dir (so its
        forward-splat reads the maps the depth stage just wrote there), not a
        fresh mkdtemp."""
        temp_dir = tmp_path / "work"
        p, stereo = _make_pipeline(temp_dir=str(temp_dir), _num_frames=2)
        _run(p, num_frames=2)
        assert len(stereo.calls) == 1
        assert stereo.calls[0][1] == str(temp_dir / "depth")


# ---------------------------------------------------------------------------
# No temp_dir: no pollution of the repo / video tree (the #163/#164 contract)
# ---------------------------------------------------------------------------


class TestNoTempDirNoPollution:
    """Without ``temp_dir`` the streaming path must not drop any directory into
    the repo or ``video/`` tree — its mkdtemp lives in the system temp."""

    def test_no_dir_created_in_cwd_or_video(self, tmp_path, monkeypatch):
        # Run from a clean cwd inside tmp_path so we can snapshot exactly what
        # appears in the working tree afterwards.
        cwd = tmp_path / "runroot"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        # A pretend 'video' sibling, to assert nothing lands there either.
        (cwd.parent / "video").mkdir(exist_ok=True)

        before = {p for p in Path(cwd).rglob("*")}
        video_before = {p for p in (cwd.parent / "video").rglob("*")}

        p, _stereo = _make_pipeline(_num_frames=2)  # no temp_dir
        _run(p, num_frames=2)

        after = {p for p in Path(cwd).rglob("*")}
        video_after = {p for p in (cwd.parent / "video").rglob("*")}
        # No new dirs/files appeared in the working tree or the video tree.
        assert after - before == set(), f"streaming polluted cwd: {after - before}"
        assert video_after - video_before == set(), f"streaming polluted video/: {video_after - video_before}"

    def test_mkdtemp_lives_outside_repo(self, tmp_path, monkeypatch):
        """When no temp_dir is given the streaming path's mkdtemp must resolve
        to the system temp (outside the repo), not the working tree."""
        cwd = tmp_path / "runroot"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        captured = {}

        real_mkdtemp = __import__("tempfile").mkdtemp

        def spy(prefix=None, dir=None):
            d = real_mkdtemp(prefix=prefix)
            captured.setdefault("dirs", []).append(d)
            return d

        with patch("pipeline.streaming_pipeline.tempfile.mkdtemp", side_effect=spy):
            p, _stereo = _make_pipeline(_num_frames=2)
            _run(p, num_frames=2)

        # At least one mkdtemp was created, and every one of them is outside the
        # repo cwd (the no-pollution contract — system temp, not the worktree).
        assert captured["dirs"], "streaming path did not call mkdtemp at all"
        for d in captured["dirs"]:
            assert not Path(d).is_relative_to(cwd), f"mkdtemp {d} landed inside the repo worktree {cwd}"


# ---------------------------------------------------------------------------
# Regression: temp_dir default is None (pre-K-21 callers unchanged)
# ---------------------------------------------------------------------------


class TestDefaultUnchanged:
    def test_temp_dir_defaults_to_none(self):
        with patch("pipeline.streaming_pipeline.EquirectangularMapper"):
            p = StreamingPipeline(output_width=100, output_height=50, device="cpu")
        assert p.temp_dir is None
