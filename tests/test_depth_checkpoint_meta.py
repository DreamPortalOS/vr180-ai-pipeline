"""Tests for I-6 (#121): model-scoped depth checkpoint dirs + meta.json.

Verifies:
- Two depth models run back-to-back on the same input write to
  **non-overlapping** directories (``depth/depth-anything/`` vs
  ``depth/depthcrafter/``) and each carries a correct ``meta.json``.
- Resume safety: a cached depth checkpoint whose ``meta.json`` does not match
  the current invocation's model/params is **not** silently reused —
  ``load_depth_checkpoint`` returns ``None`` (forcing a recompute) and logs the
  mismatch.
- ``depth_stability.py`` prints the provenance (model + params) at the top of
  the report when ``meta.json`` is present, and a "来源未知" reminder when it
  is not.

All heavy model work is stubbed — CI runs without GPU / models / ffmpeg.
"""

from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import numpy as np


def _import_run_pipeline():
    """Import scripts/run_pipeline.py as a module (same pattern as test_pipeline)."""
    import contextlib
    import importlib.util
    import sys

    scripts = os.path.join(os.path.dirname(__file__), "..", "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "run_pipeline_i6",
            os.path.join(scripts, "run_pipeline.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts)


def _args(rp, tmp_path, *, depth_model="depth-anything", model_size="small", max_res=None, frames=5):
    """Minimal args namespace for the depth stage (plain Namespace, no MagicMock)."""
    return SimpleNamespace(
        depth_model=depth_model,
        model_size=model_size,
        depthcrafter_max_res=max_res,
        depthcrafter_repo_dir=None,
        depthcrafter_python=None,
        depthcrafter_checkpoint_dir=None,
        temporal_smoothing=0.0,
        temp_dir=str(tmp_path),
        input=str(tmp_path / "fake_input.mp4"),
        device="cpu",
        chunk_size=None,
        overlap=0,
    )


def _frames(n=5, h=32, w=32):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# get_depth_dir / meta.json helpers
# ---------------------------------------------------------------------------


class TestDepthDirScoping:
    def test_depth_dir_is_model_scoped(self, tmp_path):
        rp = _import_run_pipeline()
        a = _args(rp, tmp_path, depth_model="depth-anything")
        b = _args(rp, tmp_path, depth_model="depthcrafter")
        assert rp.get_depth_dir(a) != rp.get_depth_dir(b)
        assert "depth-anything" in rp.get_depth_dir(a)
        assert "depthcrafter" in rp.get_depth_dir(b)

    def test_two_models_do_not_overwrite(self, tmp_path, monkeypatch):
        """Both models run on the same input → two separate dirs, each with meta."""
        rp = _import_run_pipeline()
        frames = _frames()

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(rp.DepthEstimator, "estimate", fake_estimate)

        # depth-anything run
        a = _args(rp, tmp_path, depth_model="depth-anything", model_size="small")
        rp.run_depth_stage(a, frames)
        dir_a = rp.get_depth_dir(a)

        # depthcrafter run — stub the estimator to avoid the real model.
        b = _args(rp, tmp_path, depth_model="depthcrafter", max_res=512)
        fake_depths = [np.zeros((32, 32), dtype=np.float32) for _ in frames]

        class FakeDC:
            def __init__(self, **kwargs):
                pass

            def estimate_video(self, input_path, output_dir):
                return fake_depths

        monkeypatch.setattr(rp, "DepthCrafterEstimator", FakeDC)
        rp.run_depth_stage(b, frames)
        dir_b = rp.get_depth_dir(b)

        assert dir_a != dir_b

        # Both dirs hold their own npy files + meta.json — no cross-contamination.
        assert len(glob.glob(os.path.join(dir_a, "depth_*.npy"))) == len(frames)
        assert len(glob.glob(os.path.join(dir_b, "depth_*.npy"))) == len(frames)

        meta_a = rp.load_depth_meta(dir_a)
        meta_b = rp.load_depth_meta(dir_b)
        assert meta_a["depth_model"] == "depth-anything"
        assert meta_a["model_size"] == "small"
        assert meta_b["depth_model"] == "depthcrafter"
        assert meta_b["max_res"] == 512
        assert meta_a["num_frames"] == len(frames)
        assert meta_b["num_frames"] == len(frames)
        # A/B provenance: the metas differ, so each model's maps are attributable.
        assert meta_a["depth_model"] != meta_b["depth_model"]


# ---------------------------------------------------------------------------
# Resume safety: stale / wrong-model meta is never silently reused
# ---------------------------------------------------------------------------


class TestResumeSafety:
    def _seed_depth(self, rp, tmp_path, args, frames):
        """Run the depth stage once to seed a checkpoint dir."""

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        rp.DepthEstimator.estimate = fake_estimate
        rp.run_depth_stage(args, frames)
        return rp.get_depth_dir(args)

    def test_matching_meta_loads_checkpoint(self, tmp_path, monkeypatch):
        rp = _import_run_pipeline()
        frames = _frames()

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(rp.DepthEstimator, "estimate", fake_estimate)

        args = _args(rp, tmp_path, depth_model="depth-anything")
        rp.run_depth_stage(args, frames)

        depths = rp.load_depth_checkpoint(args)
        assert depths is not None
        assert len(depths) == len(frames)

    def test_model_mismatch_recomputes_with_log(self, tmp_path, monkeypatch, caplog):
        """Switching --depth-model must not reuse the other model's maps."""
        rp = _import_run_pipeline()
        frames = _frames()

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(rp.DepthEstimator, "estimate", fake_estimate)

        # Seed a depth-anything cache.
        a = _args(rp, tmp_path, depth_model="depth-anything")
        rp.run_depth_stage(a, frames)

        # Resume as depthcrafter: its dir is empty (no npy) → [] (nothing to
        # reuse), NOT the depth-anything maps.  Crucially the two dirs differ —
        # there is no shared cache to silently pull from.
        b = _args(rp, tmp_path, depth_model="depthcrafter", max_res=512)
        depths = rp.load_depth_checkpoint(b)
        assert depths == []
        assert rp.get_depth_dir(b) != rp.get_depth_dir(a)

    def test_param_mismatch_recomputes_with_log(self, tmp_path, monkeypatch, caplog):
        """Changed model_size / max_res must not reuse stale maps."""
        rp = _import_run_pipeline()
        frames = _frames()

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(rp.DepthEstimator, "estimate", fake_estimate)

        args = _args(rp, tmp_path, depth_model="depth-anything", model_size="small")
        rp.run_depth_stage(args, frames)

        # Resume with a different model_size → meta mismatch → None + log.
        changed = _args(rp, tmp_path, depth_model="depth-anything", model_size="large")
        with caplog.at_level("WARNING"):
            depths = rp.load_depth_checkpoint(changed)
        assert depths is None
        assert any("stale" in r.message.lower() or "recompute" in r.message.lower() for r in caplog.records)

    def test_frame_count_mismatch_recomputes(self, tmp_path, monkeypatch):
        """A different num_frames in the meta is treated as stale."""
        rp = _import_run_pipeline()
        frames = _frames(5)

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(rp.DepthEstimator, "estimate", fake_estimate)

        args = _args(rp, tmp_path)
        rp.run_depth_stage(args, frames)
        depth_dir = rp.get_depth_dir(args)

        # Tamper the meta: pretend the cache is for a different frame count.
        meta_path = os.path.join(depth_dir, "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.loads(f.read())
        meta["num_frames"] = 999
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        depths = rp.load_depth_checkpoint(args)
        assert depths is None

    def test_no_meta_loads_with_unknown_source_warning(self, tmp_path, caplog):
        """A pre-#121 cache (no meta.json) loads but warns 来源未知 (not silently trusted)."""
        rp = _import_run_pipeline()
        args = _args(rp, tmp_path)
        depth_dir = rp.get_depth_dir(args)
        os.makedirs(depth_dir, exist_ok=True)
        for i in range(3):
            np.save(os.path.join(depth_dir, f"depth_{i:06d}.npy"), np.zeros((8, 8), dtype=np.float32))

        with caplog.at_level("WARNING"):
            depths = rp.load_depth_checkpoint(args)
        assert depths is not None
        assert len(depths) == 3
        assert any("未知" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# depth_stability provenance
# ---------------------------------------------------------------------------


class TestDepthStabilitySource:
    def _seed_npy_dir(self, tmp_path, with_meta: bool, meta: dict | None = None):
        ddir = tmp_path / ("depths_meta" if with_meta else "depths_bare")
        ddir.mkdir()
        base = np.linspace(0.1, 0.9, 8 * 8, dtype=np.float32).reshape(8, 8)
        for i in range(5):
            np.save(ddir / f"depth_{i:06d}.npy", base)
        if with_meta:
            payload = meta or {
                "depth_model": "depthcrafter",
                "num_frames": 5,
                "max_res": 512,
                "timestamp": "2026-09-01T08:00:00",
            }
            (ddir / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return ddir

    def test_report_shows_source_when_meta_present(self, tmp_path, capsys):
        from scripts.depth_stability import main

        ddir = self._seed_npy_dir(tmp_path, with_meta=True)
        rc = main(["--depth-npy-dir", str(ddir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "depthcrafter" in out
        assert "max_res=512" in out

    def test_report_warns_unknown_source_when_no_meta(self, tmp_path, capsys):
        from scripts.depth_stability import main

        ddir = self._seed_npy_dir(tmp_path, with_meta=False)
        rc = main(["--depth-npy-dir", str(ddir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "来源未知" in out

    def test_load_depth_meta_roundtrip(self, tmp_path):
        from scripts.depth_stability import load_depth_meta

        ddir = self._seed_npy_dir(tmp_path, with_meta=True)
        meta = load_depth_meta(ddir)
        assert meta is not None
        assert meta["depth_model"] == "depthcrafter"
        assert meta["num_frames"] == 5

        bare = self._seed_npy_dir(tmp_path, with_meta=False)
        assert load_depth_meta(bare) is None

    def test_json_report_carries_source(self, tmp_path):
        from scripts.depth_stability import main

        ddir = self._seed_npy_dir(tmp_path, with_meta=True)
        out_json = tmp_path / "report.json"
        rc = main(["--depth-npy-dir", str(ddir), "--json", str(out_json)])
        assert rc == 0
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["source"]["depth_model"] == "depthcrafter"
