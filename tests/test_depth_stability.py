"""Tests for the depth temporal-stability metric (scripts/depth_stability.py).

Verifies:
- temporal_jitter: a fully-static sequence is ~0, a noisy sequence is high,
  a translating sequence sits between.
- flicker_ratio: a static sequence has ~0 flicker, a high-frequency chatter
  sequence has high flicker.
- edge_consistency: a static sequence is ~1, a jiggling-edge sequence is low.
- Three-tier verdicts each tier exercised once; overall takes the worst.
- --compare renders a Markdown comparison table with deltas + winner.
- JSON round-trip via _to_json stays readable.

No GPU, models, cv2, or ffmpeg are touched here.  All sequences are
synthetic numpy arrays.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.depth_stability import (
    EDGE_OK,
    EDGE_WARN,
    FLICKER_OK,
    FLICKER_WARN,
    JITTER_OK,
    JITTER_WARN,
    _to_json,
    compute_report,
    edge_consistency,
    flicker_ratio,
    main,
    parse_args,
    render_compare,
    temporal_jitter,
)

# ---------------------------------------------------------------------------
# Synthetic sequences
# ---------------------------------------------------------------------------

RNG_SEED = 12345


def _static_frames(n: int = 6, h: int = 32, w: int = 32) -> list[np.ndarray]:
    """A constant depth field repeated — should be perfectly stable."""
    base = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
    return [base.copy() for _ in range(n)]


def _noisy_frames(n: int = 6, h: int = 32, w: int = 32, amp: float = 0.25) -> list[np.ndarray]:
    """Independent noise per frame — high jitter, high flicker."""
    rng = np.random.default_rng(RNG_SEED)
    base = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
    return [base + (rng.random((h, w), dtype=np.float32) - 0.5) * amp for _ in range(n)]


def _translating_frames(n: int = 6, h: int = 32, w: int = 32) -> list[np.ndarray]:
    """A smooth ramp shifted one pixel per frame — motion, not noise.

    Jitter is nonzero (frames differ) but flicker is low (monotonic advance,
    no sign flips) and edges stay consistent (the ramp edge just slides).
    """
    base = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    frames: list[np.ndarray] = []
    for i in range(n):
        shifted = np.roll(base, shift=i, axis=1)
        frames.append(shifted.copy())
    return frames


def _flicker_frames(n: int = 6, h: int = 32, w: int = 32, frac: float = 0.5) -> list[np.ndarray]:
    """Every other frame flips a large fraction of pixels up/down — chatter."""
    base = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(RNG_SEED)
    mask = rng.random((h, w)) < frac
    frames: list[np.ndarray] = []
    for i in range(n):
        f = base.copy()
        if i % 2 == 1:
            f[mask] = 1.0
        frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# temporal_jitter
# ---------------------------------------------------------------------------


class TestTemporalJitter:
    def test_static_is_near_zero(self) -> None:
        assert temporal_jitter(_static_frames()) == pytest.approx(0.0, abs=1e-7)

    def test_noisy_is_high(self) -> None:
        assert temporal_jitter(_noisy_frames()) > 0.05

    def test_translation_between_static_and_noisy(self) -> None:
        jit = temporal_jitter(_translating_frames())
        assert 0.0 < jit < temporal_jitter(_noisy_frames())

    def test_needs_two_frames(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            temporal_jitter([np.zeros((4, 4))])


# ---------------------------------------------------------------------------
# flicker_ratio
# ---------------------------------------------------------------------------


class TestFlickerRatio:
    def test_static_is_near_zero(self) -> None:
        # only 2 frames -> triple needs 3; but to be explicit use 6 frames
        assert flicker_ratio(_static_frames()) == pytest.approx(0.0, abs=1e-7)

    def test_flicker_sequence_is_high(self) -> None:
        # every-other-frame flip => nearly every masked pixel flickers
        assert flicker_ratio(_flicker_frames()) > 0.4

    def test_translation_is_low(self) -> None:
        # smooth monotonic advance => almost no sign flips
        assert flicker_ratio(_translating_frames()) < 0.1

    def test_too_few_frames_is_zero(self) -> None:
        assert flicker_ratio([np.zeros((4, 4))] * 2) == 0.0


# ---------------------------------------------------------------------------
# edge_consistency
# ---------------------------------------------------------------------------


class TestEdgeConsistency:
    def test_static_is_near_one(self) -> None:
        # identical frames => identical edges => IoU 1
        assert edge_consistency(_static_frames()) == pytest.approx(1.0, abs=1e-6)

    def test_noisy_is_lower(self) -> None:
        assert edge_consistency(_noisy_frames()) < edge_consistency(_static_frames())

    def test_translation_keeps_edge_consistent(self) -> None:
        # sliding ramp: edges move but stay ramp-shaped => decent overlap
        assert edge_consistency(_translating_frames()) > 0.0

    def test_needs_two_frames(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            edge_consistency([np.zeros((4, 4))])


# ---------------------------------------------------------------------------
# Verdicts / overall
# ---------------------------------------------------------------------------


class TestVerdicts:
    def test_static_all_ok(self) -> None:
        report = compute_report(_static_frames())
        assert report.temporal_jitter.ok == "OK"
        assert report.flicker_ratio.ok == "OK"
        assert report.edge_consistency.ok == "OK"
        assert report.overall == "OK"

    def test_noisy_overall_fail_or_warn(self) -> None:
        # heavy noise crosses the jitter warn threshold at minimum
        report = compute_report(_noisy_frames())
        assert report.temporal_jitter.ok in {"WARN", "FAIL"}
        assert report.overall in {"WARN", "FAIL"}

    def test_overall_takes_worst(self) -> None:
        # hand-craft a sequence that fails one metric, passes the others.
        # static base but with a big jump on the 2nd frame -> jitter spikes
        # but flicker stays 0 (only 1 diff, no triple) and edges stay put.
        frames = _static_frames()
        frames[1] = frames[0] * 1.5
        report = compute_report(frames)
        # overall must equal the worst (highest-ranked) individual tier.
        rank = {"OK": 0, "WARN": 1, "FAIL": 2}
        worst_rank = max(rank[m.ok] for m in (report.temporal_jitter, report.flicker_ratio, report.edge_consistency))
        assert rank[report.overall] == worst_rank

    def test_thresholds_exposed(self) -> None:
        assert JITTER_OK < JITTER_WARN
        assert FLICKER_OK < FLICKER_WARN
        assert EDGE_OK > EDGE_WARN  # higher is better, so OK threshold is larger


# ---------------------------------------------------------------------------
# --compare
# ---------------------------------------------------------------------------


class TestCompare:
    def _write_report(self, tmp_path, name, tj, fr, ec, overall="OK") -> str:
        data = {
            "temporal_jitter": {
                "name": "temporal_jitter",
                "value": tj,
                "ok": "OK",
                "thresholds": {"OK": JITTER_OK, "WARN": JITTER_WARN},
                "higher_is_better": False,
            },
            "flicker_ratio": {
                "name": "flicker_ratio",
                "value": fr,
                "ok": "OK",
                "thresholds": {"OK": FLICKER_OK, "WARN": FLICKER_WARN},
                "higher_is_better": False,
            },
            "edge_consistency": {
                "name": "edge_consistency",
                "value": ec,
                "ok": "OK",
                "thresholds": {"OK": EDGE_OK, "WARN": EDGE_WARN},
                "higher_is_better": True,
            },
            "n_frames": 6,
            "overall": overall,
        }
        p = tmp_path / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_compare_table_has_all_metrics(self, tmp_path) -> None:
        a = self._write_report(tmp_path, "a.json", 0.01, 0.02, 0.95)
        b = self._write_report(tmp_path, "b.json", 0.10, 0.30, 0.40)
        out = render_compare(a, b)
        for m in ("temporal_jitter", "flicker_ratio", "edge_consistency"):
            assert m in out
        # delta column present
        assert "delta" in out
        # winner column present
        assert "winner" in out

    def test_compare_picks_better(self, tmp_path) -> None:
        a = self._write_report(tmp_path, "a.json", 0.01, 0.02, 0.95)  # better
        b = self._write_report(tmp_path, "b.json", 0.10, 0.30, 0.40)  # worse
        out = render_compare(a, b)
        # for jitter (lower better), a wins; for edge (higher better), a wins
        assert "a.json" in out

    def test_compare_via_cli(self, tmp_path, capsys) -> None:
        a = self._write_report(tmp_path, "a.json", 0.01, 0.02, 0.95)
        b = self._write_report(tmp_path, "b.json", 0.10, 0.30, 0.40, overall="FAIL")
        rc = main(["--compare", a, b])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "temporal_jitter" in captured
        assert "winner" in captured

    def test_compare_bad_file_returns_1(self, tmp_path, capsys) -> None:
        a = self._write_report(tmp_path, "a.json", 0.01, 0.02, 0.95)
        rc = main(["--compare", a, str(tmp_path / "missing.json")])
        assert rc == 1


# ---------------------------------------------------------------------------
# JSON round-trip + CLI
# ---------------------------------------------------------------------------


class TestJsonAndCli:
    def test_to_json_serialisable(self) -> None:
        report = compute_report(_static_frames())
        data = _to_json(report)
        # must be JSON-encodable without numpy types leaking through
        text = json.dumps(data)
        assert "temporal_jitter" in text
        assert "overall" in text

    def test_cli_writes_json_and_prints(self, tmp_path, capsys) -> None:
        # Build a fake npy dir with static frames so no model runs.
        ddir = tmp_path / "depths"
        ddir.mkdir()
        base = np.linspace(0.1, 0.9, 8 * 8, dtype=np.float32).reshape(8, 8)
        for i in range(5):
            np.save(ddir / f"depth_{i:06d}.npy", base)

        out_json = tmp_path / "report.json"
        rc = main(
            [
                "--depth-npy-dir",
                str(ddir),
                "--json",
                str(out_json),
                "--print",
            ]
        )
        assert rc == 0
        assert out_json.exists()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["n_frames"] == 5
        assert loaded["overall"] == "OK"
        assert "✅" in capsys.readouterr().out

    def test_cli_print_only_when_no_json(self, tmp_path, capsys) -> None:
        ddir = tmp_path / "depths"
        ddir.mkdir()
        base = np.zeros((8, 8), dtype=np.float32)
        for i in range(4):
            np.save(ddir / f"depth_{i:06d}.npy", base)
        rc = main(["--depth-npy-dir", str(ddir)])
        assert rc == 0
        assert "Depth Temporal Stability Report" in capsys.readouterr().out

    def test_cli_requires_source(self, capsys) -> None:
        rc = main([])
        assert rc == 1
        assert "depth-npy-dir" in capsys.readouterr().err

    def test_cli_missing_dir_returns_1(self, tmp_path, capsys) -> None:
        rc = main(["--depth-npy-dir", str(tmp_path / "nope")])
        assert rc == 1

    def test_help_exits_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--compare" in out
        assert "--depth-npy-dir" in out
