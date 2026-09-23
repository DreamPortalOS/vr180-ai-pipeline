"""Issue #401: the Studio ``qa.dome_coverage`` node and ``scripts/dome_qa.py``
must share ONE coverage-radius algorithm and agree on every frame.

The regression that prompted this card: a production-template run reported
``level="bad"`` (content only reached zenith angle 74.5°, outer ring 29% filled)
yet ``passed=True``, because the Studio node judged pass/fail with
``coverage_deg >= min_deg`` instead of ``level``, while ``scripts/dome_qa.py``
used a stricter full-ring-fill scan. Two algorithms on the same master is how
they disagreed.

This module asserts the unification:

* a synthetic half-dome frame (lower semicircle filled to r=0.95, upper only
  to r=0.6 — the VR180→dome defect) must be rejected by BOTH consumers, and
* the shared scan produces the same ``coverage_r`` for the Studio node and
  ``dome_qa`` (one algorithm, not two).

All frames are synthetic numpy; no ffmpeg, no models, no network.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scripts.dome_qa import analyze_frame as dq_analyze_frame
from scripts.dome_qa import run_qa

from studio.coverage import analyze_frame as studio_analyze_frame
from studio.nodes.convert import DomeCoverageNode

SIZE = 256


def _half_dome(size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Asymmetric dome: bottom semicircle filled to r=0.95, top only to r=0.6.

    One azimuth reaches r≈0.95, but only ~half the circumference does, so
    full-ring-fill coverage must collapse to the short side (~0.6) and the
    master must FAIL — in both consumers.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2) / (size / 2.0)
    noise = rng.integers(0, 256, size=(size, size)).astype(np.uint8)
    bottom = yy > (size - 1) / 2.0
    fill = (bottom & (rr <= 0.95)) | (~bottom & (rr <= 0.6))
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[fill] = np.stack([noise[fill]] * 3, axis=1)
    return frame


def _full_dome(fill_r: float = 0.95, size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Symmetric noise-filled dome out to r=fill_r — the compliant counterpart."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2) / (size / 2.0)
    noise = rng.integers(0, 256, size=(size, size)).astype(np.uint8)
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[rr <= fill_r] = np.stack([noise[rr <= fill_r]] * 3, axis=1)
    return frame


# fill_r that lands the solid coverage zenith angle inside [min_deg=70, WARN_DEG=75)
# — i.e. level="bad" while coverage_deg >= min_deg. This is the exact frame from
# the bug report (74.5°, r≈0.83) that read passed=True: it separates the old
# ``coverage_deg >= min_deg`` rule from the fixed ``passed follows level``.
BUG_REPORT_FILL_R = 0.80


def _probe_data(size: int = SIZE) -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": size,
                "height": size,
                "avg_frame_rate": "30/1",
                "bit_rate": "45000000",
            }
        ],
        "format": {"bit_rate": "45000000", "duration": "12.5"},
    }


def _status(report, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"check {name!r} not found")


class TestOneAlgorithm:
    """The Studio node and dome_qa read the same coverage_r (issue #401)."""

    @pytest.mark.parametrize("frame, label", [(_half_dome(), "half"), (_full_dome(), "full")])
    def test_studio_and_dome_qa_agree(self, frame, label) -> None:
        studio = studio_analyze_frame(frame)
        dq = dq_analyze_frame(frame)
        # One algorithm ⇒ identical radius to floating point.
        assert studio.coverage_radius == pytest.approx(dq["coverage_r"], abs=1e-9), label
        # And the ring profiles match ring-for-ring.
        assert len(studio.ring_profile) == len(dq["ring_profile"])
        for a, b in zip(studio.ring_profile, dq["ring_profile"], strict=True):
            assert a["fill"] == pytest.approx(b["fill"], abs=1e-9)


class TestHalfDomeRejectedByBoth:
    """The half-dome defect: both the Studio node and dome_qa must FAIL it."""

    def test_studio_node_rejects_half_dome(self, tmp_path) -> None:
        # The old bug: a bad master read passed=True because coverage_deg >= min_deg
        # overrode level. Now passed must follow level.
        media = tmp_path / "half.png"
        assert cv2.imwrite(str(media), _half_dome())
        out = DomeCoverageNode().run(
            params={"min_deg": 70},  # the production-template default
            inputs={"video": str(media)},
            work_dir=str(tmp_path),
            node_id="cov",
        )
        report = out["report"]
        assert report["level"] == "bad"
        assert report["coverage_deg"] < 75.0  # below the warn threshold
        assert report["coverage_radius"] < 0.7  # collapsed to the short side
        # The fix itself: level=bad ⇒ passed=0, regardless of min_deg=70.
        assert out["passed"] == 0
        assert report["passed"] is False

    def test_studio_node_accepts_full_dome(self, tmp_path) -> None:
        media = tmp_path / "full.png"
        assert cv2.imwrite(str(media), _full_dome())
        out = DomeCoverageNode().run(
            params={"min_deg": 70},
            inputs={"video": str(media)},
            work_dir=str(tmp_path),
            node_id="cov",
        )
        assert out["report"]["level"] == "ok"
        assert out["passed"] == 1

    def test_bad_but_above_min_deg_still_fails(self, tmp_path) -> None:
        # The exact bug-report frame: coverage_deg lands at 73.8° (>= min_deg=70)
        # yet level="bad" (below the 75° warn band). The old ``coverage_deg >=
        # min_deg`` rule returned passed=1 here; it must now be passed=0.
        media = tmp_path / "bad-but-above-min.png"
        assert cv2.imwrite(str(media), _full_dome(BUG_REPORT_FILL_R))
        out = DomeCoverageNode().run(
            params={"min_deg": 70},
            inputs={"video": str(media)},
            work_dir=str(tmp_path),
            node_id="cov",
        )
        report = out["report"]
        assert report["level"] == "bad"
        assert report["coverage_deg"] >= 70.0  # above min_deg …
        assert report["coverage_deg"] < 75.0  # … but below the warn band
        assert out["passed"] == 0  # … yet passed follows level, not min_deg
        assert report["passed"] is False

    def test_warn_passes_but_carries_level(self, tmp_path) -> None:
        # level="warn" (75°–85°): passed=1 (it is still shippable) but the level
        # is passed through so the UI can mark it yellow, per the card.
        media = tmp_path / "warn.png"
        assert cv2.imwrite(str(media), _full_dome(0.82))  # 75.6° → warn
        out = DomeCoverageNode().run(
            params={"min_deg": 70},
            inputs={"video": str(media)},
            work_dir=str(tmp_path),
            node_id="cov",
        )
        assert out["report"]["level"] == "warn"
        assert out["passed"] == 1

    def test_dome_qa_rejects_half_dome(self, tmp_path) -> None:
        path = tmp_path / "half.mp4"
        path.write_bytes(b"fake")
        report = run_qa(
            str(path),
            size=SIZE,
            frames=[_half_dome()],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "coverage radius") == "fail"
        assert report.coverage_r < 0.7
        assert report.failed
        assert report.verdict == "FAIL"

    def test_dome_qa_accepts_full_dome(self, tmp_path) -> None:
        path = tmp_path / "full.mp4"
        path.write_bytes(b"fake")
        report = run_qa(
            str(path),
            size=SIZE,
            frames=[_full_dome()],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "coverage radius") == "pass"
        assert report.coverage_r >= 0.9
        assert not report.failed

    def test_dome_qa_worst_frame_half_dome_wins(self, tmp_path) -> None:
        # A clean full dome in the sample must not rescue the half-dome frame.
        path = tmp_path / "mix.mp4"
        path.write_bytes(b"fake")
        report = run_qa(
            str(path),
            size=SIZE,
            frames=[_full_dome(), _half_dome()],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "coverage radius") == "fail"
        assert report.verdict == "FAIL"
