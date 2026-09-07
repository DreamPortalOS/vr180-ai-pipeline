"""Tests for the source-hfov calibrator (scripts/calibrate_hfov.py, C-1 #291).

The tool answers one question: *what ``--src-hfov`` should this clip be given?*
It answers it by pushing each frame through a ``v360`` sphere round trip at
every candidate hfov and scoring how straight the result comes out — only the
correct candidate un-bends the picture.

What is covered, and how
------------------------
The card's five acceptance criteria, in order:

1. **Synthetic ground truth.**  A numpy/OpenCV grid of straight lines is warped
   into "pseudo footage" at a *known* hfov of 100° by running the analysis
   chain backwards (``flat -> hequirect -> fisheye``), then handed to the
   calibrator, which must recover 100 ± 5.  No real video, no model, no
   network — just ffmpeg, which CI already relies on for the e2e smoke test.
2. **Curve shape.**  On that same run, the 100° score must sit below both 70°
   and 130°.  Asserted as a *relative* ordering; the score's absolute units are
   an implementation detail and are never pinned.
3. **Honest abstention.**  A pure-noise frame has no straight lines to measure,
   so the report must say 置信度低 rather than invent a number.
4. **``--json``.**  Machine-readable output must survive ``json.loads`` and
   carry ``recommended_hfov`` / ``scores`` / ``confidence``.
5. **Subprocess hygiene.**  Every ``subprocess.run`` in the module is checked at
   the AST level: list form, never ``shell=True``, never an f-string command.

Everything outside the ffmpeg-gated block runs without ffmpeg: the round trip
is injected as a fake rectifier that draws deliberately bowed lines, so the
real scoring, aggregation and confidence logic are exercised with no
subprocess at all.  That keeps the bulk of the file fast and keeps CI honest
even on a runner without ``v360``.

Regression note (see :data:`~scripts.calibrate_hfov.HOUGH_GAP_FRAC`): the
Hough ``maxLineGap`` used to be a hard-coded 4 px, which made a *straight*
line fail to register wherever another line crossed it — penalising exactly
the candidate that was right.  ``test_straight_segments_span_a_crossed_grid``
and ``test_hough_gap_scales_with_min_length`` are the guards for that.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts.calibrate_hfov import (
    DEFAULT_FRAMES,
    DEFAULT_GRID,
    DEFAULT_MIN_LINE_FRAC,
    DEFAULT_OUT_FOV,
    HOUGH_MIN_GAP_PX,
    LOW_CONFIDENCE_TEXT,
    CalibrationResult,
    CandidateScore,
    V360Rectifier,
    build_parser,
    build_roundtrip_filter,
    calibrate,
    edge_map,
    equidistant_vfov,
    format_report,
    hough_gap,
    load_frames,
    main,
    parse_grid,
    pinhole_vfov,
    render_plot,
    run_v360,
    straight_segments,
    straightness_score,
)

SOURCE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_hfov.py"

# ---------------------------------------------------------------------------
# ffmpeg gate (same shape as tests/test_outpainter.py, tests/test_equirect_perf.py)
# ---------------------------------------------------------------------------


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg v360 unavailable")


# ---------------------------------------------------------------------------
# Synthetic imagery
# ---------------------------------------------------------------------------

#: Analysis size for the synthetic acceptance run — the tool's own default
#: width at 16:9, i.e. its real operating point.
SYNTH_W, SYNTH_H = 640, 360

#: The hfov the pseudo-footage is *built* at; the calibrator has to find it.
TRUE_HFOV = 100.0

#: Rectilinear fov of the imaginary world render the grid stands for.  Wider
#: than ``TRUE_HFOV`` so the fisheye it is warped into is fully covered — a
#: narrower world would leave black borders that are an artefact, not evidence.
WORLD_HFOV = 130.0

#: Coarse but *wide* candidate grid: 60,70,…,150.  It brackets the whole
#: default range and lands exactly on the 70 / 100 / 130 the card names, so the
#: answer is never a product of a conveniently narrow search.
ACCEPTANCE_GRID = "60:150:10"


def grid_frame(width: int = SYNTH_W, height: int = SYNTH_H, spacing: int | None = None) -> np.ndarray:
    """A plain grayscale lattice of straight lines — the 'world' before warping."""
    spacing = spacing or max(8, min(width, height) // 8)
    img = np.full((height, width), 20, np.uint8)
    for x in range(spacing, width, spacing):
        cv2.line(img, (x, 0), (x, height - 1), 235, 2)
    for y in range(spacing, height, spacing):
        cv2.line(img, (0, y), (width - 1, y), 235, 2)
    return img


def bowed_frame(width: int, height: int, bow: float, spacing: int | None = None) -> np.ndarray:
    """The same lattice, barrel-bowed by ``bow`` pixels. ``bow=0`` is straight.

    Used to fake the round trip without ffmpeg: a candidate that is wrong by
    ``d`` degrees is handed back a frame bowed in proportion to ``d``, so the
    *real* straightness score has to do the discriminating.
    """
    spacing = spacing or max(8, min(width, height) // 8)
    img = np.full((height, width), 20, np.uint8)
    xs = np.arange(width, dtype=np.float64)
    tx = xs / (width - 1) * 2.0 - 1.0
    for y in range(spacing, height, spacing):
        ys = y + bow * (1.0 - tx**2)
        cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, 235, 2)
    ys_all = np.arange(height, dtype=np.float64)
    ty = ys_all / (height - 1) * 2.0 - 1.0
    for x in range(spacing, width, spacing):
        xs_col = x + bow * (1.0 - ty**2)
        cv2.polylines(img, [np.stack([xs_col, ys_all], 1).astype(np.int32)], False, 235, 2)
    return img


def noise_frame(width: int = SYNTH_W, height: int = SYNTH_H, seed: int = 1234) -> np.ndarray:
    """Uniform noise: plenty of edges, not one straight line among them."""
    return np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)


def forward_warp(frame: np.ndarray, true_hfov: float, world_hfov: float = WORLD_HFOV) -> np.ndarray:
    """Run the calibrator's chain *backwards* to fabricate footage of known hfov.

    Analysis is ``fisheye(candidate) -> hequirect -> flat``; this is its inverse,
    ``flat(world) -> hequirect -> fisheye(true_hfov)``.  The result is what an
    equidistant lens of ``true_hfov`` degrees would have recorded of a world
    whose lines are straight — exactly the input the calibrator claims to solve.
    """
    height, width = frame.shape[:2]
    equirect = max(2, round(width * 2 / 2) * 2)
    vfilter = (
        f"v360=input=flat:ih_fov={world_hfov:.4f}:iv_fov={pinhole_vfov(world_hfov, width, height):.4f}:"
        f"output=hequirect:h_fov=180:v_fov=180:w={equirect}:h={equirect},"
        f"v360=input=hequirect:ih_fov=180:iv_fov=180:output=fisheye:"
        f"h_fov={true_hfov:.4f}:v_fov={equidistant_vfov(true_hfov, width, height):.4f}:w={width}:h={height}"
    )
    return run_v360([frame], vfilter, width, height)[0]


def bow_rectifier(truth: float, width: int = 320, height: int = 240, slope: float = 0.35):
    """A fake :class:`V360Rectifier`: bow grows with the distance from ``truth``."""
    full_mask = np.full((height, width), 255, np.uint8)

    def rectify(frames, hfov):
        return full_mask, [bowed_frame(width, height, abs(hfov - truth) * slope) for _ in frames]

    return rectify


def flat_rectifier(width: int = 320, height: int = 240, bow: float = 6.0):
    """A fake rectifier that ignores the candidate — the score curve is flat."""
    full_mask = np.full((height, width), 255, np.uint8)
    frame = bowed_frame(width, height, bow)

    def rectify(frames, hfov):
        del hfov  # deliberately ignored: that is what makes the curve flat
        return full_mask, [frame for _ in frames]

    return rectify


def blank_rectifier(width: int = 320, height: int = 240):
    """A fake rectifier that yields featureless frames — no line evidence at all."""
    full_mask = np.full((height, width), 255, np.uint8)
    frame = np.full((height, width), 128, np.uint8)

    def rectify(frames, hfov):
        del hfov
        return full_mask, [frame for _ in frames]

    return rectify


# ===========================================================================
# Geometry
# ===========================================================================


def test_pinhole_vfov_of_a_square_frame_equals_its_hfov():
    assert pinhole_vfov(90.0, 512, 512) == pytest.approx(90.0)


def test_pinhole_vfov_is_narrower_than_the_hfov_but_wider_than_linear():
    """16:9 pinhole: 90° across is less than 90° down, but ``tan`` is convex, so
    the vertical fov is *more* than the naive aspect-scaled 50.6°."""
    vfov = pinhole_vfov(90.0, 640, 360)
    assert 90.0 * 360 / 640 < vfov < 90.0


def test_pinhole_vfov_grows_with_hfov():
    assert pinhole_vfov(60.0, 640, 360) < pinhole_vfov(120.0, 640, 360)


def test_equidistant_vfov_scales_linearly_with_aspect():
    """Equidistant is angle-proportional, so no ``tan`` bends the aspect scaling."""
    assert equidistant_vfov(100.0, 640, 360) == pytest.approx(100.0 * 360 / 640)


def test_equidistant_and_pinhole_agree_only_on_square_frames():
    assert equidistant_vfov(80.0, 400, 400) == pytest.approx(pinhole_vfov(80.0, 400, 400))
    assert equidistant_vfov(80.0, 640, 360) != pytest.approx(pinhole_vfov(80.0, 640, 360))


@pytest.mark.parametrize("fn", [pinhole_vfov, equidistant_vfov])
@pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-4, 100)])
def test_vfov_helpers_reject_degenerate_sizes(fn, width, height):
    with pytest.raises(ValueError, match="positive"):
        fn(90.0, width, height)


# ===========================================================================
# Candidate grid parsing
# ===========================================================================


def test_parse_grid_is_inclusive_of_the_stop_value():
    assert parse_grid("60:150:10") == [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


def test_parse_grid_default_matches_the_documented_range():
    grid = parse_grid(DEFAULT_GRID)
    assert grid[0] == 60.0
    assert grid[-1] == 150.0
    assert all(b - a == pytest.approx(5.0) for a, b in pairwise(grid))


def test_parse_grid_stops_short_when_the_step_does_not_divide_the_span():
    assert parse_grid("60:95:10") == [60, 70, 80, 90]


def test_parse_grid_accepts_a_single_point():
    assert parse_grid("100:100:5") == [100.0]


@pytest.mark.parametrize(
    "spec",
    [
        "60:150",  # too few fields
        "60:150:10:2",  # too many fields
        "a:b:c",  # not numbers
        "60:150:0",  # non-positive step
        "60:150:-5",
        "150:60:5",  # stop before start
        "0:150:5",  # 0° is not a fov
        "60:180:5",  # 180° is not representable as a flat/fisheye hfov here
        "60:200:5",
    ],
)
def test_parse_grid_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_grid(spec)


# ===========================================================================
# Hough gap — regression guard for the maxLineGap=4 bug
# ===========================================================================


def test_hough_gap_scales_with_min_length():
    """The gap must track the frame, not be a fixed pixel count."""
    assert hough_gap(400.0) > hough_gap(200.0) > hough_gap(100.0)


def test_hough_gap_never_drops_below_the_floor():
    assert hough_gap(0.0) == pytest.approx(HOUGH_MIN_GAP_PX)
    assert hough_gap(1.0) == pytest.approx(HOUGH_MIN_GAP_PX)


def test_hough_gap_stays_a_small_fraction_of_the_run():
    """Generous, but not so generous that a bowed line reads as one straight run."""
    assert hough_gap(300.0) < 300.0 * 0.5


def test_straight_segments_span_a_crossed_grid():
    """A lattice line is straight *through* its crossings — Hough must see it.

    This is the exact failure the fixed ``maxLineGap`` caused: every crossing
    punched a hole in the Canny edge, and with a 4 px tolerance the detector
    refused to bridge it, so a perfectly straight 360 px line was reported as
    nothing at all.
    """
    frame = grid_frame(480, 360, spacing=40)
    edges = edge_map(frame)
    lengths = straight_segments(edges, DEFAULT_MIN_LINE_FRAC * 360)
    assert lengths.size > 0
    # At least one run covering most of the frame's short side.
    assert lengths.max() > 360 * 0.8


def test_straight_segments_returns_empty_on_a_blank_frame():
    lengths = straight_segments(np.zeros((200, 200), np.uint8), 30.0)
    assert lengths.size == 0


# ===========================================================================
# Straightness score
# ===========================================================================


def test_straightness_score_rewards_straight_lines():
    """Same lattice, same edge budget — only the curvature differs."""
    straight, _, _ = straightness_score(bowed_frame(320, 240, 0.0))
    gentle, _, _ = straightness_score(bowed_frame(320, 240, 4.0))
    severe, _, _ = straightness_score(bowed_frame(320, 240, 16.0))
    assert straight < gentle < severe


def test_straightness_score_reports_segments_and_edge_pixels():
    score, segments, edge_pixels = straightness_score(grid_frame(320, 240, spacing=40))
    assert score is not None and score > 0.0
    assert segments > 0
    assert edge_pixels > 200


def test_straightness_score_is_none_without_edges():
    score, segments, edge_pixels = straightness_score(np.full((240, 320), 128, np.uint8))
    assert score is None
    assert segments == 0
    assert edge_pixels < 200


def test_straightness_score_is_fooled_by_noise_on_a_single_frame():
    """Documents *why* noise is rejected by the curve rule and not by the score.

    Hough chains random edges into long "segments", so a noise frame scores —
    and flatteringly.  A per-frame score therefore cannot be the noise guard;
    :func:`test_noise_yields_a_flat_curve_and_low_confidence` shows the guard
    that actually works.  Pinned deliberately: if a future change ever makes
    the per-frame score reject noise, this test should fail and be deleted.
    """
    score, segments, edge_pixels = straightness_score(noise_frame(320, 240))
    assert score is not None
    assert segments > 0
    assert edge_pixels > 200


def test_noise_yields_a_flat_curve_and_low_confidence():
    """The real noise guard: the round trip turns noise into more noise, so
    every candidate scores alike and the margin collapses to nothing."""
    static = noise_frame(320, 240)
    full_mask = np.full((240, 320), 255, np.uint8)
    result = calibrate(
        [np.zeros((240, 320), np.uint8)],
        parse_grid("60:150:10"),
        rectifier=lambda frames, hfov: (full_mask, [static]),
    )
    assert result.margin == pytest.approx(0.0)
    assert result.confidence == "low"
    assert "置信度低" in format_report(result)


def test_straightness_score_honours_the_coverage_mask():
    """Masked-off borders must not contribute edges."""
    frame = grid_frame(320, 240, spacing=40)
    mask = np.zeros((240, 320), np.uint8)
    mask[60:180, 80:240] = 255
    _, _, masked_px = straightness_score(frame, mask)
    _, _, full_px = straightness_score(frame, None)
    assert 0 < masked_px < full_px


def test_edge_map_clips_to_the_mask():
    frame = grid_frame(320, 240, spacing=40)
    mask = np.zeros((240, 320), np.uint8)
    mask[:120, :] = 255
    edges = edge_map(frame, mask)
    assert np.count_nonzero(edges[120:, :]) == 0
    assert np.count_nonzero(edges[:120, :]) > 0


# ===========================================================================
# calibrate() — real scoring, injected round trip (no ffmpeg)
# ===========================================================================


def test_calibrate_recovers_the_injected_truth():
    frames = [np.zeros((240, 320), np.uint8)]
    result = calibrate(frames, parse_grid("60:150:5"), rectifier=bow_rectifier(100.0))
    assert result.recommended_hfov == 100.0
    assert result.is_confident


def test_calibrate_score_curve_dips_at_the_truth():
    frames = [np.zeros((240, 320), np.uint8)]
    result = calibrate(frames, parse_grid("60:150:5"), rectifier=bow_rectifier(100.0))
    scores = {row.hfov: row.score for row in result.scores}
    assert scores[100.0] < scores[70.0]
    assert scores[100.0] < scores[130.0]


def test_calibrate_reports_two_runners_up():
    frames = [np.zeros((240, 320), np.uint8)]
    result = calibrate(frames, parse_grid("60:150:5"), rectifier=bow_rectifier(100.0))
    assert len(result.runners_up) == 2
    assert result.recommended_hfov not in result.runners_up


def test_calibrate_scores_every_candidate():
    candidates = parse_grid("60:150:10")
    result = calibrate([np.zeros((240, 320), np.uint8)], candidates, rectifier=bow_rectifier(100.0))
    assert [row.hfov for row in result.scores] == candidates


def test_calibrate_flat_curve_is_not_confident():
    """No candidate beats the others → abstain instead of picking noise."""
    result = calibrate([np.zeros((240, 320), np.uint8)], parse_grid("60:150:5"), rectifier=flat_rectifier())
    assert result.confidence == "low"
    assert result.margin < result.min_margin


def test_calibrate_without_any_line_evidence_recommends_nothing():
    result = calibrate([np.zeros((240, 320), np.uint8)], parse_grid("60:150:5"), rectifier=blank_rectifier())
    assert result.recommended_hfov is None
    assert result.confidence == "low"
    assert result.notes
    assert all(row.score is None for row in result.scores)


def test_calibrate_flags_a_recommendation_on_the_grid_edge():
    """A winner at the boundary probably means the true value is off-grid."""
    result = calibrate([np.zeros((240, 320), np.uint8)], parse_grid("100:150:10"), rectifier=bow_rectifier(100.0))
    assert result.recommended_hfov == 100.0
    assert result.confidence == "low"
    assert any("网格" in note for note in result.notes)


def test_calibrate_respects_a_custom_min_margin():
    frames = [np.zeros((240, 320), np.uint8)]
    strict = calibrate(frames, parse_grid("60:150:5"), rectifier=bow_rectifier(100.0), min_margin=0.99)
    assert strict.recommended_hfov == 100.0
    assert strict.confidence == "low"


def test_calibrate_medians_across_frames():
    """Several frames aggregate into one score per candidate, not several."""
    frames = [np.zeros((240, 320), np.uint8) for _ in range(3)]
    result = calibrate(frames, parse_grid("90:110:10"), rectifier=bow_rectifier(100.0))
    assert all(row.frames_scored == 3 for row in result.scores)


def test_calibrate_invokes_the_progress_callback_once_per_candidate():
    seen: list[tuple[float, float | None]] = []
    candidates = parse_grid("60:150:10")
    calibrate(
        [np.zeros((240, 320), np.uint8)],
        candidates,
        rectifier=bow_rectifier(100.0),
        progress=lambda hfov, score: seen.append((hfov, score)),
    )
    assert [hfov for hfov, _ in seen] == candidates


def test_calibrate_rejects_an_empty_frame_list():
    with pytest.raises(ValueError, match="no frames"):
        calibrate([], parse_grid("60:150:10"), rectifier=bow_rectifier(100.0))


def test_calibrate_rejects_an_empty_candidate_grid():
    with pytest.raises(ValueError, match="empty candidate grid"):
        calibrate([np.zeros((240, 320), np.uint8)], [], rectifier=bow_rectifier(100.0))


# ===========================================================================
# Result serialisation, report, plot
# ===========================================================================


def _demo_result() -> CalibrationResult:
    return calibrate([np.zeros((240, 320), np.uint8)], parse_grid("60:150:5"), rectifier=bow_rectifier(100.0))


def test_result_dict_carries_the_contract_keys():
    """``recommended_hfov`` / ``scores`` / ``confidence`` are the card's ask."""
    payload = _demo_result().to_dict()
    assert {"recommended_hfov", "scores", "confidence"} <= payload.keys()
    assert payload["recommended_hfov"] == 100.0
    assert payload["confidence"] in {"high", "low"}


def test_result_dict_scores_are_rows_of_hfov_and_score():
    payload = _demo_result().to_dict()
    assert len(payload["scores"]) == len(parse_grid("60:150:5"))
    for row in payload["scores"]:
        assert {"hfov", "score", "segments", "edge_pixels", "frames_scored"} <= row.keys()


def test_result_dict_is_json_serialisable():
    payload = _demo_result().to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_candidate_score_dict_keeps_none_as_none():
    row = CandidateScore(hfov=90.0, score=None, segments=0, edge_pixels=3, frames_scored=0)
    assert row.to_dict()["score"] is None


def test_format_report_lists_every_candidate_and_marks_the_winner():
    result = _demo_result()
    report = format_report(result)
    for row in result.scores:
        assert f"{row.hfov:6.1f}" in report
    assert "recommended --src-hfov : 100" in report
    assert "runners-up" in report


def test_format_report_stays_silent_about_confidence_when_confident():
    assert "置信度低" not in format_report(_demo_result())


def test_format_report_prints_the_low_confidence_notice():
    result = calibrate([np.zeros((240, 320), np.uint8)], parse_grid("60:150:5"), rectifier=flat_rectifier())
    report = format_report(result)
    assert "置信度低" in report
    assert LOW_CONFIDENCE_TEXT in report


def test_format_report_handles_a_result_with_no_recommendation():
    result = calibrate([np.zeros((240, 320), np.uint8)], parse_grid("60:150:5"), rectifier=blank_rectifier())
    report = format_report(result)
    assert "recommended --src-hfov : (none)" in report
    assert "置信度低" in report


def test_render_plot_writes_a_readable_png(tmp_path):
    out = render_plot(_demo_result(), tmp_path / "curve" / "hfov.png")
    assert out.is_file()
    assert cv2.imread(str(out)) is not None


def test_render_plot_refuses_a_curve_with_nothing_on_it():
    empty = CalibrationResult(recommended_hfov=None, confidence="low", margin=0.0, scores=[])
    with pytest.raises(ValueError, match="nothing to plot"):
        render_plot(empty, "unused.png")


# ===========================================================================
# ffmpeg command construction — list form, no shell
# ===========================================================================


def test_roundtrip_filter_chains_fisheye_in_and_flat_out():
    vfilter = build_roundtrip_filter(TRUE_HFOV, SYNTH_W, SYNTH_H)
    stage_in, stage_out = vfilter.split(",")
    assert "input=fisheye" in stage_in and "output=hequirect" in stage_in
    assert "input=hequirect" in stage_out and "output=flat" in stage_out


def test_roundtrip_filter_uses_equidistant_in_and_pinhole_out():
    """The two stages model different lenses; mixing them up is a silent error."""
    vfilter = build_roundtrip_filter(TRUE_HFOV, SYNTH_W, SYNTH_H, out_fov=DEFAULT_OUT_FOV)
    assert f"iv_fov={equidistant_vfov(TRUE_HFOV, SYNTH_W, SYNTH_H):.4f}" in vfilter
    assert f"v_fov={pinhole_vfov(DEFAULT_OUT_FOV, SYNTH_W, SYNTH_H):.4f}" in vfilter


def test_roundtrip_filter_carries_no_shell_metacharacters():
    vfilter = build_roundtrip_filter(123.456, 640, 360)
    assert not set(vfilter) & set(";|&$`<>\n")


def test_run_v360_calls_ffmpeg_as_a_list_without_a_shell(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * (64 * 32), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_v360([np.zeros((32, 64), np.uint8)], "null", 64, 32)

    assert isinstance(seen["cmd"], list)
    assert all(isinstance(part, str) for part in seen["cmd"])
    assert seen["kwargs"].get("shell", False) is False
    assert seen["cmd"][0] == "ffmpeg"
    assert "-vf" in seen["cmd"]


def test_run_v360_honours_a_custom_ffmpeg_binary(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * (64 * 32), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_v360([np.zeros((32, 64), np.uint8)], "null", 64, 32, ffmpeg="/opt/ffmpeg")
    assert seen["cmd"][0] == "/opt/ffmpeg"


def test_run_v360_short_circuits_on_no_frames(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("run_v360 should not shell out for an empty batch")

    monkeypatch.setattr(subprocess, "run", explode)
    assert run_v360([], "null", 64, 32) == []


def test_run_v360_rejects_a_ragged_batch():
    frames = [np.zeros((32, 64), np.uint8), np.zeros((16, 64), np.uint8)]
    with pytest.raises(ValueError, match="same-sized"):
        run_v360(frames, "null", 64, 32)


def test_run_v360_rejects_colour_frames():
    with pytest.raises(ValueError, match="grayscale"):
        run_v360([np.zeros((32, 64, 3), np.uint8)], "null", 64, 32)


def test_run_v360_raises_with_the_ffmpeg_error_tail(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom: bad filter\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="bad filter"):
        run_v360([np.zeros((32, 64), np.uint8)], "null", 64, 32)


def test_run_v360_raises_on_a_short_read(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * 10, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="expected"):
        run_v360([np.zeros((32, 64), np.uint8)], "null", 64, 32)


# --- module-wide AST audit (the card's subprocess-hygiene criterion) --------


def _subprocess_run_calls() -> list[ast.Call]:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]


def test_module_actually_shells_out_somewhere():
    """Guard the guard: if the calls move, the audits below must not silently pass."""
    assert len(_subprocess_run_calls()) >= 3  # run_v360, _probe_video, _grab_frame


def test_no_subprocess_call_enables_a_shell():
    for call in _subprocess_run_calls():
        for keyword in call.keywords:
            if keyword.arg == "shell":
                assert isinstance(keyword.value, ast.Constant) and keyword.value.value is False


def test_no_call_anywhere_in_the_module_passes_shell_true():
    """Wider than the ``subprocess.run`` audit: *no* call may enable a shell.

    Checked on the AST rather than the raw text on purpose — the module's own
    prose says "always list form, never shell=True", and a substring search
    would trip over its own documentation.
    """
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "shell":
                assert isinstance(keyword.value, ast.Constant), ast.dump(keyword)
                assert keyword.value.value is False


def test_every_subprocess_call_passes_a_list_not_a_string():
    """A ``str``/f-string command is the shape that invites shell quoting bugs."""
    for call in _subprocess_run_calls():
        assert call.args, "subprocess.run must be given an argv"
        argv = call.args[0]
        assert isinstance(argv, ast.List | ast.Name), ast.dump(argv)
        assert not isinstance(argv, ast.Constant | ast.JoinedStr)


# ===========================================================================
# Frame loading
# ===========================================================================


def test_load_frames_reads_a_still_image(tmp_path):
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), grid_frame(320, 240))
    frames = load_frames(path)
    assert len(frames) == 1
    assert frames[0].shape == (240, 320)
    assert frames[0].dtype == np.uint8


def test_load_frames_downscales_a_large_still(tmp_path):
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), grid_frame(1920, 1080))
    frames = load_frames(path, target_width=640)
    assert frames[0].shape == (360, 640)


def test_load_frames_leaves_a_small_still_alone(tmp_path):
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), grid_frame(160, 120))
    assert load_frames(path, target_width=640)[0].shape == (120, 160)


def test_load_frames_reports_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="source not found"):
        load_frames(tmp_path / "nope.mp4")


def test_load_frames_rejects_a_non_positive_frame_count(tmp_path):
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), grid_frame(160, 120))
    with pytest.raises(ValueError, match="frames"):
        load_frames(path, count=0)


def test_load_frames_reports_an_unreadable_image(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a png")
    with pytest.raises(RuntimeError, match="could not read image"):
        load_frames(path)


# ===========================================================================
# CLI
# ===========================================================================


def test_parser_defaults_match_the_module_constants():
    args = build_parser().parse_args(["clip.mp4"])
    assert args.frames == DEFAULT_FRAMES
    assert args.grid == DEFAULT_GRID
    assert args.out_fov == DEFAULT_OUT_FOV
    assert args.json is None
    assert args.plot is None


def test_parser_treats_bare_json_as_stdout():
    assert build_parser().parse_args(["clip.mp4", "--json"]).json == "-"


def test_parser_takes_a_json_path():
    assert build_parser().parse_args(["clip.mp4", "--json", "out.json"]).json == "out.json"


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Wire ``main`` to the fake rectifier so the CLI runs without ffmpeg."""
    import scripts.calibrate_hfov as module

    monkeypatch.setattr(module, "load_frames", lambda *a, **k: [np.zeros((240, 320), np.uint8)])

    class FakeRectifier:
        def __init__(self, *args, **kwargs):
            self._inner = bow_rectifier(100.0)

        def __call__(self, frames, hfov):
            return self._inner(frames, hfov)

    monkeypatch.setattr(module, "V360Rectifier", FakeRectifier)


def test_cli_json_to_stdout_parses_and_carries_the_contract_keys(stubbed_pipeline, capsys):
    assert main(["clip.mp4", "--grid", "60:150:5", "--json", "--quiet"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"recommended_hfov", "scores", "confidence"} <= payload.keys()
    assert payload["recommended_hfov"] == 100.0
    assert isinstance(payload["scores"], list) and payload["scores"]


def test_cli_json_stdout_stays_clean_of_the_report(stubbed_pipeline, capsys):
    """Bare ``--json`` must emit JSON and nothing else, so it can be piped."""
    main(["clip.mp4", "--grid", "60:150:5", "--json", "--quiet"])
    out = capsys.readouterr().out
    assert "recommended --src-hfov" not in out
    json.loads(out)


def test_cli_json_path_writes_a_file_and_still_prints_the_report(stubbed_pipeline, tmp_path, capsys):
    target = tmp_path / "nested" / "hfov.json"
    assert main(["clip.mp4", "--grid", "60:150:5", "--json", str(target), "--quiet"]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["recommended_hfov"] == 100.0
    assert "recommended --src-hfov : 100" in capsys.readouterr().out


def test_cli_records_the_source_and_analysis_size(stubbed_pipeline, capsys):
    main(["clip.mp4", "--grid", "60:150:5", "--json", "--quiet"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "clip.mp4"
    assert payload["analysis_size"] == [320, 240]
    assert payload["frames"] == 1


def test_cli_writes_a_plot_when_asked(stubbed_pipeline, tmp_path):
    plot = tmp_path / "curve.png"
    assert main(["clip.mp4", "--grid", "60:150:5", "--plot", str(plot), "--quiet"]) == 0
    assert plot.is_file()


def test_cli_progress_goes_to_stderr_not_stdout(stubbed_pipeline, capsys):
    main(["clip.mp4", "--grid", "60:150:5", "--json"])
    captured = capsys.readouterr()
    assert "scanning" in captured.err
    json.loads(captured.out)


def test_cli_quiet_suppresses_progress(stubbed_pipeline, capsys):
    main(["clip.mp4", "--grid", "60:150:5", "--json", "--quiet"])
    assert "scanning" not in capsys.readouterr().err


def test_cli_reports_a_missing_source_without_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mp4")]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_reports_a_bad_grid_without_a_traceback(capsys):
    assert main(["clip.mp4", "--grid", "nonsense"]) == 1
    assert "Error:" in capsys.readouterr().err


# ===========================================================================
# Acceptance — real ffmpeg, synthetic ground truth (card criteria 1-4)
# ===========================================================================


@pytest.fixture(scope="module")
def synthetic_hfov_100():
    """Pseudo footage built at a known 100°, plus the calibrator's verdict.

    Module-scoped: the round trip is ~10 ffmpeg calls, and every acceptance
    assertion below interrogates the *same* run.
    """
    warped = forward_warp(grid_frame(), TRUE_HFOV)
    result = calibrate([warped], parse_grid(ACCEPTANCE_GRID), rectifier=V360Rectifier())
    return warped, result


@_FFMPEG
def test_forward_warp_actually_bends_the_grid(synthetic_hfov_100):
    """Sanity-check the fixture: if the 'pseudo footage' were not warped, the
    whole acceptance run would be measuring nothing."""
    warped, _ = synthetic_hfov_100
    original = grid_frame()
    assert warped.shape == original.shape
    straight_score, _, _ = straightness_score(original)
    warped_score, _, _ = straightness_score(warped)
    assert warped_score > straight_score


@_FFMPEG
def test_synthetic_recommendation_lands_within_five_degrees(synthetic_hfov_100):
    """Card criterion 1: known hfov=100 → recommendation inside 100 ± 5."""
    _, result = synthetic_hfov_100
    assert result.recommended_hfov is not None
    assert abs(result.recommended_hfov - TRUE_HFOV) <= 5.0


@_FFMPEG
def test_synthetic_recommendation_is_confident(synthetic_hfov_100):
    """A clean synthetic grid is the easiest case there is; low confidence here
    would mean the margin rule is useless in the field."""
    _, result = synthetic_hfov_100
    assert result.confidence == "high"
    assert result.margin >= result.min_margin


@_FFMPEG
def test_synthetic_score_curve_dips_at_100(synthetic_hfov_100):
    """Card criterion 2: 100 scores below both 70 and 130 (relative only)."""
    _, result = synthetic_hfov_100
    scores = {row.hfov: row.score for row in result.scores}
    assert scores[70.0] is not None and scores[100.0] is not None and scores[130.0] is not None
    assert scores[100.0] < scores[70.0]
    assert scores[100.0] < scores[130.0]


@_FFMPEG
def test_synthetic_curve_is_lowest_at_the_truth(synthetic_hfov_100):
    _, result = synthetic_hfov_100
    valid = [row for row in result.scores if row.score is not None]
    assert min(valid, key=lambda row: row.score).hfov == TRUE_HFOV


@_FFMPEG
def test_noise_source_refuses_to_commit():
    """Card criterion 3: no straight lines → the report must say 置信度低."""
    result = calibrate([noise_frame()], parse_grid(ACCEPTANCE_GRID), rectifier=V360Rectifier())
    assert result.confidence == "low"
    assert "置信度低" in format_report(result)


@_FFMPEG
def test_cli_end_to_end_on_a_synthetic_still(tmp_path, capsys):
    """Card criterion 4, over the real chain: ``--json`` parses and is right."""
    still = tmp_path / "pseudo_hfov100.png"
    cv2.imwrite(str(still), forward_warp(grid_frame(), TRUE_HFOV))
    assert main([str(still), "--grid", "70:130:30", "--json", "--quiet"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert {"recommended_hfov", "scores", "confidence"} <= payload.keys()
    assert abs(payload["recommended_hfov"] - TRUE_HFOV) <= 5.0
    assert [row["hfov"] for row in payload["scores"]] == [70.0, 100.0, 130.0]


@_FFMPEG
def test_coverage_mask_excludes_the_uncovered_border():
    """The round trip leaves constant-filled corners; scoring them would mean
    scoring a curved artefact instead of the picture."""
    rectifier = V360Rectifier()
    mask, rectified = rectifier([grid_frame(320, 180)], 70.0)
    assert mask.shape == (180, 320)
    covered = np.count_nonzero(mask) / mask.size
    assert 0.0 < covered <= 1.0
    assert len(rectified) == 1
    assert rectified[0].shape == (180, 320)
