"""Tests for the pre-pipeline source health check (scripts/check_source_quality.py, W-1 #305).

The tool answers one question before a 40-minute conversion starts: *is this
clip worth converting?*  Four checks — ``decodable`` / ``square`` /
``forward_motion`` / ``edges`` — and an exit code, so it can gate a run.

Where the value is
------------------
``forward_motion`` is the check the card exists for.  The VR180 route converts
*parallax*; a locked-off camera has none, so a static clip produces a flat
picture pasted on a sphere after the full GPU spend.  A generation prompt that
says "camera flies forward" is a wish — models return static shots and lateral
pans routinely — so the motion has to be **measured**.

The discriminator, and why all three synthetic cases are here
------------------------------------------------------------
Dense Farnebäck flow, projected onto the radial direction from the frame
centre.  Three fixtures, built with numpy/OpenCV only (no real video, no
network, no model), pin the three cases the operator actually hits:

===============  ==================================  ====================
fixture          what it is                          must be judged
===============  ==================================  ====================
``radial_outflow_frames``  texture zooming out of the centre   PASS
``static_frames``          one frame + noise                   FAIL, 提示含「静态」
``panning_frames``         the whole frame sliding sideways    FAIL
===============  ==================================  ====================

The pan is the fixture that earns the method.  A naive "is there motion?"
test passes a pan happily — its flow magnitude is *large*.  What separates it
is the **sign structure**: a lateral move projects to ``+|v|`` on the leading
half of the frame and ``-|v|`` on the trailing half, so the mean radial
component cancels to ~0 while ``flow_mean`` stays big.  Forward motion instead
has the flow scale with radius (``≈ (s-1)·r``), which is the second condition —
outer ring above inner disc.

Measured on these fixtures at 240×240 with 8 pairs, before any assertion was
written (this is the margin the assertions are allowed to rely on):

* radial outflow — ``radial ≈ +3.61 px``, inner ``+1.37`` → outer ``+4.78``
* static —         ``radial ≈ -0.0002 px``
* pan —            ``radial ≈ -0.003 px`` with ``flow ≈ 6 px``

Three orders of magnitude between the good case and both bad ones, so the
thresholds are nowhere near the fixtures' noise and the tests are not
knife-edge.

The verdict is taken on a **scale-free rate** (radial flow ÷ the frame's
half-diagonal ``R``) rather than on raw pixels, so the same threshold holds for
a 480 px analysis frame and a 2880 px source.  ``test_radial_rate_is_scale_free``
pins that property.

Everything here runs on synthetic arrays: no test reads ``video/``, decodes a
real file, or shells out to ffmpeg.  The ffprobe/ffmpeg layer is covered by
parsing captured ffprobe JSON and by an AST sweep
(``test_subprocess_calls_are_list_form_without_shell``) that holds every
``subprocess.run`` in the module to list form with no ``shell=True``.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import check_source_quality as csq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_source_quality.py"

FRAME_SIZE = 240
N_FRAMES = 9  # 9 frames -> 8 adjacent pairs, the CLI default


# ---------------------------------------------------------------------------
# Synthetic footage (numpy/OpenCV only — no video files, no ffmpeg)
# ---------------------------------------------------------------------------


def _texture(height: int, width: int, seed: int = 7) -> np.ndarray:
    """A blobby grayscale texture: enough structure for Farnebäck to track.

    Built by upsampling coarse noise rather than using per-pixel noise — flow
    needs a *trackable* pattern, and white noise has no correspondence between
    frames at all.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(max(2, height // 8), max(2, width // 8)), dtype=np.uint8)
    img = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0)


def radial_outflow_frames(
    n: int = N_FRAMES,
    size: int = FRAME_SIZE,
    zoom_per_frame: float = 1.04,
) -> list[np.ndarray]:
    """Positive sample: texture streaming outward from the centre.

    Each frame is the previous scene scaled up about the frame centre, which is
    exactly what a forward-flying camera does to a static world — everything
    moves away from the vanishing point, and the further out it already is, the
    faster it goes.  Rendered from a 2× oversized canvas so the growing crop
    never runs out of source pixels and never has to invent border content.
    """
    base = _texture(size * 2, size * 2)
    frames: list[np.ndarray] = []
    for i in range(n):
        matrix = cv2.getRotationMatrix2D((size, size), 0, zoom_per_frame**i)
        big = cv2.warpAffine(base, matrix, (2 * size, 2 * size), flags=cv2.INTER_LINEAR)
        frames.append(big[size // 2 : size // 2 + size, size // 2 : size // 2 + size].copy())
    return frames


def static_frames(n: int = N_FRAMES, size: int = FRAME_SIZE, sigma: float = 3.0) -> list[np.ndarray]:
    """Negative sample 1: a locked-off camera — one scene plus sensor noise.

    The noise matters: a pixel-identical repeat would give an exactly-zero flow
    field, which is an easier problem than reality.  A little noise makes
    Farnebäck return a small *random* field, which is what a real static shot
    looks like and what the threshold has to survive.
    """
    rng = np.random.default_rng(11)
    base = _texture(size, size, seed=3).astype(np.int16)
    return [np.clip(base + rng.normal(0.0, sigma, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def panning_frames(n: int = N_FRAMES, size: int = FRAME_SIZE, dx: int = 6) -> list[np.ndarray]:
    """Negative sample 2: the whole frame sliding sideways.

    The hard case.  There is plenty of motion — more raw flow than the positive
    sample has — so anything that merely asks "does this move?" passes it.  Only
    the radial projection sees that the motion is not *outward*.
    """
    base = _texture(size, size + dx * n)
    return [base[:, i * dx : i * dx + size].copy() for i in range(n)]


def _stats_for(frames: list[np.ndarray]) -> list[csq.RadialStats]:
    return [csq.radial_flow_stats(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


# ---------------------------------------------------------------------------
# forward_motion — the card's three acceptance cases
# ---------------------------------------------------------------------------


def test_radial_outflow_is_recognised_as_forward_motion() -> None:
    """Acceptance 1: a radial-outflow clip passes ``forward_motion``.

    Both of §QA's conditions are asserted explicitly, not just the verdict, so
    a future regression says *which* half broke: the radial component must be
    positive, and it must grow with radius (outer ring above inner disc) —
    the signature of ``flow ≈ (s-1)·r``.
    """
    stats = _stats_for(radial_outflow_frames())
    assert len(stats) == N_FRAMES - 1

    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_PASS, f"radial outflow must PASS, got {result.status}: {result.detail}"
    assert result.measured["radial_positive"] is True
    assert result.measured["outer_exceeds_inner"] is True
    assert result.measured["radial_rate"] > csq.DEFAULT_MIN_RADIAL_RATE
    assert result.measured["outer_mean_px"] > result.measured["inner_mean_px"]


def test_static_clip_fails_and_says_static() -> None:
    """Acceptance 2: a locked-off clip FAILs and the message contains 「静态」.

    The wording is load-bearing, not cosmetic: the operator reads one line and
    has to know the clip needs regenerating, not re-encoding.  The advice names
    静态 for every ``forward_motion`` failure by design
    (:data:`~scripts.check_source_quality.FORWARD_FAIL_ADVICE`) because the
    remedy is the same whichever sub-mode was detected.
    """
    result = csq.check_forward_motion(_stats_for(static_frames()))
    assert result.status == csq.STATUS_FAIL, f"static clip must FAIL, got {result.status}: {result.detail}"
    assert "静态" in result.detail + result.advice
    assert result.measured["radial_positive"] is False
    # The card's own description of a locked-off shot: sub-pixel and negative.
    assert abs(result.measured["radial_mean_px"]) <= 1.0


def test_panning_clip_fails_because_radial_component_is_not_positive() -> None:
    """Acceptance 3: a lateral pan FAILs — motion, but not *outward* motion.

    The second assertion is the point of the whole method: the pan's raw flow
    is comparable to (here, larger than) the positive sample's, so a
    magnitude-only test would wave it through.  It is the radial *projection*
    that collapses to ~0.
    """
    result = csq.check_forward_motion(_stats_for(panning_frames()))
    assert result.status == csq.STATUS_FAIL, f"pan must FAIL, got {result.status}: {result.detail}"
    assert result.measured["radial_positive"] is False
    assert result.measured["flow_mean_px"] > 1.0, "the pan fixture must actually contain plenty of motion"
    assert abs(result.measured["radial_rate"]) < csq.DEFAULT_MIN_RADIAL_RATE


def test_the_three_cases_are_separated_by_orders_of_magnitude() -> None:
    """The margin itself is the contract — the thresholds must not sit on a knife edge.

    Asserting only pass/fail would let a future tweak shrink the separation to
    a hair without any test noticing.  This pins the actual daylight: the good
    case clears the threshold by ≥5×, and both bad cases sit ≥5× *below* it.
    """
    good = csq.check_forward_motion(_stats_for(radial_outflow_frames())).measured["radial_rate"]
    still = csq.check_forward_motion(_stats_for(static_frames())).measured["radial_rate"]
    pan = csq.check_forward_motion(_stats_for(panning_frames())).measured["radial_rate"]
    floor = csq.DEFAULT_MIN_RADIAL_RATE

    assert good > floor * 5.0, f"forward-motion margin too thin: {good} vs threshold {floor}"
    assert abs(still) < floor / 5.0, f"static clip too close to the threshold: {still}"
    assert abs(pan) < floor / 5.0, f"pan too close to the threshold: {pan}"


def test_slow_forward_motion_passes_with_a_weak_warning() -> None:
    """Barely-moving-but-moving is a WARN, not a FAIL.

    The distinction is a real operator decision: a slow push still yields
    parallax, so blocking the run would be wrong, but the stereo will be subtle
    and they should know before they spend the GPU hour.
    """
    radius = 100.0
    rate = csq.DEFAULT_MIN_RADIAL_RATE * 1.2  # inside the weak band (< 2x)
    stats = [
        csq.RadialStats(
            radial_mean=rate * radius,
            inner_mean=0.2 * rate * radius,
            outer_mean=1.6 * rate * radius,
            flow_mean=rate * radius,
            radius=radius,
        )
    ] * 3
    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_WARN
    assert result.advice == csq.FORWARD_WEAK_ADVICE


def test_outward_flow_that_does_not_grow_with_radius_fails() -> None:
    """Positive radial mean alone is not enough — condition 2 stands on its own.

    Guards against a future "simplification" that drops the ring comparison:
    forward motion scales the flow with radius, and a field that is uniformly
    outward everywhere is not the geometry of flying into a scene.
    """
    stats = [csq.RadialStats(radial_mean=5.0, inner_mean=6.0, outer_mean=4.0, flow_mean=5.0, radius=100.0)] * 3
    result = csq.check_forward_motion(stats)
    assert result.status == csq.STATUS_FAIL
    assert result.measured["radial_positive"] is True
    assert result.measured["outer_exceeds_inner"] is False


def test_no_usable_pairs_fails_rather_than_silently_passing() -> None:
    """Zero sampled pairs must FAIL: "we measured nothing" is not "it's fine"."""
    result = csq.check_forward_motion([])
    assert result.status == csq.STATUS_FAIL
    assert result.measured["pairs"] == 0


def test_forward_motion_verdict_is_a_median_not_a_mean() -> None:
    """One wild pair (a scene cut) must not flip a clip's verdict.

    A mean over 5 pairs would be dragged negative by a single outlier; the
    median ignores it, which is why the aggregation is specified rather than
    incidental.
    """
    radius = 100.0
    good = csq.RadialStats(radial_mean=3.0, inner_mean=1.0, outer_mean=4.0, flow_mean=3.0, radius=radius)
    cut = csq.RadialStats(radial_mean=-400.0, inner_mean=-400.0, outer_mean=-400.0, flow_mean=400.0, radius=radius)
    assert csq.check_forward_motion([good, good, cut, good, good]).status == csq.STATUS_PASS


def test_radial_rate_is_scale_free() -> None:
    """The same clip at two resolutions yields the same rate.

    This is why the threshold is a fraction of the half-diagonal ``R`` rather
    than a pixel count: the analysis width is an implementation detail
    (``--flow-width``), and a resolution-dependent threshold would quietly mean
    something different on every source.
    """
    small = _stats_for(radial_outflow_frames(size=160))
    large = _stats_for(radial_outflow_frames(size=320))
    rate_small = float(np.median([s.radial_rate for s in small]))
    rate_large = float(np.median([s.radial_rate for s in large]))
    assert rate_small == pytest.approx(rate_large, rel=0.25), (
        f"radial rate must be resolution independent: {rate_small} vs {rate_large}"
    )
    # ...and the raw pixel figures genuinely do differ, so the test above is
    # not passing for the trivial reason that nothing changed.
    px_small = float(np.median([s.radial_mean for s in small]))
    px_large = float(np.median([s.radial_mean for s in large]))
    assert px_large > px_small * 1.5


def test_radial_flow_stats_rejects_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="same-sized grayscale"):
        csq.radial_flow_stats(np.zeros((10, 10), np.uint8), np.zeros((12, 10), np.uint8))


# ---------------------------------------------------------------------------
# edges — letterbox bars and blown highlights
# ---------------------------------------------------------------------------


def _letterboxed_frame(size: int = FRAME_SIZE, bar: int = 20) -> np.ndarray:
    """A frame with pure-black bars top and bottom — the classic 16:9-in-1:1."""
    frame = _texture(size, size)
    frame[:bar, :] = 0
    frame[size - bar :, :] = 0
    return frame


def test_letterbox_bars_fail_and_name_the_top_and_bottom_edges() -> None:
    """Acceptance 4: black bars top/bottom FAIL, and the report says *which* edges.

    Naming the edge is the actionable half: "there is a bar somewhere" leaves
    the operator to hunt, while 上边/下边 tells them what to crop.  The left and
    right edges must stay clean in the same run, otherwise the check is just
    flagging everything.
    """
    frames = [csq.edge_band_stats(_letterboxed_frame()) for _ in range(4)]
    result = csq.check_edges(frames)

    assert result.status == csq.STATUS_FAIL
    assert result.measured["letterboxed_edges"] == ["top", "bottom"]
    assert "上边" in result.detail and "下边" in result.detail
    assert "左边" not in result.detail and "右边" not in result.detail


def test_dark_but_textured_edges_are_not_mistaken_for_letterbox_bars() -> None:
    """A dark *scene* must pass: the variance half of the rule earns its place.

    Night footage has edge bands with a low mean.  Without the variance
    condition this check would fail every dark clip — the fastest way to make
    an operator start ignoring the tool.
    """
    rng = np.random.default_rng(5)
    dark = rng.integers(0, 30, size=(FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    result = csq.check_edges([csq.edge_band_stats(dark) for _ in range(4)])
    assert result.status == csq.STATUS_PASS
    assert result.measured["letterboxed_edges"] == []


def test_blown_out_edge_fails_and_names_that_edge() -> None:
    """A dead-white band on one side FAILs and is attributed to that side."""
    frame = _texture(FRAME_SIZE, FRAME_SIZE)
    frame[:, FRAME_SIZE - 12 :] = 255
    result = csq.check_edges([csq.edge_band_stats(frame) for _ in range(4)])
    assert result.status == csq.STATUS_FAIL
    assert result.measured["overexposed_edges"] == ["right"]
    assert "右边" in result.detail


def test_one_black_frame_does_not_brand_the_clip_as_letterboxed() -> None:
    """A single fade-to-black frame is outvoted by the median across frames.

    Fades are ordinary; a bar is not. The distinction is that a bar is present
    in *every* frame, which is exactly what a median across the sampled frames
    tests for.
    """
    clean = [csq.edge_band_stats(_texture(FRAME_SIZE, FRAME_SIZE)) for _ in range(4)]
    black = csq.edge_band_stats(np.zeros((FRAME_SIZE, FRAME_SIZE), np.uint8))
    assert csq.check_edges([*clean, black]).status == csq.STATUS_PASS


def test_edge_band_stats_reports_every_side() -> None:
    stats = csq.edge_band_stats(_texture(64, 64), band=4)
    assert set(stats) == set(csq.EDGE_LABELS)
    assert all({"mean", "var", "bright_frac"} <= set(v) for v in stats.values())


def test_no_frames_fails_rather_than_silently_passing() -> None:
    assert csq.check_edges([]).status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# square — aspect ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "status"),
    [
        (2880, 2880, csq.STATUS_PASS),
        (1024, 1024, csq.STATUS_PASS),
        (1015, 1000, csq.STATUS_PASS),  # 1.5% off — inside the ±2% tolerance
        (1030, 1000, csq.STATUS_WARN),  # 3% off — outside it
        (1280, 720, csq.STATUS_WARN),
        (1920, 1080, csq.STATUS_WARN),
    ],
)
def test_square_check_warns_but_never_fails_on_shape(width: int, height: int, status: str) -> None:
    """Non-square WARNs and never FAILs — the fisheye and 16:9 routes are legitimate.

    Turning this into a FAIL would block sources the pipeline handles fine
    (§QB), so the exit code must stay 0 for a plain 16:9 clip.
    """
    result = csq.check_square(width, height)
    assert result.status == status
    assert result.measured["width"] == width
    assert result.measured["height"] == height


def test_square_check_fails_only_on_a_nonsensical_size() -> None:
    assert csq.check_square(0, 0).status == csq.STATUS_FAIL


# ---------------------------------------------------------------------------
# decodable — ffprobe metadata (parsed, never executed)
# ---------------------------------------------------------------------------


def test_self_consistent_metadata_passes_and_reports_pix_fmt() -> None:
    """10-bit is reported, never judged (#292/#298 proved it runs)."""
    info = csq.ProbeInfo(
        width=2880, height=2880, fps=24.0, duration=10.0, nb_frames=240, pix_fmt="yuv420p10le", codec="hevc"
    )
    result = csq.check_decodable(info)
    assert result.status == csq.STATUS_PASS
    assert result.measured["pix_fmt"] == "yuv420p10le"
    assert "yuv420p10le" in result.detail


def test_frame_count_drift_warns_without_blocking_the_run() -> None:
    """Broken timestamps are worth saying out loud, but the file still decodes."""
    info = csq.ProbeInfo(width=1280, height=720, fps=24.0, duration=10.0, nb_frames=120, pix_fmt="yuv420p")
    result = csq.check_decodable(info)
    assert result.status == csq.STATUS_WARN
    assert result.measured["frame_count_drift"] == pytest.approx(0.5)


def test_missing_frame_count_still_passes() -> None:
    """Plenty of containers omit ``nb_frames``; that is not a defect."""
    info = csq.ProbeInfo(width=1280, height=720, fps=24.0, duration=10.0, nb_frames=0, pix_fmt="yuv420p")
    assert csq.check_decodable(info).status == csq.STATUS_PASS


def test_unreadable_dimensions_fail() -> None:
    assert csq.check_decodable(csq.ProbeInfo()).status == csq.STATUS_FAIL


def test_probe_source_parses_captured_ffprobe_json(monkeypatch, tmp_path: Path) -> None:
    """The ffprobe parser is exercised against real captured output, not a live call.

    CI has ffprobe, but a test that decodes a real file would need a real file;
    this pins the JSON shape (including ffprobe's ``"24/1"`` frame-rate
    fractions and string-typed ``nb_frames``) with no subprocess at all.
    """
    payload = {
        "streams": [
            {
                "width": 2880,
                "height": 2880,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
                "nb_frames": "240",
                "pix_fmt": "yuv420p10le",
                "codec_name": "hevc",
                "duration": "10.000000",
            }
        ],
        "format": {"duration": "10.000000"},
    }
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload).encode("utf-8"), b"")

    monkeypatch.setattr(csq.subprocess, "run", fake_run)
    info = csq.probe_source(tmp_path / "clip.mp4")

    assert (info.width, info.height, info.fps, info.nb_frames) == (2880, 2880, 24.0, 240)
    assert info.pix_fmt == "yuv420p10le"
    assert isinstance(captured["cmd"], list) and captured["cmd"][0] == "ffprobe"


def test_probe_source_raises_when_there_is_no_video_stream(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, b'{"streams": []}', b"")

    monkeypatch.setattr(csq.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no video stream"):
        csq.probe_source(tmp_path / "audio.m4a")


# ---------------------------------------------------------------------------
# Report / JSON / exit code
# ---------------------------------------------------------------------------


def _report_with(*checks: csq.CheckResult) -> csq.SourceReport:
    return csq.SourceReport(source="clip.mp4", checks=list(checks))


def test_json_payload_is_loadable_and_carries_name_status_measured() -> None:
    """Acceptance 5a: ``--json`` is machine-readable and complete.

    The card names the three fields a consumer needs on every check; the
    ``measured`` numbers are what let a caller disagree with the verdict
    without re-running the analysis.
    """
    report = _report_with(
        csq.check_square(2880, 2880),
        csq.check_forward_motion(_stats_for(radial_outflow_frames())),
    )
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))

    assert {c["name"] for c in payload["checks"]} == {"square", "forward_motion"}
    for check in payload["checks"]:
        assert {"name", "status", "measured"} <= set(check)
        assert isinstance(check["measured"], dict)
    assert payload["summary"]["overall"] == csq.STATUS_PASS
    assert payload["exit_code"] == 0


def test_exit_code_is_one_when_anything_fails_and_zero_for_warn_only() -> None:
    """Acceptance 5b: WARNs never gate a run; FAILs always do.

    A non-square 16:9 source is a WARN and must still exit 0 — otherwise
    wiring this into preflight would block sources the pipeline handles fine.
    """
    failing = _report_with(csq.CheckResult("forward_motion", csq.STATUS_FAIL, "static", {}))
    assert failing.exit_code == 1
    assert failing.summary["overall"] == csq.STATUS_FAIL

    warn_only = _report_with(csq.check_square(1280, 720))
    assert warn_only.exit_code == 0
    assert warn_only.summary["overall"] == csq.STATUS_WARN

    clean = _report_with(csq.check_square(1024, 1024))
    assert clean.exit_code == 0
    assert clean.summary["overall"] == csq.STATUS_PASS


def test_human_report_shows_an_icon_and_the_advice_for_every_check() -> None:
    """The human report has to be readable on its own — icon, numbers, next step."""
    report = _report_with(
        csq.check_square(1280, 720),
        csq.check_forward_motion(_stats_for(static_frames())),
    )
    text = csq.format_report(report)

    assert "❌" in text and "⚠️" in text
    assert "静态" in text
    assert csq.NON_SQUARE_ADVICE in text
    assert "不要进管线" in text


def test_skipping_a_check_marks_it_skipped_instead_of_dropping_it(monkeypatch, tmp_path: Path) -> None:
    """``--skip`` must leave a visible trace, not a silent gap.

    A skipped check that simply vanished from the report would read as "this
    clip has three checks", and nobody would notice the motion test was never
    run.  The frame sampler is monkeypatched away here, which also asserts the
    orchestrator does not decode anything when both frame-hungry checks are
    skipped.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=1024, height=1024, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no frames should be decoded when both frame checks are skipped")

    monkeypatch.setattr(csq, "iter_frame_pairs", fail_if_called)
    report = csq.run_checks(clip, skip=["forward_motion", "edges"])

    statuses = {c.name: c.status for c in report.checks}
    assert statuses["forward_motion"] == csq.STATUS_SKIP
    assert statuses["edges"] == csq.STATUS_SKIP
    assert statuses["square"] == csq.STATUS_PASS
    assert report.exit_code == 0
    assert [c.name for c in report.checks] == list(csq.CHECK_NAMES)


def test_unknown_skip_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown check name"):
        csq.run_checks("clip.mp4", skip=["forwards"])


def test_missing_file_fails_without_touching_ffprobe(monkeypatch, tmp_path: Path) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("probe_source must not run for a missing file")

    monkeypatch.setattr(csq, "probe_source", fail_if_called)
    report = csq.run_checks(tmp_path / "nope.mp4")
    assert report.exit_code == 1
    assert report.checks[0].name == "decodable"


def test_run_checks_wires_the_sampled_pairs_into_both_frame_checks(monkeypatch, tmp_path: Path) -> None:
    """End-to-end through ``run_checks`` with the decoder replaced by fixtures.

    This is the only test that exercises the orchestration — sampling, the
    native-resolution edge bands, the downscaled flow analysis and the report
    assembly — and it does so on synthetic frames, so it stays honest on a
    runner with no video files.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    info = csq.ProbeInfo(width=FRAME_SIZE, height=FRAME_SIZE, fps=24.0, duration=4.0, nb_frames=96, pix_fmt="yuv420p")
    monkeypatch.setattr(csq, "probe_source", lambda *a, **k: info)

    frames = radial_outflow_frames()
    monkeypatch.setattr(
        csq,
        "iter_frame_pairs",
        lambda *a, **k: iter([(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]),
    )
    report = csq.run_checks(clip)

    statuses = {c.name: c.status for c in report.checks}
    assert statuses == {
        "decodable": csq.STATUS_PASS,
        "square": csq.STATUS_PASS,
        "forward_motion": csq.STATUS_PASS,
        "edges": csq.STATUS_PASS,
    }
    assert report.exit_code == 0


def test_downscale_for_flow_shrinks_wide_frames_and_leaves_small_ones_alone() -> None:
    """A 2880² source is analysed at ``--flow-width``; a 320 px one is untouched."""
    big = np.zeros((2880, 2880), np.uint8)
    assert csq.downscale_for_flow(big, 480).shape[1] == 480
    small = np.zeros((320, 320), np.uint8)
    assert csq.downscale_for_flow(small, 480) is small


def test_radial_basis_is_centred_and_normalised() -> None:
    """The geometry the whole check rests on: unit vectors pointing outward.

    ``R`` is the half-diagonal, so ``r/R`` spans [0, 1] over the frame and the
    corner pixels — the fastest-moving ones under forward motion — are inside
    the outer ring rather than off the end of it.
    """
    ux, uy, r, radius = csq._radial_basis(101, 101)
    assert radius == pytest.approx(0.5 * math.hypot(101, 101))
    assert r[50, 50] == pytest.approx(0.0)
    assert ux[50, 100] == pytest.approx(1.0, abs=1e-3)
    assert uy[100, 50] == pytest.approx(1.0, abs=1e-3)
    assert float(np.hypot(ux, uy)[0, 0]) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# CLI contract + repo discipline
# ---------------------------------------------------------------------------


def test_cli_help_runs_without_pythonpath() -> None:
    """``--help`` exits 0 in a subprocess with PYTHONPATH stripped (K-15 shape).

    argparse only reaches its exit-0 path after the module has fully imported,
    so this catches an import-time crash — the failure mode a pure unit test
    that already has the module loaded cannot see.
    """
    import os

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    for name in csq.CHECK_NAMES:
        assert name in proc.stdout


def test_cli_rejects_an_unknown_skip_name() -> None:
    """``--skip`` is a closed set, so a typo is caught at parse time."""
    with pytest.raises(SystemExit):
        csq.build_parser().parse_args(["clip.mp4", "--skip", "forwards"])


def test_subprocess_calls_are_list_form_without_shell() -> None:
    """Repo discipline, checked at the AST level rather than by convention.

    Every ``subprocess.run`` in the module must take a list literal and must
    never pass ``shell=True``, and no command may be assembled from an f-string
    — the shape that turns a filename with a space (or a quote) into a shell
    injection.
    """
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert calls, "expected at least one subprocess.run in check_source_quality.py"
    for call in calls:
        for kw in call.keywords:
            assert kw.arg != "shell", "shell=True is forbidden"
        assert call.args, "subprocess.run must be given an explicit command"
        command = call.args[0]
        assert isinstance(command, ast.Name | ast.List), f"command must be a list (or a list variable), got {command}"
        assert not isinstance(command, ast.JoinedStr), "command must never be an f-string"


def test_script_never_writes_next_to_the_source() -> None:
    """The tool is read-only on the input: no ``"w"`` opens, no ffmpeg output file.

    A health check that mutates the thing it is checking would be a nasty
    surprise on an operator's only copy of a 4k render.  The one write path in
    the module is the explicit ``--json PATH`` destination.
    """
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.count("write_text") == 1, "the only write should be the explicit --json destination"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            raise AssertionError("check_source_quality must not open files for writing")
