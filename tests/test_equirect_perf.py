"""Issue #256 — persistent ffmpeg v360 worker in :mod:`pipeline.equirectangular_mapper`.

Before #256 ``_map_via_ffmpeg`` spawned one ``ffmpeg`` process per frame,
round-tripped PNGs on disk, and re-ran an ``ffmpeg -filters`` probe on every
call — equirect was 96 % of the pipeline wall clock.  These tests pin the
*structure* of the fix rather than wall-clock numbers:

* **Plumbing** tests drive the mapper against a fake ``Popen`` built on real OS
  pipes.  They run on a CPU-only, ffmpeg-less CI runner and can simulate a
  worker that dies or stalls — something a real ffmpeg cannot do on demand.
* **Geometry / pixel-equivalence** tests use the real ffmpeg + v360 filter when
  present and are skipped otherwise (same convention as
  ``test_equirectangular_mapper``).  The OpenCV geometry test always runs.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import shutil
import subprocess
import threading
from typing import ClassVar

import numpy as np
import pytest

import pipeline.equirectangular_mapper as eqm
from pipeline.equirectangular_mapper import EquirectangularMapper

# The genuine (cached) probe wrapper, captured before any fixture stubs it.
_REAL_FFMPEG_AVAILABLE = EquirectangularMapper._ffmpeg_available

# --------------------------------------------------------------------------- #
# Fixtures: a fake ffmpeg worker on real OS pipes
# --------------------------------------------------------------------------- #

_SRC_H, _SRC_W = 18, 32  # tiny source frame
_OUT_W, _OUT_H = 16, 16  # tiny per-eye output


class _FakeFfmpegProc:
    """Emulates ``ffmpeg -f rawvideo -i pipe:0 … -f rawvideo pipe:1`` on OS pipes.

    Every input frame of ``in_nbytes`` yields one output frame of ``out_nbytes``
    filled with that input frame's *first byte*, so a test can prove which input
    a given output came from — a one-frame hold or off-by-one shows up as the
    previous frame's value.

    ``die_after=k``: the worker "crashes" (rc=1, stdout EOF) when asked for its
    ``k+1``-th frame.  ``stall=True``: the worker swallows frames and never
    answers until killed.
    """

    spawned: ClassVar[list[_FakeFfmpegProc]] = []

    def __init__(
        self,
        cmd: list[str],
        in_nbytes: int,
        out_nbytes: int,
        *,
        die_after: int | None = None,
        stall: bool = False,
    ):
        self.cmd = cmd
        self.pid = 10_000 + len(self.spawned)
        self.returncode: int | None = None
        self.killed = False
        self.frames_served = 0
        self._in_nbytes = in_nbytes
        self._out_nbytes = out_nbytes
        self._die_after = die_after
        self._stall = stall
        self._release = threading.Event()
        r_in, w_in = os.pipe()
        r_out, w_out = os.pipe()
        self.stdin = os.fdopen(w_in, "wb")  # the mapper writes here
        self.stdout = os.fdopen(r_out, "rb")  # the mapper reads here
        self.stderr = io.BytesIO(b"")
        self._rin = os.fdopen(r_in, "rb")
        self._wout = os.fdopen(w_out, "wb")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.spawned.append(self)

    def _serve(self) -> None:
        try:
            while True:
                data = self._rin.read(self._in_nbytes)
                if len(data) < self._in_nbytes:
                    break  # EOF: the mapper closed our stdin
                if self._die_after is not None and self.frames_served >= self._die_after:
                    self.returncode = 1
                    break
                if self._stall:
                    self._release.wait()
                    break
                self._wout.write(bytes([data[0]]) * self._out_nbytes)
                self._wout.flush()
                self.frames_served += 1
        finally:
            with contextlib.suppress(OSError):
                self._wout.close()
            if self.returncode is None:
                self.returncode = 0

    # -- subprocess.Popen surface used by the mapper -------------------------

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
        return self.returncode if self.returncode is not None else 0

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9
        self._release.set()


def _fake_popen_factory(**proc_kwargs):
    """Build a ``Popen`` stand-in that sizes the fake worker from the real ffmpeg argv."""

    def fake_popen(cmd, stdin=None, stdout=None, stderr=None, **_ignored):
        assert cmd[0] == "ffmpeg"
        w, h = (int(x) for x in cmd[cmd.index("-s") + 1].split("x"))
        vf = cmd[cmd.index("-vf") + 1]
        mo = re.search(r":w=(\d+):h=(\d+)", vf)
        assert mo, f"cannot find output size in filter {vf!r}"
        ow, oh = int(mo.group(1)), int(mo.group(2))
        last_pix_fmt_idx = len(cmd) - 1 - cmd[::-1].index("-pix_fmt")
        ch = 4 if cmd[last_pix_fmt_idx + 1] == "rgba" else 3
        return _FakeFfmpegProc(cmd, w * h * 3, ow * oh * ch, **proc_kwargs)

    return fake_popen


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Route the mapper's ffmpeg path onto :class:`_FakeFfmpegProc`.

    Returns an ``install(**kwargs)`` callable to re-arm the factory with
    ``die_after`` / ``stall``; the list of spawned fakes is
    ``_FakeFfmpegProc.spawned``.  ``subprocess.run`` is booby-trapped so any
    per-frame ``ffmpeg -filters`` probe or one-shot ffmpeg call fails loudly.
    """
    _FakeFfmpegProc.spawned.clear()
    monkeypatch.setattr(EquirectangularMapper, "_ffmpeg_available", lambda self: True)

    def _no_run(*_a, **_k):
        raise AssertionError("subprocess.run must not be called on the per-frame path")

    monkeypatch.setattr(eqm.subprocess, "run", _no_run)

    def install(**kwargs):
        monkeypatch.setattr(eqm.subprocess, "Popen", _fake_popen_factory(**kwargs))
        return _FakeFfmpegProc.spawned

    install()
    return install


def _mapper(**kw) -> EquirectangularMapper:
    kw.setdefault("output_width", _OUT_W)
    kw.setdefault("output_height", _OUT_H)
    kw.setdefault("src_hfov", 126.0)
    return EquirectangularMapper(**kw)


def _frame(value: int, h: int = _SRC_H, w: int = _SRC_W) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Plumbing (CI-safe, no ffmpeg)
# --------------------------------------------------------------------------- #


class TestPersistentWorker:
    def test_100_stereo_frames_use_exactly_one_worker_and_no_per_frame_run(self, fake_ffmpeg):
        # The headline structural criterion: 100 calls (= 200 eye-frames) must
        # not spawn 200 processes.  ``subprocess.run`` is booby-trapped by the
        # fixture, so a per-frame ``ffmpeg -filters`` probe would also fail.
        m = _mapper()
        for i in range(100):
            left, right = _frame(i % 256), _frame((i * 7 + 3) % 256)
            sbs = m.map_stereo_pair(left, right)
            assert sbs.shape == (_OUT_H, _OUT_W * 2, 3)
            # Output i must be *this* frame's answer, not frame i-1's (no hold).
            assert int(sbs[:, :_OUT_W].min()) == int(sbs[:, :_OUT_W].max()) == i % 256
            assert int(sbs[:, _OUT_W:].min()) == int(sbs[:, _OUT_W:].max()) == (i * 7 + 3) % 256
        assert len(_FakeFfmpegProc.spawned) == 1
        assert _FakeFfmpegProc.spawned[0].frames_served == 200
        assert len(m._pipes) == 1
        m.close()

    def test_one_worker_per_source_size_and_alpha_flag(self, fake_ffmpeg):
        m = _mapper()
        m.map_single(_frame(1))
        m.map_single(_frame(2))  # same key → reuse
        m.map_single(_frame(3), with_alpha=True)  # alpha → second worker
        m.map_single(_frame(4, h=12, w=20))  # other size → third worker
        m.map_single(_frame(5, h=12, w=20))
        assert len(_FakeFfmpegProc.spawned) == 3
        assert set(m._pipes) == {(_SRC_W, _SRC_H, False), (_SRC_W, _SRC_H, True), (20, 12, False)}
        rgba = m.map_single(_frame(9), with_alpha=True)
        assert rgba.shape == (_OUT_H, _OUT_W, 4)
        m.close()

    def test_worker_argv_single_threads_the_rawvideo_encoder(self, fake_ffmpeg):
        # ``-threads 1`` on the *output* side is what stops ffmpeg's frame-
        # threaded rawvideo encoder holding one frame back.  Guard it, plus the
        # raw-pipe transport and the #255 black-composite chain.
        m = _mapper()
        m.map_single(_frame(0))
        cmd = _FakeFfmpegProc.spawned[0].cmd
        i_in = cmd.index("-i")
        assert cmd[i_in + 1] == "pipe:0"
        assert cmd[-1] == "pipe:1"
        assert cmd[cmd.index("-s") + 1] == f"{_SRC_W}x{_SRC_H}"
        assert "-threads" in cmd[i_in:] and cmd[cmd.index("-threads", i_in) + 1] == "1"
        assert cmd.count("rawvideo") == 2  # -f rawvideo in and out
        assert cmd[cmd.index("-flush_packets") + 1] == "1"
        vf = cmd[cmd.index("-vf") + 1]
        assert vf == m._v360_filter(_SRC_W, _SRC_H)
        assert "alpha_mask=1" in vf and "overlay" in vf and vf.endswith("format=rgb24")
        m.close()

    def test_dead_worker_is_restarted_once_and_output_stays_correct(self, fake_ffmpeg):
        fake_ffmpeg(die_after=3)
        m = _mapper()
        for i in range(10):
            out = m.map_single(_frame(i))
            assert int(out.min()) == int(out.max()) == i
        # Worker 1 served 3 frames and died on the 4th; worker 2 picked up —
        # but it also has die_after=3, so a restart happens every 3 frames.
        # 10 frames → ceil(10/3) = 4 workers, never disabled, no data lost.
        assert len(_FakeFfmpegProc.spawned) == 4
        assert m._pipe_disabled is False
        m.close()

    def test_two_consecutive_failures_fall_back_to_oneshot(self, fake_ffmpeg, monkeypatch, caplog):
        fake_ffmpeg(die_after=0)  # every worker dies on its very first frame
        sentinel = np.full((_OUT_H, _OUT_W, 3), 77, dtype=np.uint8)
        calls: list[tuple] = []

        def _oneshot(self, frame, with_alpha=False):
            calls.append((frame.shape, with_alpha))
            return sentinel

        monkeypatch.setattr(EquirectangularMapper, "_map_via_ffmpeg_oneshot", _oneshot)
        m = _mapper()
        with caplog.at_level(logging.WARNING, logger=eqm.__name__):
            out = m.map_single(_frame(1))
        assert out is sentinel
        assert m._pipe_disabled is True
        assert len(_FakeFfmpegProc.spawned) == 2  # first worker + one retry, then give up
        assert any("falling back to one-shot" in r.getMessage() for r in caplog.records)
        # Once disabled, later frames go straight to the one-shot path — no new workers.
        m.map_single(_frame(2))
        m.map_single(_frame(3), with_alpha=True)
        assert len(_FakeFfmpegProc.spawned) == 2
        assert calls == [((_SRC_H, _SRC_W, 3), False), ((_SRC_H, _SRC_W, 3), False), ((_SRC_H, _SRC_W, 3), True)]
        assert m._pipes == {}
        m.close()

    def test_stalled_worker_is_killed_after_timeout(self, fake_ffmpeg, monkeypatch):
        fake_ffmpeg(stall=True)
        monkeypatch.setattr(
            EquirectangularMapper,
            "_map_via_ffmpeg_oneshot",
            lambda self, frame, with_alpha=False: np.zeros((_OUT_H, _OUT_W, 3), dtype=np.uint8),
        )
        m = _mapper()
        m.pipe_timeout = 0.2
        out = m.map_single(_frame(5))
        assert out.shape == (_OUT_H, _OUT_W, 3)
        # Both the first worker and its retry stalled → both killed, then fallback.
        assert len(_FakeFfmpegProc.spawned) == 2
        assert all(p.killed for p in _FakeFfmpegProc.spawned)
        assert all(p.returncode is not None for p in _FakeFfmpegProc.spawned)
        assert m._pipe_disabled is True
        m.close()

    def test_close_reaps_workers_and_context_manager_closes(self, fake_ffmpeg):
        with _mapper() as m:
            m.map_single(_frame(1))
            m.map_single(_frame(2), with_alpha=True)
            procs = list(_FakeFfmpegProc.spawned)
            assert len(procs) == 2 and all(p.returncode is None for p in procs)
        assert m._pipes == {}
        assert all(p.returncode is not None for p in procs), "workers must have exited on close()"
        assert all(p.stdin.closed for p in procs)
        m.close()  # idempotent

    def test_del_closes_workers(self, fake_ffmpeg):
        m = _mapper()
        m.map_single(_frame(1))
        proc = _FakeFfmpegProc.spawned[0]
        del m
        import gc

        gc.collect()
        assert proc.returncode is not None

    def test_ffmpeg_probe_runs_once_per_mapper(self, monkeypatch, fake_ffmpeg):
        probes: list[int] = []
        monkeypatch.setattr(EquirectangularMapper, "_probe_ffmpeg_v360", staticmethod(lambda: probes.append(1) or True))
        # Put the real (caching) ``_ffmpeg_available`` back in place of the fixture's stub.
        monkeypatch.setattr(EquirectangularMapper, "_ffmpeg_available", _REAL_FFMPEG_AVAILABLE)
        m = _mapper()
        for i in range(50):
            m.map_single(_frame(i))
        assert probes == [1], "the ffmpeg -filters probe must be cached, not re-run per frame"
        m.close()

    def test_rejects_frames_the_raw_protocol_cannot_carry(self, fake_ffmpeg):
        m = _mapper()
        with pytest.raises(ValueError):
            m.map_single(np.zeros((_SRC_H, _SRC_W, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            m.map_single(np.zeros((_SRC_H, _SRC_W, 4), dtype=np.uint8))
        with pytest.raises(ValueError):
            m.map_single(np.zeros((_SRC_H, _SRC_W), dtype=np.uint8))
        assert _FakeFfmpegProc.spawned == []
        m.close()


# --------------------------------------------------------------------------- #
# Geometry — OpenCV path always runs; ffmpeg path when v360 is present
# --------------------------------------------------------------------------- #

_H, _W = 36, 64
_CREAM = (237, 218, 193)
_FRAME = np.full((_H, _W, 3), _CREAM, dtype=np.uint8)
_FRAME[12:24, 22:42] = (255, 0, 0)  # red patch dead-centre


def _black_fraction(rgb: np.ndarray) -> float:
    return float((rgb[:, :, :3].sum(axis=2) == 0).mean())


def _assert_geometry(m: EquirectangularMapper) -> None:
    rgb = m.map_single(_FRAME)
    assert rgb.shape == (_H, _W, 3) and rgb.dtype == np.uint8
    # Centre of the dome samples the centre of the source (the red patch).
    assert tuple(int(x) for x in rgb[_H // 2, _W // 2]) == (255, 0, 0)
    # A 90° source in a 180° dome leaves the corners uncovered → pure black.
    for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        assert tuple(int(x) for x in rgb[r, c]) == (0, 0, 0), f"corner ({r},{c}) = {rgb[r, c]}"
    assert _black_fraction(rgb) > 0.2
    rgba = m.map_single(_FRAME, with_alpha=True)
    assert rgba.shape == (_H, _W, 4)
    assert rgba[_H // 2, _W // 2, 3] == 255 and rgba[0, 0, 3] == 0
    assert np.array_equal(rgb, rgba[:, :, :3])


def test_opencv_geometry_center_from_source_center_and_black_outside_fov():
    _assert_geometry(EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0, use_ffmpeg=False))


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg v360 unavailable")


@_FFMPEG
class TestRealFfmpegWorker:
    def test_geometry_center_from_source_center_and_black_outside_fov(self):
        with EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0) as m:
            _assert_geometry(m)
            assert len(m._pipes) == 2 and m._pipe_disabled is False  # rgb + rgba workers, no fallback

    @pytest.mark.parametrize("src_hfov", [90.0, 126.0])
    @pytest.mark.parametrize("with_alpha", [False, True])
    def test_pixels_identical_to_oneshot_reference(self, src_hfov, with_alpha):
        # Acceptance criterion: same frame through the old transport (one ffmpeg
        # + PNG round trip per frame) and the new one (persistent raw pipe) →
        # mean absolute error < 2 on the uint8 scale.  Same filter chain, so the
        # expectation is actually bit-identical; assert that, and report MAE.
        rng = np.random.default_rng(256)
        noise = rng.integers(0, 256, (54, 96, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:54, 0:96]
        gradient = np.dstack([xx * 255 // 95, yy * 255 // 53, (xx + yy) * 255 // 148]).astype(np.uint8)
        with EquirectangularMapper(output_width=64, output_height=64, src_hfov=src_hfov) as m:
            for frame in (noise, gradient, _FRAME):
                ref = m._map_via_ffmpeg_oneshot(frame, with_alpha=with_alpha)
                new = m.map_single(frame, with_alpha=with_alpha)
                assert new.shape == ref.shape and new.dtype == ref.dtype
                mae = float(np.abs(new.astype(np.int16) - ref.astype(np.int16)).mean())
                assert mae < 2.0, f"MAE {mae} exceeds the #256 bound"
                assert np.array_equal(new, ref), f"pipe vs one-shot differ (MAE={mae})"
            assert m._pipe_disabled is False

    def test_100_calls_spawn_one_process_and_close_reaps_it(self, monkeypatch):
        real_popen = subprocess.Popen
        spawned: list[list[str]] = []

        def counting_popen(cmd, *a, **k):
            if "pipe:1" in cmd:  # count workers only, not the one-off ``ffmpeg -filters`` probe
                spawned.append(list(cmd))
            return real_popen(cmd, *a, **k)

        monkeypatch.setattr(eqm.subprocess, "Popen", counting_popen)
        m = EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0)
        first = m.map_stereo_pair(_FRAME, _FRAME)
        pid = m._pipes[(_W, _H, False)].proc.pid
        for _ in range(99):
            sbs = m.map_stereo_pair(_FRAME, _FRAME)
            assert np.array_equal(sbs, first)
        assert len(spawned) == 1, f"expected one persistent worker, got {len(spawned)} spawns"
        assert m._pipes[(_W, _H, False)].proc.pid == pid
        proc = m._pipes[(_W, _H, False)].proc
        m.close()
        assert proc.poll() is not None, "ffmpeg worker must be reaped on close()"
        assert m._pipes == {}
