"""Tests for pipeline.spherical_injector — ISOBMFF box building and injection."""

import functools
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import pipeline.spherical_injector as si
from pipeline.spherical_injector import (
    _EQUI_BOUNDS_OFFSET,
    _EQUI_BOUNDS_VR180,
    _EQUI_BOX_SIZE,
    _FIXED_0_32_QUARTER,
    _SAMPLE_ENTRY_HEADER_SIZE,
    _STEREO_LEFT_RIGHT,
    _STEREO_MONO,
    _STEREO_TOP_BOTTOM,
    _box4,
    _build_st3d,
    _build_sv3d,
    _bump_box_size,
    _find_box_at,
    _find_box_recursive,
    _find_equi_offset,
    _find_visual_sample_entry,
    _full_box,
    _inject_via_python_isobmff,
    _rewrite_projection_bounds,
    _spatialmedia_bounds_arg,
    _spherical_structure_problems,
    _stereo_mode_byte,
    _u8,
    _u32,
    _verify_injection,
    _walk_boxes,
    inject_spherical_metadata,
    read_projection_bounds,
)


class TestSt3dBox:
    """Test st3d box construction per Google Spherical Video V2 spec."""

    def test_sbs_stereo_mode(self):
        assert _stereo_mode_byte("sbs") == _STEREO_LEFT_RIGHT

    def test_tb_stereo_mode(self):
        assert _stereo_mode_byte("tb") == _STEREO_TOP_BOTTOM

    def test_mono_stereo_mode(self):
        assert _stereo_mode_byte("mono") == _STEREO_MONO

    def test_st3d_box_structure_sbs(self):
        box = _build_st3d("sbs")
        # full_box = size(4) + type(4) + version_flags(4) + payload
        size = struct.unpack(">I", box[:4])[0]
        assert box[4:8] == b"st3d"
        # version=0, flags=0
        assert box[8:12] == b"\x00\x00\x00\x00"
        # stereo mode byte: 2 = left-right
        assert box[12] == _STEREO_LEFT_RIGHT
        assert size == 13  # 4+4+4+1

    def test_st3d_box_structure_tb(self):
        box = _build_st3d("tb")
        assert box[12] == _STEREO_TOP_BOTTOM

    def test_st3d_no_string_payload(self):
        """st3d must NOT contain 'side-by-side' or 'top-bottom' strings."""
        box = _build_st3d("sbs")
        assert b"side-by-side" not in box
        assert b"top-bottom" not in box


class TestSv3dBox:
    """sv3d must be exactly what the Spherical Video V2 RFC — and ffmpeg's
    ``mov_read_sv3d`` — expect: ``sv3d { svhd, proj { prhd, equi } }``.

    Issue #279: the old builder emitted ``sv3d { svhd, svv3d, svmi }`` (box
    names that exist in no spec) and ffmpeg answered "Missing projection box".
    """

    @staticmethod
    def _children(box: bytes) -> list[tuple[bytes, bytes]]:
        """[(type, payload after the 8-byte header), ...] for the direct children of *box*."""
        buf = bytearray(box)
        return [(t, bytes(buf[o + 8 : o + s])) for o, t, s in _walk_boxes(buf, 8, len(buf))]

    def test_sv3d_no_nested_sv3d(self):
        """sv3d box must NOT contain another sv3d inside it."""
        sv3d = _build_sv3d(7680, 1920, "sbs")
        body = sv3d[8:]  # Skip outer size+type
        assert body.count(b"sv3d") == 0, "sv3d should not be nested inside itself"

    def test_sv3d_contains_svhd(self):
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert b"svhd" in sv3d

    def test_sv3d_does_not_contain_st3d(self):
        """st3d should NOT be inside sv3d — it's a sibling per spec."""
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert b"st3d" not in sv3d

    def test_sv3d_outer_type_is_sv3d(self):
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert sv3d[4:8] == b"sv3d"

    def test_sv3d_size_consistency(self):
        sv3d = _build_sv3d(7680, 1920, "sbs")
        size = struct.unpack(">I", sv3d[:4])[0]
        assert size == len(sv3d)

    def test_sv3d_children_are_svhd_then_proj(self):
        """sv3d is a plain Box whose children are svhd then proj — the order ffmpeg parses."""
        types = [t for t, _ in self._children(_build_sv3d(7680, 1920, "sbs"))]
        assert types == [b"svhd", b"proj"]

    def test_svhd_is_fullbox_v0_with_null_terminated_source(self):
        (_svhd, payload), _proj = self._children(_build_sv3d(7680, 1920, "sbs"))
        assert payload[:4] == b"\x00\x00\x00\x00"  # version 0, flags 0
        assert payload[4:].endswith(b"\x00")
        assert payload[4:-1] == b"vr180-ai-pipeline"

    def test_proj_children_are_prhd_then_equi(self):
        """proj holds prhd followed by exactly one projection data box (equi)."""
        _svhd, (_proj, proj_payload) = self._children(_build_sv3d(7680, 1920, "sbs"))
        types = [t for t, _ in self._children(_box4(b"proj", proj_payload))]
        assert types == [b"prhd", b"equi"]

    def test_prhd_is_fullbox_v0_with_zero_pose(self):
        """prhd: version/flags + yaw/pitch/roll as int32 16.16 fixed point, all 0."""
        _svhd, (_proj, proj_payload) = self._children(_build_sv3d(7680, 1920, "sbs"))
        (_prhd, prhd_payload), _equi = self._children(_box4(b"proj", proj_payload))
        assert len(prhd_payload) == 4 + 3 * 4
        assert prhd_payload[:4] == b"\x00\x00\x00\x00"
        assert struct.unpack(">iii", prhd_payload[4:]) == (0, 0, 0)

    def test_equi_bounds_describe_180_degree_crop(self):
        """equi bounds are 0.32 fixed-point crop proportions (RFC): a 180x180
        per-eye frame crops 90/360 = 0.25 from the left and from the right and
        nothing from top/bottom."""
        _svhd, (_proj, proj_payload) = self._children(_build_sv3d(7680, 1920, "sbs"))
        _prhd, (_equi, equi_payload) = self._children(_box4(b"proj", proj_payload))
        assert len(equi_payload) == 4 + 4 * 4
        assert equi_payload[:4] == b"\x00\x00\x00\x00"
        top, bottom, left, right = struct.unpack(">IIII", equi_payload[4:])
        assert (top, bottom, left, right) == _EQUI_BOUNDS_VR180
        assert (top, bottom) == (0, 0)
        assert left == right == _FIXED_0_32_QUARTER
        assert _FIXED_0_32_QUARTER / 2**32 == 0.25
        # ffmpeg's validity rule: bottom < UINT_MAX - top and right < UINT_MAX - left
        assert bottom < 0xFFFFFFFF - top and right < 0xFFFFFFFF - left

    def test_no_legacy_box_names(self):
        """The invented pre-#279 box names must be gone for good."""
        sv3d = _build_sv3d(7680, 1920, "sbs")
        for bogus in (b"svv3d", b"svproj", b"svmi"):
            assert bogus not in sv3d, bogus

    def test_sv3d_rejects_unknown_stereo_mode(self):
        with pytest.raises(ValueError):
            _build_sv3d(7680, 1920, "diagonal")


class TestIsobmffHelpers:
    """Test low-level ISOBMFF helper functions."""

    def test_u32(self):
        assert _u32(0) == b"\x00\x00\x00\x00"
        assert _u32(1) == b"\x00\x00\x00\x01"
        assert _u32(256) == b"\x00\x00\x01\x00"

    def test_u8(self):
        assert _u8(0) == b"\x00"
        assert _u8(1) == b"\x01"
        assert _u8(255) == b"\xff"

    def test_box4(self):
        box = _box4(b"test", b"hello")
        size = struct.unpack(">I", box[:4])[0]
        assert size == 8 + 5  # header + body
        assert box[4:8] == b"test"
        assert box[8:] == b"hello"

    def test_full_box(self):
        box = _full_box(b"test", 0, 0, b"\x01")
        assert box[4:8] == b"test"
        assert box[8:12] == b"\x00\x00\x00\x00"  # version=0, flags=0
        assert box[12:] == b"\x01"


class TestBoxFinding:
    """Test ISOBMFF box search functions."""

    def _make_box(self, type_: bytes, body: bytes = b"") -> bytes:
        size = 8 + len(body)
        return struct.pack(">I", size) + type_ + body

    def test_find_box_at_simple(self):
        buf = bytearray(self._make_box(b"moov", b"hello"))
        pos = _find_box_at(buf, b"moov", 0, len(buf))
        assert pos == 0

    def test_find_box_at_not_found(self):
        buf = bytearray(self._make_box(b"moov", b"hello"))
        pos = _find_box_at(buf, b"trak", 0, len(buf))
        assert pos == -1

    def test_find_box_at_multiple(self):
        box1 = self._make_box(b"ftyp", b"mp42")
        box2 = self._make_box(b"moov", b"hello")
        buf = bytearray(box1 + box2)
        pos = _find_box_at(buf, b"moov", 0, len(buf))
        assert pos == len(box1)

    def test_find_box_recursive_in_container(self):
        inner = self._make_box(b"stsd", b"data")
        outer = self._make_box(b"moov", inner)
        buf = bytearray(outer)
        pos = _find_box_recursive(buf, b"stsd", 0, len(buf))
        assert pos == 8  # moov header size


class TestSpecLayoutScanning:
    """Regression tests for issue #46: sv3d/st3d live inside stsd sample entries.

    Per Google Spherical Video V2, sv3d/st3d hang inside the visual sample
    entry (avc1/hvc1/...) of the stsd box — not directly under stbl. The old
    scanner only descended moov/trak/mdia/minf/stbl, so real injected files
    were false-negatives. Structures here are hand-built from raw bytes, no
    ffmpeg or real media needed.
    """

    def _make_box(self, type_: bytes, body: bytes = b"") -> bytes:
        size = 8 + len(body)
        return struct.pack(">I", size) + type_ + body

    def _make_stsd(self, entries: bytes) -> bytes:
        """stsd FullBox: header(8) + version/flags(4) + entry_count(4) + entries."""
        return self._make_box(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + entries)

    def _make_visual_sample_entry(self, codec: bytes, children: bytes) -> bytes:
        """VisualSampleEntry: header(8) + 78 bytes fixed fields + child boxes."""
        fixed = b"\x00" * 78
        return self._make_box(codec, fixed + children)

    def _make_spec_tree(self, leaf: bytes) -> bytes:
        """moov > trak > mdia > minf > stbl > stsd > hvc1 > leaf."""
        stsd = self._make_stsd(self._make_visual_sample_entry(b"hvc1", leaf))
        stbl = self._make_box(b"stbl", stsd)
        minf = self._make_box(b"minf", stbl)
        mdia = self._make_box(b"mdia", minf)
        trak = self._make_box(b"trak", mdia)
        return self._make_box(b"moov", trak)

    def test_finds_st3d_and_sv3d_in_sample_entry(self):
        st3d = self._make_box(b"st3d", b"\x00\x00\x00\x00\x02")
        sv3d = self._make_box(b"sv3d", b"payload")
        buf = bytearray(self._make_spec_tree(st3d + sv3d))

        st3d_pos = _find_box_recursive(buf, b"st3d", 0, len(buf))
        sv3d_pos = _find_box_recursive(buf, b"sv3d", 0, len(buf))

        assert st3d_pos != -1
        assert sv3d_pos != -1
        assert bytes(buf[st3d_pos + 4 : st3d_pos + 8]) == b"st3d"
        assert bytes(buf[sv3d_pos + 4 : sv3d_pos + 8]) == b"sv3d"
        # sv3d immediately follows st3d in the sample entry
        st3d_size = struct.unpack(">I", buf[st3d_pos : st3d_pos + 4])[0]
        assert sv3d_pos == st3d_pos + st3d_size

    def test_offset_is_absolute(self):
        """Found offset must be absolute from buffer start, verifiable by re-parse."""
        st3d = self._make_box(b"st3d", b"\x00\x00\x00\x00\x02")
        buf = bytearray(self._make_spec_tree(st3d))
        pos = _find_box_recursive(buf, b"st3d", 0, len(buf))
        # Re-read size/type at the returned offset to prove it points at the box
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        assert size == len(st3d)
        assert bytes(buf[pos + 4 : pos + 8]) == b"st3d"

    def test_each_visual_sample_entry_codec(self):
        """All supported visual sample entry types must be descended into."""
        for codec in (b"avc1", b"avc3", b"hvc1", b"hev1", b"av01", b"vp09", b"mp4v"):
            st3d = self._make_box(b"st3d", b"\x00\x00\x00\x00\x02")
            stsd = self._make_stsd(self._make_visual_sample_entry(codec, st3d))
            buf = bytearray(self._make_box(b"stbl", stsd))
            assert _find_box_recursive(buf, b"st3d", 0, len(buf)) != -1, codec

    def test_returns_minus_one_when_absent(self):
        """Same tree shape but no spherical boxes → -1, no false positive."""
        other = self._make_box(b"avcC", b"\x01\x02\x03")
        buf = bytearray(self._make_spec_tree(other))
        assert _find_box_recursive(buf, b"st3d", 0, len(buf)) == -1
        assert _find_box_recursive(buf, b"sv3d", 0, len(buf)) == -1

    def test_truncated_sample_entry_does_not_raise(self):
        """A sample entry whose declared size is smaller than its fixed header
        must be skipped (bounds check), not crash with a bare exception."""
        # hvc1 claims size 20 — far less than 8+78 header; children unreachable
        bad_entry = struct.pack(">I", 20) + b"hvc1" + b"\x00" * 12
        stsd = self._make_stsd(bad_entry)
        buf = bytearray(self._make_box(b"stbl", stsd))
        assert _find_box_recursive(buf, b"st3d", 0, len(buf)) == -1

    def test_truncated_stsd_does_not_raise(self):
        """stsd claiming fewer bytes than its 16-byte FullBox header is skipped."""
        bad_stsd = struct.pack(">I", 12) + b"stsd" + b"\x00" * 4
        buf = bytearray(self._make_box(b"stbl", bad_stsd))
        assert _find_box_recursive(buf, b"st3d", 0, len(buf)) == -1

    def test_stbl_direct_children_still_found(self):
        """Boxes directly under stbl (old behavior) must keep working."""
        st3d = self._make_box(b"st3d", b"\x00\x00\x00\x00\x02")
        buf = bytearray(self._make_box(b"stbl", st3d))
        assert _find_box_recursive(buf, b"st3d", 0, len(buf)) == 8


# ---------------------------------------------------------------------------
# issue #91: injection self-check + ancestor-size-bump regression
# ---------------------------------------------------------------------------


def _make_tiny_mp4(path: str, w: int = 100, h: int = 100, frames: int = 3, fps: int = 24) -> None:
    """Produce a tiny real H.264 mp4 via ffmpeg (rawvideo pipe)."""
    frame = bytes([128]) * (w * h) + bytes([128]) * ((w // 2) * (h // 2)) * 2
    raw = b"".join(frame for _ in range(frames))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            path,
        ],
        input=raw,
        capture_output=True,
        check=True,
    )


class TestInjectSelfCheck:
    """issue #91: injection must self-verify via _find_box_recursive and raise
    on failure — no silent bad output."""

    def test_inject_produces_findable_boxes(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _make_tiny_mp4(str(src))
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        data = bytearray(out.read_bytes())
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1

    def test_inject_bumps_ancestor_sizes(self, tmp_path):
        """issue #91 root cause: ancestor box sizes (stsd, stbl, minf, mdia,
        trak, moov) must grow by the inserted payload so the box tree stays
        parseable. A flat top-level walk must reach every box without an
        early 'break' on a size overrun."""
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _make_tiny_mp4(str(src))
        before = bytearray(src.read_bytes())
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        after = bytearray(out.read_bytes())

        def first_box_size(buf, btype):
            # Walk top-level/known-container boxes to read the size at the
            # type field (a naive byte-find hits false matches like avc1 in
            # sample-entry names / udta tags).
            i = buf.find(btype)
            return struct.unpack(">I", buf[i - 4 : i])[0] if i >= 0 else None

        # stsd..moov must all have grown by the same delta (the sv3d+st3d
        # payload length), proving every ancestor was bumped. stsd is the
        # critical one: it is the FullBox ancestor that the naive injector
        # used to miss, leaving the avc1 sample entry pointing past stsd's
        # declared end (scanner false-negative).
        delta = first_box_size(after, b"stsd") - first_box_size(before, b"stsd")
        assert delta > 0
        for btype in (b"stsd", b"stbl", b"minf", b"mdia", b"trak", b"moov"):
            assert first_box_size(after, btype) - first_box_size(before, btype) == delta, (
                f"{btype!r} size not bumped by the insertion delta"
            )

    def test_inject_raises_on_uninjectable_input(self, tmp_path):
        """An input with no moov/visual sample entry must raise, not silently
        produce a plain-2D file."""
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00\x00\x00\x08ftypmp42")  # ftyp only, no moov
        out = tmp_path / "out.mp4"
        with pytest.raises(RuntimeError):
            inject_spherical_metadata(str(bad), str(out), stereo_mode="sbs")

    def test_inject_is_idempotent_safe_after_audio_remux(self, tmp_path):
        """issue #91: ffmpeg -c copy (audio remux) strips sv3d/st3d; a
        re-injection on the remuxed file must restore them. This locks the
        run_pipeline 'remux -> re-inject' contract."""
        src = tmp_path / "src.mp4"
        inj = tmp_path / "inj.mp4"
        aud = tmp_path / "aud.mp4"
        # 24 frames @ 24fps = 1s, no faststart (avoids the mdat-before-moov
        # layout that some ffmpeg builds mishandle with -c copy on tiny files).
        _make_tiny_mp4(str(src), w=128, h=128, frames=24)
        # silent 1s aac audio source
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "-c:a", "aac", str(aud)],
            capture_output=True,
            check=True,
        )
        inject_spherical_metadata(str(src), str(inj), stereo_mode="sbs")
        # audio remux strips the boxes (regression under test)
        from pipeline.audio_mux import copy_audio_to

        copy_audio_to(str(inj), str(aud))
        # Whether ``-c copy`` strips the (now spec-compliant, hence parseable)
        # boxes or carries them through depends on the ffmpeg version.  The
        # contract under test holds either way: after re-injection there is
        # exactly one st3d and one sv3d, and the file decodes (#279).
        inject_spherical_metadata(str(inj), str(inj) + ".vr.mp4", stereo_mode="sbs")
        os.replace(str(inj) + ".vr.mp4", str(inj))
        data = bytearray(inj.read_bytes())
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1
        children = _entry_children(data)
        assert children.count(b"st3d") == 1 and children.count(b"sv3d") == 1, children
        assert _ffmpeg_decode(inj)[0] == 0


# ---------------------------------------------------------------------------
# issue #91 acceptance: real end-to-end conversion produces findable boxes
# ---------------------------------------------------------------------------


class TestRealFileEndToEnd:
    """Slow, real-ffmpeg regression test (issue #91).

    Mocks cannot catch this class of regression — the whole point of #91 is
    that the inject *path* produced a plain-2D file while still logging
    success.  We therefore run a tiny *real* conversion end-to-end
    (ffmpeg rawvideo → libx264 → in-process ISOBMFF injection, with and
    without an audio remux) and assert the final artefact's byte stream
    contains sv3d + st3d.
    """

    @pytest.mark.slow
    def test_embed_single_frame_batch_produces_vr180_metadata(self, tmp_path):
        """The metadata stage's real code path (vr_metadata.embed_single_frame_batch)
        must leave findable sv3d/st3d in the final file."""
        import numpy as np

        from pipeline.vr_metadata import VRMetadataEmbedder

        frames = [np.full((128, 256, 3), 128, dtype=np.uint8) for _ in range(12)]
        out = tmp_path / "out.mp4"
        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        embedder.embed_single_frame_batch(frames, str(out), width=256, height=128)

        data = bytearray(out.read_bytes())
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1

    @pytest.mark.slow
    def test_audio_remux_then_reinject_keeps_both_audio_and_metadata(self, tmp_path):
        """With audio, the 'remux -> re-inject' sequence must leave a file
        that has BOTH an AAC audio stream AND sv3d/st3d (issue #91 acceptance
        item 2: 带 --copy-audio-from 时音频与元数据同时存在)."""
        import numpy as np

        from pipeline.audio_mux import has_audio_stream
        from pipeline.vr_metadata import VRMetadataEmbedder

        frames = [np.full((128, 256, 3), 128, dtype=np.uint8) for _ in range(24)]
        video = tmp_path / "video.mp4"
        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        embedder.embed_single_frame_batch(frames, str(video), width=256, height=128)

        # Build a short AAC audio source.
        aud = tmp_path / "aud.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "2", "-c:a", "aac", str(aud)],
            capture_output=True,
            check=True,
        )

        # Remux audio in (strips sv3d/st3d), then re-inject — the run_pipeline
        # contract.
        from pipeline.audio_mux import copy_audio_to

        copy_audio_to(str(video), str(aud))
        inject_spherical_metadata(str(video), str(video) + ".vr.mp4", stereo_mode="sbs")
        os.replace(str(video) + ".vr.mp4", str(video))

        data = bytearray(video.read_bytes())
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1
        assert has_audio_stream(str(video))


# ---------------------------------------------------------------------------
# issue #279: the in-process fallback writer must produce *decodable* files
#
# CI has no spatialmedia, so the fallback is the path that actually ships
# there.  Every clip below is a 1-second ``-f lavfi testsrc`` render — no
# model, no media asset — and every assertion is either a byte-level parse of
# the boxes or the card's acceptance command ``ffmpeg -v error -i f -f null -``.
# ---------------------------------------------------------------------------

_TESTSRC = "testsrc=size=256x128:rate=24"
_VARIANTS = [
    pytest.param("h264", True, id="h264-faststart"),
    pytest.param("h264", False, id="h264-mdat-first"),
    pytest.param("hevc", True, id="hevc-faststart"),
    pytest.param("hevc", False, id="hevc-mdat-first"),
]


@functools.cache
def _ffmpeg_encoders() -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout


def _encode_testsrc(path: Path, codec: str, faststart: bool, *, audio_first: bool = False) -> None:
    """Render a 1-second synthetic clip. ``faststart`` puts moov before mdat
    (the pipeline's real layout); ``audio_first`` adds an AAC track *before*
    the video track."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if audio_first:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]
    cmd += ["-f", "lavfi", "-i", _TESTSRC]
    if audio_first:
        cmd += ["-map", "0:a", "-map", "1:v", "-c:a", "aac"]
    if codec == "h264":
        cmd += ["-c:v", "libx264"]
    else:
        cmd += ["-c:v", "libx265", "-tag:v", "hvc1"]
    cmd += ["-pix_fmt", "yuv420p", "-t", "1"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(path))
    subprocess.run(cmd, capture_output=True, check=True)


def _ffmpeg_decode(path: Path) -> tuple[int, str]:
    """(exit code, stderr) of the acceptance command ``ffmpeg -v error -i <file> -f null -``."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return result.returncode, result.stderr


def _top_level_types(buf: bytes) -> list[bytes]:
    data = bytearray(buf)
    return [t for _o, t, _s in _walk_boxes(data, 0, len(data))]


def _chunk_tables(buf: bytes) -> list[tuple[bytes, list[int]]]:
    """Every stco/co64 table under moov, in file order, parsed straight from the bytes."""
    data = bytearray(buf)
    tables: list[tuple[bytes, list[int]]] = []

    def walk(start: int, end: int) -> None:
        for off, btype, size in _walk_boxes(data, start, end):
            if btype in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                walk(off + 8, off + size)
            elif btype in (b"stco", b"co64"):
                n = struct.unpack_from(">I", data, off + 12)[0]
                fmt = f">{n}I" if btype == b"stco" else f">{n}Q"
                tables.append((btype, list(struct.unpack_from(fmt, data, off + 16))))

    walk(0, len(data))
    return tables


def _entry_children(buf: bytes) -> list[bytes]:
    """Child box types of the first visual sample entry (after its 8+78 byte header)."""
    data = bytearray(buf)
    off, size, _chain = _find_visual_sample_entry(data)
    return [t for _o, t, _s in _walk_boxes(data, off + _SAMPLE_ENTRY_HEADER_SIZE, off + size)]


def _sv3d_tree(buf: bytes) -> tuple[list[bytes], list[bytes]]:
    """(sv3d child types, proj child types) parsed from the file bytes."""
    data = bytearray(buf)
    off = _find_box_recursive(data, b"sv3d", 0, len(data))
    assert off != -1, "sv3d not found"
    size = struct.unpack_from(">I", data, off)[0]
    kids = list(_walk_boxes(data, off + 8, off + size))
    proj = [(o, s) for o, t, s in kids if t == b"proj"]
    proj_kids = [t for _o, t, _s in _walk_boxes(data, proj[0][0] + 8, proj[0][0] + proj[0][1])] if proj else []
    return [t for _o, t, _s in kids], proj_kids


def _traks(buf: bytes) -> list[tuple[bytes, int]]:
    """[(handler_type, trak size), ...] in moov order."""
    data = bytearray(buf)
    out = []
    for moov_off, mtype, moov_sz in _walk_boxes(data, 0, len(data)):
        if mtype != b"moov":
            continue
        for off, btype, size in _walk_boxes(data, moov_off + 8, moov_off + moov_sz):
            if btype == b"trak":
                hdlr = _find_box_recursive(data, b"hdlr", off, off + size)
                out.append((bytes(data[hdlr + 16 : hdlr + 20]), size))
    return out


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    """Encode each variant once per module; tests copy from here."""
    d = tmp_path_factory.mktemp("i279")
    for codec, faststart in (("h264", True), ("h264", False), ("hevc", True), ("hevc", False)):
        if codec == "hevc" and "libx265" not in _ffmpeg_encoders():
            continue
        _encode_testsrc(d / f"{codec}_{'fs' if faststart else 'nofs'}.mp4", codec, faststart)
    return d


def _sample(sample_dir: Path, codec: str, faststart: bool) -> Path:
    if codec == "hevc" and "libx265" not in _ffmpeg_encoders():
        pytest.skip("this ffmpeg build has no libx265 encoder")
    return sample_dir / f"{codec}_{'fs' if faststart else 'nofs'}.mp4"


@pytest.fixture
def force_fallback(monkeypatch):
    """Simulate CI / a user box without spatialmedia: the CLI path reports failure."""
    monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", lambda *_a, **_k: False)


class TestFallbackWriterDecodes:
    """Forced fallback, H.264 and HEVC, moov-first and mdat-first: the output
    must decode, and the bytes must show *why* (offsets, sizes, structure)."""

    @pytest.mark.parametrize(("codec", "faststart"), _VARIANTS)
    def test_output_decodes_with_ffmpeg(self, sample_dir, tmp_path, force_fallback, codec, faststart):
        src = _sample(sample_dir, codec, faststart)
        tops = _top_level_types(src.read_bytes())
        assert (tops.index(b"moov") < tops.index(b"mdat")) == faststart, tops  # layout precondition
        out = tmp_path / "out.mp4"

        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")

        rc, err = _ffmpeg_decode(out)
        assert rc == 0, err
        assert "Missing projection box" not in err
        assert "Invalid NAL" not in err

    @pytest.mark.parametrize(("codec", "faststart"), _VARIANTS)
    def test_chunk_offsets_follow_moov_growth(self, sample_dir, tmp_path, force_fallback, codec, faststart):
        """Every stco/co64 entry == original + moov growth for data that moved
        (moov before mdat), and == original when moov sits after mdat."""
        src = _sample(sample_dir, codec, faststart)
        out = tmp_path / "out.mp4"
        before = src.read_bytes()
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        after = out.read_bytes()

        delta = len(after) - len(before)
        assert delta > 0
        moov_before = next(s for _o, t, s in _walk_boxes(bytearray(before), 0, len(before)) if t == b"moov")
        moov_after = next(s for _o, t, s in _walk_boxes(bytearray(after), 0, len(after)) if t == b"moov")
        assert moov_after - moov_before == delta
        entry_off, entry_sz, _chain = _find_visual_sample_entry(bytearray(before))
        moved_from = entry_off + entry_sz  # bytes at/after here shifted by delta

        tables_before, tables_after = _chunk_tables(before), _chunk_tables(after)
        assert tables_before, "sample has no chunk offset table"
        assert len(tables_before) == len(tables_after)
        for (t0, offs0), (t1, offs1) in zip(tables_before, tables_after, strict=True):
            assert t0 == t1 and len(offs0) == len(offs1)
            for o0, o1 in zip(offs0, offs1, strict=True):
                assert o1 == o0 + (delta if o0 >= moved_from else 0), (o0, o1, delta, moved_from)
        if faststart:
            assert all(
                o1 == o0 + delta
                for (_, a), (_, b) in zip(tables_before, tables_after, strict=True)
                for o0, o1 in zip(a, b, strict=True)
            )
        else:
            assert tables_after == tables_before

    @pytest.mark.parametrize(("codec", "faststart"), _VARIANTS)
    def test_box_structure_and_sizes(self, sample_dir, tmp_path, force_fallback, codec, faststart):
        """sv3d { svhd, proj { prhd, equi } } + st3d inside the sample entry, and
        the entry plus every ancestor (stsd..moov) grew by exactly delta."""
        src = _sample(sample_dir, codec, faststart)
        out = tmp_path / "out.mp4"
        before = src.read_bytes()
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        after = out.read_bytes()
        delta = len(after) - len(before)

        sv3d_kids, proj_kids = _sv3d_tree(after)
        assert sv3d_kids == [b"svhd", b"proj"]
        assert proj_kids == [b"prhd", b"equi"]
        kids = _entry_children(after)
        assert kids[0] in (b"avcC", b"hvcC"), kids  # codec config box untouched and first
        assert kids[-2:] == [b"st3d", b"sv3d"], kids  # st3d precedes sv3d
        assert kids.count(b"st3d") == 1 and kids.count(b"sv3d") == 1
        st3d_off = _find_box_recursive(bytearray(after), b"st3d", 0, len(after))
        assert after[st3d_off + 12] == _STEREO_LEFT_RIGHT

        e0, s0, chain0 = _find_visual_sample_entry(bytearray(before))
        e1, s1, chain1 = _find_visual_sample_entry(bytearray(after))
        assert (e1, s1 - s0) == (e0, delta)  # sample entry itself
        assert [t for _o, _s, t in chain0] == [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"]
        for (o0, sz0, t0), (o1, sz1, t1) in zip(chain0, chain1, strict=True):
            assert (t1, o1, sz1 - sz0) == (t0, o0, delta), t0

    def test_ffprobe_sees_stereo_and_spherical_side_data(self, sample_dir, tmp_path, force_fallback):
        """What a player will read: Stereo 3D side-by-side + an equirectangular projection."""
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "out.mp4"
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream_side_data_list", str(out)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        assert probe.returncode == 0, probe.stderr
        assert "side by side" in probe.stdout
        assert "equirectangular" in probe.stdout


# ---------------------------------------------------------------------------
# The pre-#279 writer, copied as-is, serves as the negative reference: it must
# be *rejected* by the new checks — proof that the tests above can bite.
# (_find_visual_sample_entry / _bump_box_size behave identically on these
# single-track clips, so they are imported rather than duplicated.)
# ---------------------------------------------------------------------------


def _legacy_build_sv3d(stereo_mode: str) -> bytes:
    """Pre-#279 _build_sv3d: sv3d { svhd, svv3d { svproj }, svmi }."""
    svhd = _full_box(b"svhd", 0, 0, b"vr180-ai-pipeline\x00")
    svproj = _box4(b"svproj", _u32(0))
    svv3d = _box4(b"svv3d", svproj)
    svmi = _full_box(b"svmi", 0, 0, _u8(_stereo_mode_byte(stereo_mode)))
    return _box4(b"sv3d", svhd + svv3d + svmi)


def _legacy_bump_offsets(buf: bytearray, box_off: int, box_size: int, delta: int, entry_bytes: int) -> None:
    body_end = box_off + box_size
    candidates = [(box_off + 12, box_off + 16), (box_off + 16, box_off + 20)]
    for ec_off, off_start in candidates:
        if off_start > body_end:
            continue
        region_len = body_end - off_start
        if region_len < 0 or region_len % entry_bytes != 0:
            continue
        n_entries = struct.unpack(">I", buf[ec_off : ec_off + 4])[0]
        if n_entries * entry_bytes == region_len:
            for i in range(n_entries):
                at = off_start + i * entry_bytes
                if entry_bytes == 4:
                    old = struct.unpack(">I", buf[at : at + 4])[0]
                    struct.pack_into(">I", buf, at, old + delta)
                else:
                    old = struct.unpack(">Q", buf[at : at + 8])[0]
                    struct.pack_into(">Q", buf, at, old + delta)
            return


def _legacy_bump_chunk_offsets(buf: bytearray, start: int, end: int, delta: int) -> None:
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = buf[pos + 4 : pos + 8]
        if btype == b"stco":
            _legacy_bump_offsets(buf, pos, size, delta, entry_bytes=4)
        elif btype == b"co64":
            _legacy_bump_offsets(buf, pos, size, delta, entry_bytes=8)
        pos += size


def _legacy_inject_via_python_isobmff(output_path: str, stereo_mode: str) -> None:
    """Pre-#279 writer: non-spec sv3d + unconditional stco bump of the owning trak only."""
    buf = bytearray(Path(output_path).read_bytes())
    sv3d = _legacy_build_sv3d(stereo_mode)
    st3d = _build_st3d(stereo_mode)
    payload = st3d + sv3d
    loc = _find_visual_sample_entry(buf)
    if loc is None:
        raise RuntimeError("no injectable visual sample entry (avc1/hvc1/...) found in moov tree")
    entry_off, entry_sz, chain = loc
    insert_at = entry_off + entry_sz
    buf[insert_at:insert_at] = payload
    delta = len(payload)
    struct.pack_into(">I", buf, entry_off, entry_sz + delta)
    for anc_off, _anc_sz, _anc_type in chain:
        _bump_box_size(buf, anc_off, delta)
    stbl_off = next(off for off, _sz, btype in chain if btype == b"stbl")
    stbl_sz = struct.unpack(">I", buf[stbl_off : stbl_off + 4])[0]
    _legacy_bump_chunk_offsets(buf, stbl_off + 8, stbl_off + stbl_sz, delta)
    Path(output_path).write_bytes(bytes(buf))


class TestLegacyWriterIsCaught:
    @pytest.mark.parametrize(("codec", "faststart"), _VARIANTS)
    def test_legacy_output_is_rejected(self, sample_dir, tmp_path, codec, faststart):
        out = tmp_path / "legacy.mp4"
        shutil.copy2(_sample(sample_dir, codec, faststart), out)
        _legacy_inject_via_python_isobmff(str(out), "sbs")
        data = bytearray(out.read_bytes())

        # The old presence-only self-check passed on this file — that was the bug.
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1

        rc, err = _ffmpeg_decode(out)
        assert "Missing projection box" in err, err
        if not faststart:
            # mdat never moved, yet every stco entry was bumped -> garbage NAL sizes.
            assert rc != 0
            assert "Invalid NAL" in err
        assert rc != 0 or "Missing projection box" in err

        problems = _spherical_structure_problems(data)
        assert any("proj" in p for p in problems), problems
        with pytest.raises(RuntimeError):
            _verify_injection(str(out))


class TestSpatialmediaPathUnchanged:
    """Regression: when spatialmedia succeeds, the fallback writer is never run
    and the CLI argv is exactly what it was — plus the RFC VR180 bounds (#281)."""

    def test_cli_success_skips_fallback_writer(self, sample_dir, tmp_path, monkeypatch):
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "out.mp4"
        real_writer = si._inject_via_python_isobmff

        def fake_cli(inp: str, outp: str, mode: str) -> bool:
            # Stand-in for a spatialmedia run: a valid V2 file appears at outp.
            shutil.copy2(inp, outp)
            real_writer(outp, mode)
            return True

        fallback_calls: list[tuple] = []
        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", fake_cli)
        monkeypatch.setattr(si, "_inject_via_python_isobmff", lambda *a: fallback_calls.append(a))

        assert inject_spherical_metadata(str(src), str(out), stereo_mode="sbs") == str(out)
        assert fallback_calls == []
        assert _ffmpeg_decode(out)[0] == 0

    def test_cli_failure_runs_fallback_exactly_once(self, sample_dir, tmp_path, monkeypatch):
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "out.mp4"
        real_writer = si._inject_via_python_isobmff
        calls: list[tuple] = []

        def spy(path: str, mode: str) -> None:
            calls.append((path, mode))
            real_writer(path, mode)

        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", lambda *_a: False)
        monkeypatch.setattr(si, "_inject_via_python_isobmff", spy)
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        assert calls == [(str(out), "sbs")]
        assert _ffmpeg_decode(out)[0] == 0

    def test_cli_success_without_boxes_still_fails_self_check(self, sample_dir, tmp_path, monkeypatch):
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "out.mp4"

        def lying_cli(inp: str, outp: str, _mode: str) -> bool:
            shutil.copy2(inp, outp)  # plain 2D file, "success"
            return True

        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", lying_cli)
        with pytest.raises(RuntimeError, match="missing st3d"):
            inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")

    def test_cli_argv_carries_rfc_bounds(self, monkeypatch, tmp_path):
        """#281: ``-b top:bottom:left:right`` (0.32 fixed point) asks spatial-media
        for the VR180 crop; without it the CLI writes all zeros (full 360)."""
        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "", "No module named spatialmedia")

        monkeypatch.setattr(si.subprocess, "run", fake_run)
        assert si._inject_via_spatialmedia_cli("in.mp4", "out.mp4", "sbs") is False
        assert seen == [
            [
                sys.executable,
                "-m",
                "spatialmedia",
                "-i",
                "-2",
                "-s",
                "left-right",
                "-p",
                "equirectangular",
                "-b",
                "0:0:1073741824:1073741824",
                "in.mp4",
                "out.mp4",
            ]
        ]
        bounds_arg = seen[0][seen[0].index("-b") + 1]
        assert bounds_arg == _spatialmedia_bounds_arg(_EQUI_BOUNDS_VR180)
        # spatial-media's Metadata() parses each field with int(x, 0): must round-trip to the RFC tuple.
        assert tuple(int(x, 0) for x in bounds_arg.split(":")) == _EQUI_BOUNDS_VR180 == (0, 0, 0x40000000, 0x40000000)

    def test_cli_maps_tb_and_never_spawns_for_mono(self, monkeypatch):
        """``-s`` knows none|top-bottom|left-right and ``none`` writes no st3d at
        all, so ``mono`` (st3d 0) must go straight to the in-process writer."""
        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "", "")

        monkeypatch.setattr(si.subprocess, "run", fake_run)
        assert si._inject_via_spatialmedia_cli("in.mp4", "out.mp4", "tb") is False
        assert seen[0][seen[0].index("-s") + 1] == "top-bottom"
        seen.clear()
        assert si._inject_via_spatialmedia_cli("in.mp4", "out.mp4", "mono") is False
        assert seen == []

    def test_mono_end_to_end_uses_fallback_writer(self, sample_dir, tmp_path, monkeypatch):
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "mono.mp4"
        real_writer = si._inject_via_python_isobmff
        calls: list[tuple] = []

        def spy(path: str, mode: str) -> None:
            calls.append((path, mode))
            real_writer(path, mode)

        monkeypatch.setattr(si, "_inject_via_python_isobmff", spy)
        inject_spherical_metadata(str(src), str(out), stereo_mode="mono")
        assert calls == [(str(out), "mono")]
        data = out.read_bytes()
        st3d_off = _find_box_recursive(bytearray(data), b"st3d", 0, len(data))
        assert data[st3d_off + 12] == _STEREO_MONO
        assert read_projection_bounds(out) == _EQUI_BOUNDS_VR180
        assert _ffmpeg_decode(out)[0] == 0

    def test_unknown_mode_raises_before_any_subprocess(self, sample_dir, tmp_path, monkeypatch):
        def no_subprocess(*_a, **_k):
            raise AssertionError("subprocess.run must not be reached for an unknown stereo mode")

        monkeypatch.setattr(si.subprocess, "run", no_subprocess)
        with pytest.raises(ValueError, match="diagonal"):
            inject_spherical_metadata(
                str(_sample(sample_dir, "h264", True)), str(tmp_path / "x.mp4"), stereo_mode="diagonal"
            )
        assert not (tmp_path / "x.mp4").exists()


class TestVerifyInjection:
    """_verify_injection = structural check + real ffmpeg decode; failures raise."""

    @pytest.fixture
    def injected(self, sample_dir, tmp_path) -> Path:
        out = tmp_path / "injected.mp4"
        shutil.copy2(_sample(sample_dir, "h264", True), out)
        _inject_via_python_isobmff(str(out), "sbs")
        return out

    def test_passes_on_good_file(self, injected):
        _verify_injection(str(injected))

    def test_rejects_plain_file(self, sample_dir):
        with pytest.raises(RuntimeError, match="missing st3d box; missing sv3d box"):
            _verify_injection(str(_sample(sample_dir, "h264", True)))

    def test_structure_check_accepts_spec_layout_and_rejects_legacy(self):
        fixed = b"\x00" * 78
        for sv3d, expect_ok in ((_build_sv3d(7680, 1920, "sbs"), True), (_legacy_build_sv3d("sbs"), False)):
            entry = _box4(b"hvc1", fixed + _build_st3d("sbs") + sv3d)
            stsd = _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + entry)
            tree = bytearray(_box4(b"moov", _box4(b"trak", _box4(b"mdia", _box4(b"minf", _box4(b"stbl", stsd))))))
            problems = _spherical_structure_problems(tree)
            assert (problems == []) is expect_ok, problems

    def test_rejects_nonzero_ffmpeg_exit(self, injected, monkeypatch):
        monkeypatch.setattr(
            si.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 69, "", "[hevc] Invalid NAL unit size (-49532146 > 1518)."
            ),
        )
        with pytest.raises(RuntimeError, match="exit 69"):
            _verify_injection(str(injected))

    def test_rejects_fatal_pattern_even_with_exit_zero(self, injected, monkeypatch):
        """ffmpeg logs a malformed sv3d at error level but exits 0 — still a failure."""
        monkeypatch.setattr(
            si.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", "[in#0 @ 0x1] Missing projection box\n"),
        )
        with pytest.raises(RuntimeError, match="Missing projection box"):
            _verify_injection(str(injected))

    def test_degrades_to_structure_check_without_ffmpeg(self, injected, monkeypatch, capsys):
        def no_ffmpeg(*_a, **_k):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(si.subprocess, "run", no_ffmpeg)
        _verify_injection(str(injected))  # must not raise
        assert "WARNING" in capsys.readouterr().out

    def test_decode_command_is_list_form_ffmpeg_null(self, injected, monkeypatch):
        seen: list[tuple[list[str], dict]] = []

        def fake_run(cmd, **kwargs):
            seen.append((list(cmd), kwargs))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(si.subprocess, "run", fake_run)
        _verify_injection(str(injected))
        assert len(seen) == 1
        cmd, kwargs = seen[0]
        assert cmd[0] == "ffmpeg" and cmd[-3:] == ["-f", "null", "-"]
        assert "-v" in cmd and cmd[cmd.index("-v") + 1] == "error"
        assert cmd[cmd.index("-i") + 1] == str(injected)
        assert kwargs.get("shell") is not True


class TestMultiTrackAndReinject:
    def test_audio_track_first_shifts_every_table_and_only_video_trak_grows(self, tmp_path, force_fallback):
        src = tmp_path / "audio_first.mp4"
        _encode_testsrc(src, "h264", True, audio_first=True)
        before = src.read_bytes()
        assert [h for h, _ in _traks(before)] == [b"soun", b"vide"]
        out = tmp_path / "out.mp4"

        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")

        after = out.read_bytes()
        delta = len(after) - len(before)
        tables_before, tables_after = _chunk_tables(before), _chunk_tables(after)
        assert len(tables_before) == 2  # one per trak
        for (_t0, offs0), (_t1, offs1) in zip(tables_before, tables_after, strict=True):
            assert offs1 == [o + delta for o in offs0]  # moov precedes mdat: everything moved
        (h0a, audio_before), (h0v, video_before) = _traks(before)
        (h1a, audio_after), (h1v, video_after) = _traks(after)
        assert (h1a, h1v) == (h0a, h0v)
        assert audio_after == audio_before  # audio trak untouched
        assert video_after == video_before + delta
        assert _ffmpeg_decode(out)[0] == 0

    def test_reinjection_replaces_existing_boxes(self, sample_dir, tmp_path, force_fallback):
        """Injecting an already-injected file swaps the boxes instead of
        appending a second st3d/sv3d (which ffmpeg rejects)."""
        src = _sample(sample_dir, "h264", True)
        first = tmp_path / "first.mp4"
        second = tmp_path / "second.mp4"
        inject_spherical_metadata(str(src), str(first), stereo_mode="sbs")
        inject_spherical_metadata(str(first), str(second), stereo_mode="tb")

        assert second.stat().st_size == first.stat().st_size
        data = second.read_bytes()
        kids = _entry_children(data)
        assert kids.count(b"st3d") == 1 and kids.count(b"sv3d") == 1, kids
        st3d_off = _find_box_recursive(bytearray(data), b"st3d", 0, len(data))
        assert data[st3d_off + 12] == _STEREO_TOP_BOTTOM
        assert _ffmpeg_decode(second)[0] == 0


# ---------------------------------------------------------------------------
# issue #281: both injection paths must leave the same RFC VR180 equi bounds
#
# The CLI path (spatial-media, owner's box) used to write all-zero bounds —
# "this frame covers the full 360x180 sphere" — while the fallback (CI) wrote
# the RFC's 0.25 left/right crop.  The tests below drive the CLI branch with a
# stand-in that reproduces the old CLI output and prove the delivered file is
# byte-identical to the fallback's; the last test runs the real CLI where it
# is installed.
# ---------------------------------------------------------------------------

_RFC_VR180_BOUNDS = (0, 0, 0x40000000, 0x40000000)
_FULL_SPHERE_BOUNDS = (0, 0, 0, 0)


def _equi_box_bytes(path: Path) -> bytes:
    """The whole equi box (header + version/flags + 4 bounds) as stored in *path*."""
    data = bytearray(path.read_bytes())
    off = _find_equi_offset(data)
    assert off != -1, "equi not found"
    return bytes(data[off : off + _EQUI_BOX_SIZE])


def _pre_281_cli_stand_in(inp: str, outp: str, mode: str) -> bool:
    """Stand-in for a pre-#281 spatial-media run: valid V2 boxes, all-zero equi bounds."""
    shutil.copy2(inp, outp)
    _inject_via_python_isobmff(outp, mode)
    assert _rewrite_projection_bounds(outp, _FULL_SPHERE_BOUNDS)
    return True


class TestProjectionBounds:
    @pytest.fixture
    def injected(self, sample_dir, tmp_path) -> Path:
        """A fallback-written file (RFC bounds) to poke at directly."""
        out = tmp_path / "injected.mp4"
        shutil.copy2(_sample(sample_dir, "h264", True), out)
        _inject_via_python_isobmff(str(out), "sbs")
        return out

    def test_fallback_path_writes_rfc_bounds(self, sample_dir, tmp_path, force_fallback):
        out = tmp_path / "fallback.mp4"
        inject_spherical_metadata(str(_sample(sample_dir, "h264", True)), str(out), stereo_mode="sbs")
        assert read_projection_bounds(out) == _RFC_VR180_BOUNDS == _EQUI_BOUNDS_VR180

    def test_cli_path_ends_byte_identical_to_fallback(self, sample_dir, tmp_path, monkeypatch):
        """Acceptance: the same source through both paths -> identical
        read_projection_bounds and an identical equi box, even when the CLI
        wrote the 360 default."""
        src = _sample(sample_dir, "h264", True)
        ref = tmp_path / "fallback.mp4"
        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", lambda *_a, **_k: False)
        inject_spherical_metadata(str(src), str(ref), stereo_mode="sbs")

        out = tmp_path / "cli.mp4"
        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", _pre_281_cli_stand_in)
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")

        assert read_projection_bounds(out) == read_projection_bounds(ref) == _RFC_VR180_BOUNDS
        assert _equi_box_bytes(out) == _equi_box_bytes(ref)
        assert out.stat().st_size == ref.stat().st_size
        assert out.read_bytes() == ref.read_bytes()
        rc, err = _ffmpeg_decode(out)
        assert rc == 0, err

    def test_cli_output_with_rfc_bounds_is_not_touched(self, sample_dir, tmp_path, monkeypatch):
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "cli.mp4"
        cli_bytes: list[bytes] = []

        def compliant_cli(inp: str, outp: str, mode: str) -> bool:
            shutil.copy2(inp, outp)
            _inject_via_python_isobmff(outp, mode)
            cli_bytes.append(Path(outp).read_bytes())
            return True

        rewrites: list[bool] = []
        real_rewrite = si._rewrite_projection_bounds

        def spy(path, bounds):
            result = real_rewrite(path, bounds)
            rewrites.append(result)
            return result

        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", compliant_cli)
        monkeypatch.setattr(si, "_rewrite_projection_bounds", spy)
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        assert rewrites == [False]
        assert out.read_bytes() == cli_bytes[0]

    def test_rewrite_changes_only_the_four_bounds_fields(self, injected):
        before = injected.read_bytes()
        equi_off = _find_equi_offset(bytearray(before))
        fields = set(range(equi_off + _EQUI_BOUNDS_OFFSET, equi_off + _EQUI_BOX_SIZE))

        assert _rewrite_projection_bounds(injected, _FULL_SPHERE_BOUNDS) is True
        after = injected.read_bytes()

        assert len(after) == len(before)
        changed = {i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b}
        assert changed and changed <= fields, (changed, fields)
        assert after[equi_off : equi_off + _EQUI_BOUNDS_OFFSET] == before[equi_off : equi_off + _EQUI_BOUNDS_OFFSET]
        assert read_projection_bounds(injected) == _FULL_SPHERE_BOUNDS
        # ...and back: the rewrite is its own inverse, byte for byte.
        assert _rewrite_projection_bounds(injected, _EQUI_BOUNDS_VR180) is True
        assert injected.read_bytes() == before

    def test_rewrite_is_noop_when_bounds_already_match(self, injected):
        before = injected.read_bytes()
        assert _rewrite_projection_bounds(injected, _EQUI_BOUNDS_VR180) is False
        assert injected.read_bytes() == before

    def test_rewrite_leaves_file_without_equi_untouched(self, sample_dir, tmp_path):
        plain = tmp_path / "plain.mp4"
        shutil.copy2(_sample(sample_dir, "h264", True), plain)
        before = plain.read_bytes()
        assert _rewrite_projection_bounds(plain, _EQUI_BOUNDS_VR180) is False
        assert plain.read_bytes() == before

    def test_read_raises_without_equi(self, sample_dir):
        with pytest.raises(RuntimeError, match="equi"):
            read_projection_bounds(_sample(sample_dir, "h264", True))

    def test_read_matches_the_raw_bytes(self, injected):
        data = bytearray(injected.read_bytes())
        equi_off = _find_equi_offset(data)
        assert bytes(data[equi_off + 4 : equi_off + 8]) == b"equi"
        raw = struct.unpack_from(">IIII", data, equi_off + _EQUI_BOUNDS_OFFSET)
        assert read_projection_bounds(injected) == raw == _RFC_VR180_BOUNDS

    def test_self_check_rejects_full_sphere_bounds(self, injected):
        """The 360 default the CLI used to write must not pass _verify_injection."""
        _rewrite_projection_bounds(injected, _FULL_SPHERE_BOUNDS)
        problems = _spherical_structure_problems(bytearray(injected.read_bytes()))
        assert any("bounds" in p for p in problems), problems
        with pytest.raises(RuntimeError, match="bounds"):
            _verify_injection(str(injected))

    def test_self_check_accepts_rfc_bounds(self, injected):
        assert _spherical_structure_problems(bytearray(injected.read_bytes())) == []

    @pytest.mark.skipif(
        importlib.util.find_spec("spatialmedia") is None,
        reason="spatialmedia not installed here (CI has none; the owner's venv does)",
    )
    def test_real_spatialmedia_cli_writes_rfc_bounds(self, sample_dir, tmp_path, monkeypatch):
        """Owner's box: the real CLI honours ``-b`` — neither the fallback writer
        nor the in-place rewrite is needed, and the equi box equals the fallback's."""
        src = _sample(sample_dir, "h264", True)
        out = tmp_path / "real_cli.mp4"
        real_writer = si._inject_via_python_isobmff
        real_rewrite = si._rewrite_projection_bounds
        rewrites: list[bool] = []

        def spy(path, bounds):
            result = real_rewrite(path, bounds)
            rewrites.append(result)
            return result

        monkeypatch.setattr(si, "_inject_via_python_isobmff", lambda *_a: pytest.fail("fallback writer ran"))
        monkeypatch.setattr(si, "_rewrite_projection_bounds", spy)
        inject_spherical_metadata(str(src), str(out), stereo_mode="sbs")
        assert rewrites == [False]
        assert read_projection_bounds(out) == _RFC_VR180_BOUNDS

        ref = tmp_path / "fallback.mp4"
        monkeypatch.setattr(si, "_inject_via_spatialmedia_cli", lambda *_a, **_k: False)
        monkeypatch.setattr(si, "_inject_via_python_isobmff", real_writer)
        inject_spherical_metadata(str(src), str(ref), stereo_mode="sbs")
        assert _equi_box_bytes(out) == _equi_box_bytes(ref)
        assert _ffmpeg_decode(out)[0] == 0
