"""Tests for pipeline.spherical_injector — ISOBMFF box building and injection."""

import struct

from pipeline.spherical_injector import (
    _STEREO_LEFT_RIGHT,
    _STEREO_MONO,
    _STEREO_TOP_BOTTOM,
    _box4,
    _build_st3d,
    _build_sv3d,
    _find_box_at,
    _find_box_recursive,
    _full_box,
    _stereo_mode_byte,
    _u8,
    _u32,
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
    """Test sv3d box construction per Google Spherical Video V2 spec."""

    def test_sv3d_no_nested_sv3d(self):
        """sv3d box must NOT contain another sv3d inside it."""
        sv3d = _build_sv3d(7680, 1920, "sbs")
        body = sv3d[8:]  # Skip outer size+type
        assert body.count(b"sv3d") == 0, "sv3d should not be nested inside itself"

    def test_sv3d_contains_svv3d(self):
        """sv3d must contain svv3d as inner box."""
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert b"svv3d" in sv3d

    def test_sv3d_contains_svproj(self):
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert b"svproj" in sv3d

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

    def test_sv3d_contains_svmi(self):
        sv3d = _build_sv3d(7680, 1920, "sbs")
        assert b"svmi" in sv3d


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
