"""Tests for pipeline.spherical_injector — ISOBMFF box building and injection."""

import os
import struct
import subprocess

import pytest

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
    inject_spherical_metadata,
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
        data_after_remux = bytearray(inj.read_bytes())
        assert _find_box_recursive(data_after_remux, b"sv3d", 0, len(data_after_remux)) == -1, (
            "precondition: remux must strip boxes for this test to be meaningful"
        )
        # re-inject restores them
        inject_spherical_metadata(str(inj), str(inj) + ".vr.mp4", stereo_mode="sbs")
        os.replace(str(inj) + ".vr.mp4", str(inj))
        data = bytearray(inj.read_bytes())
        assert _find_box_recursive(data, b"sv3d", 0, len(data)) != -1
        assert _find_box_recursive(data, b"st3d", 0, len(data)) != -1


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
