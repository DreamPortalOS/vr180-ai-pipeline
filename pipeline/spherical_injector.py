"""Inject Spherical Video V2 (sv3d) metadata into MP4 files.

Uses Google's spatial-media CLI for reliable ISOBMFF metadata injection.
Falls back to ffmpeg -movflags remux if spatialmedia is unavailable.

References:
- Google spatial-media: https://github.com/google/spatial-media
- Google Spherical Video V2 spec
"""

import contextlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# ─── ISOBMFF constants ────────────────────────────────────────────────────────

_STEREO_MONO = 0
_STEREO_TOP_BOTTOM = 1
_STEREO_LEFT_RIGHT = 2


def _u32(val: int) -> bytes:
    """Pack an unsigned 32-bit big-endian integer."""
    return struct.pack(">I", val)


def _u8(val: int) -> bytes:
    """Pack an unsigned 8-bit integer."""
    return struct.pack("B", val & 0xFF)


def _box4(box_type: bytes, payload: bytes) -> bytes:
    """Build a basic ISOBMFF box: size(4) + type(4) + payload."""
    size = 8 + len(payload)
    return _u32(size) + box_type + payload


def _full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    """Build a full ISOBMFF box: size(4) + type(4) + version_flags(4) + payload."""
    size = 12 + len(payload)
    version_flags = struct.pack(">I", (version << 24) | (flags & 0x00FFFFFF))
    return _u32(size) + box_type + version_flags + payload


def _stereo_mode_byte(mode: str) -> int:
    """Convert stereo mode string to st3d stereo mode byte."""
    mapping = {"mono": _STEREO_MONO, "tb": _STEREO_TOP_BOTTOM, "sbs": _STEREO_LEFT_RIGHT}
    if mode not in mapping:
        raise ValueError(f"Unknown stereo mode: {mode!r} (expected mono/tb/sbs)")
    return mapping[mode]


def _build_st3d(stereo_mode: str) -> bytes:
    """Build Google Spherical Video V2 st3d ISOBMFF box.

    st3d = full_box(version=0, flags=0) + stereo_mode(1)
    """
    return _full_box(b"st3d", 0, 0, _u8(_stereo_mode_byte(stereo_mode)))


def _build_svhd() -> bytes:
    """Build svhd box: metadata source string."""
    return _full_box(b"svhd", 0, 0, b"vr180-ai-pipeline\x00")


def _build_proj_yaw_pitch_roll() -> bytes:
    """Build svhd projection header."""
    return _full_box(b"svhd", 0, 0, b"vr180-ai-pipeline\x00")


def _build_svproj(width: int, height: int) -> bytes:
    """Build svproj box containing equirectangular projection data."""
    # svproj (equirectangular): 4-byte projection type (0 = equirectangular)
    proj_header = _u32(0)  # equirectangular
    return _box4(b"svproj", proj_header)


def _build_svv3d(width: int, height: int) -> bytes:
    """Build svv3d box with stereo video viewport info."""
    # Minimal svv3d: contains proj box
    proj = _build_svproj(width, height)
    return _box4(b"svv3d", proj)


def _build_svmi(stereo_mode: str) -> bytes:
    """Build svmi box (stereo video metadata indicator)."""
    return _full_box(b"svmi", 0, 0, _u8(_stereo_mode_byte(stereo_mode)))


def _build_sv3d(width: int, height: int, stereo_mode: str) -> bytes:
    """Build complete Google Spherical Video V2 sv3d ISOBMFF box.

    sv3d contains: svhd, svv3d (which contains proj), svmi.
    st3d is a sibling, NOT nested inside sv3d.
    """
    svhd = _build_svhd()
    svv3d = _build_svv3d(width, height)
    svmi = _build_svmi(stereo_mode)
    return _box4(b"sv3d", svhd + svv3d + svmi)


def _find_box_at(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Find an ISOBMFF box by type in a buffer range.

    Returns the byte offset of the box, or -1 if not found.
    """
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8:
            break
        if buf[pos + 4 : pos + 8] == box_type:
            return pos
        pos += size
    return -1


def _find_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Recursively search for an ISOBMFF box inside containers.

    Searches top-level boxes, and recurses into containers (moov, trak, mdia, minf, stbl).
    Returns byte offset of the found box, or -1.
    """
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = bytes(buf[pos + 4 : pos + 8])
        if btype == box_type:
            return pos
        if btype in containers:
            inner_start = pos + 8
            inner_end = pos + size
            found = _find_box_recursive(buf, box_type, inner_start, inner_end)
            if found != -1:
                return found
        pos += size
    return -1


def inject_spherical_metadata(
    input_path: str,
    output_path: str,
    width: int = 7680,
    height: int = 1920,
    stereo_mode: str = "sbs",
) -> str:
    """Inject Google Spherical Video V2 metadata into an MP4 file.

    Uses Google's spatial-media CLI for correct ISOBMFF sv3d/st3d box injection.
    Falls back to ffmpeg metadata remux if spatialmedia is unavailable.

    Args:
        input_path: Path to input MP4
        output_path: Path to output MP4 with sv3d atom injected
        width: Full panorama width in pixels
        height: Full panorama height in pixels
        stereo_mode: "sbs" (side-by-side) or "tb" (top-bottom)

    Returns:
        Path to output file
    """
    success = _inject_via_spatialmedia_cli(input_path, output_path, stereo_mode)
    if not success:
        # Fallback: copy file and inject via udta XML
        print("[Metadata] spatial-media not available, using ffmpeg fallback")
        shutil.copy2(input_path, output_path)
        _inject_via_ffmpeg_udta(output_path, stereo_mode)

    return output_path


def _inject_via_spatialmedia_cli(
    input_path: str,
    output_path: str,
    stereo_mode: str,
) -> bool:
    """Inject metadata using Google's spatial-media CLI tool.

    Uses V2 spec (-2 flag) which injects sv3d + st3d ISOBMFF boxes.
    """
    try:
        sm_stereo = "left-right" if stereo_mode == "sbs" else "top-bottom"
        cmd = [
            sys.executable,
            "-m",
            "spatialmedia",
            "-i",
            "-2",
            "-s",
            sm_stereo,
            "-p",
            "equirectangular",
            input_path,
            output_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[Metadata] ✅ VR180 sv3d+st3d injected via spatial-media")
            return True
        else:
            print(f"[Metadata] spatialmedia error: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("[Metadata] python3/spatialmedia not found")
        return False
    except subprocess.TimeoutExpired:
        print("[Metadata] spatialmedia timed out")
        return False
    except Exception as e:
        print(f"[Metadata] spatialmedia error: {e}")
        return False


def _inject_via_ffmpeg_udta(output_path: str, stereo_mode: str):
    """Fallback: inject Spherical Video V1 XML metadata via ffmpeg remux."""
    stereo_tag = "left-right" if stereo_mode == "sbs" else "top-bottom"
    xml = f"""<?xml version="1.0"?>
<rdf:SphericalVideo
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
<GSpherical:Spherical>true</GSpherical:Spherical>
<GSpherical:Stitched>true</GSpherical:Stitched>
<GSpherical:StitchingSoftware>vr180-ai-pipeline</GSpherical:StitchingSoftware>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:StereoMode>{stereo_tag}</GSpherical:StereoMode>
</rdf:SphericalVideo>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        xml_path = f.name

    try:
        tmp = output_path + ".remux.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            output_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-metadata:s:v",
            f"spherical-video={xml}",
            tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            shutil.move(tmp, output_path)
            print("[Metadata] Injected via ffmpeg metadata remux (V1 XML)")
        else:
            print(f"[Metadata] ffmpeg remux failed: {result.stderr[:200]}")
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(xml_path)
