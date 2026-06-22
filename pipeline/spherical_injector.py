"""Inject Spherical Video V2 (sv3d) metadata into MP4 files.

Direct binary manipulation of ISOBMFF boxes. Places the sv3d box
inside the avc1/hvc1 video sample entry of the stsd atom.

References:
- Google Spherical Video V2 spec: https://github.com/google/spatial-media/blob/master/docs/spherical-video-rfc.md
- ISOBMFF (ISO 14496-12) box structure
"""
import os
import struct
from typing import Union, List, Tuple, Optional


def inject_spherical_metadata(
    input_path: str,
    output_path: str,
    width: int = 7680,
    height: int = 1920,
    stereo_mode: str = "sbs",
) -> str:
    """Inject Google Spherical Video V2 metadata into an MP4 file.

    Args:
        input_path: Path to input MP4 (may already exist)
        output_path: Path to output MP4 with sv3d atom injected
        width: Full panorama width in pixels
        height: Full panorama height in pixels
        stereo_mode: "sbs" (side-by-side) or "tb" (top-bottom)

    Returns:
        Path to output file
    """
    import shutil

    shutil.copy2(input_path, output_path)
    if not _inject_sv3d_in_file(output_path, width, height, stereo_mode):
        print("[Metadata] Warning: sv3d injection failed, file copied as-is")
    return output_path


# ---------------------------------------------------------------------------
# ISOBMFF box helpers
# ---------------------------------------------------------------------------

def _u32(n: int) -> bytes:
    return struct.pack(">I", n)


def _u8(n: int) -> bytes:
    return struct.pack(">B", n)


def _box4(type_: bytes, *parts: Union[bytes, int]) -> bytes:
    """Build an ISOBMFF box: type_ + concatenated parts, prefixed by size."""
    body = b"".join(
        _u32(p) if isinstance(p, int) else p for p in parts
    )
    return _u32(8 + len(body)) + type_ + body


def _full_box(type_: bytes, version: int, flags: int, *parts: Union[bytes, int]) -> bytes:
    """Build a full box (version + flags) with body parts."""
    version_flags = struct.pack(">I", (version << 24) | (flags & 0x00FFFFFF))
    body = b"".join(
        _u32(p) if isinstance(p, int) else p for p in parts
    )
    return _u32(8 + 4 + len(body)) + type_ + version_flags + body


# ---------------------------------------------------------------------------
# st3d box builder (Google Spherical Video V2 spec)
# ---------------------------------------------------------------------------

# Stereo mode constants per Google spec
_STEREO_MONO = 0
_STEREO_TOP_BOTTOM = 1
_STEREO_LEFT_RIGHT = 2  # aka side-by-side


def _stereo_mode_byte(stereo_mode: str) -> int:
    """Convert stereo mode string to spec-compliant byte value."""
    if stereo_mode == "sbs":
        return _STEREO_LEFT_RIGHT
    elif stereo_mode == "tb":
        return _STEREO_TOP_BOTTOM
    else:
        return _STEREO_MONO


def _build_st3d(stereo_mode: str) -> bytes:
    """Build the st3d box per Google Spherical Video V2 spec.

    st3d contains a single uint8 stereo mode:
        0 = mono, 1 = top-bottom, 2 = left-right (side-by-side)
    """
    return _full_box(b"st3d", 0, 0, _u8(_stereo_mode_byte(stereo_mode)))


# ---------------------------------------------------------------------------
# sv3d box builder (Google Spherical Video V2 spec)
# ---------------------------------------------------------------------------

def _build_sv3d(width: int, height: int, stereo_mode: str) -> bytes:
    """Build the complete sv3d box for VR180 SBS.

    Structure:
        sv3d
        ├── svhd (spherical video header)
        ├── svv3d (spherical video view box)
        │   └── svmi (stereo video mode info)
        └── svproj (spherical video projection)
            ├── svpj (projection type)
            ├── svpw (projection window)
            └── svgp (global projection header)
    """
    # svhd — header (version=0, flags=0, empty payload)
    svhd = _full_box(b"svhd", 0, 0)

    # svmi — stereo video mode info
    # Byte 0: has_left (1), Byte 1: has_right (1)
    svmi = _full_box(b"svmi", 0, 0, _u8(1), _u8(1))
    # svv3d — view box containing svmi
    svv3d = _box4(b"svv3d", svmi)

    # svpj — projection type (1 = equirectangular)
    svpj = _full_box(b"svpj", 0, 0, _u32(1))
    # svpw — projection window (yaw, pitch, roll in degrees)
    svpw = _full_box(b"svpw", 0, 0, _u32(180), _u32(180), _u32(0))
    # svgp — global projection header (empty)
    svgp = _full_box(b"svgp", 0, 0)
    # svproj — projection container
    svproj_body = svpj + svpw + svgp
    svproj = _box4(b"svproj", svproj_body)

    # Assemble sv3d (st3d is NOT inside sv3d — it's a sibling per spec)
    sv3d_body = svhd + svv3d + svproj
    sv3d = _box4(b"sv3d", sv3d_body)

    return sv3d


# ---------------------------------------------------------------------------
# MP4 parser / writer
# ---------------------------------------------------------------------------

def _read_atoms(path: str) -> List[Tuple[int, int, bytes]]:
    """Walk the MP4 file and return a flat list of (offset, size, type_) tuples.

    Only reads top-level atoms (does not recurse into containers).
    """
    atoms = []
    with open(path, "rb") as f:
        offset = 0
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        while offset < file_size - 7:
            header = f.read(8)
            if len(header) < 8:
                break
            size = struct.unpack(">I", header[:4])[0]
            type_ = header[4:8]
            if size < 8:
                break
            actual_size = size
            # Handle large boxes (64-bit size)
            if size == 1:
                large_header = f.read(8)
                if len(large_header) < 8:
                    break
                actual_size = struct.unpack(">Q", large_header)[0]
            atoms.append((offset, actual_size, type_))
            offset += actual_size
            f.seek(offset)
    return atoms


def _find_box_at(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Find an ISOBMFF box by type, scanning buf[start:end] by box boundaries.

    Returns the offset of the box, or -1 if not found.
    Scans by box size (not byte-by-byte) for correctness and performance.
    """
    i = start
    while i < end - 7:
        sz = struct.unpack(">I", buf[i:i + 4])[0]
        if sz < 8 or i + sz > end:
            break  # Invalid box, stop scanning
        if buf[i + 4:i + 8] == box_type:
            return i
        i += sz  # Jump to next box by its size
    return -1


# Known container boxes that can contain child boxes
_CONTAINER_TYPES = frozenset([
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf",
    b"edts", b"udta", b"stsd", b"sinf", b"schi",
])


def _find_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Recursively find a box by type, scanning container boxes.

    Uses proper box-boundary-based scanning for correctness and performance.
    """
    i = start
    while i < end - 7:
        sz = struct.unpack(">I", buf[i:i + 4])[0]
        if sz < 8 or i + sz > end:
            break  # Invalid box, stop scanning
        if buf[i + 4:i + 8] == box_type:
            return i
        # If this is a known container box, recurse inside it
        typ = bytes(buf[i + 4:i + 8])
        if typ in _CONTAINER_TYPES:
            result = _find_box_recursive(buf, box_type, i + 8, i + sz)
            if result >= 0:
                return result
        i += sz  # Jump to next box by its size
    return -1


def _update_box_sizes(buf: bytearray, box_start: int, delta: int, chain: List[bytes]):
    """Walk up the ISOBMFF box tree and update sizes for all parent boxes.

    Args:
        buf: The full MP4 file buffer
        box_start: Starting offset of the innermost box that was modified
        delta: Number of bytes added/removed
        chain: List of parent box types to walk up (innermost first)
    """
    current_pos = box_start
    for parent_type in chain:
        # Find the parent box that contains current_pos
        parent_pos = _find_parent_box(buf, parent_type, current_pos)
        if parent_pos >= 0:
            cur_size = struct.unpack(">I", buf[parent_pos:parent_pos + 4])[0]
            struct.pack_into(">I", buf, parent_pos, cur_size + delta)


def _find_parent_box(buf: bytearray, box_type: bytes, child_offset: int) -> int:
    """Find a box of the given type that contains child_offset.

    This is a simplified approach: scan backwards from child_offset to find
    a box of the right type whose range includes child_offset.
    """
    # For moov/trak/mdia etc, we scan from the beginning of the file
    # to find the matching container. This is O(n) but only done once.
    i = 0
    file_end = len(buf)
    while i < file_end - 7:
        sz = struct.unpack(">I", buf[i:i + 4])[0]
        if sz < 8 or i + sz > file_end:
            break
        if buf[i + 4:i + 8] == box_type:
            if i < child_offset < i + sz:
                return i
        i += sz
    return -1


def _inject_sv3d_in_file(path: str, width: int, height: int, stereo_mode: str) -> bool:
    """Read path, find stsd→avc1/hvc1, inject sv3d + st3d boxes, rewrite file.

    Injects:
    - sv3d box: inside the avc1/hvc1 entry (after the fixed header)
    - st3d box: as a sibling of sv3d inside the avc1/hvc1 entry
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # Build the metadata boxes
    sv3d_box = _build_sv3d(width, height, stereo_mode)
    st3d_box = _build_st3d(stereo_mode)
    metadata_payload = sv3d_box + st3d_box
    payload_len = len(metadata_payload)

    # Find moov atom
    moov_pos = _find_box_at(data, b"moov", 0, len(data))
    if moov_pos < 0:
        print("[Metadata] moov box not found")
        return False

    moov_size = struct.unpack(">I", data[moov_pos:moov_pos + 4])[0]
    moov_end = moov_pos + moov_size

    # Find stsd box inside moov (recursive search through trak→mdia→minf→stbl)
    stsd_pos = _find_box_recursive(data, b"stsd", moov_pos, moov_end)
    if stsd_pos < 0:
        print("[Metadata] stsd box not found")
        return False

    stsd_size = struct.unpack(">I", data[stsd_pos:stsd_pos + 4])[0]
    stsd_end = stsd_pos + stsd_size

    # Find avc1 or hvc1 entry inside stsd
    # stsd has: size(4) + type(4) + version_flags(4) + entry_count(4) = 16 bytes header
    entry_pos = _find_box_at(data, b"avc1", stsd_pos + 16, stsd_end)
    if entry_pos < 0:
        entry_pos = _find_box_at(data, b"hvc1", stsd_pos + 16, stsd_end)
    if entry_pos < 0:
        print("[Metadata] No avc1/hvc1 entry found in stsd")
        return False

    entry_size = struct.unpack(">I", data[entry_pos:entry_pos + 4])[0]
    entry_type = bytes(data[entry_pos + 4:entry_pos + 8])

    # avc1/hvc1 entry structure:
    #   size(4) + type(4) + reserved(6) + data_ref_index(2) +
    #   pre_defined(2) + reserved(2) + pre_defined(12) +
    #   width(2) + height(2) + horiz_resolution(4) + vert_resolution(4) +
    #   reserved(4) + frame_count(2) + compressor_name(32) + depth(2) +
    #   pre_defined(2) = 78 bytes of fixed header before child boxes
    FIXED_HEADER_SIZE = 78

    # We insert after the fixed header
    insert_pos = entry_pos + 8 + FIXED_HEADER_SIZE

    # Insert the metadata payload
    data[insert_pos:insert_pos] = metadata_payload

    # Update all parent box sizes up the chain
    # Chain: avc1/hvc1 → stsd → stbl → minf → mdia → trak → moov
    parent_chain = [entry_type, b"stsd", b"stbl", b"minf", b"mdia", b"trak", b"moov"]
    for parent_type in parent_chain:
        parent_pos = _find_parent_box(data, parent_type, entry_pos)
        if parent_pos >= 0:
            cur_size = struct.unpack(">I", data[parent_pos:parent_pos + 4])[0]
            struct.pack_into(">I", data, parent_pos, cur_size + payload_len)

    with open(path, "wb") as f:
        f.write(data)

    print(f"[Metadata] sv3d + st3d boxes injected ({payload_len} bytes)")
    return True