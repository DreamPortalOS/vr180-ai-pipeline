"""Inject Spherical Video V2 (sv3d) metadata into MP4 files.

Direct binary manipulation of ISOBMFF boxes. Places the sv3d box
inside the avc1/hvc1 video sample entry of the stsd atom.
"""
import os
import struct
from typing import Union


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


def _box4(type_: bytes, *parts: Union[bytes, int]) -> bytes:
    """Build an ISOBMFF box: type_ + concatenated parts, prefixed by size."""
    body = b"".join(
        _u32(p) if isinstance(p, int) else p for p in parts
    )
    return _u32(8 + len(body)) + type_ + body


def _flat_box(type_: bytes, *parts: Union[bytes, int]) -> bytes:
    """Build a simple box with version=0, flags=0, then parts."""
    return _box4(type_, _u32(0), *parts)


def _write_box(f, box: bytes):
    f.write(box)


# ---------------------------------------------------------------------------
# sv3d box builders
# ---------------------------------------------------------------------------

def _build_sv3d(width: int, height: int, stereo_mode: str) -> bytes:
    """Build the complete sv3d box for VR180 SBS."""
    # svhd — header (version=0, flags=0, empty payload)
    svhd = _flat_box(b"svhd")

    # svv3d → svmi (has_left=1, has_right=1)
    svmi = _flat_box(b"svmi", b"\x01\x00\x00\x00\x01")
    svv3d = _box4(b"sv3d", b"\x00\x00\x00\x00", svmi)

    # svproj → svpj + svpw + svgp
    svpj = _flat_box(b"svpj", _u32(1))                          # 1 = equirectangular
    svpw = _flat_box(b"svpw", _u32(180), _u32(180), _u32(0))    # yaw=180, pitch=180, roll=0
    svgp = _flat_box(b"svgp")                                     # empty projection header
    svproj_body = svpj + svpw + svgp
    svproj = _box4(b"svproj", _u32(0), svproj_body)               # version=0

    # st3d — stereo mode
    stereo_atom = b"side-by-side\x00" if stereo_mode == "sbs" else b"top-bottom\x00"
    st3d = _flat_box(b"st3d", stereo_atom)

    # Assemble sv3d
    sv3d_body = svhd + svv3d + svproj + st3d
    sv3d = _box4(b"sv3d", _u32(0), sv3d_body)
    return sv3d


# ---------------------------------------------------------------------------
# MP4 parser / writer
# ---------------------------------------------------------------------------

def _read_atoms(path: str):
    """Walk the MP4 file and return a flat list of (offset, size, type_) tuples."""
    atoms = []
    with open(path, "rb") as f:
        offset = 0
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        while offset < file_size - 7:
            size = struct.unpack(">I", f.read(4))[0]
            type_ = f.read(4)
            if size < 8:
                break
            atoms.append((offset, size, type_))
            # Handle large boxes (64-bit size)
            if size == 1:
                if offset + 16 <= file_size:
                    size = struct.unpack(">Q", f.read(8))[0]
                else:
                    break
            offset += size
            f.seek(offset)
    return atoms


def _inject_sv3d_in_file(path: str, width: int, height: int, stereo_mode: str) -> bool:
    """Read path, find stsd→avc1, inject sv3d box, rewrite file."""
    atoms = _read_atoms(path)
    if not atoms:
        return False

    sv3d_box = _build_sv3d(width, height, stereo_mode)
    sv3d_len = len(sv3d_box)

    # Find moov atom
    moov_atom = None
    for off, sz, typ in atoms:
        if typ == b"moov":
            moov_atom = (off, sz)
            break
    if not moov_atom:
        return False

    # Parse inside moov to find stsd → avc1/hvc1
    with open(path, "rb") as f:
        data = bytearray(f.read())

    def find_box(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
        """Find an ISOBMFF box start offset given type, scanning buf[start:end]."""
        i = start
        while i < end - 7:
            sz = struct.unpack(">I", buf[i:i + 4])[0]
            if sz < 8 or i + sz > end:
                i += 1
                continue
            if buf[i + 4:i + 8] == box_type:
                return i
            i += 1
        return -1

    def find_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int, depth: int = 0) -> int:
        """Recursively find a box by scanning container boxes."""
        i = start
        while i < end - 7:
            sz = struct.unpack(">I", buf[i:i + 4])[0]
            if sz < 8 or i + sz > end:
                i += 1
                continue
            if buf[i + 4:i + 8] == box_type:
                return i
            # If this is a container box (not a data box), recurse inside it
            typ = buf[i + 4:i + 8].decode("ascii", errors="replace")
            # Known container boxes: moov, trak, mdia, minf, stbl, dinf, edts, udta
            if typ in ("moov", "trak", "mdia", "minf", "stbl", "dinf", "edts", "udta",
                       "stsd", "stss", "stco", "stsz", "stsc", "stts"):
                result = find_box_recursive(buf, box_type, i + 8, i + sz, depth + 1)
                if result >= 0:
                    return result
            i += sz
        return -1

    moov_start, moov_size = moov_atom
    moov_end = moov_start + moov_size

    # Find stsd box inside moov
    stsd_pos = find_box_recursive(data, b"stsd", moov_start, moov_end)
    if stsd_pos < 0:
        print("[Metadata] stsd box not found")
        return False

    stsd_size = struct.unpack(">I", data[stsd_pos:stsd_pos + 4])[0]
    stsd_end = stsd_pos + stsd_size

    # Find avc1 entry inside stsd
    entry_pos = find_box(data, b"avc1", stsd_pos + 8, stsd_end)
    if entry_pos < 0:
        entry_pos = find_box(data, b"hvc1", stsd_pos + 8, stsd_end)
    if entry_pos < 0:
        print("[Metadata] No avc1/hvc1 entry found in stsd")
        return False

    entry_size = struct.unpack(">I", data[entry_pos:entry_pos + 4])[0]

    # avc1 entry structure:
    #   size (4) + type (4) + reserved(6) + data_ref_index(2) +
    #   version(2) + revision(2) + vendor(4) + temporal_quality(4) +
    #   width(2) + height(2) + h_res(4) + v_res(4) + data_size(4) +
    #   frame_count(2) + compressor(32) + depth(2) + color_table_id(2)
    #   = 78 bytes of header before child boxes begin
    # hvc1 has same structure
    AV1_HEADER_SIZE = 78
    insert_pos = entry_pos + 8 + AV1_HEADER_SIZE

    # Verify insert_pos is within the entry
    if insert_pos + sv3d_len > entry_pos + entry_size:
        # The entry might not have enough room; extend it
        extra = sv3d_len

        # Insert sv3d box
        data[insert_pos:insert_pos] = sv3d_box

        # Update sizes up the chain
        def update_size(buf: bytearray, box_start: int, delta: int):
            cur = struct.unpack(">I", buf[box_start:box_start + 4])[0]
            struct.pack_into(">I", buf, box_start, cur + delta)

        # Walk up the atom tree to update all parent sizes
        # Find the chain: stsd -> stbl -> minf -> mdia -> trak -> moov
        for parent_type in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"):
            parent_pos = 0
            # Find innermost container that contains entry_pos
            # We need to find the chain bottom-up, not top-down
            pass

        # Simpler: update entry size and walk parents
        update_size(data, entry_pos, sv3d_len)
        update_size(data, stsd_pos, sv3d_len)
        # Update moov
        update_size(data, moov_start, sv3d_len)

        with open(path, "wb") as f:
            f.write(data)

        print(f"[Metadata] sv3d box injected ({sv3d_len} bytes)")
        return True

    # If there's room, just insert
    data[insert_pos:insert_pos] = sv3d_box

    # Update sizes
    def update_size(buf: bytearray, box_start: int, delta: int):
        cur = struct.unpack(">I", buf[box_start:box_start + 4])[0]
        struct.pack_into(">I", buf, box_start, cur + delta)

    update_size(data, entry_pos, sv3d_len)
    update_size(data, stsd_pos, sv3d_len)
    update_size(data, moov_start, sv3d_len)

    with open(path, "wb") as f:
        f.write(data)

    print(f"[Metadata] sv3d box injected ({sv3d_len} bytes)")
    return True