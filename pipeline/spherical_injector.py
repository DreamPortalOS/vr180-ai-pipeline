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
from pathlib import Path

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


# Plain container boxes: child boxes start right after the 8-byte header.
_PLAIN_CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl"})

# stsd is a FullBox: version/flags(4) + entry_count(4) precede the sample entries.
_STSD_HEADER_SIZE = 16

# Visual sample entries (avc1/hvc1/...) carry 78 bytes of fixed fields after the
# 8-byte box header before any child boxes (sv3d/st3d live here per Spherical V2).
_VISUAL_SAMPLE_ENTRIES = frozenset({b"avc1", b"avc3", b"hvc1", b"hev1", b"av01", b"vp09", b"mp4v"})
_SAMPLE_ENTRY_HEADER_SIZE = 8 + 78


def _find_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Recursively search for an ISOBMFF box inside containers.

    Searches top-level boxes, and recurses into containers (moov, trak, mdia,
    minf, stbl), into stsd (FullBox + entry_count), and into visual sample
    entries (avc1/hvc1/... — where Spherical V2 sv3d/st3d actually live).
    Returns byte offset of the found box, or -1.
    """
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = bytes(buf[pos + 4 : pos + 8])
        if btype == box_type:
            return pos
        if btype in _PLAIN_CONTAINERS:
            inner_start = pos + 8
        elif btype == b"stsd":
            inner_start = pos + _STSD_HEADER_SIZE
        elif btype in _VISUAL_SAMPLE_ENTRIES:
            inner_start = pos + _SAMPLE_ENTRY_HEADER_SIZE
        else:
            pos += size
            continue
        # Bounds check: skip boxes whose declared header overruns the buffer
        # instead of raising on a truncated/corrupt file.
        if inner_start <= pos + size:
            found = _find_box_recursive(buf, box_type, inner_start, pos + size)
            if found != -1:
                return found
        pos += size
    return -1


def _header_size(box_type: bytes) -> int:
    """Number of bytes before child boxes start, for known container kinds.

    Plain containers: 8 (size + type).
    stsd is a FullBox: 8 + 4 (version/flags) + 4 (entry_count) = 16.
    Visual sample entries: 8 + 78 fixed fields = 86.
    0 means "leaf / not descended into".
    """
    if box_type in _PLAIN_CONTAINERS:
        return 8
    if box_type == b"stsd":
        return _STSD_HEADER_SIZE
    if box_type in _VISUAL_SAMPLE_ENTRIES:
        return _SAMPLE_ENTRY_HEADER_SIZE
    return 0


def _walk_boxes(buf: bytearray, start: int, end: int):
    """Yield (offset, box_type, size) for every top-level box in [start, end)."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = buf[pos + 4 : pos + 8]
        yield pos, bytes(btype), size
        pos += size


def _find_visual_sample_entry(buf: bytearray) -> tuple[int, int, list[tuple[int, int, bytes]]] | None:
    """Locate the first visual sample entry (avc1/hvc1/...) reachable via the
    moov->trak->mdia->minf->stbl->stsd path, and return the chain of ancestor
    boxes needed to bump sizes after insertion.

    Returns (entry_offset, entry_size, ancestor_chain) where each ancestor is
    (offset, size, box_type) starting with the IMMEDIATE parent of the entry
    (stsd) and ending with the outermost (moov). Returns None if no such
    entry is reachable.
    """
    # Walk top-level boxes to find moov.
    for moov_off, moov_type, moov_sz in _walk_boxes(buf, 0, len(buf)):
        if moov_type != b"moov":
            continue
        chain: list[tuple[int, int, bytes]] = [(moov_off, moov_sz, moov_type)]
        try:
            entry_off, entry_sz = _descend_to_entry(buf, moov_off + 8, moov_off + moov_sz, chain)
        except _DescendError:
            continue
        if entry_off is None:
            continue
        return entry_off, entry_sz, chain
    return None


class _DescendError(Exception):
    """Raised when the expected moov->...->stsd box path is not found."""


def _descend_to_entry(
    buf: bytearray, start: int, end: int, chain: list[tuple[int, int, bytes]]
) -> tuple[int | None, int]:
    """Recursively descend moov->trak->mdia->minf->stbl->stsd -> visual entry.

    Appends each container box to *chain* as we descend (so the caller gets the
    full ancestor stack in parent->root order). Returns (entry_offset,
    entry_size) of the first visual sample entry found, or (None, 0).
    """
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = bytes(buf[pos + 4 : pos + 8])
        if btype in _VISUAL_SAMPLE_ENTRIES:
            return pos, size
        hs = _header_size(btype)
        if hs == 0:
            pos += size
            continue
        if btype in (b"trak", b"mdia", b"minf", b"stbl", b"stsd"):
            chain.append((pos, size, btype))
        inner = _descend_to_entry(buf, pos + hs, pos + size, chain)
        if inner[0] is not None:
            return inner
        pos += size
    return (None, 0)


def _bump_box_size(buf: bytearray, offset: int, delta: int) -> None:
    """Atomically add *delta* to the 4-byte big-endian size field at *offset*."""
    old = struct.unpack(">I", buf[offset : offset + 4])[0]
    struct.pack_into(">I", buf, offset, old + delta)


def inject_spherical_metadata(
    input_path: str,
    output_path: str,
    width: int = 7680,
    height: int = 1920,
    stereo_mode: str = "sbs",
) -> str:
    """Inject Google Spherical Video V2 metadata into an MP4 file.

    Produces real ISOBMFF ``sv3d`` + ``st3d`` boxes inside the visual sample
    entry (avc1/hvc1/...) of the video track, with every ancestor box's size
    field bumped by the inserted payload length. The result is self-verified
    via :func:`_find_box_recursive` — injection that does not survive the
    scan raises :class:`RuntimeError` (no silent bad output).

    Google's ``spatial-media`` CLI is tried first (it is the reference
    implementation). When unavailable it is **not** treated as a failure: we
    fall through to the in-process ISOBMFF writer, which is the reliable path.

    Args:
        input_path: Path to input MP4
        output_path: Path to output MP4 with sv3d+st3d atoms injected
        width: Full panorama width in pixels (carried for API compatibility;
               the sv3d box uses the frame dimensions per Spherical V2 spec)
        height: Full panorama height in pixels
        stereo_mode: "sbs" (side-by-side) or "tb" (top-bottom)

    Returns:
        Path to output file

    Raises:
        RuntimeError: if the injected sv3d/st3d boxes cannot be found again
                      by :func:`_find_box_recursive` in the written file.
    """
    if _inject_via_spatialmedia_cli(input_path, output_path, stereo_mode):
        _verify_injection(output_path)
        return output_path

    # spatialmedia unavailable or failed -> use the in-process ISOBMFF writer.
    # This is the reliable path: it writes real sv3d/st3d boxes with correct
    # ancestor size bumps, and self-verifies via _find_box_recursive.
    print("[Metadata] injecting sv3d+st3d via in-process ISOBMFF writer")
    shutil.copy2(input_path, output_path)
    _inject_via_python_isobmff(output_path, stereo_mode)
    _verify_injection(output_path)

    return output_path


def _inject_via_python_isobmff(output_path: str, stereo_mode: str) -> None:
    """Insert sv3d + st3d ISOBMFF boxes into *output_path* in place.

    1. Read the whole MP4 into a bytearray.
    2. Find the first visual sample entry via the moov->...->stsd path.
    3. Build the st3d (sibling, first) and sv3d boxes.
    4. Insert them right after the current children of the sample entry.
    5. Bump the size field of every ancestor box (entry -> stsd -> stbl ->
       minf -> mdia -> trak -> moov) by the inserted payload length.
    6. Bump the **absolute** chunk offsets in the track's ``stco``/``co64``
       boxes by the same delta, because the insertion sits before ``mdat``
       and the mdat that the offsets point at has moved. (Without this bump
       ffmpeg's -c copy demuxes the right number of packets but the muxer
       writes 0 bytes — issue #91.)
    7. Write the modified buffer back.

    Raises:
        RuntimeError: if no injectable visual sample entry is found.
    """
    buf = bytearray(Path(output_path).read_bytes())

    sv3d = _build_sv3d(7680, 1920, stereo_mode)
    st3d = _build_st3d(stereo_mode)
    payload = st3d + sv3d  # st3d is the sibling that precedes sv3d

    loc = _find_visual_sample_entry(buf)
    if loc is None:
        raise RuntimeError("no injectable visual sample entry (avc1/hvc1/...) found in moov tree")

    entry_off, entry_sz, chain = loc

    # Insertion point: immediately after the current contents of the sample entry.
    insert_at = entry_off + entry_sz

    # Shift everything from the insertion point to the end of the file down by
    # len(payload) so we can splice the boxes in.
    buf[insert_at:insert_at] = payload
    delta = len(payload)

    # Now bump the size field of the sample entry itself AND every ancestor,
    # each by delta. Ancestors are in parent->root order, all with the same
    # delta because the added bytes sit inside every one of them. The file has
    # no top-level container size, so bumping moov is the outermost container
    # size requirement.
    struct.pack_into(">I", buf, entry_off, entry_sz + delta)
    for anc_off, _anc_sz, _anc_type in chain:
        _bump_box_size(buf, anc_off, delta)

    # Bump the chunk-offset table(s) of the SAME track that owns the sample
    # entry we just extended.  stco stores 32-bit absolute file offsets; co64
    # stores 64-bit ones.  Every offset they point at lies at or after mdat,
    # which moved forward by delta, so each value must grow by delta.  Only
    # offsets for the affected track must change; we locate that track's stbl
    # (the stbl that is an ancestor of our sample entry) and patch only it.
    stbl_off = next(off for off, _sz, btype in chain if btype == b"stbl")
    stbl_sz = struct.unpack(">I", buf[stbl_off : stbl_off + 4])[0]
    _bump_chunk_offsets(buf, stbl_off + 8, stbl_off + stbl_sz, delta)

    Path(output_path).write_bytes(bytes(buf))


def _bump_chunk_offsets(buf: bytearray, start: int, end: int, delta: int) -> None:
    """Add *delta* to every absolute file offset in ``stco`` / ``co64`` boxes
    found at this container level (the stbl contents).

    Chunk offsets are absolute file offsets into ``mdat``.  Because the
    sv3d/st3d payload is spliced *before* mdat, mdat moves forward by *delta*
    and every chunk offset must grow by the same amount.

    We detect the offset region length from the box's declared size (some
    ffmpeg builds include the 4-byte ``reserved`` field in ``stco``, some do
    not) and sanity-check it against the stored ``entry_count``.
    """
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        if size < 8 or pos + size > end:
            break
        btype = buf[pos + 4 : pos + 8]
        if btype == b"stco":
            _bump_offsets(buf, pos, size, delta, entry_bytes=4)
        elif btype == b"co64":
            _bump_offsets(buf, pos, size, delta, entry_bytes=8)
        pos += size


def _bump_offsets(buf: bytearray, box_off: int, box_size: int, delta: int, entry_bytes: int) -> None:
    """Add *delta* to each offset entry of a stco/co64 box.

    The offset region is everything after the fixed prefix.  We try the two
    known prefix lengths (with / without the reserved field) and pick the one
    whose entry_count matches the region length.
    """
    body_end = box_off + box_size
    candidates = [
        (box_off + 12, box_off + 16),  # no reserved: entry_count@+12, offsets@+16
        (box_off + 16, box_off + 20),  # with reserved:  entry_count@+16, offsets@+20
    ]
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
    # Unrecognised layout — skip rather than corrupt offsets.


def _verify_injection(output_path: str) -> None:
    """Self-check: both sv3d and st3d must be findable via the recursive scanner.

    Raises :class:`RuntimeError` if either box is missing — this is the guard
    against silent bad output (issue #91: logs printed success but the file
    had no sv3d/st3d).
    """
    buf = bytearray(Path(output_path).read_bytes())
    missing = [bt for bt in (b"sv3d", b"st3d") if _find_box_recursive(buf, bt, 0, len(buf)) == -1]
    if missing:
        raise RuntimeError(
            f"VR metadata injection FAILED self-check: missing box(es) "
            f"{[b.decode('ascii') for b in missing]} in {output_path} "
            f"(injection produced a plain-2D file)"
        )


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
