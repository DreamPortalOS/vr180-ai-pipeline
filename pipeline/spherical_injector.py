"""Inject Spherical Video V2 (st3d + sv3d) metadata into MP4 files.

Google's spatial-media CLI is tried first; when it is unavailable (it is not on
PyPI) an in-process ISOBMFF writer splices spec-compliant boxes into the video
track's visual sample entry and fixes up every affected size / chunk offset.
Either way the result is verified structurally *and* by a real ffmpeg decode
before it is handed back (issue #279: the old writer produced undecodable
files and its self-check still said OK).

Both paths must leave a byte-identical ``equi`` box (issue #281): the CLI is
asked for the RFC VR180 bounds via ``--bounds``, the four bounds fields are
patched in place should a spatial-media build still write the all-zero 360
default, and the self-check refuses anything but the RFC values.
:func:`read_projection_bounds` exposes the fields for tests and QA.

Box headers come in three forms (ISO/IEC 14496-12 §4.2): a 32-bit size, a
64-bit ``largesize`` (what ffmpeg writes for an ``mdat`` past 4 GiB — ten
seconds of 8K HEVC) and size 0 ("to end of file").  Every walker here handles
all three, and a 32-bit ``stco`` that can no longer hold its shifted offsets
is widened to ``co64`` rather than truncated (issue #282).

References:
- Google spatial-media: https://github.com/google/spatial-media
- Spherical Video V2 RFC: docs/spherical-video-v2-rfc.md in that repo
- ffmpeg parser: libavformat/mov.c ``mov_read_sv3d`` / ``mov_read_st3d``
"""

import contextlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

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


# ─── Spherical Video V2 box layout ───────────────────────────────────────────
#
#   VisualSampleEntry (avc1 / hvc1 / ...)
#     ├── st3d  FullBox(v0)  stereo_mode: u8
#     └── sv3d  Box
#           ├── svhd  FullBox(v0)  metadata_source: null-terminated UTF-8
#           └── proj  Box
#                 ├── prhd  FullBox(v0)  pose_yaw/pitch/roll: int32, 16.16 fixed
#                 └── equi  FullBox(v0)  bounds top/bottom/left/right: u32, 0.32 fixed
#
# ffmpeg's mov_read_sv3d walks svhd -> proj -> prhd -> {equi|cbmp|mshp}
# *sequentially* and logs "Missing projection box" when the box after svhd is
# anything but proj — which is exactly what the pre-#279 writer emitted.

_METADATA_SOURCE = b"vr180-ai-pipeline\x00"

# equi projection bounds are 0.32 fixed-point *crop* proportions of the full
# 360x180 sphere ("the proportion of projection cropped from each edge not
# covered by the video frame").  Each eye of our SBS output is a 180x180
# equirect (h_fov=180:v_fov=180 in equirectangular_mapper): the full vertical
# range and the middle half of the horizontal range, i.e. crop 90/360 = 0.25
# from the left and from the right, nothing from top/bottom.
_FIXED_0_32_QUARTER = 0x40000000  # 0.25 in 0.32 fixed point
_EQUI_BOUNDS_VR180 = (0, 0, _FIXED_0_32_QUARTER, _FIXED_0_32_QUARTER)  # top, bottom, left, right

# equi is a FullBox: size(4) type(4) version/flags(4) then the four u32 bounds.
# The offset/size below assume the compact 8-byte header both writers emit;
# :func:`_equi_bounds_at` handles a (legal, never seen) largesize header too.
_EQUI_BOUNDS_OFFSET = 12
_EQUI_BOUNDS_SIZE = 4 * 4
_EQUI_BOX_SIZE = _EQUI_BOUNDS_OFFSET + _EQUI_BOUNDS_SIZE

# Projection data boxes a proj box may carry (exactly one, right after prhd).
_PROJECTION_DATA_BOXES = frozenset({b"equi", b"cbmp", b"mshp"})


def _build_svhd() -> bytes:
    """svhd: FullBox(v0) + null-terminated ``metadata_source`` string."""
    return _full_box(b"svhd", 0, 0, _METADATA_SOURCE)


def _build_prhd(yaw: int = 0, pitch: int = 0, roll: int = 0) -> bytes:
    """prhd: FullBox(v0) + pose yaw / pitch / roll as int32 16.16 fixed-point degrees."""
    return _full_box(b"prhd", 0, 0, struct.pack(">iii", yaw, pitch, roll))


def _build_equi(bounds: tuple[int, int, int, int] = _EQUI_BOUNDS_VR180) -> bytes:
    """equi: FullBox(v0) + projection_bounds top / bottom / left / right (u32, 0.32 fixed)."""
    top, bottom, left, right = bounds
    return _full_box(b"equi", 0, 0, struct.pack(">IIII", top, bottom, left, right))


def _build_proj(bounds: tuple[int, int, int, int] = _EQUI_BOUNDS_VR180) -> bytes:
    """proj: plain Box holding prhd followed by exactly one projection data box (equi)."""
    return _box4(b"proj", _build_prhd() + _build_equi(bounds))


def _build_sv3d(width: int, height: int, stereo_mode: str) -> bytes:
    """Build a spec-compliant ``sv3d { svhd, proj { prhd, equi } }`` box.

    ``width`` / ``height`` / ``stereo_mode`` are accepted for API compatibility
    (existing callers and tests pass them).  The box content does not depend on
    them: the stereo layout lives in the sibling ``st3d`` box, and the equi
    bounds are fixed for the pipeline's 180x180-per-eye output.  ``stereo_mode``
    is still validated so an unknown mode fails as early as it used to.
    """
    _stereo_mode_byte(stereo_mode)
    return _box4(b"sv3d", _build_svhd() + _build_proj())


# ─── ISOBMFF box headers ──────────────────────────────────────────────────────
#
# ISO/IEC 14496-12 §4.2 encodes a box's length in one of three ways (#282):
#
#   size >= 8   the box is ``size`` bytes long; header = size(4) + type(4)
#   size == 1   a 64-bit ``largesize`` follows the type; header = 16 bytes —
#               what ffmpeg writes for an ``mdat`` past 4 GiB (ten seconds of
#               8K HEVC), and what the pre-#282 walker misread as "size < 8".
#   size == 0   the box runs to the end of the file (or of the enclosing range)
#
# Every function that walks boxes goes through _read_box_header / _iter_boxes so
# the three forms are handled identically for locating, sizing and shifting.

_BOX_HEADER_SIZE = 8
_LARGE_BOX_HEADER_SIZE = 16


def _read_box_header(buf: bytearray, pos: int, end: int) -> tuple[int, bytes, int] | None:
    """Parse the box header at *pos*, bounded by *end*: ``(size, box_type, header_len)``.

    ``size`` is the full box length including its header, so ``pos + size`` is
    the next box.  Returns None when the header is truncated, the size field is
    malformed (2..7, or a largesize below 16) or the box would overrun *end* —
    walkers stop there instead of mis-reading whatever follows.
    """
    if pos + _BOX_HEADER_SIZE > end:
        return None
    size = struct.unpack_from(">I", buf, pos)[0]
    box_type = bytes(buf[pos + 4 : pos + 8])
    header_len = _BOX_HEADER_SIZE
    if size == 1:
        if pos + _LARGE_BOX_HEADER_SIZE > end:
            return None
        size = struct.unpack_from(">Q", buf, pos + 8)[0]
        header_len = _LARGE_BOX_HEADER_SIZE
    elif size == 0:
        size = end - pos
    if size < header_len or pos + size > end:
        return None
    return size, box_type, header_len


def _box_header_len(buf: bytearray, pos: int) -> int:
    """Header length of the box at *pos*: 16 when it carries a largesize field, else 8."""
    return _LARGE_BOX_HEADER_SIZE if struct.unpack_from(">I", buf, pos)[0] == 1 else _BOX_HEADER_SIZE


def _iter_boxes(buf: bytearray, start: int, end: int):
    """Yield ``(offset, box_type, size, header_len)`` for every box in [start, end).

    Stops at the first malformed or overrunning header (see :func:`_read_box_header`).
    """
    pos = start
    while pos < end:
        header = _read_box_header(buf, pos, end)
        if header is None:
            break
        size, box_type, header_len = header
        yield pos, box_type, size, header_len
        pos += size


def _walk_boxes(buf: bytearray, start: int, end: int):
    """Yield ``(offset, box_type, size)`` for every box in [start, end).

    ``size`` includes the box's own 8- or 16-byte header, so ``offset + size``
    is always the next box; the payload starts at ``offset +
    _box_header_len(buf, offset)``.  A ``size == 0`` box reports the distance
    to *end*.
    """
    for off, box_type, size, _header_len in _iter_boxes(buf, start, end):
        yield off, box_type, size


def _locate_box_at(buf: bytearray, box_type: bytes, start: int, end: int) -> tuple[int, int, int] | None:
    """``(offset, size, header_len)`` of the first *box_type* directly inside [start, end), or None."""
    for off, btype, size, header_len in _iter_boxes(buf, start, end):
        if btype == box_type:
            return off, size, header_len
    return None


def _find_box_at(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Byte offset of the first *box_type* directly inside [start, end), or -1 if not found."""
    found = _locate_box_at(buf, box_type, start, end)
    return -1 if found is None else found[0]


# Plain container boxes: child boxes start right after the box header.
_PLAIN_CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl"})

# stsd is a FullBox: version/flags(4) + entry_count(4) precede the sample entries.
_STSD_FIXED_FIELDS = 8
_STSD_HEADER_SIZE = _BOX_HEADER_SIZE + _STSD_FIXED_FIELDS

# Visual sample entries (avc1/hvc1/...) carry 78 bytes of fixed fields after the
# box header before any child boxes (sv3d/st3d live here per Spherical V2).
_VISUAL_SAMPLE_ENTRIES = frozenset({b"avc1", b"avc3", b"hvc1", b"hev1", b"av01", b"vp09", b"mp4v"})
_SAMPLE_ENTRY_FIXED_FIELDS = 78
_SAMPLE_ENTRY_HEADER_SIZE = _BOX_HEADER_SIZE + _SAMPLE_ENTRY_FIXED_FIELDS


def _children_start(box_type: bytes, box_off: int, header_len: int) -> int | None:
    """Offset of the first child box of the container at *box_off*, or None for a leaf.

    *header_len* is the box's own header (8, or 16 for largesize).  Plain
    containers: children right after it.  stsd: after version/flags +
    entry_count.  Visual sample entries: after their 78 fixed fields.
    """
    if box_type in _PLAIN_CONTAINERS:
        return box_off + header_len
    if box_type == b"stsd":
        return box_off + header_len + _STSD_FIXED_FIELDS
    if box_type in _VISUAL_SAMPLE_ENTRIES:
        return box_off + header_len + _SAMPLE_ENTRY_FIXED_FIELDS
    return None


def _locate_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int) -> tuple[int, int, int] | None:
    """``(offset, size, header_len)`` of the first *box_type* found depth-first in [start, end), or None.

    Descends into containers (moov, trak, mdia, minf, stbl), into stsd
    (FullBox + entry_count) and into visual sample entries (avc1/hvc1/... —
    where Spherical V2 sv3d/st3d actually live).
    """
    for off, btype, size, header_len in _iter_boxes(buf, start, end):
        if btype == box_type:
            return off, size, header_len
        inner = _children_start(btype, off, header_len)
        # Bounds check: skip boxes whose declared header overruns the box
        # (truncated/corrupt file) instead of raising.
        if inner is not None and inner <= off + size:
            found = _locate_box_recursive(buf, box_type, inner, off + size)
            if found is not None:
                return found
    return None


def _find_box_recursive(buf: bytearray, box_type: bytes, start: int, end: int) -> int:
    """Byte offset of the first *box_type* found by :func:`_locate_box_recursive`, or -1."""
    found = _locate_box_recursive(buf, box_type, start, end)
    return -1 if found is None else found[0]


def _find_visual_sample_entry(buf: bytearray) -> tuple[int, int, list[tuple[int, int, bytes]]] | None:
    """Locate the first visual sample entry (avc1/hvc1/...) reachable via the
    moov->trak->mdia->minf->stbl->stsd path, and return the chain of ancestor
    boxes needed to bump sizes after insertion.

    Returns (entry_offset, entry_size, ancestor_chain) where each ancestor is
    (offset, size, box_type) in root->parent order: moov first, then trak,
    mdia, minf, stbl and finally stsd (the entry's immediate parent).  Only
    containers that actually own the entry are in the chain.  Returns None if
    no such entry is reachable.  A largesize ``mdat`` ahead of moov (mdat-first
    layout past 4 GiB) is stepped over like any other box.
    """
    for moov_off, moov_type, moov_sz, moov_hl in _iter_boxes(buf, 0, len(buf)):
        if moov_type != b"moov":
            continue
        chain: list[tuple[int, int, bytes]] = [(moov_off, moov_sz, moov_type)]
        entry_off, entry_sz = _descend_to_entry(buf, moov_off + moov_hl, moov_off + moov_sz, chain)
        if entry_off is None:
            continue
        return entry_off, entry_sz, chain
    return None


def _descend_to_entry(
    buf: bytearray, start: int, end: int, chain: list[tuple[int, int, bytes]]
) -> tuple[int | None, int]:
    """Recursively descend moov->trak->mdia->minf->stbl->stsd -> visual entry.

    Appends each container box to *chain* as we descend (so the caller gets the
    full ancestor stack in root->parent order). Returns (entry_offset,
    entry_size) of the first visual sample entry found, or (None, 0).
    """
    for off, btype, size, header_len in _iter_boxes(buf, start, end):
        if btype in _VISUAL_SAMPLE_ENTRIES:
            return off, size
        inner = _children_start(btype, off, header_len)
        if inner is None or inner > off + size:
            continue
        tracked = btype in (b"trak", b"mdia", b"minf", b"stbl", b"stsd")
        if tracked:
            chain.append((off, size, btype))
        found = _descend_to_entry(buf, inner, off + size, chain)
        if found[0] is not None:
            return found
        if tracked:
            # Backtrack: this container (e.g. an audio trak that precedes the
            # video trak) does not own the entry and must not get its size bumped.
            chain.pop()
    return (None, 0)


def _bump_box_size(buf: bytearray, offset: int, delta: int) -> None:
    """Add *delta* to the size of the box at *offset*.

    Writes the 32-bit size field, or the 64-bit largesize field when the box
    uses one (size == 1).  A size-0 box runs to the end of the file and needs
    no update.
    """
    size = struct.unpack_from(">I", buf, offset)[0]
    if size == 1:
        large = struct.unpack_from(">Q", buf, offset + 8)[0]
        struct.pack_into(">Q", buf, offset + 8, large + delta)
    elif size != 0:
        struct.pack_into(">I", buf, offset, size + delta)


def inject_spherical_metadata(
    input_path: str,
    output_path: str,
    width: int = 7680,
    height: int = 1920,
    stereo_mode: str = "sbs",
) -> str:
    """Inject Google Spherical Video V2 metadata into an MP4 file.

    Writes real ISOBMFF ``st3d`` + ``sv3d { svhd, proj { prhd, equi } }`` boxes
    inside the visual sample entry (avc1/hvc1/...) of the video track.

    Google's ``spatial-media`` CLI is tried first (the reference
    implementation).  When it is unavailable or fails, the in-process ISOBMFF
    writer (:func:`_inject_via_python_isobmff`) takes over.  Whichever path
    wrote the file, :func:`_verify_injection` then checks the box structure and
    runs a real ffmpeg decode; any failure raises instead of delivering a
    broken file (issue #279: the old writer produced undecodable output and
    its presence-only self-check still said OK).

    Both paths end with the same ``equi`` bytes (issue #281): the CLI is passed
    the RFC VR180 bounds, and should it still write the all-zero 360 default,
    :func:`_rewrite_projection_bounds` patches the four fields in place before
    the self-check (which also insists on the RFC values).

    Args:
        input_path: Path to input MP4
        output_path: Path to output MP4 with sv3d+st3d atoms injected
        width: Full panorama width in pixels (carried for API compatibility)
        height: Full panorama height in pixels (carried for API compatibility)
        stereo_mode: "sbs" (side-by-side), "tb" (top-bottom) or "mono"

    Returns:
        Path to output file

    Raises:
        ValueError: on an unknown ``stereo_mode``.
        RuntimeError: if no injectable sample entry exists, or the written
                      file fails the structural / decode verification.
    """
    _stereo_mode_byte(stereo_mode)  # fail early on an unknown mode, before any subprocess

    if _inject_via_spatialmedia_cli(input_path, output_path, stereo_mode):
        if _rewrite_projection_bounds(output_path, _EQUI_BOUNDS_VR180):
            print(
                "[Metadata] equi bounds rewritten in place to the RFC VR180 values (spatial-media wrote the 360 default)"
            )
        _verify_injection(output_path)
        return output_path

    # spatialmedia unavailable or failed -> in-process ISOBMFF writer, verified
    # the same way (structure + ffmpeg decode) before the file is handed back.
    print("[Metadata] injecting sv3d+st3d via in-process ISOBMFF writer")
    shutil.copy2(input_path, output_path)
    _inject_via_python_isobmff(output_path, stereo_mode)
    _verify_injection(output_path)

    return output_path


def _inject_via_python_isobmff(output_path: str, stereo_mode: str) -> None:
    """Insert (or replace) st3d + sv3d in the first visual sample entry, in place.

    1. Read the whole MP4 into a bytearray and locate the first visual sample
       entry (avc1/hvc1/...) via moov->trak->mdia->minf->stbl->stsd, together
       with that chain of ancestor boxes.  Boxes of every size form (32-bit,
       largesize, size 0) are stepped over correctly (#282).
    2. Rebuild the sample entry: keep its header + 78 fixed bytes and every
       existing child box except a previous st3d/sv3d, then append fresh st3d
       + sv3d (st3d first, as the spec asks).  Replacing rather than appending
       keeps a re-injection idempotent — ffmpeg ``-c copy`` (audio remux)
       carries a valid sv3d/st3d through, and ffmpeg rejects a sample entry
       with two st3d.
    3. ``entry_delta`` = new entry length - old entry length (0 or negative
       when the boxes being replaced were at least as large).
    4. Plan the chunk-offset shift: every ``stco``/``co64`` entry of **every**
       trak that points at or past the end of the old entry moves by the total
       growth of moov — those are the bytes that physically moved.  With
       ``+faststart`` (moov before mdat) that is all of them; with moov after
       mdat none qualify and the tables are left untouched.  (The pre-#279
       writer added the delta unconditionally, which corrupted the mdat-first
       layout.)  A 32-bit ``stco`` whose shifted values would not fit is
       widened to ``co64`` — never truncated (#282); widening grows moov by 4
       bytes per entry, so :func:`_plan_chunk_offset_shift` settles the total.
    5. Apply: patch the tables that keep their width in place, then splice the
       new sample entry and every widened table in from the highest offset
       down, bumping each one's ancestors (stsd/stbl/minf/mdia/trak/moov for
       the entry, stbl/minf/mdia/trak/moov for a table) by its own growth.

    Raises:
        RuntimeError: if no injectable visual sample entry is found or a chunk
                      offset table is malformed.
    """
    buf = bytearray(Path(output_path).read_bytes())

    loc = _find_visual_sample_entry(buf)
    if loc is None:
        raise RuntimeError("no injectable visual sample entry (avc1/hvc1/...) found in moov tree")
    entry_off, entry_sz, chain = loc
    moov_off, moov_sz, _moov_type = chain[0]
    old_entry_end = entry_off + entry_sz

    # st3d is the sibling that precedes sv3d.
    payload = _build_st3d(stereo_mode) + _build_sv3d(7680, 1920, stereo_mode)
    new_entry = _rebuild_sample_entry(buf, entry_off, entry_sz, payload)
    entry_delta = len(new_entry) - entry_sz

    # Everything below is planned on the *unmodified* buffer so every coordinate
    # (threshold, box positions, ancestor size fields) is in original-file terms.
    tables = list(_iter_chunk_tables(buf, moov_off + _box_header_len(buf, moov_off), moov_off + moov_sz))
    delta, widen = _plan_chunk_offset_shift(tables, threshold=old_entry_end, entry_delta=entry_delta)

    # Splices (replacements that change a box's length) are applied from the
    # highest offset down: the other splice points and every ancestor's size
    # field all sit *before* the bytes they own, so they stay valid throughout.
    splices: list[tuple[int, int, bytes, tuple[int, ...]]] = [
        (entry_off, entry_sz, new_entry, tuple(off for off, _sz, _type in chain)),
    ]
    for table in tables:
        shifted = [value + delta if value >= old_entry_end else value for value in table.values]
        if table.offset in widen:
            print(
                f"[Metadata] stco at offset {table.offset} widened to co64: {len(shifted)} chunk offsets "
                f"would exceed 4 GiB after moov grew by {delta} bytes"
            )
            splices.append((table.offset, table.size, _build_co64(shifted), (moov_off, *table.ancestors)))
        else:
            _write_chunk_offsets(buf, table, shifted)
    for off, old_size, new_bytes, ancestors in sorted(splices, key=lambda splice: splice[0], reverse=True):
        buf[off : off + old_size] = new_bytes
        for ancestor in ancestors:
            _bump_box_size(buf, ancestor, len(new_bytes) - old_size)

    Path(output_path).write_bytes(bytes(buf))


def _rebuild_sample_entry(buf: bytearray, entry_off: int, entry_sz: int, new_children: bytes) -> bytes:
    """Return the visual sample entry at *entry_off* with st3d/sv3d replaced by *new_children*.

    Existing child boxes other than st3d/sv3d (avcC/hvcC, pasp, colr, ...) are
    kept in order; any unparseable tail bytes are preserved after the new boxes
    so a quirky-but-working file is not made worse.  The entry is always
    emitted with the compact 8-byte header (a largesize sample entry is legal
    but pointless) and its size field is already correct.
    """
    entry_end = entry_off + entry_sz
    header_len = _box_header_len(buf, entry_off)
    fields_end = entry_off + header_len + _SAMPLE_ENTRY_FIXED_FIELDS
    if fields_end > entry_end:
        raise RuntimeError(
            f"visual sample entry at offset {entry_off} is shorter than its "
            f"{header_len + _SAMPLE_ENTRY_FIXED_FIELDS}-byte header"
        )

    kept = bytearray()
    consumed = fields_end
    for off, btype, size in _walk_boxes(buf, fields_end, entry_end):
        if btype not in (b"st3d", b"sv3d"):
            kept += buf[off : off + size]
        consumed = off + size

    body = bytes(buf[entry_off + header_len : fields_end]) + bytes(kept) + new_children + bytes(buf[consumed:entry_end])
    return _box4(bytes(buf[entry_off + 4 : entry_off + 8]), body)


# ─── Chunk offset tables (stco / co64) ───────────────────────────────────────

# Containers to descend through when looking for chunk-offset tables.
_CHUNK_OFFSET_PARENTS = frozenset({b"trak", b"mdia", b"minf", b"stbl"})

_U32_MAX = 0xFFFF_FFFF

#: Largest file offset a 32-bit ``stco`` entry may hold after the shift.  A
#: table that would have to point past it once moov has grown is rewritten as
#: ``co64`` (#282); tests lower this to force the widening on small clips.
_STCO_MAX_OFFSET = _U32_MAX

# stco/co64 are FullBoxes: header, version+flags(4), entry_count(4), then the
# 32-bit (stco) or 64-bit (co64) absolute file offsets.
_CHUNK_TABLE_FIXED_FIELDS = 8


class _ChunkTable(NamedTuple):
    """One stco/co64 box as found in the unmodified file."""

    offset: int
    box_type: bytes
    size: int
    header_len: int
    ancestors: tuple[int, ...]  # offsets of the trak/mdia/minf/stbl boxes owning it, outermost first
    values: list[int]  # chunk offsets as stored


def _chunk_offset_format(box_type: bytes, count: int) -> str:
    return f">{count}{'I' if box_type == b'stco' else 'Q'}"


def _read_chunk_offsets(buf: bytearray, off: int, box_type: bytes, size: int, header_len: int) -> list[int]:
    """The chunk offsets stored in the stco/co64 box at *off*.

    Raises:
        RuntimeError: if the table's entry_count does not fit inside its box.
    """
    width = 4 if box_type == b"stco" else 8
    count = struct.unpack_from(">I", buf, off + header_len + 4)[0]
    table = off + header_len + _CHUNK_TABLE_FIXED_FIELDS
    if table + count * width > off + size:
        raise RuntimeError(
            f"malformed {box_type.decode('ascii')} box at offset {off}: entry_count {count} does not fit in size {size}"
        )
    return list(struct.unpack_from(_chunk_offset_format(box_type, count), buf, table))


def _iter_chunk_tables(buf: bytearray, start: int, end: int, ancestors: tuple[int, ...] = ()):
    """Yield a :class:`_ChunkTable` for every stco/co64 box under [start, end), in file order."""
    for off, btype, size, header_len in _iter_boxes(buf, start, end):
        if btype in (b"stco", b"co64"):
            yield _ChunkTable(
                off, btype, size, header_len, ancestors, _read_chunk_offsets(buf, off, btype, size, header_len)
            )
        elif btype in _CHUNK_OFFSET_PARENTS:
            yield from _iter_chunk_tables(buf, off + header_len, off + size, (*ancestors, off))


def _write_chunk_offsets(buf: bytearray, table: _ChunkTable, values: list[int]) -> None:
    """Overwrite the offsets of *table* in place (same count, same width).

    Raises:
        RuntimeError: if a value does not fit a 32-bit stco entry — the planner
                      must have widened that table; nothing is ever truncated.
    """
    if table.box_type == b"stco" and any(value > _U32_MAX for value in values):
        raise RuntimeError(
            f"stco at offset {table.offset} cannot hold a chunk offset past 4 GiB; it must be widened to co64"
        )
    at = table.offset + table.header_len + _CHUNK_TABLE_FIXED_FIELDS
    struct.pack_into(_chunk_offset_format(table.box_type, len(values)), buf, at, *values)


def _build_co64(values: list[int]) -> bytes:
    """A ``co64`` FullBox(v0) holding *values* as 64-bit chunk offsets."""
    return _full_box(
        b"co64", 0, 0, _u32(len(values)) + struct.pack(_chunk_offset_format(b"co64", len(values)), *values)
    )


def _plan_chunk_offset_shift(
    tables: list[_ChunkTable], *, threshold: int, entry_delta: int
) -> tuple[int, frozenset[int]]:
    """Settle how far the media moves and which stco tables must become co64.

    Every chunk offset at/after *threshold* moves by the total growth of moov:
    *entry_delta* plus 4 bytes per entry of every stco widened to co64.  A
    32-bit table is widened when any of its moved offsets would exceed
    :data:`_STCO_MAX_OFFSET`.  Widening one table grows moov, which can push
    another table over the limit, so this iterates to a fixed point (at most
    one round per stco table).  Returns ``(delta, offsets of tables to widen)``.
    """
    widen: set[int] = set()
    delta = entry_delta
    while True:
        overflowing = {
            table.offset
            for table in tables
            if table.box_type == b"stco"
            and table.offset not in widen
            and any(value >= threshold and value + delta > _STCO_MAX_OFFSET for value in table.values)
        }
        if not overflowing:
            return delta, frozenset(widen)
        widen |= overflowing
        delta = entry_delta + sum(4 * len(table.values) for table in tables if table.offset in widen)


# ffmpeg stderr fragments that mean the file is broken even when the process
# exits 0 (a malformed sv3d is logged at error level, then ffmpeg carries on).
_FFMPEG_FATAL_PATTERNS = (
    "Missing projection box",
    "Missing spherical video header",
    "Missing projection header box",
    "Unknown projection type",
    "Invalid NAL",
)

#: Seconds of media the decode check runs through.  The failure modes it guards
#: against (malformed sv3d, chunk offsets pointing into garbage) surface in the
#: header / first packets, and a uniform stco shift is either right for every
#: chunk or wrong for every chunk — so a bounded decode catches them while
#: keeping the check cheap on long 8K outputs.  ``None`` decodes the whole file.
VERIFY_DECODE_SECONDS: float | None = 10.0
_VERIFY_TIMEOUT_SECONDS = 600


def _verify_injection(output_path: str) -> None:
    """Self-check the written file: compliant box structure AND a clean ffmpeg decode.

    1. Structure: st3d present; sv3d present and shaped
       ``sv3d { svhd, proj { prhd, equi|cbmp|mshp } }`` — what ffmpeg's
       ``mov_read_sv3d`` requires.  (The pre-#279 writer passed a presence-only
       scan while emitting ``sv3d { svhd, svv3d, svmi }``.)  An equi box must
       carry exactly :data:`_EQUI_BOUNDS_VR180` (#281: the CLI path used to
       ship the all-zero 360 default while the fallback wrote 0.25).
    2. Decode: ``ffmpeg -v error -i <file> -f null -`` must exit 0 and its
       stderr must not contain any of :data:`_FFMPEG_FATAL_PATTERNS`.  When
       ffmpeg is not installed this step degrades to a printed warning.

    Raises:
        RuntimeError: on any structural problem or decode failure — a broken
                      file is never delivered silently (issues #91, #279).
    """
    buf = bytearray(Path(output_path).read_bytes())
    problems = _spherical_structure_problems(buf)
    if problems:
        raise RuntimeError(f"VR metadata injection FAILED self-check in {output_path}: {'; '.join(problems)}")
    _ffmpeg_decode_check(output_path)


def _spherical_structure_problems(buf: bytearray) -> list[str]:
    """Return human-readable problems with the st3d/sv3d layout in *buf* (empty when compliant)."""
    problems: list[str] = []
    if _find_box_recursive(buf, b"st3d", 0, len(buf)) == -1:
        problems.append("missing st3d box")
    sv3d = _locate_box_recursive(buf, b"sv3d", 0, len(buf))
    if sv3d is None:
        problems.append("missing sv3d box")
        return problems

    sv3d_off, sv3d_sz, sv3d_hl = sv3d
    sv3d_children = list(_iter_boxes(buf, sv3d_off + sv3d_hl, sv3d_off + sv3d_sz))
    sv3d_types = [btype for _off, btype, _sz, _hl in sv3d_children]
    if sv3d_types[:1] != [b"svhd"]:
        problems.append(f"sv3d must start with svhd, found {sv3d_types}")
    if len(sv3d_types) < 2 or sv3d_types[1] != b"proj":
        problems.append(f"sv3d must carry proj right after svhd, found {sv3d_types}")
        return problems

    proj_off, _proj_type, proj_sz, proj_hl = sv3d_children[1]
    proj_children = list(_iter_boxes(buf, proj_off + proj_hl, proj_off + proj_sz))
    proj_types = [btype for _off, btype, _sz, _hl in proj_children]
    if proj_types[:1] != [b"prhd"]:
        problems.append(f"proj must start with prhd, found {proj_types}")
    if len(proj_types) < 2 or proj_types[1] not in _PROJECTION_DATA_BOXES:
        problems.append(f"proj must carry one of equi/cbmp/mshp after prhd, found {proj_types}")
        return problems

    equi_off, equi_type, equi_sz, equi_hl = proj_children[1]
    if equi_type == b"equi":
        equi_min_size = equi_hl + (_EQUI_BOX_SIZE - _BOX_HEADER_SIZE)
        if equi_sz < equi_min_size:
            problems.append(f"equi box is {equi_sz} bytes, expected at least {equi_min_size}")
        else:
            bounds = _unpack_equi_bounds(buf, equi_off)
            if bounds != _EQUI_BOUNDS_VR180:
                problems.append(
                    f"equi bounds {_format_bounds(bounds)} are not the RFC VR180 values {_format_bounds(_EQUI_BOUNDS_VR180)}"
                )
    return problems


# ─── equi projection bounds: read / patch in place ───────────────────────────


def _find_equi_offset(buf: bytearray) -> int:
    """Byte offset of the ``equi`` box under ``sv3d -> proj``, or -1 when any of the three is missing."""
    sv3d = _locate_box_recursive(buf, b"sv3d", 0, len(buf))
    if sv3d is None:
        return -1
    sv3d_off, sv3d_sz, sv3d_hl = sv3d
    proj = _locate_box_at(buf, b"proj", sv3d_off + sv3d_hl, sv3d_off + sv3d_sz)
    if proj is None:
        return -1
    proj_off, proj_sz, proj_hl = proj
    equi = _locate_box_at(buf, b"equi", proj_off + proj_hl, proj_off + proj_sz)
    if equi is None or _equi_bounds_at(buf, equi[0]) + _EQUI_BOUNDS_SIZE > len(buf):
        return -1
    return equi[0]


def _equi_bounds_at(buf: bytearray, equi_off: int) -> int:
    """Offset of the four bounds fields: past the box header (8, or 16 for largesize) and version/flags."""
    return equi_off + _box_header_len(buf, equi_off) + 4


def _unpack_equi_bounds(buf: bytearray, equi_off: int) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) u32 fields of the equi box at *equi_off*."""
    top, bottom, left, right = struct.unpack_from(">IIII", buf, _equi_bounds_at(buf, equi_off))
    return top, bottom, left, right


def _format_bounds(bounds: tuple[int, int, int, int]) -> str:
    return "(" + ", ".join(f"0x{b:08x}" for b in bounds) + ")"


def read_projection_bounds(path: str | os.PathLike[str]) -> tuple[int, int, int, int]:
    """Return the ``equi`` projection bounds ``(top, bottom, left, right)`` stored in *path*.

    Each value is a 0.32 fixed-point crop proportion exactly as written in the
    file (VR180 per the Spherical Video V2 RFC: ``(0, 0, 0x40000000,
    0x40000000)``).  Meant for tests and QA tooling to compare what the two
    injection paths actually wrote.

    Raises:
        RuntimeError: when the file has no ``sv3d -> proj -> equi`` box.
    """
    buf = bytearray(Path(path).read_bytes())
    equi_off = _find_equi_offset(buf)
    if equi_off == -1:
        raise RuntimeError(f"no sv3d/proj/equi projection box found in {os.fspath(path)}")
    return _unpack_equi_bounds(buf, equi_off)


def _rewrite_projection_bounds(path: str | os.PathLike[str], bounds: tuple[int, int, int, int]) -> bool:
    """Overwrite only the four bounds fields of the existing ``equi`` box, in place.

    The box keeps its size, so no ancestor size or chunk offset changes and the
    16 bytes are written with a seek instead of rewriting the file.  Returns
    True when bytes changed, False when the file already carried *bounds* or
    has no equi box at all (then it is left untouched — :func:`_verify_injection`
    is where a missing box is reported).
    """
    buf = bytearray(Path(path).read_bytes())
    equi_off = _find_equi_offset(buf)
    if equi_off == -1:
        return False
    packed = struct.pack(">IIII", *bounds)
    at = _equi_bounds_at(buf, equi_off)
    if buf[at : at + len(packed)] == packed:
        return False
    with open(path, "r+b") as fh:
        fh.seek(at)
        fh.write(packed)
    return True


def _ffmpeg_decode_check(path: str) -> None:
    """Run ``ffmpeg -v error -i <path> -f null -`` and raise unless it decodes cleanly."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path]
    if VERIFY_DECODE_SECONDS is not None:
        cmd += ["-t", f"{VERIFY_DECODE_SECONDS:g}"]
    cmd += ["-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=_VERIFY_TIMEOUT_SECONDS)
    except FileNotFoundError:
        print("[Metadata] WARNING: ffmpeg not found — sv3d/st3d verified structurally only, not by decoding")
        return
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"VR metadata decode check timed out after {_VERIFY_TIMEOUT_SECONDS}s for {path}") from exc

    stderr = result.stderr or ""
    hits = [pattern for pattern in _FFMPEG_FATAL_PATTERNS if pattern in stderr]
    if result.returncode != 0 or hits:
        raise RuntimeError(
            f"VR metadata injection FAILED decode check for {path}: ffmpeg exit {result.returncode}, "
            f"fatal patterns {hits}: {stderr.strip()[-600:]}"
        )


# spatial-media ``-s`` vocabulary for the modes it can express (``mono`` is not one of them).
_SPATIALMEDIA_STEREO_ARG = {"sbs": "left-right", "tb": "top-bottom"}


def _spatialmedia_bounds_arg(bounds: tuple[int, int, int, int]) -> str:
    """``-b`` value: ``top:bottom:left:right`` as decimal 0.32 fixed-point integers."""
    return ":".join(str(b) for b in bounds)


def _inject_via_spatialmedia_cli(
    input_path: str,
    output_path: str,
    stereo_mode: str,
) -> bool:
    """Inject metadata using Google's spatial-media CLI tool.

    Uses V2 spec (-2 flag) which injects sv3d + st3d ISOBMFF boxes, and passes
    the RFC VR180 equi bounds via ``-b top:bottom:left:right`` (0.32 fixed
    point; spatial-media parses each with ``int(x, 0)``).  Without ``-b`` the
    CLI writes all-zero bounds, i.e. claims a full 360x180 sphere (#281).

    ``mono`` cannot be expressed on this CLI: ``-s`` only accepts
    ``none | top-bottom | left-right`` and ``none`` writes *no* st3d box, so the
    caller's in-process writer (which does emit ``st3d`` mode 0) is used instead.
    """
    if stereo_mode not in _SPATIALMEDIA_STEREO_ARG:
        print(f"[Metadata] spatialmedia CLI has no st3d {stereo_mode!r} mode; using in-process writer")
        return False
    try:
        cmd = [
            sys.executable,
            "-m",
            "spatialmedia",
            "-i",
            "-2",
            "-s",
            _SPATIALMEDIA_STEREO_ARG[stereo_mode],
            "-p",
            "equirectangular",
            "-b",
            _spatialmedia_bounds_arg(_EQUI_BOUNDS_VR180),
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
