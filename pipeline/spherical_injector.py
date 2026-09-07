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
_EQUI_BOUNDS_OFFSET = 12
_EQUI_BOX_SIZE = _EQUI_BOUNDS_OFFSET + 4 * 4

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
    (offset, size, box_type) in root->parent order: moov first, then trak,
    mdia, minf, stbl and finally stsd (the entry's immediate parent).  Only
    containers that actually own the entry are in the chain.  Returns None if
    no such entry is reachable.
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
        tracked = btype in (b"trak", b"mdia", b"minf", b"stbl", b"stsd")
        if tracked:
            chain.append((pos, size, btype))
        inner = _descend_to_entry(buf, pos + hs, pos + size, chain)
        if inner[0] is not None:
            return inner
        if tracked:
            # Backtrack: this container (e.g. an audio trak that precedes the
            # video trak) does not own the entry and must not get its size bumped.
            chain.pop()
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
       with that chain of ancestor boxes.
    2. Rebuild the sample entry: keep its 8+78 byte header and every existing
       child box except a previous st3d/sv3d, then append fresh st3d + sv3d
       (st3d first, as the spec asks).  Replacing rather than appending keeps a
       re-injection idempotent — ffmpeg ``-c copy`` (audio remux) carries a
       valid sv3d/st3d through, and ffmpeg rejects a sample entry with two st3d.
    3. ``delta`` = new entry length - old entry length (0 or negative when the
       boxes being replaced were at least as large).
    4. Shift every ``stco``/``co64`` entry of **every** trak that points at or
       past the end of the old entry by ``delta`` — those are the bytes that
       physically moved.  With ``+faststart`` (moov before mdat) that is all of
       them; with moov after mdat none qualify and the same code path leaves
       them untouched.  The pre-#279 writer added delta unconditionally, which
       is what corrupted the mdat-first layout.
    5. Splice the new entry in and bump the size field of each ancestor
       (stsd, stbl, minf, mdia, trak, moov) by ``delta``.

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
    delta = len(new_entry) - entry_sz

    # Chunk offsets are patched on the *unmodified* buffer so every coordinate
    # (threshold and box positions) is still expressed in original-file terms.
    _shift_chunk_offsets(buf, moov_off, moov_sz, threshold=old_entry_end, delta=delta)

    buf[entry_off:old_entry_end] = new_entry
    for anc_off, _anc_sz, _anc_type in chain:
        _bump_box_size(buf, anc_off, delta)

    Path(output_path).write_bytes(bytes(buf))


def _rebuild_sample_entry(buf: bytearray, entry_off: int, entry_sz: int, new_children: bytes) -> bytes:
    """Return the visual sample entry at *entry_off* with st3d/sv3d replaced by *new_children*.

    Existing child boxes other than st3d/sv3d (avcC/hvcC, pasp, colr, ...) are
    kept in order; any unparseable tail bytes are preserved after the new boxes
    so a quirky-but-working file is not made worse.  The size field of the
    returned entry is already correct.
    """
    entry_end = entry_off + entry_sz
    head_end = entry_off + _SAMPLE_ENTRY_HEADER_SIZE
    if head_end > entry_end:
        raise RuntimeError(
            f"visual sample entry at offset {entry_off} is shorter than its {_SAMPLE_ENTRY_HEADER_SIZE}-byte header"
        )

    kept = bytearray()
    consumed = head_end
    for off, btype, size in _walk_boxes(buf, head_end, entry_end):
        if btype not in (b"st3d", b"sv3d"):
            kept += buf[off : off + size]
        consumed = off + size

    new_entry = bytearray(buf[entry_off:head_end]) + kept + new_children + buf[consumed:entry_end]
    struct.pack_into(">I", new_entry, 0, len(new_entry))
    return bytes(new_entry)


# Containers to descend through when looking for chunk-offset tables.
_CHUNK_OFFSET_PARENTS = frozenset({b"trak", b"mdia", b"minf", b"stbl"})


def _iter_chunk_offset_boxes(buf: bytearray, start: int, end: int):
    """Yield (offset, box_type, size) for every stco/co64 box under [start, end)."""
    for off, btype, size in _walk_boxes(buf, start, end):
        if btype in (b"stco", b"co64"):
            yield off, btype, size
        elif btype in _CHUNK_OFFSET_PARENTS:
            yield from _iter_chunk_offset_boxes(buf, off + 8, off + size)


def _shift_chunk_offsets(buf: bytearray, moov_off: int, moov_sz: int, *, threshold: int, delta: int) -> int:
    """Add *delta* to every stco/co64 entry >= *threshold* in every trak of the moov.

    stco/co64 are FullBoxes: size(4) type(4) version+flags(4) entry_count(4)
    followed by 32-bit (stco) or 64-bit (co64) absolute file offsets.  Only
    offsets at/after *threshold* — the bytes that actually moved — are shifted.
    Returns the number of entries shifted.

    Raises:
        RuntimeError: if a table's entry_count does not fit inside its box.
    """
    shifted = 0
    for off, btype, size in _iter_chunk_offset_boxes(buf, moov_off + 8, moov_off + moov_sz):
        fmt, width = (">I", 4) if btype == b"stco" else (">Q", 8)
        count = struct.unpack(">I", buf[off + 12 : off + 16])[0]
        table = off + 16
        if table + count * width > off + size:
            raise RuntimeError(
                f"malformed {btype.decode('ascii')} box at offset {off}: entry_count {count} does not fit in size {size}"
            )
        for i in range(count):
            at = table + i * width
            (value,) = struct.unpack(fmt, buf[at : at + width])
            if value >= threshold:
                struct.pack_into(fmt, buf, at, value + delta)
                shifted += 1
    return shifted


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
    sv3d_off = _find_box_recursive(buf, b"sv3d", 0, len(buf))
    if sv3d_off == -1:
        problems.append("missing sv3d box")
        return problems

    sv3d_sz = struct.unpack(">I", buf[sv3d_off : sv3d_off + 4])[0]
    sv3d_children = list(_walk_boxes(buf, sv3d_off + 8, sv3d_off + sv3d_sz))
    sv3d_types = [btype for _off, btype, _sz in sv3d_children]
    if sv3d_types[:1] != [b"svhd"]:
        problems.append(f"sv3d must start with svhd, found {sv3d_types}")
    if len(sv3d_types) < 2 or sv3d_types[1] != b"proj":
        problems.append(f"sv3d must carry proj right after svhd, found {sv3d_types}")
        return problems

    proj_off, _proj_type, proj_sz = sv3d_children[1]
    proj_children = list(_walk_boxes(buf, proj_off + 8, proj_off + proj_sz))
    proj_types = [btype for _off, btype, _sz in proj_children]
    if proj_types[:1] != [b"prhd"]:
        problems.append(f"proj must start with prhd, found {proj_types}")
    if len(proj_types) < 2 or proj_types[1] not in _PROJECTION_DATA_BOXES:
        problems.append(f"proj must carry one of equi/cbmp/mshp after prhd, found {proj_types}")
        return problems

    equi_off, equi_type, equi_sz = proj_children[1]
    if equi_type == b"equi":
        if equi_sz < _EQUI_BOX_SIZE:
            problems.append(f"equi box is {equi_sz} bytes, expected at least {_EQUI_BOX_SIZE}")
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
    sv3d_off = _find_box_recursive(buf, b"sv3d", 0, len(buf))
    if sv3d_off == -1:
        return -1
    sv3d_sz = struct.unpack_from(">I", buf, sv3d_off)[0]
    proj_off = _find_box_at(buf, b"proj", sv3d_off + 8, sv3d_off + sv3d_sz)
    if proj_off == -1:
        return -1
    proj_sz = struct.unpack_from(">I", buf, proj_off)[0]
    equi_off = _find_box_at(buf, b"equi", proj_off + 8, proj_off + proj_sz)
    if equi_off == -1 or equi_off + _EQUI_BOX_SIZE > len(buf):
        return -1
    return equi_off


def _unpack_equi_bounds(buf: bytearray, equi_off: int) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) u32 fields of the equi box at *equi_off*."""
    top, bottom, left, right = struct.unpack_from(">IIII", buf, equi_off + _EQUI_BOUNDS_OFFSET)
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
    at = equi_off + _EQUI_BOUNDS_OFFSET
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
