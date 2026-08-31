"""D-1: sidecar JSON metadata for each output artefact.

DreamPortal (the VR playback consumer) needs per-artefact metadata that is
not fully recoverable from the container alone (generation params, upstream
pipeline version, QA verdict). Every finished video is written alongside a
JSON sibling:

    scene01_matterhorn_seg01.mp4   ->   scene01_matterhorn_seg01.json

The JSON schema mirrors ``Tools/Media/media_manifest.json`` in the
``DreamPortalOS/nextscene`` repo so playback can pick the render path
(180 vs 360, SBS vs mono) and keep artefacts reproducible.

Fields sourced from *actual* files:

  - top-level ``name`` / ``bytes`` / ``sha256`` / ``duration_sec``
  - ``video`` block from ffprobe (codec / width / height / fps / bitrate /
    pixel format)
  - ``audio`` block from ffprobe (codec / channels / sample_rate) or ``None``
    when the artefact is silent
  - ``qa`` block computed from :func:`scripts.vr180_qa.run_qa`

Fields supplied by the *caller*:

  - ``immersive`` — projection / fov / stereo layout / eye size / boxes
  - ``generation`` — pipeline version, route, backends, prompt, seed, etc.

All heavy lifting goes through ffprobe and :mod:`pipeline.spherical_injector`,
both callable as subprocess list commands with no ``shell=True``. The module
is fully unit-testable on CI (CPU-only, no model inference).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pipeline

log = logging.getLogger("sidecar")

# ---------------------------------------------------------------------------
# D-3: projection contract (issue #80) — the machine-readable anchor that
# DreamPortal's EPanoProjection switches on.  Values here are the single
# source of truth; the enum literal strings MUST match the DreamPortal enum
# names (Equirect360 / Equirect180_SBS / Fisheye_Dome) lowercased for JSON.
# ---------------------------------------------------------------------------

# Canonical projection tags.  Ordered to mirror the historical adoption
# path (VR180 → dome → future 360) but extensible: a downstream that sees an
# unknown value should fall back to a generic viewer, never crash.
PROJECTION_EQUIRECT360 = "equirect360"  # 360° mono equirect (future)
PROJECTION_EQUIRECT180 = "equirect"  # 180° equirect (VR180 family)
PROJECTION_FISHEYE_DOME = "fisheye_domemaster"  # Domemaster azimuthal-equidistant
PROJECTIONS = (PROJECTION_EQUIRECT360, PROJECTION_EQUIRECT180, PROJECTION_FISHEYE_DOME)

# Canonical stereo layout tags.  "mono" = single frame (dome, 360, flat 2D);
# "side_by_side" = left-right SBS equirect (VR180).
STEREO_MONO = "mono"
STEREO_SIDE_BY_SIDE = "side_by_side"
STEREO_LAYOUTS = (STEREO_MONO, STEREO_SIDE_BY_SIDE)

# Field names that MUST be present in every published immersive block (D-3).
IMMERSIVE_REQUIRED_FIELDS = ("projection", "fov_deg", "stereo_layout", "eye_resolution")

# ffprobe / ffmpeg binary resolution (mirrors pipeline.audio_mux).
_FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"
_FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


def _probe_json(path: str, ffprobe: str = _FFPROBE_BIN) -> dict[str, Any]:
    """Read ffprobe JSON for a media file. Raises :class:`RuntimeError` on
    tool failure."""
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path!r} (exit {result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned non-JSON for {path!r}: {exc}") from None


def _sha256_file(path: str | Path) -> str:
    """Stream the SHA-256 hex digest of a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise RuntimeError("no video stream found")


def _audio_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def _parse_fps(stream: dict[str, Any]) -> float:
    """Turn a ffmpeg ``30/1``-style rational into a float (0.0 on failure)."""
    from fractions import Fraction

    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        frac = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(frac) if frac.denominator else 0.0


# ---------------------------------------------------------------------------
# Box (sv3d/st3d) scanning — mirrors scripts.vr180_qa._scan_boxes using the
# same pipeline.spherical_injector primitive. Kept here (rather than imported
# from vr180_qa, which lives in scripts/) so pipeline-sidecar.py stays
# resolvable from tests without adding scripts/ to the package namespace.
# ---------------------------------------------------------------------------

_BOX_SCAN = (b"sv3d", b"st3d")
_STEREO_LEFT_RIGHT = 2  # matches pipeline.spherical_injector._STEREO_LEFT_RIGHT


def _scan_boxes(path: str | Path) -> list[str]:
    """Return the list of sv3d/st3d box names present in *path*."""
    from pipeline.spherical_injector import _find_box_recursive

    data = bytearray(Path(path).read_bytes())
    present: list[str] = []
    for box in _BOX_SCAN:
        off = _find_box_recursive(data, box, 0, len(data))
        if off != -1:
            present.append(box.decode("ascii"))
    return present


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SidecarError(Exception):
    """Raised when the sidecar cannot be written (missing file, ffprobe fail)."""


class ImmersiveError(ValueError, SidecarError):
    """Raised when the ``immersive`` block violates the D-3 projection contract
    (missing required field, unknown projection / stereo tag, invalid FOV)."""


def build_sidecar(
    video_path: str | Path,
    *,
    immersive: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    qa: dict[str, Any] | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Build the sidecar dict for *video_path* without writing it to disk.

    Parameters
    ----------
    video_path:
        Path to the finished video file.
    immersive:
        Optional caller-supplied immersive block. If ``None`` a minimal
        ``immersive`` block is inferred from the file's box presence (sv3d/st3d
        present → equirect SBS; otherwise a fallback mono block). See
        :func:`_default_immersive`.
    generation:
        Optional caller-supplied generation block. If ``None`` a minimal block
        (``pipeline_version`` only) is written.
    qa:
        Optional pre-computed QA block. If ``None`` :func:`scripts.vr180_qa.run_qa`
        is invoked (best-effort; a failure logs a warning and writes
        ``passed=false`` with the error).
    ffprobe:
        Path to the ffprobe executable (default: ``ffprobe`` from PATH, or the
        :data:`_FFPROBE_BIN` resolved at import time).

    Returns
    -------
    dict
        A JSON-serialisable sidecar record.

    Raises
    ------
    SidecarError
        When the input file does not exist or ffprobe cannot read it.
    """
    path = Path(video_path)
    if not path.is_file():
        raise SidecarError(f"Video file not found: {path}")

    probe_ff = ffprobe or _FFPROBE_BIN
    try:
        probe = _probe_json(str(path), ffprobe=probe_ff)
    except RuntimeError as exc:
        raise SidecarError(f"ffprobe failed: {exc}") from None

    vstream = _video_stream(probe)
    astream = _audio_stream(probe)
    fmt = probe.get("format", {})

    duration_s = float(fmt.get("duration", 0.0) or 0.0)

    video = {
        "codec": vstream.get("codec_name", "unknown"),
        "width": int(vstream.get("width", 0)),
        "height": int(vstream.get("height", 0)),
        "fps": _parse_fps(vstream),
        "bitrate_bps": int(vstream.get("bit_rate") or fmt.get("bit_rate") or 0),
        "pix_fmt": vstream.get("pix_fmt", "yuv420p"),
    }

    audio: dict[str, Any] | None = None
    if astream is not None:
        audio = {
            "codec": astream.get("codec_name", "unknown"),
            "channels": int(astream.get("channels", 0)),
            "sample_rate": int(astream.get("sample_rate", 0)),
        }

    immersive = _default_immersive(video_path, video) if immersive is None else normalize_immersive(immersive)

    gen = dict(generation or {})
    gen.setdefault("pipeline_version", pipeline.__version__)

    if qa is None:
        qa = _run_qa(str(path), probe_ff)

    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "duration_sec": duration_s,
        "video": video,
        "audio": audio,
        "immersive": immersive,
        "generation": gen,
        "qa": qa,
    }


def write_sidecar(
    video_path: str | Path,
    *,
    immersive: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    qa: dict[str, Any] | None = None,
    ffprobe: str | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Build and write the sidecar JSON for *video_path*.

    Writes to ``<video_path>.json`` by default, or (when ``out_dir`` is given)
    to ``out_dir/<video_path>.name.json``.

    Returns
    -------
    Path
        The path the JSON was written to.

    Raises
    ------
    SidecarError
        If the video cannot be probed or the file cannot be written.
    """
    record = build_sidecar(
        video_path,
        immersive=immersive,
        generation=generation,
        qa=qa,
        ffprobe=ffprobe,
    )
    out_path = _out_path(video_path, out_dir=out_dir)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SidecarError(f"failed to write sidecar {out_path}: {exc}") from None
    log.info("📄 Sidecar written → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _out_path(video_path: str | Path, out_dir: str | Path | None) -> Path:
    """Determine the JSON path for a video (optionally inside *out_dir*)."""
    p = Path(video_path)
    parent = Path(out_dir) if out_dir is not None else p.parent
    return parent / f"{p.stem}.json"


def _default_immersive(video_path: str | Path, video: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer a minimal immersive block from the file itself.

    Presence of *both* sv3d and st3d ISOBMFF boxes signals a VR180 SBS equirect
    artefact; otherwise (e.g. fulldome or plain mono) we fall back to a neutral
    mono-fisheye-like block with empty ``spatial_metadata``. ``eye_resolution``
    is inferred from the video stream dimensions when the caller has not
    supplied it (D-3 requires the field to always be present). The caller
    should overwrite when they know more (route, fov, exact eye size).
    """
    present = _scan_boxes(video_path)
    # eye_resolution: [left_width, left_height] for SBS; [width, height] for
    # mono.  For SBS the full frame is 2×eye-width, so halve the width.
    v = video or {}
    w = int(v.get("width", 0))
    h = int(v.get("height", 0))
    eye_resolution = [w, h]
    if "sv3d" in present and "st3d" in present and w > 0:
        eye_resolution = [w // 2, h]

    if "sv3d" in present and "st3d" in present:
        return {
            "projection": PROJECTION_EQUIRECT180,
            "fov_deg": 180,
            "stereo_layout": STEREO_SIDE_BY_SIDE,
            "eye_resolution": eye_resolution,
            "spatial_metadata": ["sv3d", "st3d"],
        }
    return {
        "projection": PROJECTION_FISHEYE_DOME,
        "fov_deg": 180,
        "stereo_layout": STEREO_MONO,
        "eye_resolution": eye_resolution,
        "spatial_metadata": [b.decode("ascii") for b in _BOX_SCAN if b in [p.encode("ascii") for p in present]],
    }


# ---------------------------------------------------------------------------
# D-3: immersive block validation / normalisation
# ---------------------------------------------------------------------------


def normalize_immersive(immersive: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a caller-supplied ``immersive`` block.

    Enforces the D-3 projection contract: the four fields
    ``projection`` / ``fov_deg`` / ``stereo_layout`` / ``eye_resolution``
    are required and must carry recognised values.  An unknown projection
    tag is **not** rejected — the contract is extensible for future
    projections (e.g. a hypothetical 180-over-under) — but unknown
    ``stereo_layout`` values are rejected because the downstream stereo
    renderer has a closed set.

    Parameters
    ----------
    immersive:
        The caller-supplied block (a mutable dict; this function never mutates
        the input).

    Returns
    -------
    dict
        A clean copy with all four required fields present and valid.

    Raises
    ------
    ImmersiveError
        If a required field is missing, projection is unknown, stereo layout
        is unknown, ``fov_deg`` is out of the supported range, or
        ``eye_resolution`` is malformed.
    """
    out = dict(immersive)

    # projection
    proj = out.get("projection")
    if proj is None:
        raise ImmersiveError("immersive block missing required field 'projection'")
    if proj not in PROJECTIONS:
        # Allow unknown projections (extensibility) but warn so a typo is
        # surfaced in logs; the downstream maps unknown → generic viewer.
        log.warning("immersive.projection=%r is not in %s — passing through for extensibility", proj, list(PROJECTIONS))
    out["projection"] = proj

    # fov_deg
    fov = out.get("fov_deg")
    if fov is None:
        raise ImmersiveError("immersive block missing required field 'fov_deg'")
    try:
        fov = float(fov)
    except (TypeError, ValueError):
        raise ImmersiveError(f"immersive.fov_deg must be numeric, got {fov!r}") from None
    if not (0 < fov <= 360):
        raise ImmersiveError(f"immersive.fov_deg must be in (0, 360], got {fov}")
    # Keep int-looking FOVs as ints for clean JSON (180 not 180.0).
    out["fov_deg"] = int(fov) if fov == int(fov) else fov

    # stereo_layout
    layout = out.get("stereo_layout")
    if layout is None:
        raise ImmersiveError("immersive block missing required field 'stereo_layout'")
    if layout not in STEREO_LAYOUTS:
        raise ImmersiveError(f"immersive.stereo_layout {layout!r} not in {list(STEREO_LAYOUTS)}")
    out["stereo_layout"] = layout

    # eye_resolution
    eye = out.get("eye_resolution")
    if eye is None:
        raise ImmersiveError("immersive block missing required field 'eye_resolution'")
    out["eye_resolution"] = _coerce_eye_resolution(eye)

    return out


def _coerce_eye_resolution(eye: Any) -> list[int]:
    """Coerce ``eye_resolution`` to a two-element int list [width, height]."""
    try:
        w, h = eye
    except (TypeError, ValueError):
        raise ImmersiveError(f"immersive.eye_resolution must be a 2-tuple/list [w, h], got {eye!r}") from None
    try:
        return [int(w), int(h)]
    except (TypeError, ValueError):
        raise ImmersiveError(f"immersive.eye_resolution must be numeric, got {eye!r}") from None


def _run_qa(path: str, ffprobe: str) -> dict[str, Any]:
    """Best-effort QA run. On success mirrors the QAReport checks into a map;
    on failure returns a small block with ``passed=false`` and the error text."""
    try:
        from scripts.vr180_qa import run_qa

        report = run_qa(path, ffprobe=ffprobe)
        checks = {c.name: {"status": c.status, "detail": c.detail} for c in report.checks}
        return {
            "passed": not report.failed,
            "verdict": report.verdict,
            "checks": checks,
        }
    except Exception as exc:
        log.warning("QA failed while building sidecar for %s: %s", path, exc)
        return {
            "passed": False,
            "verdict": "unknown",
            "checks": {"error": {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}},
        }
