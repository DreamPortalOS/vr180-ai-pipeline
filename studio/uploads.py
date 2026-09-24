"""Content-addressed upload store for the Studio canvas (issue #417).

Files dropped onto the canvas land here, de-duplicated by the first 12 hex
digits of their SHA-256. The store returns the path plus lightweight media
metadata (image dimensions / video duration + resolution + a poster frame)
so a freshly-created input node can show a thumbnail without a second round
trip.

Kept dependency-free on the FastAPI side: ``parse_multipart`` uses the
stdlib ``email`` module, so the server imports and the test suite run
without ``python-multipart`` (CI is CPU-only and the dev venv does not
ship it; see CLAUDE.md boundary lock).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any

# Accept only what the canvas can meaningfully wire into the graph.
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".mov"})
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS

#: Default upload ceiling (2 GB). Overridable via ``UploadStore(max_bytes=...)``.
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class UploadError(ValueError):
    """Raised for unsupported extensions / oversize files / malformed parts."""

    #: Maps to an HTTP status on the server. 400 = client gave us something we
    #: refuse to store; 413 = too big. Kept as an int so the route handler can
    #: read it without an extra mapping table.
    status_code: int = 400


class UnsupportedFileTypeError(UploadError):
    pass


class FileTooLargeError(UploadError):
    status_code = 413


class MalformedMediaError(UploadError):
    """A file whose extension is allowed but whose bytes do not decode.

    Without this the endpoint would 500 on an extension-valid-but-garbage
    upload (``UnidentifiedImageError`` / ffmpeg exit code). The extension is
    only a first filter — the bytes have to be probed too.
    """

    status_code = 400


@dataclass(frozen=True)
class UploadMeta:
    """Metadata returned to the canvas after a successful upload."""

    path: str
    kind: str  # "image" | "video"
    width: int | None
    height: int | None
    duration: float | None
    poster: str | None  # absolute path to a poster frame for videos, else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "poster": self.poster,
        }


@dataclass(frozen=True)
class _ParsedPart:
    filename: str
    data: bytes


def parse_multipart(content_type: str, body: bytes) -> _ParsedPart:
    """Parse a ``multipart/form-data`` body and return the first file part.

    Uses the stdlib ``email`` package so the server has no hard dependency on
    ``python-multipart``. The incoming bytes are framed as a synthetic MIME
    message (the form's own ``Content-Type`` header prepended), then walked for
    the first part carrying a ``filename`` (i.e. an actual file upload, not a
    plain form field).
    """
    if not content_type or "multipart/form-data" not in content_type.lower():
        raise UploadError("expected multipart/form-data body")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = BytesParser(policy=default).parsebytes(header + body)
    for part in msg.walk():
        if part.is_multipart():
            continue
        disposition = part.get("Content-Disposition", "") or ""
        if "filename" not in disposition.lower():
            continue
        filename = part.get_filename() or ""
        # ``get_payload(decode=True)`` honours Content-Transfer-Encoding and
        # returns the raw file bytes for binary parts; under the ``default``
        # policy non-text parts come back as bytes, not mangled text.
        payload = part.get_payload(decode=True)
        data = payload if payload is not None else (part.get_content() or b"")
        return _ParsedPart(filename=filename, data=bytes(data))
    raise UploadError("no file part in multipart body")


_EXT_RE = re.compile(r"\.([a-zA-Z0-9]+)$")


def _split_ext(filename: str) -> str:
    """Return the lower-cased extension *with* leading dot, or "" if none."""
    if not filename:
        return ""
    # os.path would treat ``.bashrc`` as having no ext; we only care about the
    # trailing dot-segment of the basename, so a regex on the whole name is fine.
    m = _EXT_RE.search(filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    return ("." + m.group(1).lower()) if m else ""


def classify(ext: str) -> str:
    """Map an extension to a canvas port kind, or raise."""
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise UnsupportedFileTypeError(f"unsupported file type {ext or '(none)'}; allowed: {sorted(ALLOWED_EXTS)}")


def _image_dims(path: Path) -> tuple[int, int]:
    """Read image WxH via Pillow (already a core dep)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is in requirements
        raise RuntimeError("Pillow is required to read image dimensions") from exc
    try:
        with Image.open(path) as img:
            # ``load()`` forces a full decode; a .png named over garbage
            # passes the extension check but must not take the server down.
            img.load()
        return int(img.width), int(img.height)
    except Exception as exc:
        raise MalformedMediaError(f"could not decode {path.name} as an image: {exc}") from exc


def probe_video(
    path: str | Path,
    *,
    poster_dir: str | Path | None = None,
    sha: str | None = None,
) -> tuple[int, int, float, str]:
    """Probe video resolution/duration and extract a poster frame via ffmpeg.

    All ffmpeg invocations use the list form (CLAUDE.md boundary). ffprobe is
    not assumed to be present, so duration/resolution come from one ``ffmpeg
    -i`` parse pass and the poster from a second pass — both via the same
    binary that is already on PATH. ``probe_video`` is the single shared
    implementation used by both the upload store and the ``input.video`` node
    so the card readout and the run-time meta can never disagree.
    """
    p = Path(path)
    poster_root = Path(poster_dir) if poster_dir else p.parent
    poster_root.mkdir(parents=True, exist_ok=True)
    label = sha or hashlib.sha256(p.name.encode()).hexdigest()[:12]
    poster = poster_root / f"{label}.png"
    # Grab the very first frame as a PNG the canvas can <img>. Seeking to a
    # fixed offset like 1s would skip past short clips (issue #417: a 0.5s
    # upload produced an empty poster). A failed poster is non-fatal — we
    # still return the probe numbers and let the node show the path instead.
    poster_cmd = [_FFMPEG, "-y", "-i", str(p), "-frames:v", "1", "-q:v", "2", str(poster)]
    try:
        proc = subprocess.run(poster_cmd, capture_output=True, text=True, check=False, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MalformedMediaError(f"ffmpeg probe of {p.name} failed: {exc}") from exc
    if proc.returncode != 0 or not poster.is_file():
        raise MalformedMediaError(
            f"could not decode {p.name} as a video: "
            f"{(proc.stderr or '').strip()[-200:] or 'poster frame extraction failed'}"
        )

    # Resolution + duration from stderr (ffmpeg prints stream info there when
    # no output is requested). One pass, regex out the numbers we need. The
    # poster already proved the file is a real video, so this is best-effort:
    # an unparseable banner yields zeros rather than a second failure.
    info_cmd = [_FFMPEG, "-i", str(p)]
    width = height = 0
    duration = 0.0
    try:
        proc = subprocess.run(info_cmd, capture_output=True, text=True, check=False, timeout=60)
        banner = (proc.stderr or "") + (proc.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        banner = ""
    dim = re.search(r"(\d{2,5})x(\d{2,5})", banner)
    if dim:
        width, height = int(dim.group(1)), int(dim.group(2))
    dur = re.search(r"Duration:\s*(\d+):(\d{2}):(\d+(?:\.\d+)?)", banner)
    if dur:
        h, mi, s = int(dur.group(1)), int(dur.group(2)), float(dur.group(3))
        duration = h * 3600 + mi * 60 + s
    return width, height, duration, str(poster)


class UploadStore:
    """Content-addressed store: same bytes ⇒ same path, no duplicate files."""

    def __init__(self, root: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root) / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)

    def _dedup_path(self, sha: str, ext: str) -> Path:
        return self.root / f"{sha}{ext}"

    def save(self, *, filename: str, data: bytes) -> UploadMeta:
        """Validate + persist one uploaded file, returning its metadata."""
        if len(data) > self.max_bytes:
            raise FileTooLargeError(f"file is {len(data)} bytes; limit is {self.max_bytes} bytes")
        ext = _split_ext(filename)
        kind = classify(ext)
        sha = hashlib.sha256(data).hexdigest()[:12]
        dest = self._dedup_path(sha, ext)
        # Content-addressed: if the exact bytes already landed, skip the write.
        if not dest.is_file():
            dest.write_bytes(data)
        try:
            return self._probe(dest, sha, ext, kind)
        except UploadError:
            # The bytes did not decode despite an allowed extension. Drop the
            # bogus file so a re-upload with real bytes is not mistaken for a
            # cache hit (same sha) and silently served back as malformed.
            import contextlib

            with contextlib.suppress(OSError):
                dest.unlink()
            raise

    def _probe(self, path: Path, sha: str, ext: str, kind: str) -> UploadMeta:
        width = height = None
        duration = None
        poster = None
        if kind == "image":
            width, height = _image_dims(path)
        else:
            w, h, dur, poster_path = probe_video(path, poster_dir=self.root, sha=sha)
            width, height = (w or None), (h or None)
            duration = dur or None
            poster = poster_path or None
        return UploadMeta(
            path=str(path),
            kind=kind,
            width=width,
            height=height,
            duration=duration,
            poster=poster,
        )

    def meta_for(self, path: str | Path) -> UploadMeta:
        """Re-probe an existing stored file (used by input nodes at run time)."""
        p = Path(path)
        ext = p.suffix.lower()
        kind = classify(ext)
        sha = p.stem
        return self._probe(p, sha, ext, kind)
