"""Scene-oriented output naming + segment identifiers (issue #81, D-4).

The DreamPortal playback end organises an experience around **scenes**: one
scene = one or more video segments plus transitions between them.  To let the
downstream assemble a scene manifest purely from filenames (no manual mapping),
this module defines a stable, parseable naming convention:

    <scene_id>_<scene_name>_seg<NN>_<route>_<preset>.<ext>
    e.g. s03_santorini_seg01_vr180_standalone.mp4

Field contract (frozen so it can serve as a cross-version asset key):

- ``scene_id`` — stable short id, **does not change across versions**.  The
  downstream uses it as the asset key.  Lowercased, ``[a-z0-9_-]+`` (e.g.
  ``s03``).  An explicit ``None`` collapses to ``scene`` so an unnamed run
  still round-trips.
- ``scene_name`` — human-readable slug (e.g. ``santorini``).  Slugified on
  compose: lowercased, non ``[a-z0-9]+`` runs collapsed to a single ``_``.
  An explicit ``None`` collapses to ``unnamed``.
- ``segment_index`` — 1-based order within a scene.  Zero-padded to 2 digits
  in the filename (``seg01``) so lexical sort matches assembly order; stored
  verbatim as an int in the sidecar JSON so renames never lose it.  ``None``
  collapses to ``1`` (single-segment scene).
- ``route`` — projection route: ``vr180`` or ``fulldome`` (mirrors
  ``SOLUTION_ARCHITECTURE.md`` and the ``route`` field planned in #78/#79).
- ``preset`` — downstream playback preset (``pcvr`` / ``standalone`` /
  ``source`` per #79).  ``None`` collapses to ``standalone`` (the Quest
  self-contained default).

Round-trip invariant: ``parse_scene_name(compose_scene_name(spec))`` reproduces
``spec`` (modulo default-collapsing of ``None`` fields).  This is the property
that lets a script rebuild a scene manifest from a directory of files.

This module is pure-Python (stdlib only) so it imports on CI with no models /
no ffmpeg — the naming convention is the one piece of the scene-assembly
contract that does not depend on the heavy pipeline machinery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical routes (mirror SOLUTION_ARCHITECTURE.md's two delivery routes).
ROUTES = ("vr180", "fulldome")

# Canonical playback presets (mirror #79's downstream platform tiers).  ``source``
# means "no re-encode target chosen" — the file is left at the encode the
# pipeline produced.  Extensible: a preset unknown to this tuple still composes
# (and round-trips) but is flagged by :func:`validate_spec`.
PRESETS = ("pcvr", "standalone", "source")

# Field token validators.  The naming convention is intentionally restrictive
# so a filename is an unambiguous, single-parse contract — no quoted spaces,
# no dots inside a field (a dot would collide with the extension separator).
_SCENE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SLUG_RUN_RE = re.compile(r"[^a-z0-9]+")

# Segment token as it appears in a filename, e.g. ``seg01``.  Zero-padded to 2
# digits; more digits are accepted on parse (``seg001``) so segment counts
# above 99 survive a round-trip without renumbering.
_SEG_TOKEN_RE = re.compile(r"^seg0*(\d+)$")

_DEFAULT_SCENE_ID = "scene"
_DEFAULT_SCENE_NAME = "unnamed"
_DEFAULT_SEGMENT = 1
_DEFAULT_PRESET = "standalone"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NamingError(ValueError):
    """Raised when a spec or filename violates the naming convention."""


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneAssetSpec:
    """Resolved identity of one scene segment.

    Frozen so it is hashable and safe to use as a dict key / set member — the
    downstream manifest builder treats a spec as an immutable asset identity.

    ``None`` fields are allowed at construction and collapse to documented
    defaults on compose (see :func:`compose_scene_name`); a fully-resolved spec
    has no ``None`` fields.  Use :func:`validate_spec` to surface unknown
    route/preset values as a hard error rather than silently composing.
    """

    scene_id: str | None = None
    scene_name: str | None = None
    segment_index: int | None = None
    route: str = "vr180"
    preset: str | None = None

    def resolved(self) -> SceneAssetSpec:
        """Return a copy with ``None`` fields replaced by their defaults."""
        return replace(
            self,
            scene_id=self.scene_id or _DEFAULT_SCENE_ID,
            scene_name=_slugify(self.scene_name) if self.scene_name else _DEFAULT_SCENE_NAME,
            segment_index=_DEFAULT_SEGMENT if self.segment_index is None else self.segment_index,
            preset=self.preset or _DEFAULT_PRESET,
        )


def validate_spec(spec: SceneAssetSpec) -> None:
    """Raise :class:`NamingError` if *spec* has fields this convention forbids.

    Composing/parsing is permissive enough to round-trip, but a caller that
    wants to *publish* a name should validate first — an unknown ``route`` or
    ``preset`` means the downstream's field-definition contract (the point of
    this issue) is being violated.
    """
    r = spec.resolved()
    if not _SCENE_ID_RE.match(r.scene_id):
        raise NamingError(f"scene_id {r.scene_id!r} must match {_SCENE_ID_RE.pattern} (lowercase alnum, '_', '-')")
    if r.segment_index < 1:
        raise NamingError(f"segment_index must be >= 1, got {r.segment_index}")
    if r.route not in ROUTES:
        raise NamingError(f"route {r.route!r} not in {list(ROUTES)}")
    if r.preset not in PRESETS:
        raise NamingError(f"preset {r.preset!r} not in {list(PRESETS)}")


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


def _slugify(value: str) -> str:
    """Collapse *value* to a lowercase ``[a-z0-9_]+`` slug.

    Any run of non-alphanumeric characters becomes a single ``_``; leading /
    trailing runs are stripped.  Empty result collapses to ``_DEFAULT_SCENE_NAME``
    so a compose never emits a dangling ``__`` separator.
    """
    slug = _SLUG_RUN_RE.sub("_", value.lower()).strip("_")
    return slug or _DEFAULT_SCENE_NAME


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def compose_scene_name(spec: SceneAssetSpec, extension: str = "mp4") -> str:
    """Build a scene-oriented filename from *spec*.

    Parameters
    ----------
    spec:
        The segment identity.  ``None`` fields collapse to defaults; the
        ``scene_name`` is slugified.
    extension:
        File extension without the leading dot (default ``"mp4"``).

    Returns
    -------
    str
        ``<scene_id>_<scene_name>_seg<NN>_<route>_<preset>.<extension>``.

    Raises
    ------
    NamingError
        If a resolved field is empty or ``segment_index`` is non-positive.
    """
    ext = extension.lstrip(".").lower()
    if not ext:
        raise NamingError("extension must be non-empty (e.g. 'mp4')")

    r = spec.resolved()
    if not _SCENE_ID_RE.match(r.scene_id):
        raise NamingError(f"scene_id {r.scene_id!r} must match {_SCENE_ID_RE.pattern} (lowercase alnum, '_', '-')")
    if r.segment_index < 1:
        raise NamingError(f"segment_index must be >= 1, got {r.segment_index}")

    seg_token = f"seg{r.segment_index:02d}"
    name = f"{r.scene_id}_{r.scene_name}_{seg_token}_{r.route}_{r.preset}.{ext}"
    return name


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_scene_name(filename: str) -> SceneAssetSpec:
    """Parse a scene-oriented filename back into a :class:`SceneAssetSpec`.

    Accepts either a bare name (``"s03_santorini_seg01_vr180_standalone.mp4"``)
    or a path (only the final component is read).  The extension is optional on
    input — a caller that strips extensions can still parse the stem.

    Raises
    ------
    NamingError
        If the name does not match the convention, or a field is malformed
        (e.g. a non-numeric segment index).
    """
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, _ext = base.rpartition(".")
    name = stem if dot else base

    # 5 ordered fields.  Splitting on ``_`` from the left keeps scene_name
    # flexible: the route/preset are anchored at the tail by their fixed
    # positions, so a scene_name with no underscores inside it still parses.
    parts = name.split("_")
    if len(parts) < 5:
        raise NamingError(
            f"{filename!r}: expected ≥5 underscore-separated fields (scene_id, scene_name, segNN, route, preset)"
        )

    preset = parts[-1]
    route = parts[-2]
    seg_token = parts[-3]
    scene_id = parts[0]
    # A scene_name may itself contain underscores (e.g. "santorini_sunset");
    # rejoin everything between scene_id and the segment token.
    scene_name = "_".join(parts[1:-3])

    seg_match = _SEG_TOKEN_RE.match(seg_token)
    if seg_match is None:
        raise NamingError(f"{filename!r}: segment token {seg_token!r} must match 'seg<digits>' (e.g. 'seg01')")
    segment_index = int(seg_match.group(1))

    if not scene_id:
        raise NamingError(f"{filename!r}: scene_id is empty")

    return SceneAssetSpec(
        scene_id=scene_id,
        scene_name=scene_name,
        segment_index=segment_index,
        route=route,
        preset=preset,
    )


# ---------------------------------------------------------------------------
# Sidecar JSON helper
# ---------------------------------------------------------------------------


def sidecar_scene_fields(spec: SceneAssetSpec) -> dict:
    """Return the ``scene`` block to embed in a sidecar JSON (#78).

    The filename is the *convenience* view; the sidecar is the **authoritative**
    source (filenames get renamed).  This helper emits the stable, machine-read
    fields a downstream manifest builder keys on — ``scene_id`` as the asset
    key and ``segment_index`` as the assembly order — plus the human-readable
    ``scene_name`` and the composed ``filename`` for traceability.

    The returned dict is intentionally flat under a ``scene`` key so it can be
    ``json.dumps``-ed directly into a sidecar's top-level ``scene`` field.
    """
    r = spec.resolved()
    return {
        "scene_id": r.scene_id,
        "scene_name": r.scene_name,
        "segment_index": r.segment_index,
        "route": r.route,
        "preset": r.preset,
        "filename": compose_scene_name(r),
    }
