"""Prompt Library — reusable, composable templates for VR180-friendly shots.

This module sits one layer *upstream* of ``pipeline/prompt_builder.py``:

- ``prompt_builder.wrap_prompt_for_vr180`` *wraps* a finished prompt text with
  output-target-aware suffixes (FOV, anti-nausea negatives, projection notes).
- ``prompt_library`` *produces* that finished prompt text from a template key +
  a set of placeholder fields (subject / setting / etc.).

The two do not overlap in responsibility. A typical call chain is::

    text = prompt_library.render("slow_dolly_in", subject="a sailboat",
                                  setting="a calm harbour at dawn")
    wrapped = prompt_builder.wrap_prompt_for_vr180(text, scene_type="orbit")

The value of this library is the curated template *bodies*: every template
embeds the descriptive constraints that make a shot convert cleanly to VR180
stereo — single continuous take, slow & constant camera motion, stable
lighting, clear foreground/mid-ground/background layering, and avoidance of
depth-hostile surfaces (mirrors / glass / fog).  These are the opposite of the
"obviously AI, discontinuous" footage the operator flagged on the first batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TEMPLATES",
    "PromptLibraryError",
    "PromptTemplate",
    "get_template",
    "list_templates",
    "render",
]


class PromptLibraryError(ValueError):
    """Raised when a template cannot be rendered (missing / extra fields)."""


@dataclass(frozen=True)
class PromptTemplate:
    """A reusable shot template.

    Attributes
    ----------
    key : str
        Stable registry identifier, e.g. ``"slow_dolly_in"``.
    summary : str
        One-line description of when the template fits.
    body : str
        Template text containing ``{placeholder}`` markers.  Every marker
        listed in ``placeholders`` must appear in ``body`` and vice-versa.
    camera_motion : str
        Canonical motion tag, e.g. ``"dolly_in"`` / ``"orbit_left"`` /
        ``"static"``.  Used by :func:`list_templates` for filtering.
    recommended_duration : float
        Suggested clip length in seconds, tuned for the seedance tier.
    placeholders : tuple[str, ...]
        Names of the required placeholders.  Order is cosmetic; rendering
        accepts fields in any order.
    """

    key: str
    summary: str
    body: str
    camera_motion: str
    recommended_duration: float
    placeholders: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Keep the dataclass honest: declared placeholders must match the
        # placeholders actually referenced in the body.  This is the same
        # invariant the test suite checks per-template, enforced here so a
        # hand-edited template can never silently drift.
        actual = _placeholders_in(self.body)
        declared = set(self.placeholders)
        if actual != declared:
            missing = actual - declared
            extra = declared - actual
            raise PromptLibraryError(
                f"template {self.key!r} placeholder mismatch — "
                f"in body but not declared: {sorted(missing)}; "
                f"declared but not in body: {sorted(extra)}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholders_in(body: str) -> set[str]:
    """Return the set of ``{name}`` placeholder names referenced in *body*.

    Uses :class:`string.Formatter` field-name parsing so only real field
    references are counted, not literal braces in prose.
    """
    import string

    return {name for _, name, _, _ in string.Formatter().parse(body) if name}


# ---------------------------------------------------------------------------
# Curated templates
# ---------------------------------------------------------------------------
#
# Every body below carries the same backbone of VR180-friendly constraints:
#   - one continuous shot, no cuts / no flashbacks
#   - slow, constant-speed camera motion (fast motion flickers depth → nausea)
#   - stable lighting, no abrupt brightness swings
#   - clear foreground / mid-ground / background layering (depth needs layers)
#   - avoid large mirrors / glass / fog (depth estimation breaks on these)
# The per-template variation is the camera motion + how the subject sits in
# the frame.  Durations are tuned to the seedance 5s/10s tiers.

_CONSTRAINT_TAIL = (
    "single continuous shot with no cuts, jump-cuts or flashbacks; "
    "slow and constant-speed camera motion throughout; stable even lighting "
    "with no abrupt brightness changes; clear foreground, mid-ground and "
    "background depth layering; avoid large mirrors, transparent glass and "
    "dense fog; photorealistic, sharp focus, 8K"
)

_TEMPLATES_DEF: list[PromptTemplate] = [
    PromptTemplate(
        key="locked_static",
        summary="Static locked-off shot — minimal motion, lets depth + stereo settle.",
        body=(
            "A locked-off static camera holds steady on {subject} in {setting}. "
            "No camera movement at all — the frame breathes only through subtle "
            "natural motion in the scene (water ripple, leaves, drifting clouds). " + _CONSTRAINT_TAIL
        ),
        camera_motion="static",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="slow_dolly_in",
        summary="Slow forward dolly toward subject — builds presence without depth flicker.",
        body=(
            "A slow, constant-speed forward dolly pushes gently toward {subject} "
            "set in {setting}. The camera advances at an unhurried walking pace, "
            "keeping {subject} roughly framed and the horizon level. " + _CONSTRAINT_TAIL
        ),
        camera_motion="dolly_in",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="slow_dolly_out",
        summary="Slow pull-back dolly — reveals context while preserving stereo layers.",
        body=(
            "A slow, constant-speed backward dolly pulls gently away from {subject} "
            "in {setting}, gradually revealing the surrounding environment. The "
            "retreat is smooth and even, horizon stays level, {subject} remains "
            "the anchor of the frame. " + _CONSTRAINT_TAIL
        ),
        camera_motion="dolly_out",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="orbit_left",
        summary="Slow leftward orbit around subject — parallax that stereo can follow.",
        body=(
            "A slow, constant-speed orbital camera arcs leftward around {subject} "
            "in {setting}, holding a near-constant distance and a level horizon "
            "throughout the pass. {subject} stays framed at centre. " + _CONSTRAINT_TAIL
        ),
        camera_motion="orbit_left",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="orbit_right",
        summary="Slow rightward orbit around subject — mirror of orbit_left for variety.",
        body=(
            "A slow, constant-speed orbital camera arcs rightward around {subject} "
            "in {setting}, holding a near-constant distance and a level horizon "
            "throughout the pass. {subject} stays framed at centre. " + _CONSTRAINT_TAIL
        ),
        camera_motion="orbit_right",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="pan_slow",
        summary="Slow lateral pan across scene — reveals layered depth left-to-right.",
        body=(
            "A slow, constant-speed horizontal pan glides left-to-right across "
            "{setting}, sweeping past {subject} without stopping. The pan is even "
            "and gentle, horizon held level, no vertical wobble. " + _CONSTRAINT_TAIL
        ),
        camera_motion="pan_slow",
        recommended_duration=8.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="tilt_up",
        summary="Slow upward tilt — reveals vertical scale (sky / architecture / trees).",
        body=(
            "A slow, constant-speed upward tilt raises the camera gaze from the "
            "base of {subject} up through {setting}, revealing vertical scale. The "
            "tilt is smooth and even, with no jerky acceleration. " + _CONSTRAINT_TAIL
        ),
        camera_motion="tilt_up",
        recommended_duration=8.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="push_through",
        summary="Slow push-through an opening — strong stereo parallax, depth-friendly.",
        body=(
            "A slow, constant-speed forward push carries the camera through {setting}, "
            "passing a foreground layer {subject} on one side as it advances into "
            "the space beyond. Motion is steady and deliberate, foreground elements "
            "parallax smoothly against the background. " + _CONSTRAINT_TAIL
        ),
        camera_motion="push_through",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
    PromptTemplate(
        key="aerial_descent",
        summary="Slow aerial descent toward subject — grand reveal, stable horizon.",
        body=(
            "A slow, constant-speed aerial descent lowers the camera toward {subject} "
            "in {setting}, keeping the horizon level and the descent smooth. The "
            "approach is gentle so depth structure stays stable as ground detail "
            "resolves. " + _CONSTRAINT_TAIL
        ),
        camera_motion="dolly_in",
        recommended_duration=10.0,
        placeholders=("subject", "setting"),
    ),
]

TEMPLATES: dict[str, PromptTemplate] = {t.key: t for t in _TEMPLATES_DEF}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_templates(*, camera_motion: str | None = None) -> list[PromptTemplate]:
    """List templates, optionally filtered by ``camera_motion``.

    Returned in stable definition order (not insertion-stable-by-dict, but the
    order templates are declared in ``_TEMPLATES_DEF``).  ``camera_motion``
    matching is exact; an unrecognized motion simply yields an empty list.
    """
    if camera_motion is None:
        return list(_TEMPLATES_DEF)
    return [t for t in _TEMPLATES_DEF if t.camera_motion == camera_motion]


def get_template(key: str) -> PromptTemplate:
    """Return the template for *key*.

    Raises :class:`KeyError` with a message listing every available key when
    *key* is not registered — so a typo surfaces immediately instead of
    silently rendering an empty string.
    """
    try:
        return TEMPLATES[key]
    except KeyError:
        available = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"unknown prompt template key: {key!r}. available keys: {available}") from None


def render(key: str, **fields: str) -> str:
    """Render template *key* with the given placeholder *fields*.

    Parameters
    ----------
    key : str
        A registered template key (see :func:`get_template`).
    **fields : str
        Values for every placeholder declared by the template.

    Returns
    -------
    str
        The fully rendered prompt text — safe to pass directly to
        :func:`pipeline.prompt_builder.wrap_prompt_for_vr180`.

    Raises
    ------
    PromptLibraryError
        If a required placeholder is missing, or if an unexpected field is
        supplied (to prevent a typo'd field name from being silently
        swallowed by :meth:`str.format`).
    KeyError
        If *key* is not a registered template (via :func:`get_template`).
    """
    template = get_template(key)

    declared = set(template.placeholders)
    supplied = set(fields)

    missing = declared - supplied
    if missing:
        raise PromptLibraryError(
            f"render({key!r}) missing required placeholder(s): "
            f"{sorted(missing)}. declared placeholders: {sorted(declared)}"
        )

    extra = supplied - declared
    if extra:
        raise PromptLibraryError(
            f"render({key!r}) got unexpected field(s): {sorted(extra)}. declared placeholders: {sorted(declared)}"
        )

    return template.body.format(**fields)
