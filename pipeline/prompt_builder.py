"""Prompt Builder — wraps user prompts with output-target-aware constraints.

Supports three output targets:
- vr180_flight  : VR180 stereoscopic (headset), ~120° FOV, strict anti-nausea
- fulldome_180  : Monoscopic dome cinema (flight theatre), ~150-180° FOV, relaxed
- vr360_dome    : Full 360° equirectangular (omni-directional), notes for backend

For backward compatibility, wrap_prompt_for_vr180() delegates to wrap_prompt()
with target="vr180_flight" and returns the same dict structure as before.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Scene-type templates: motion descriptors + scene-specific constraints
# ---------------------------------------------------------------------------

_SCENE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "fpv": {
        "motion": [
            "first-person FPV view",
            "smooth continuous forward motion at moderate speed",
            "level and stable horizon",
            "stable low altitude",
        ],
        "composition": [
            "rich depth layers (foreground/mid-ground/background)",
            "main subject centered",
            "wide cinematic ~120° field of view",
        ],
        "quality": [
            "soft natural lighting",
            "ultra-detailed",
            "sharp focus",
            "8K",
            "photorealistic",
        ],
        "negative": [
            "rapid turns",
            "barrel rolls",
            "camera shake",
            "motion blur",
            "sudden cuts",
            "extreme close-ups at frame edges",
            "flat composition",
            "fast zoom",
            "rushing past frame edges",
        ],
    },
    "walkthrough": {
        "motion": [
            "first-person walkthrough perspective",
            "smooth steady walking pace",
            "continuous forward movement",
            "level and stable horizon",
        ],
        "composition": [
            "rich depth layers (foreground/mid-ground/background)",
            "main subject centered in view",
            "wide cinematic ~100° field of view",
        ],
        "quality": [
            "soft ambient lighting",
            "ultra-detailed",
            "sharp focus",
            "8K",
            "photorealistic",
        ],
        "negative": [
            "running motion",
            "camera shake",
            "motion blur",
            "sudden cuts",
            "teleporting",
            "jittery movement",
            "flat composition",
            "extreme fisheye",
        ],
    },
    "orbit": {
        "motion": [
            "smooth orbital camera path",
            "continuous circular motion around subject",
            "constant distance from subject",
            "level and stable horizon",
            "moderate rotation speed",
        ],
        "composition": [
            "main subject at center of frame",
            "rich depth layers (foreground/mid-ground/background)",
            "wide cinematic field of view",
        ],
        "quality": [
            "dramatic lighting",
            "ultra-detailed",
            "sharp focus",
            "8K",
            "photorealistic",
        ],
        "negative": [
            "erratic orbit",
            "camera shake",
            "motion blur",
            "sudden zoom",
            "subject leaving frame",
            "spinning",
            "barrel rolls",
            "flat composition",
        ],
    },
    "static": {
        "motion": [
            "locked-off camera position",
            "completely static camera",
            "no camera movement",
        ],
        "composition": [
            "rich depth layers (foreground/mid-ground/background)",
            "main subject centered",
            "wide cinematic field of view",
        ],
        "quality": [
            "cinematic lighting",
            "ultra-detailed",
            "sharp focus",
            "8K",
            "photorealistic",
        ],
        "negative": [
            "camera movement",
            "camera shake",
            "motion blur",
            "sudden cuts",
            "zooming",
            "panning",
            "flat composition",
        ],
    },
}

# Fallback for unknown scene types
_DEFAULT_SCENE = "fpv"

# ---------------------------------------------------------------------------
# Target-specific templates overriding / amending base scene templates
# ---------------------------------------------------------------------------

# Valid target values
_VALID_TARGETS = ("vr180_flight", "fulldome_180", "vr360_dome")
_DEFAULT_TARGET = "vr180_flight"

_TARGET_TEMPLATES: dict[str, dict[str, list[str] | str]] = {
    "fulldome_180": {
        "composition_overrides": [
            "ultra-wide cinematic ~150-180° field of view",
        ],
        "negative_exclude": [
            "extreme close-ups at frame edges",
            "rushing past frame edges",
            "barrel rolls",
        ],
        "extra_positive": [
            "stable horizon throughout",
        ],
        "notes": "",
    },
    "vr360_dome": {
        "composition_overrides": [
            "full 360° equirectangular spherical coverage",
            "seamless wraparound environment",
            "omni-directional field of view",
        ],
        "extra_positive": [
            "continuous 360° surround view",
        ],
        "negative_exclude": [],
        "notes": (
            "NOTE: Most mainstream 2D video models do not natively support "
            "360° equirectangular output. This prompt is best used with a "
            "360-capable backend, multi-view generation, or AI outpainting "
            "to fill the unseen rear hemisphere."
        ),
    },
}


@dataclass
class PromptResult:
    """Result of VR180 prompt wrapping."""

    positive: str
    negative: str

    def to_dict(self) -> dict[str, str]:
        return {"positive": self.positive, "negative": self.negative}


@dataclass
class ExtendedPromptResult:
    """Extended result with target and notes metadata."""

    positive: str
    negative: str
    target: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "target": self.target,
            "notes": self.notes,
        }


def wrap_prompt(
    user_prompt: str,
    scene_type: str = "fpv",
    target: str = "vr180_flight",
) -> dict[str, str]:
    """Wrap a user prompt with output-target-aware constraints.

    Parameters
    ----------
    user_prompt : str
        The original creative prompt from the user. Preserved as-is.
    scene_type : str
        One of 'fpv', 'walkthrough', 'orbit', 'static'.
    target : str
        One of 'vr180_flight', 'fulldome_180', 'vr360_dome'.
        Unknown targets fall back to 'vr180_flight'.

    Returns
    -------
    dict
        {"positive": str, "negative": str, "target": str, "notes": str}
        — positive prompt, negative prompt, target identifier, and notes.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")

    scene = scene_type.lower().strip()
    template = _SCENE_TEMPLATES.get(scene, _SCENE_TEMPLATES[_DEFAULT_SCENE])

    target = target.lower().strip()
    if target not in _VALID_TARGETS:
        target = _DEFAULT_TARGET

    target_conf = _TARGET_TEMPLATES.get(target, {})

    # Build positive prompt
    parts = [user_prompt.strip()]
    parts.extend(template["motion"])

    # Use target-specific composition overrides if present, else base
    composition_overrides = target_conf.get("composition_overrides", [])
    if composition_overrides:
        parts.extend(composition_overrides)
    else:
        parts.extend(template["composition"])

    parts.extend(template["quality"])
    extra_positive = target_conf.get("extra_positive", [])
    parts.extend(extra_positive)

    positive = ", ".join(parts)

    # Build negative prompt: start from base, exclude target-specific items
    negative_terms = list(template["negative"])
    negative_exclude = target_conf.get("negative_exclude", [])
    for term in negative_exclude:
        while term in negative_terms:
            negative_terms.remove(term)

    negative = ", ".join(negative_terms)

    notes = str(target_conf.get("notes", ""))

    return ExtendedPromptResult(positive=positive, negative=negative, target=target, notes=notes).to_dict()


def wrap_prompt_for_vr180(
    user_prompt: str,
    scene_type: str = "fpv",
) -> dict[str, str]:
    """Legacy alias — delegates to wrap_prompt(..., target='vr180_flight').

    Returns the same dict keys {"positive", "negative"} as before for
    backward compatibility with existing callers and tests.

    Parameters
    ----------
    user_prompt : str
        The original creative prompt from the user.
    scene_type : str
        One of 'fpv', 'walkthrough', 'orbit', 'static'.

    Returns
    -------
    dict
        {"positive": str, "negative": str}
    """
    full = wrap_prompt(user_prompt, scene_type=scene_type, target="vr180_flight")
    return {"positive": full["positive"], "negative": full["negative"]}
