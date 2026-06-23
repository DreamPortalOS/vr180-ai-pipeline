"""VR180 Prompt Builder — wraps user prompts with VR180-friendly constraints.

Based on docs/PROMPT_GUIDE_VR180.md:
- Appends VR180 motion/composition/quality constraints to user prompt
- Does NOT modify or delete user's original content
- Returns positive and negative prompts separated for API compatibility
- Supports scene_type: fpv / walkthrough / orbit / static
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


@dataclass
class PromptResult:
    """Result of VR180 prompt wrapping."""

    positive: str
    negative: str

    def to_dict(self) -> dict[str, str]:
        return {"positive": self.positive, "negative": self.negative}


def wrap_prompt_for_vr180(
    user_prompt: str,
    scene_type: str = "fpv",
) -> dict[str, str]:
    """Wrap a user prompt with VR180-friendly constraints.

    Parameters
    ----------
    user_prompt : str
        The original creative prompt from the user. This is preserved as-is
        and only appended to with VR180 constraints.
    scene_type : str
        One of 'fpv', 'walkthrough', 'orbit', 'static'. Determines the
        motion template applied. Falls back to 'fpv' for unknown types.

    Returns
    -------
    dict
        {"positive": str, "negative": str} — positive prompt and negative
        prompt, ready for API consumption.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")

    scene = scene_type.lower().strip()
    template = _SCENE_TEMPLATES.get(scene, _SCENE_TEMPLATES[_DEFAULT_SCENE])

    # Build positive prompt: user's original text + motion + composition + quality
    parts = [user_prompt.strip()]
    parts.extend(template["motion"])
    parts.extend(template["composition"])
    parts.extend(template["quality"])

    positive = ", ".join(parts)

    # Build negative prompt
    negative = ", ".join(template["negative"])

    return PromptResult(positive=positive, negative=negative).to_dict()
