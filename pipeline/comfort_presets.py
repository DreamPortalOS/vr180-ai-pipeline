"""I-3: Stereo comfort presets — owner-tuned low-parallax tiers.

Owner real-device feedback on the first VR180 samples was blunt: *"非常晕，两只
眼睛对不上"* ("very dizzy, the two eyes can't converge").  Two levers fix the
symptom — depth quality (covered by #77/#83) and, independently, **parallax
strength + convergence plane**.  This module is the second lever: a set of
matched ``max_disparity`` / ``convergence`` / ``temporal_smooth`` tiers, switched
with a single ``--comfort {safe,balanced,strong}`` flag so the owner can A/B them
in the headset without juggling three numbers.

Three tiers, calibrated against the observed comfort cliff:

- ``safe``     — ``max_disparity=0.02``, ``convergence=0.5``.  The "久看不晕"
  (watch-long-no-nausea) tier: weak stereo pop, but the convergence plane is far
  (0.5 → most of the scene recedes behind the screen), so the eyes barely
  diverge.  Use this when depth is unstable or the viewer is stereo-sensitive.
  This was the original StereoRenderer default ``max_disparity``.
- ``balanced`` — ``max_disparity=0.035``, ``convergence=0.35``.  The new default.
  A middle ground: enough stereo to read 3D structure, convergence pulled in
  enough that mid-field objects sit near the screen plane (low divergence).
  Chosen because the owner's 0.06 sample was *clearly* too strong when depth was
  shaky, but 0.02 read as nearly flat — 0.035 is the midpoint that survived a
  headset session.
- ``strong``   — ``max_disparity=0.06``, ``convergence=0.2``.  The "立体强"
  (strong stereo) tier: matches the sample the owner found punchy.  Convergence
  near (0.2 → much of the scene pops *out* of the screen) so only use this when
  depth quality is solid (DepthCrafter / StereoCrafter backends); on shaky
  per-frame depth it reproduces the original "eyes can't converge" complaint.

All three tiers keep ``temporal_smooth=True`` — flickering disparity is a
comfort killer independent of magnitude.

Resolution rule (enforced by :func:`resolve_comfort`): a preset only fills in
values the caller did *not* explicitly pass.  An explicit ``--max-disparity`` (or
``--convergence`` / ``--no-temporal``) always wins, so a preset is a starting
point, not a cage.  This mirrors the ``--quality`` preset precedence in
:mod:`pipeline.streaming_pipeline`.

This module is pure-Python (stdlib only) so it imports on CI with no models /
no ffmpeg.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Each preset maps 1:1 onto StereoRenderer's __init__ kwargs.  Keys are kept
# tight (the three comfort-relevant knobs) so resolve_comfort stays predictable:
# adding a fourth knob here is a one-line change with no special-casing.
COMFORT_PRESETS: dict[str, dict[str, Any]] = {
    "safe": {
        "max_disparity": 0.02,
        "convergence": 0.5,
        "temporal_smooth": True,
    },
    "balanced": {
        "max_disparity": 0.035,
        "convergence": 0.35,
        "temporal_smooth": True,
    },
    "strong": {
        "max_disparity": 0.06,
        "convergence": 0.2,
        "temporal_smooth": True,
    },
}

# The preset applied when the caller does not name one.  ``balanced`` per the
# card: the owner's 0.06 sample was too strong on shaky depth, 0.02 too flat.
DEFAULT_COMFORT = "balanced"


def resolve_comfort(name: str | None, explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a comfort preset, letting *explicit* overrides win per-key.

    Args:
        name: preset name (``"safe"`` / ``"balanced"`` / ``"strong"``).  When
            ``None``, :data:`DEFAULT_COMFORT` is used so callers can pass the
            CLI default straight through without a special case.
        explicit: caller-supplied overrides.  Any key present here (even
            ``None``-valued ones the caller wants to "unset") replaces the
            preset value.  Only the three comfort knobs are recognised —
            ``max_disparity`` / ``convergence`` / ``temporal_smooth`` — but
            unknown keys are passed through untouched so the dict stays a
            drop-in for ``StereoRenderer(**resolved)``-style construction.

    Returns:
        A fresh dict (never the preset itself) of resolved comfort values.

    Raises:
        ValueError: if *name* is not ``None`` and not in :data:`COMFORT_PRESETS`.
    """
    if name is None:
        name = DEFAULT_COMFORT
    if name not in COMFORT_PRESETS:
        valid = ", ".join(sorted(COMFORT_PRESETS))
        raise ValueError(f"unknown comfort preset: {name!r} — choose from {valid}")

    # Copy so the caller can mutate the result without touching the preset.
    resolved = dict(COMFORT_PRESETS[name])
    if explicit:
        resolved.update(explicit)
    return resolved
