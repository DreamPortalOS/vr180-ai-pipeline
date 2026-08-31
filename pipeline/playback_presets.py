"""D-2: downstream playback presets — PCVR / Quest standalone / source.

DreamPortal (the VR playback consumer) targets two delivery platforms whose
video specs differ (see ``Docs/R1_Video_Pipeline_Plan.md`` in the DreamPortal
repo):

  =============== =========================== ======== ============ ==========================
  tier             target                       codec    bitrate      notes
  =============== =========================== ======== ============ ==========================
  ``pcvr``         RTX 4070S + Quest Link      HEVC     high         PC NVDEC headroom big;
                                                          (CRF 18)    quality first
  ``standalone``   Quest 3 (XR2 Gen2)           HEVC     mid          standalone decode cap
                                                          (CRF 23)    8192x4096@60; control
                                                                       bitrate + power
  ``source``       (passthrough / dev)         source   source       no re-encode tuning;
                                                                       keeps --codec/--crf as-is
  =============== =========================== ======== ============ ==========================

Key constraints surfaced by real-device testing on the playback side
(encapsulated here so the encode stages stay dumb):

  1. **Segmented files > ultra-long video.**  The playback end retired the
     "one long video + time-pointer seek" design in favour of segmented files
     + dual-MediaPlayer crossfade.  *This pipeline keeps emitting single short
     clips* — do not merge.
  2. **GOP / keyframe interval drives seek precision.**  Fixed 1-second GOP,
     segment head MUST be an IDR frame.  Encoded as ``-g <fps> -keyint_min <fps>
     -sc_threshold 0`` plus ``-force_key_frames`` on the open GOP path.
  3. ``+faststart`` (moov atom前置) for fast start.  Already set by the encode
     stages; preserved by every preset.

The three presets below are a **starting point, not a cage**: an explicit
``--codec`` / ``--crf`` / ``--bitrate`` always wins (mirrors the
:mod:`pipeline.comfort_presets` and :mod:`pipeline.streaming_pipeline`
``--quality`` precedence rules).  The bitrate numbers are calibrated against
this repo's test corpus, not the downstream doc's unverified suggestions —
see the card note ("建议以 Quest 3 真机实测结果标定后再固化默认值").

This module is pure-Python (stdlib only) so it imports on CI with no models /
no ffmpeg.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Each preset carries the encode knobs a downstream playback target cares
# about.  Keys map 1:1 onto the ffmpeg command-builder's expectations:
#
#   codec        : 'h264' | 'h265' — forces the codec family (HEVC for both
#                 VR tiers; ``source`` leaves it to the caller).
#   crf          : constant-rate factor (quality-first).  Only used when the
#                 caller did NOT pass an explicit --bitrate.
#   bitrate      : None = use CRF; a string like '80M' = forced target.  Both
#                 VR tiers default to CRF (quality-first, no ceiling hunting)
#                 but an explicit --bitrate overrides as usual.
#   gop_seconds  : fixed GOP length in *seconds* (1s per the playback-side
#                 seek-precision guidance).  Translated to a frame count by
#                 the caller using the run fps.
#   force_idr    : force IDR at segment head (closed GOP) — the playback
#                 side needs a clean seek target.  Encoded as
#                 ``-force_key_frames`` when the path supports it.
#   faststart    : ``+faststart`` moov-前置 flag (always True for the VR tiers;
#                 ``source`` keeps whatever the caller set).
#
# ``source`` is the passthrough / dev tier: it fills in *nothing* so the raw
# --codec / --crf / --bitrate the operator typed flow straight through
# unchanged — useful for A/B and for reproducing the pre-D-2 behaviour.
PLAYBACK_PRESETS: dict[str, dict[str, Any]] = {
    "pcvr": {
        "codec": "h265",
        "crf": 18,
        "bitrate": None,
        "gop_seconds": 1,
        "force_idr": True,
        "faststart": True,
    },
    "standalone": {
        "codec": "h265",
        "crf": 23,
        "bitrate": None,
        "gop_seconds": 1,
        "force_idr": True,
        "faststart": True,
    },
    "source": {
        "codec": None,
        "crf": None,
        "bitrate": None,
        "gop_seconds": None,
        "force_idr": False,
        "faststart": None,
    },
}

# The preset applied when the caller does not name one.  ``source`` keeps the
# pre-D-2 behaviour bit-exact (no surprise codec/crf change), so existing
# runs and tests are unaffected unless the operator opts in.
DEFAULT_PLAYBACK = "source"

# Fixed GOP in seconds (the playback-side seek-precision guidance).  Exposed
# as a module constant so callers/tests can reference the documented value
# without re-deriving it from the preset dict.
DEFAULT_GOP_SECONDS = 1


def resolve_playback(name: str | None, explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a playback preset, letting *explicit* overrides win per-key.

    Args:
        name: preset name (``"pcvr"`` / ``"standalone"`` / ``"source"``).
            When ``None``, :data:`DEFAULT_PLAYBACK` is used so callers can
            pass the CLI default straight through without a special case.
        explicit: caller-supplied overrides.  Any key present here (even
            ``None``-valued ones the caller wants to "unset") replaces the
            preset value.  Recognised keys are ``codec`` / ``crf`` /
            ``bitrate`` / ``gop_seconds`` / ``force_idr`` / ``faststart``;
            unknown keys pass through untouched so the dict stays a drop-in
            for the command-builder.

    Returns:
        A fresh dict (never the preset itself) of resolved playback values.
        ``source`` resolves to an all-``None`` block (passthrough).

    Raises:
        ValueError: if *name* is not ``None`` and not in
            :data:`PLAYBACK_PRESETS`.
    """
    if name is None:
        name = DEFAULT_PLAYBACK
    if name not in PLAYBACK_PRESETS:
        valid = ", ".join(sorted(PLAYBACK_PRESETS))
        raise ValueError(f"unknown playback preset: {name!r} — choose from {valid}")

    # Copy so the caller can mutate the result without touching the preset.
    resolved = dict(PLAYBACK_PRESETS[name])
    if explicit:
        resolved.update(explicit)
    return resolved
