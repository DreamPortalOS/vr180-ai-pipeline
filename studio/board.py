"""Manual storyboard board (分镜板) — text → image per shot, operator-driven.

The owner wants the basic loop working before any automatic shot splitting:
write a prompt per shot, generate a few candidate stills, pick one, adjust the
camera move and duration, then render the shot to video.  This module holds
the server-side pieces the drawer calls; the ``storyboard.board`` node only
re-emits what the operator built so downstream nodes (video / concat / dome
convert) keep working unchanged.

Every gateway call is injectable (``client``) so CI never touches the network.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

#: Board aspect presets: key → (API aspect ratio, prompt constraint).
#: ``dome`` is a 1:1 domemaster; its constraint is the fixed header the lead
#: uses for every Gemini / gateway dome prompt (DOME_SCRIPT_V1.md).
DOME_CONSTRAINT = (
    "Fisheye fulldome domemaster, 1:1 square frame, circular image area inscribed in the square, "
    "corners pure black. Image centre is the zenith (straight up); the circle edge is the horizon; "
    "the bottom of the circle is the direction the audience faces. The scene fills the whole circle "
    "out to the rim."
)
FLAT_CONSTRAINT = "cinematic still, wide field of view, rich depth layers, level horizon"
VR180_CONSTRAINT = "wide 2:1 panoramic still, level horizon, subject centred, rich depth layers"

BOARD_ASPECTS: dict[str, tuple[str, str]] = {
    "dome": ("1:1", DOME_CONSTRAINT),
    "16:9": ("16:9", FLAT_CONSTRAINT),
    "2:1": ("2:1", VR180_CONSTRAINT),
}

MOTIONS = ("static", "dolly_in", "dolly_out", "pan_left", "pan_right")

MAX_VARIANTS = 4
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


def board_prompt(prompt: str, aspect: str = "dome", style: str = "") -> str:
    """The prompt actually sent: aspect constraint + shot text + optional style."""
    _, constraint = BOARD_ASPECTS.get(aspect, BOARD_ASPECTS["dome"])
    body = str(prompt or "").strip()
    if not body:
        raise ValueError("分镜描述为空：请先写这一镜要画什么")
    parts = [constraint, body]
    style = str(style or "").strip()
    if style and style not in body:
        parts.append(style)
    return " ".join(p.rstrip(".") + "." for p in parts)


def safe_shot_id(shot_id: str) -> str:
    cleaned = _SAFE_ID.sub("_", str(shot_id or "")).strip("_")[:40]
    return cleaned or "shot"


def generate_shot_images(
    *,
    prompt: str,
    aspect: str,
    n: int,
    out_dir: Path,
    shot_id: str,
    style: str = "",
    settings: Any = None,
    client: Any = None,
    model: str = "",
) -> dict[str, Any]:
    """Generate ``n`` candidate stills for one shot.

    Returns ``{"images": [...], "errors": [...], "provider": "gateway"|"mock",
    "prompt": <sent prompt>}``.  With no gateway configured and no injected
    client, writes labelled placeholders and says so (``provider="mock"``) —
    the drawer shows the same "未配置出图网关" warning as the run path.
    """
    from studio.nodes.image_gateway import DEFAULT_IMAGE_MODEL, GatewayImageClient, ImageJob, size_for_aspect

    n = max(1, min(MAX_VARIANTS, int(n or 1)))
    full = board_prompt(prompt, aspect, style)
    ratio, _ = BOARD_ASPECTS.get(aspect, BOARD_ASPECTS["dome"])
    sid = safe_shot_id(shot_id)
    batch = uuid.uuid4().hex[:6]
    out_dir.mkdir(parents=True, exist_ok=True)

    if client is None and settings is not None and settings.litellm_base_url and settings.litellm_api_key:
        client = GatewayImageClient(
            settings.litellm_base_url,
            settings.litellm_api_key,
            model=model or DEFAULT_IMAGE_MODEL,
            size=size_for_aspect(ratio),
        )
    if client is None:
        from studio.nodes.production import _write_placeholder_png

        w, h = (int(x) for x in size_for_aspect(ratio, long_side=512).split("x"))
        images = []
        for v in range(1, n + 1):
            path = out_dir / f"{sid}_{batch}_v{v}.png"
            _write_placeholder_png(path, w=w, h=h, seed=v + len(sid), label=str(prompt)[:40])
            images.append(str(path))
        return {"images": images, "errors": [], "provider": "mock", "prompt": full}

    jobs = [
        ImageJob(key=f"{sid}|{v}", prompt=full, out_path=out_dir / f"{sid}_{batch}_v{v}.png") for v in range(1, n + 1)
    ]
    results = client.generate_many(jobs, concurrency=min(n, 3))
    return {
        "images": [r.path for r in results if r.path],
        "errors": [r.error for r in results if r.error],
        "provider": "gateway",
        "prompt": full,
    }


def board_to_stills(shots: list[dict[str, Any]], aspect: str = "dome") -> dict[str, Any]:
    """Board params → the ``stills`` json shape downstream nodes consume.

    Only shots with a picked image are emitted, in board order; the rest are
    listed under ``pending`` so the node can report them instead of failing.
    """
    out: list[dict[str, Any]] = []
    pending: list[str] = []
    for idx, shot in enumerate(shots or []):
        if not isinstance(shot, dict):
            continue
        sid = str(shot.get("id") or f"shot_{idx + 1:02d}")
        image = shot.get("image")
        if not image or not Path(str(image)).is_file():
            pending.append(sid)
            continue
        motion = str(shot.get("motion") or "static")
        out.append(
            {
                "id": sid,
                "index": idx,
                "description": str(shot.get("prompt") or ""),
                "image": str(image),
                "duration": float(shot.get("duration") or 4),
                "motion": motion if motion in MOTIONS else "static",
                "variants": [str(v) for v in shot.get("variants") or []],
                "video": shot.get("video"),
            }
        )
    ratio, _ = BOARD_ASPECTS.get(aspect, BOARD_ASPECTS["dome"])
    return {"shots": out, "pending": pending, "summary": {"aspect_ratio": ratio, "board_aspect": aspect}}
