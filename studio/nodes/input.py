"""Input nodes — text / image / video sources for the canvas (issue #417).

These are the ``category="input"`` nodes the operator creates by dragging a
file onto the canvas (``input.image`` / ``input.video``) or typing into the
card (``input.text``). They are pure pass-throughs with validation, so a
project document can be a genuine starting point for a graph instead of
needing a ``script.storyboard`` node to produce the first value.

Only the *path* lands in project JSON (never file contents) — the bytes live
in ``work_root/uploads/``, de-duplicated by content hash (see
``studio.uploads``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode
from studio.uploads import IMAGE_EXTS, VIDEO_EXTS


class InputTextNode(StudioNode):
    type_name = "input.text"
    category = "input"
    label = "文本输入"
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = (PortSpec(name="text", type="text"),)

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "label", "type": "string", "default": "input", "label": "标签"},
            {"name": "text", "type": "string", "default": "", "label": "文本"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = inputs, work_dir, node_id
        text = str(params.get("text") or "").strip()
        if not text:
            raise ValueError("input.text requires non-empty text")
        return {"text": text}


class _PathInputNode(StudioNode):
    """Shared machinery for input.image / input.video."""

    @classmethod
    def _resolve(cls, params: dict[str, Any], inputs: dict[str, Any], work_dir: str) -> Path:
        raw = params.get("path") or inputs.get("value")
        if not raw:
            raise ValueError(f"{cls.type_name} requires a file path (params.path or input 'value')")
        path = Path(str(raw))
        if not path.is_absolute():
            # Relative paths are resolved against the run's work_dir so a
            # project JSON saved on one machine stays portable.
            path = Path(work_dir) / path
        if not path.is_file():
            raise FileNotFoundError(f"{cls.type_name}: source not found: {path}")
        return path

    @staticmethod
    def _suffix_ok(path: Path, allowed: frozenset[str]) -> None:
        suffix = path.suffix.lower()
        if suffix not in allowed:
            raise ValueError(f"{suffix or '(none)'} is not an accepted input type; allowed: {sorted(allowed)}")


class InputImageNode(_PathInputNode):
    type_name = "input.image"
    category = "input"
    label = "图片输入"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="value", type="image"),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="image", type="image"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "path", "type": "string", "default": "", "label": "图片路径"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = node_id
        path = self._resolve(params, inputs, work_dir)
        self._suffix_ok(path, IMAGE_EXTS)
        meta: dict[str, Any] = {"path": str(path), "kind": "image"}
        try:
            from PIL import Image

            with Image.open(path) as img:
                meta["width"] = int(img.width)
                meta["height"] = int(img.height)
        except Exception:
            meta["width"] = meta["height"] = None
        meta["size_bytes"] = path.stat().st_size
        return {"image": str(path), "meta": meta}


class InputVideoNode(_PathInputNode):
    type_name = "input.video"
    category = "input"
    label = "视频输入"
    inputs: tuple[PortSpec, ...] = (PortSpec(name="value", type="video"),)
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "path", "type": "string", "default": "", "label": "视频路径"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = node_id
        path = self._resolve(params, inputs, work_dir)
        self._suffix_ok(path, VIDEO_EXTS)
        meta: dict[str, Any] = {"path": str(path), "kind": "video"}
        try:
            from studio.uploads import probe_video

            # Reuse the upload store's ffmpeg probe so the card's poster frame
            # + duration/resolution readout matches what the server reported
            # at upload time — one algorithm, not two. The poster frame lands
            # under this node's own studio_out dir (not a stray uploads/ dir).
            poster_dir = Path(work_dir) / "studio_out" / node_id
            w, h, dur, poster_path = probe_video(path, poster_dir=poster_dir)
            meta["width"] = w or None
            meta["height"] = h or None
            meta["duration"] = dur or None
            meta["poster"] = poster_path or None
        except Exception:
            meta.setdefault("width", None)
            meta.setdefault("height", None)
            meta.setdefault("duration", None)
            meta.setdefault("poster", None)
        meta["size_bytes"] = path.stat().st_size
        return {"video": str(path), "meta": meta}
