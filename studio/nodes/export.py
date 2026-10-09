"""Export node — copy artefacts and write a run summary under work_dir."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from studio.nodes.base import PortSpec, StudioNode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_paths(params: dict[str, Any], work_dir: str, node_id: str) -> tuple[Path, Path]:
    # Keep the established node directory and basename-only filename behavior.
    filename = Path(str(params.get("filename") or f"{node_id}.mp4")).name
    subdir = Path(str(params.get("subdir") or "studio_export")).name
    out_dir = Path(work_dir) / subdir / node_id
    return out_dir / filename, out_dir / "manifest.json"


def _signature(path: Path) -> tuple | None:
    if path.is_symlink():
        raise ValueError(f"export target must not be a symlink: {path}")
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, _sha256(path)


def _publish_bundle(source: Path, dest: Path, receipt: Path, prompt: Any) -> dict[str, Any]:
    """Preserve prior files on ordinary failures, not a crash-atomic transaction.

    The exclusive lock coordinates this exporter only. External noncooperating
    writes are checked before publish/rollback, not protected by a filesystem CAS.
    A failed rollback retains its private recovery directory for the operator.
    """
    if dest == receipt or dest.name == ".export.lock":
        raise ValueError("export filename conflicts with reserved bundle metadata")
    lock = dest.parent / ".export.lock"
    token = uuid.uuid4().hex
    stage = None
    preserve = False
    identity = None
    handle = lock.open("x", encoding="utf-8")
    try:
        stat = os.fstat(handle.fileno())
        identity = stat.st_dev, stat.st_ino
        handle.write(token)
        handle.close()
        stage = Path(tempfile.mkdtemp(prefix=".export-", dir=dest.parent))
        targets = (dest, receipt)
        before = [_signature(path) for path in targets]
        backups = [stage / "previous-video", stage / "previous-manifest"]
        for path, signature, backup in zip(targets, before, backups, strict=True):
            if signature is not None:
                shutil.copy2(path, backup)
        video = stage / "video"
        metadata = stage / "manifest.json"
        shutil.copy2(source, video)
        manifest = {
            "export_path": str(dest),
            "source_video": str(source),
            "prompt": prompt,
            "size_bytes": video.stat().st_size,
            "sha256": _sha256(video),
        }
        metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if [_signature(path) for path in targets] != before:
            raise RuntimeError("export targets changed before publication")
        os.replace(video, dest)
        installed = None
        try:
            # Windows may change identity metadata during replace. No further
            # payload hashing is required until rollback verifies ownership.
            stat = dest.stat()
            installed = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, manifest["sha256"]
            if _signature(receipt) != before[1]:
                raise RuntimeError("export manifest changed during publication")
            os.replace(metadata, receipt)
        except Exception as publish_error:
            try:
                if installed is None or _signature(dest) != installed:
                    raise RuntimeError("export video changed before rollback")
                if before[0] is None:
                    dest.unlink()
                else:
                    os.replace(backups[0], dest)
            except Exception as rollback_error:
                preserve = True
                raise RuntimeError(
                    f"export publication failed; recovery retained at {stage}: {rollback_error}"
                ) from publish_error
            raise
        return manifest
    finally:
        primary = sys.exc_info()[1]
        cleanup_errors = []
        try:
            if not handle.closed:
                handle.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        finally:
            try:
                if stage is not None and not preserve:
                    shutil.rmtree(stage)
            except Exception as exc:
                cleanup_errors.append(exc)
            finally:
                try:
                    # Identity, not token contents: a partial/failed write is
                    # still our lock, but an unknown replacement never is.
                    if identity is not None:
                        stat = lock.lstat()
                        if not lock.is_symlink() and (stat.st_dev, stat.st_ino) == identity:
                            lock.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(str(exc) for exc in cleanup_errors)
            raise RuntimeError(f"{primary or 'export cleanup failed'}; cleanup: {detail}; staging: {stage}") from (
                primary or cleanup_errors[0]
            )


class ExportBundleNode(StudioNode):
    type_name = "export.bundle"
    category = "export"
    label = "导出"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="video", type="video"),
        PortSpec(name="prompt", type="text"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="path", type="text"),
        PortSpec(name="manifest", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {"name": "filename", "type": "string", "default": "export.mp4", "label": "文件名"},
            {"name": "subdir", "type": "string", "default": "studio_export", "label": "子目录"},
        ]

    @classmethod
    def cache_is_valid(
        cls,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> bool:
        """Reuse only an export whose current files still match the receipt."""
        try:
            dest, manifest_path = _export_paths(params, work_dir, node_id)
            source = Path(str(inputs["video"]))
            manifest = outputs["manifest"]
            digest = manifest["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                return False
            expected = {
                "export_path": str(dest),
                "source_video": str(source),
                "prompt": inputs.get("prompt"),
                "size_bytes": dest.stat().st_size,
                "sha256": digest,
            }
            if outputs["path"] != str(dest) or manifest != {**expected, "manifest_path": str(manifest_path)}:
                return False
            disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            return disk == expected and _sha256(source) == digest and _sha256(dest) == digest
        except (OSError, ValueError, TypeError, KeyError):
            # Missing, old or malformed cache records must rematerialize safely.
            return False

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        video = inputs.get("video")
        if not video:
            raise ValueError("export node requires a video input")
        video_path = Path(str(video))
        if not video_path.is_file():
            raise ValueError(f"export video not found: {video_path}")

        dest, manifest_path = _export_paths(params, work_dir, node_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        manifest = _publish_bundle(video_path, dest, manifest_path, inputs.get("prompt"))
        manifest["manifest_path"] = str(manifest_path)
        return {"path": str(dest), "manifest": manifest}
