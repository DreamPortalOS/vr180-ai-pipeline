"""Studio project store — JSON files on disk (secrets never in project docs)."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from studio.models import Project, StudioModelError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ProjectStoreError(ValueError):
    pass


def _safe_id(name: str, project_id: str | None = None) -> str:
    raw = project_id or name or "untitled"
    cleaned = _SAFE_NAME.sub("_", raw).strip("._") or "untitled"
    return cleaned[:80]


class ProjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self.root / f"{project_id}.studio.json"

    def list_projects(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        suffix = ".studio.json"
        for path in sorted(self.root.glob("*" + suffix), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = Project.load(path).to_dict()
            except (StudioModelError, OSError, UnicodeDecodeError):
                continue
            st = path.stat()
            pid = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
            items.append(
                {
                    "id": pid,
                    "name": data.get("name") or pid,
                    "nodes": len(data.get("nodes") or []),
                    "edges": len(data.get("edges") or []),
                    "mtime": st.st_mtime,
                    "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    "path": str(path),
                }
            )
        return items

    def save(self, project_dict: dict[str, Any], project_id: str | None = None) -> dict[str, Any]:
        try:
            project = Project.from_dict(project_dict)
        except StudioModelError as exc:
            raise ProjectStoreError(str(exc)) from exc
        pid = _safe_id(project.name, project_id)
        path = self._path(pid)
        project.save(path)
        st = path.stat()
        return {
            "id": pid,
            "name": project.name,
            "path": str(path),
            "mtime": st.st_mtime,
            "nodes": len(project.nodes),
            "edges": len(project.edges),
        }

    def load(self, project_id: str) -> dict[str, Any]:
        pid = _safe_id(project_id)
        path = self._path(pid)
        if not path.is_file():
            raise ProjectStoreError(f"project not found: {pid}")
        return Project.load(path).to_dict()

    def delete(self, project_id: str) -> bool:
        pid = _safe_id(project_id)
        path = self._path(pid)
        if not path.is_file():
            return False
        path.unlink()
        return True
