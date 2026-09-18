"""FastAPI gateway for Immersive Node Studio (M0)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from studio.graph import GraphError, extract_gallery, list_node_types, run_graph
from studio.models import Project, StudioModelError, empty_demo_project
from studio.projects import ProjectStore, ProjectStoreError
from studio.templates import production_pipeline_project

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ProjectPayload(BaseModel):
    project: dict[str, Any]
    work_dir: str | None = Field(default=None, description="Optional absolute output directory")


class RunRequest(ProjectPayload):
    only_downstream_of: str | None = None
    dirty_from: str | None = None


class SaveProjectRequest(BaseModel):
    project: dict[str, Any]
    project_id: str | None = None


def create_app(*, default_work_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Immersive Node Studio", version="0.1.0")
    work_root = Path(default_work_dir) if default_work_dir else Path(tempfile.gettempdir()) / "vr180-studio"
    work_root.mkdir(parents=True, exist_ok=True)
    run_cache: dict[str, dict[str, Any]] = {}
    project_store = ProjectStore(work_root / "projects")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "work_root": str(work_root), "projects_root": str(project_store.root)}

    @app.get("/api/node-types")
    def node_types() -> list[dict[str, Any]]:
        return list_node_types()

    @app.get("/api/demo-project")
    def demo_project() -> dict[str, Any]:
        return empty_demo_project().to_dict()

    @app.get("/api/templates/dual-export")
    def template_dual_export() -> dict[str, Any]:
        # kept for older clients
        return production_pipeline_project().to_dict()

    @app.get("/api/templates/production")
    def template_production() -> dict[str, Any]:
        return production_pipeline_project().to_dict()

    @app.post("/api/validate")
    def validate(payload: ProjectPayload) -> dict[str, Any]:
        try:
            project = Project.from_dict(payload.project)
        except StudioModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "name": project.name, "nodes": len(project.nodes), "edges": len(project.edges)}

    @app.post("/api/run")
    def run(request: RunRequest) -> dict[str, Any]:
        try:
            project = Project.from_dict(request.project)
        except StudioModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        work_dir = Path(request.work_dir) if request.work_dir else work_root
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"invalid work_dir: {exc}") from exc
        if not work_dir.is_absolute():
            raise HTTPException(status_code=400, detail="work_dir must be absolute")

        try:
            report = run_graph(
                project,
                work_dir=str(work_dir),
                only_downstream_of=request.only_downstream_of,
                dirty_from=request.dirty_from,
                cache=run_cache,
            )
        except GraphError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = report.to_dict()
        payload["gallery"] = extract_gallery(report)
        return payload

    @app.get("/api/projects")
    def list_saved_projects() -> list[dict[str, Any]]:
        return project_store.list_projects()

    @app.post("/api/projects")
    def save_project(req: SaveProjectRequest) -> dict[str, Any]:
        try:
            return project_store.save(req.project, req.project_id)
        except ProjectStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def load_project(project_id: str) -> dict[str, Any]:
        try:
            return project_store.load(project_id)
        except ProjectStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        ok = project_store.delete(project_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
        return {"deleted": project_id}

    # Serve assets at BOTH /static/* (absolute) and /* (relative) so
    # index.html works from the FastAPI root and from file:// next to the files.
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

        @app.get("/style.css")
        def style_css() -> FileResponse:
            return FileResponse(STATIC_DIR / "style.css", media_type="text/css")

        @app.get("/app.js")
        def app_js() -> FileResponse:
            return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")

        @app.get("/preview3d.js")
        def preview3d_js() -> FileResponse:
            return FileResponse(STATIC_DIR / "preview3d.js", media_type="application/javascript")

        @app.get("/dome3d.html")
        def dome3d_html() -> FileResponse:
            return FileResponse(STATIC_DIR / "dome3d.html", headers={"Cache-Control": "no-store"})

    return app


app = create_app()


def main() -> None:
    """Dev entry: ``python -m studio.server``."""
    import uvicorn

    uvicorn.run("studio.server:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
