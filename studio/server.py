"""FastAPI gateway for Immersive Node Studio (M0)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from studio.coverage import _downscale, analyze_frame
from studio.graph import GraphError, RunHistory, extract_gallery, list_node_types, run_graph
from studio.models import Project, StudioModelError, empty_demo_project
from studio.projects import ProjectStore, ProjectStoreError
from studio.settings import StudioSettings
from studio.templates import production_pipeline_project
from studio.uploads import DEFAULT_MAX_BYTES, UploadError, UploadStore, parse_multipart

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


class SettingsPatch(BaseModel):
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    litellm_model: str | None = None
    sensenova_base_url: str | None = None
    sensenova_api_key: str | None = None
    sensenova_model: str | None = None
    seedance_provider: str | None = None


def create_app(*, default_work_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Immersive Node Studio", version="0.1.0")
    work_root = Path(default_work_dir) if default_work_dir else Path(tempfile.gettempdir()) / "vr180-studio"
    work_root.mkdir(parents=True, exist_ok=True)
    #: content-addressed output cache (same as run_graph's ``cache``)
    run_cache: dict[str, dict[str, Any]] = {}
    #: per-node result history, keyed by node id (issue #418)
    run_history: dict[str, RunHistory] = {}
    project_store = ProjectStore(work_root / "projects")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "work_root": str(work_root),
            "projects_root": str(project_store.root),
            "settings_path": str(work_root / "studio_settings.json"),
        }

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        """Non-secret view of provider settings (keys masked)."""
        s = StudioSettings.load(work_root / "studio_settings.json")

        def mask(v: str) -> str:
            if not v:
                return ""
            return v[:3] + "…" + v[-2:] if len(v) > 8 else "***"

        return {
            "litellm_base_url": s.litellm_base_url,
            "litellm_model": s.litellm_model,
            "litellm_api_key_set": bool(s.litellm_api_key),
            "litellm_api_key_masked": mask(s.litellm_api_key),
            "sensenova_base_url": s.sensenova_base_url,
            "sensenova_model": s.sensenova_model,
            "sensenova_api_key_set": bool(s.sensenova_api_key),
            "seedance_provider": s.seedance_provider,
            "ark_api_key_set": bool(s.ark_api_key),
        }

    @app.post("/api/settings")
    def put_settings(patch: SettingsPatch) -> dict[str, Any]:
        """Merge provider settings into studio_settings.json under work_root."""
        path = work_root / "studio_settings.json"
        current: dict[str, Any] = {}
        if path.is_file():
            import json

            current = json.loads(path.read_text(encoding="utf-8"))
        payload = patch.model_dump(exclude_none=True)
        current.update(payload)
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return get_settings()

    @app.get("/api/media")
    def media(request: Request, path: str) -> FileResponse:
        """Serve a local artefact (stills/sheet/video) for canvas thumbnails.

        Only files under work_root or the system temp studio dir are allowed.
        """
        target = Path(path).resolve()
        allowed_roots = [
            work_root.resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ]
        if not any(target.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail=f"path outside studio work_root: {target}")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {target}")
        suffix = target.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".json": "application/json",
        }.get(suffix, "application/octet-stream")
        return FileResponse(target, media_type=media_type)

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

    @app.post("/api/coverage")
    async def analyze_coverage(request: Request) -> dict[str, Any]:
        """Measure the coverage radius of a domemaster frame sent by the 3D preview.

        The request body is a decodable image (png/jpg/webp). It is analysed by
        ``studio.coverage.analyze_frame`` — the single shared full-ring-fill
        scan also used by ``scripts/dome_qa.py`` and the ``qa.dome_coverage``
        node — so the browser readout, the Studio node and the release gate
        all report the same number for the same master (issue #405: the panel
        previously ran a third, client-side scan at ``edgeAt(0.98)`` that could
        disagree). The frontend now only draws the circle; the reading is the
        server's.
        """
        import cv2
        import numpy as np

        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty body: POST a domemaster image")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] < 2 or frame.shape[1] < 2:
            raise HTTPException(status_code=400, detail="unsupported or truncated image")
        return analyze_frame(_downscale(frame)).to_dict()

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
                history=run_history,
            )
        except GraphError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = report.to_dict()
        payload["gallery"] = extract_gallery(report)
        return payload

    @app.get("/api/node/{node_id}/history")
    def node_history(node_id: str) -> dict[str, Any]:
        """Return the last 5 run results for one node id (issue #418).

        The canvas ◀▶ buttons on a node card cycle through these entries.
        Unknown / never-run nodes return an empty list, not 404, so the
        frontend can treat "no history" as a first-run state.
        """
        hist = run_history.get(node_id)
        entries = hist.recent() if hist else []
        return {"node_id": node_id, "history": entries, "count": len(entries)}

    @app.post("/api/upload")
    async def upload(request: Request) -> dict[str, Any]:
        """Accept a single dropped file, store it content-addressed, return
        metadata. The canvas creates an ``input.image``/``input.video`` node
        from the reply (issue #417).

        Multipart is parsed from the raw body via the stdlib ``email`` module
        so the server has no hard dependency on ``python-multipart``; the dev
        venv and the CPU-only CI runner both import cleanly.
        """
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty upload body")
        try:
            part = parse_multipart(request.headers.get("content-type", ""), body)
            store = UploadStore(work_root, max_bytes=DEFAULT_MAX_BYTES)
            meta = store.save(filename=part.filename, data=part.data)
        except UploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return meta.to_dict()

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
