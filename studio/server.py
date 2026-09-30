"""FastAPI gateway for Immersive Node Studio (M0)."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from studio.coverage import _downscale, analyze_frame
from studio.graph import GraphError, RunHistory, extract_gallery, list_node_types, run_graph
from studio.models import NodeSpec, Project, StudioModelError, empty_demo_project
from studio.projects import ProjectStore, ProjectStoreError
from studio.settings import StudioSettings
from studio.templates import production_pipeline_project
from studio.uploads import DEFAULT_MAX_BYTES, UploadError, UploadStore, parse_multipart
from studio.xr_https import CertError, ensure_self_signed_cert

STATIC_DIR = Path(__file__).resolve().parent / "static"
XR_DIR = STATIC_DIR / "xr"

#: Master formats the Quest dome viewer can texture (issue #434).  Mirrors the
#: canvas's image/video accept lists in studio/uploads.py but adds ``.webm``
#: (Quest Browser plays it) and keeps ``.gif`` out (no audio, no gain over a
#: poster here).
XR_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
XR_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm"})
XR_MEDIA_EXTS = XR_IMAGE_EXTS | XR_VIDEO_EXTS


class ProjectPayload(BaseModel):
    project: dict[str, Any]
    work_dir: str | None = Field(default=None, description="Optional absolute output directory")


class RunRequest(ProjectPayload):
    only_downstream_of: str | None = None
    dirty_from: str | None = None
    run_mode: str = "all"
    run_node: str | None = None


class JobRequest(ProjectPayload):
    """Enqueue a background run (issue #419 queue + stop).

    ``label`` is what the queue panel shows (node name + mode); ``run_mode``
    and ``run_node`` select the same scopes as the synchronous /api/run.
    """

    run_mode: str = "all"
    run_node: str | None = None
    label: str | None = None


class ShotSpec(BaseModel):
    """One checked shot from the storyboard drawer (issue #419)."""

    id: str
    description: str = ""
    duration: float | None = None
    #: storyboard still on disk; when present the shot renders from it
    image: str | None = None
    motion: str | None = None


class BatchShotRequest(BaseModel):
    """Submit one video-generation job per checked shot (issue #419).

    The frontend already holds each shot's id/description/duration from the
    gallery, so the server can build a tiny per-shot mock graph without re-running
    the whole storyboard. Exactly one job is enqueued per shot.
    """

    shots: list[ShotSpec] = Field(default_factory=list)
    work_dir: str | None = None
    label: str | None = None


class SaveProjectRequest(BaseModel):
    project: dict[str, Any]
    project_id: str | None = None


class BoardImageRequest(BaseModel):
    """分镜板: generate candidate stills for one shot."""

    shot_id: str = "shot"
    prompt: str = ""
    aspect: str = "dome"
    style: str = ""
    n: int = 2
    model: str = ""


class BoardConcatRequest(BaseModel):
    """分镜板: join the shot videos in board order into one film."""

    videos: list[str] = Field(default_factory=list)
    filename: str = "board_film.mp4"


class SettingsPatch(BaseModel):
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    litellm_model: str | None = None
    sensenova_base_url: str | None = None
    sensenova_api_key: str | None = None
    sensenova_model: str | None = None
    seedance_provider: str | None = None


class RunJob:
    """One queued graph run (issue #419 队列与停止).

    ``done`` counts the nodes whose status has been reported, so the queue panel
    can show a progress bar without the browser knowing about the graph.
    """

    def __init__(self, job_id: str, label: str, total: int) -> None:
        self.id = job_id
        self.label = label
        self.total = max(1, total)
        self.status = "queued"  # queued | running | ok | error | cancelled
        self.done = 0
        self.cancel_event = threading.Event()
        self.cancelled = False
        self.report: dict[str, Any] | None = None
        self.error: str | None = None
        self.started_at: float = time.time()
        self.finished_at: float | None = None

    @property
    def progress(self) -> float:
        return min(1.0, self.done / self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "progress": self.progress,
            "done": self.done,
            "total": self.total,
            "cancelled": self.cancelled,
            "error": self.error,
            "report": self.report,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """In-process queue of background graph runs.

    Cancellation is cooperative: ``cancel()`` sets the event that ``run_graph``
    checks at each node boundary, so the pending node lands as ``cancelled`` and
    the nodes that already finished keep their outputs.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, RunJob] = {}
        self._lock = threading.Lock()

    def create(self, label: str, total: int) -> RunJob:
        job = RunJob(uuid.uuid4().hex[:12], label or "运行", total)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> RunJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.started_at)
        return [j.to_dict() for j in jobs]

    def list_dicts(self) -> list[dict[str, Any]]:
        return self.list_jobs()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.finished_at is not None:
                return False
            job.cancelled = True
            job.cancel_event.set()
            return True

    def clear_finished(self) -> int:
        """Drop every finished job from the list; returns how many were removed."""
        with self._lock:
            done = [k for k, j in self._jobs.items() if j.finished_at is not None]
            for k in done:
                del self._jobs[k]
        return len(done)

    def cancel_all(self) -> int:
        """Stop every running job; returns how many were actually cancelled."""
        with self._lock:
            running = [j for j in self._jobs.values() if j.finished_at is None]
        for job in running:
            job.cancelled = True
            job.cancel_event.set()
        return len(running)


def create_app(*, default_work_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Immersive Node Studio", version="0.1.0")
    work_root = Path(default_work_dir) if default_work_dir else Path(tempfile.gettempdir()) / "vr180-studio"
    work_root.mkdir(parents=True, exist_ok=True)
    #: content-addressed output cache (same as run_graph's ``cache``)
    run_cache: dict[str, dict[str, Any]] = {}
    last_outputs: dict[str, dict[str, Any]] = {}
    jobs = JobManager()
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

            current = json.loads(path.read_text(encoding="utf-8-sig"))
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

    def _xr_source_entry(path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        return {
            "path": str(path),
            "name": path.name,
            "kind": "image" if suffix in XR_IMAGE_EXTS else "video",
            "size": path.stat().st_size,
            "media_url": f"/api/media?path={urllib.parse.quote(str(path))}",
        }

    @app.get("/api/xr/sources")
    def xr_sources(src: str | None = None) -> dict[str, Any]:
        """Master list for the Quest dome viewer's 2D 选片 page (issue #434).

        Read-only, work_root-scoped: returns the most recent studio_out
        image/video files plus the explicit ``?src=`` path (the URL form
        `?src=<测试 domemaster>` the lead uses).  The path whitelist reuses
        ``/api/media``'s roots, so a traversal attempt resolves outside and is
        rejected — 400 for an out-of-root path, 404 for a missing file.
        """
        allowed_roots = [work_root.resolve(), Path(tempfile.gettempdir()).resolve()]

        def allowed(target: Path) -> bool:
            return any(target.is_relative_to(root) for root in allowed_roots)

        sources: list[dict[str, Any]] = []
        if src:
            target = Path(src).resolve()
            if not allowed(target):
                raise HTTPException(status_code=400, detail=f"src outside studio work_root: {target}")
            if not target.is_file():
                raise HTTPException(status_code=404, detail=f"src file not found: {target}")
            sources.append(_xr_source_entry(target))
        # Newest first (the last export lands at the top of the pick list).
        media = (
            p
            for p in sorted(work_root.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True)
            if p.is_file() and p.suffix.lower() in XR_MEDIA_EXTS
        )
        for p in media:
            sources.append(_xr_source_entry(p))
            if len(sources) >= 30:
                break
        return {"sources": sources, "work_root": str(work_root)}

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
                run_mode=request.run_mode,
                run_node=request.run_node,
                only_downstream_of=request.only_downstream_of,
                dirty_from=request.dirty_from,
                cache=run_cache,
                history=run_history,
                last_outputs=last_outputs,
            )
        except GraphError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = report.to_dict()
        payload["gallery"] = extract_gallery(report)
        return payload

    def _enqueue(
        *,
        project: Project,
        work_dir: Path,
        run_mode: str,
        run_node: str | None,
        label: str,
    ) -> RunJob:
        """Run a graph on a worker thread, reporting per-node progress to a job.

        The job owns its ``cancel_event``; ``run_graph`` checks it at each node
        boundary, and ``/api/jobs/{id}/cancel`` sets it — so the two share one
        event and a cancel lands on the next pending node.
        """
        total = len(project.nodes)
        job = jobs.create(label, total)
        cancel_event = job.cancel_event

        def runner() -> None:
            job.status = "running"

            def on_status(_nid: str, status: str) -> None:
                # A node reaching a terminal state is one unit of progress; do
                # not count cancelled nodes after the stop, or the bar would
                # jump past the cancellation point.
                if status in {"ok", "skipped", "locked", "error"} and not cancel_event.is_set():
                    job.done += 1

            try:
                report = run_graph(
                    project,
                    work_dir=str(work_dir),
                    run_mode=run_mode,
                    run_node=run_node,
                    cache=run_cache,
                    history=run_history,
                    last_outputs=last_outputs,
                    on_status=on_status,
                    cancel=cancel_event,
                )
            except GraphError as exc:
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()
                return
            job.done = job.total
            job.status = "cancelled" if cancel_event.is_set() else "ok"
            job.report = {
                **report.to_dict(),
                "gallery": extract_gallery(report),
            }
            job.finished_at = time.time()

        thread = threading.Thread(target=runner, name=f"studio-run-{job.id}", daemon=True)
        thread.start()
        return job

    @app.post("/api/jobs")
    def enqueue_job(request: JobRequest) -> dict[str, Any]:
        try:
            project = Project.from_dict(request.project)
        except StudioModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        work_dir = Path(request.work_dir) if request.work_dir else work_root
        if not work_dir.is_absolute():
            raise HTTPException(status_code=400, detail="work_dir must be absolute")
        work_dir.mkdir(parents=True, exist_ok=True)
        label = request.label or (f"{request.run_node} · {request.run_mode}" if request.run_node else request.run_mode)
        job = _enqueue(
            project=project,
            work_dir=work_dir,
            run_mode=request.run_mode,
            run_node=request.run_node,
            label=label,
        )
        return {"job_id": job.id, "label": job.label}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs.list_dicts()

    @app.post("/api/jobs/stop-all")
    def stop_all_jobs() -> dict[str, Any]:
        n = jobs.cancel_all()
        return {"stopped": n}

    @app.post("/api/jobs/clear")
    def clear_jobs() -> dict[str, Any]:
        return {"cleared": jobs.clear_finished()}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        ok = jobs.cancel(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"job not running: {job_id}")
        return {"cancelled": job_id}

    def _clip_frame(image: Path, short_side: int = 512) -> list[int]:
        """Clip size that keeps the still's aspect (16:9 / 2:1 boards stay wide)."""
        try:
            from PIL import Image

            with Image.open(image) as im:
                w, h = im.size
        except Exception:  # unreadable still: fall back to the square clip
            return [short_side, short_side]
        if w <= 0 or h <= 0 or abs(w - h) <= 2:
            return [short_side, short_side]
        scale = short_side / min(w, h)
        return [round(w * scale / 2) * 2, round(h * scale / 2) * 2]

    @app.post("/api/batch-shots")
    def batch_shots(request: BatchShotRequest) -> dict[str, Any]:
        """Enqueue one video job per checked shot (issue #419 只跑勾选镜头).

        Each shot becomes a one-node mock project so the queue panel lists the
        shots individually. The count is what the acceptance check is about: N
        checked shots enqueue exactly N jobs, and every enqueued project has a
        node_id matching a requested shot id.
        """
        if not request.shots:
            raise HTTPException(status_code=400, detail="no shots selected")
        work_dir = Path(request.work_dir) if request.work_dir else work_root
        if not work_dir.is_absolute():
            raise HTTPException(status_code=400, detail="work_dir must be absolute")
        work_dir.mkdir(parents=True, exist_ok=True)
        enqueued: list[dict[str, Any]] = []
        for shot in request.shots:
            prompt = (shot.description or shot.id or "shot")[:512]
            duration = max(1.0, min(10.0, float(shot.duration or 1)))
            node_id = f"n_shot_{shot.id[:24]}"
            if shot.image and Path(shot.image).is_file():
                # T1 09-29: render the shot from its real storyboard still (with
                # the storyboard's camera move) instead of a 64px mock clip.
                spec = NodeSpec(
                    id=node_id,
                    type="video.from_stills",
                    pos=(0, 0),
                    params={
                        "shots": [
                            {
                                "id": shot.id,
                                "description": shot.description,
                                "image": shot.image,
                                "duration": duration,
                                "motion": shot.motion or "static",
                                "frame": _clip_frame(Path(shot.image)),
                            }
                        ],
                        "size": 512,
                        "fps": 24,
                        "max_clips": 1,
                    },
                )
            else:
                spec = NodeSpec(
                    id=node_id,
                    type="video.mock",
                    pos=(0, 0),
                    params={"prompt": prompt, "duration": duration, "size": 64, "fps": 5},
                )
            project = Project(name=f"batch:{shot.id}", nodes=[spec], edges=[])
            job = _enqueue(
                project=project,
                work_dir=work_dir / "batch" / shot.id[:24],
                run_mode="all",
                run_node=None,
                label=f"分镜 {shot.id}",
            )
            enqueued.append({"job_id": job.id, "shot_id": shot.id, "node_id": node_id, "label": job.label})
        return {"submitted": len(enqueued), "jobs": enqueued}

    def _inside_work_root(path: str) -> Path:
        target = Path(path).resolve()
        roots = [work_root.resolve(), Path(tempfile.gettempdir()).resolve()]
        if not any(target.is_relative_to(r) for r in roots):
            raise HTTPException(status_code=403, detail=f"path outside studio work_root: {target}")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {target}")
        return target

    @app.post("/api/board/images")
    def board_images(request: BoardImageRequest) -> dict[str, Any]:
        """Text → image for one 分镜板 shot: ``n`` candidates to pick from.

        Runs synchronously (the gateway takes ~10–20 s per image and the
        candidates run concurrently); with no gateway configured it returns
        labelled placeholders and ``provider="mock"`` so the drawer can say so.
        """
        from studio.board import BOARD_ASPECTS, generate_shot_images, safe_shot_id

        if request.aspect not in BOARD_ASPECTS:
            raise HTTPException(status_code=400, detail=f"unknown aspect: {request.aspect}")
        # STUDIO_OFFLINE (set by the test suite) never reaches the gateway.
        settings = None if os.environ.get("STUDIO_OFFLINE") else StudioSettings.load(work_root / "studio_settings.json")
        try:
            result = generate_shot_images(
                prompt=request.prompt,
                aspect=request.aspect,
                n=request.n,
                style=request.style,
                shot_id=request.shot_id,
                out_dir=work_root / "board" / safe_shot_id(request.shot_id),
                settings=settings,
                model=request.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result["images"]:
            raise HTTPException(status_code=502, detail="出图失败: " + "; ".join(result["errors"])[:400])
        return result

    @app.post("/api/board/concat")
    def board_concat(request: BoardConcatRequest) -> dict[str, Any]:
        """Queue one job that joins the given shot videos in order."""
        if not request.videos:
            raise HTTPException(status_code=400, detail="no shot videos to join")
        clips = [{"video": str(_inside_work_root(v))} for v in request.videos]
        name = Path(request.filename).name or "board_film.mp4"
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        node_id = "n_board_concat"
        project = Project(
            name="board:concat",
            nodes=[NodeSpec(id=node_id, type="video.concat", pos=(0, 0), params={"clips": clips, "filename": name})],
            edges=[],
        )
        job = _enqueue(
            project=project,
            work_dir=work_root / "board" / "films" / uuid.uuid4().hex[:8],
            run_mode="all",
            run_node=None,
            label=f"拼接成片 {len(clips)} 镜",
        )
        return {"job_id": job.id, "label": job.label, "clips": len(clips)}

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

        @app.get("/dome_proj.js")
        def dome_proj_js() -> FileResponse:
            """Pure domemaster projection math (issue #416); loaded before
            preview3d.js so the preview's shaders match it."""
            return FileResponse(STATIC_DIR / "dome_proj.js", media_type="application/javascript")

        @app.get("/dome3d.html")
        def dome3d_html() -> FileResponse:
            return FileResponse(STATIC_DIR / "dome3d.html", headers={"Cache-Control": "no-store"})

        @app.get("/xr")
        def xr_page() -> FileResponse:
            """Quest dome viewer — 2D 选片 page + VR scene, no CDN (issue #434)."""
            return FileResponse(XR_DIR / "dome_xr.html", headers={"Cache-Control": "no-store"})

        @app.get("/xr/dome_xr.js")
        def xr_js() -> FileResponse:
            return FileResponse(XR_DIR / "dome_xr.js", media_type="application/javascript")

        @app.get("/xr/dome_xr_proj.js")
        def xr_proj_js() -> FileResponse:
            """The pure UV helper the GLSL is kept identical to (issue #434)."""
            return FileResponse(XR_DIR / "dome_xr_proj.js", media_type="application/javascript")

    return app


app = create_app()


def _build_parser():
    """``python -m studio.server`` argv — ``--host/--port/--https`` (issue #434).

    Default behaviour is unchanged (127.0.0.1:8787, http) when no flags are
    given; USB access (``adb reverse``) needs nothing more.  ``--https``
    generates a self-signed cert (Wi-Fi Quest access) and must stay opt-in so
    the default `python -m studio.server` never silently changes.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m studio.server")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="bind port (default 8787)")
    parser.add_argument(
        "--https",
        action="store_true",
        help="serve over https with a self-signed cert (Wi-Fi access for the Quest dome viewer; "
        "USB via `adb reverse tcp:8787 tcp:8787` needs no https)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dev entry: ``python -m studio.server [--host H] [--port P] [--https]``."""
    import uvicorn

    args = _build_parser().parse_args(argv)
    kwargs: dict[str, Any] = {"host": args.host, "port": args.port, "reload": False}
    if args.https:
        # Wi-Fi Quest access needs a secure context.  The cert lives in the
        # same work_root as the module-level app so /api/xr/sources and the
        # cert share a root.  Catch CertError so the operator gets the USB
        # hint on stderr (instead of a traceback) and main() exits non-zero.
        work_root = Path(tempfile.gettempdir()) / "vr180-studio"
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            cert, key = ensure_self_signed_cert(work_root)
        except CertError as exc:
            print(f"[studio.server] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
    uvicorn.run("studio.server:app", **kwargs)


if __name__ == "__main__":
    main()
