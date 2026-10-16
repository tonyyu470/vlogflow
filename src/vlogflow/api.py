from __future__ import annotations

import asyncio
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings, data_root, load_settings, public_settings, save_settings
from .domain import ProjectEdit
from .jobs import Jobs, public_job
from .processes import Runner
from .providers import Provider
from .store import Conflict, Store
from .uploads import receive_uploads


class NewProject(BaseModel):
    brief: str = Field(min_length=1, max_length=10000)
    language: Literal["zh", "en"] = "zh"


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["import", "analyze", "reference", "subtitles", "render", "audition"]
    paths: list[str] = Field(default_factory=list, max_length=50)
    path: str = Field(default="", max_length=4096)
    transcript: str = Field(default="", max_length=4000)


class Epoch(BaseModel):
    epoch: int


class Transcript(Epoch):
    transcript: str = Field(min_length=1, max_length=4000)


def create_app(root: Path | None = None):
    root = (root or data_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    store = Store(root)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = Jobs(store)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.jobs.close)

    app = FastAPI(title="VlogFlow", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
            if not hmac.compare_digest(request.headers.get("x-vlogflow-token", ""), token):
                return JSONResponse(
                    {"detail": "Reload this local page before making changes"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Project, task or artifact does not exist"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/bootstrap")
    def bootstrap():
        return dict(token=token, version="0.1.0", data_directory=str(root))

    @app.get("/api/settings")
    def get_settings():
        return public_settings(load_settings(root))

    @app.put("/api/settings")
    def put_settings(body: dict):
        previous = load_settings(root).model_dump()
        if body.pop("clear_api_key", False):
            previous["api_key"] = ""
        if body.get("api_key") == "":
            body.pop("api_key")
        body.pop("has_api_key", None)
        settings = Settings.model_validate(previous | body)
        save_settings(root, settings)
        return public_settings(settings)

    @app.post("/api/settings/test")
    async def test_settings():
        settings = load_settings(root)
        runner = Runner(root / "diagnostics", cancel=app.state.jobs.stopping.is_set)
        try:
            result = await asyncio.wrap_future(app.state.jobs.pool.submit(Provider(settings, runner).test))
            return {"ok": True, "response": result}
        except (RuntimeError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/projects")
    def projects():
        return store.list()

    @app.post("/api/projects")
    def new_project(body: NewProject):
        settings = load_settings(root)
        settings.last_language = body.language
        save_settings(root, settings)
        return store.create(body.brief, body.language)

    @app.get("/api/projects/{pid}")
    def project(pid: str):
        return store.get(pid)

    @app.patch("/api/projects/{pid}")
    def edit_project(pid: str, body: ProjectEdit):
        changes = body.model_dump(exclude_none=True)
        epoch = changes.pop("epoch")
        return store.edit(pid, changes, epoch)

    @app.post("/api/projects/{pid}/approve")
    def approve(pid: str, body: Epoch):
        return store.approve(pid, body.epoch)

    @app.post("/api/projects/{pid}/voice/transcript")
    def transcript(pid: str, body: Transcript):
        p = store.get(pid)
        if not p["voice"]:
            raise ValueError("Import a reference recording first")
        if not body.transcript.strip():
            raise ValueError("Reference transcript cannot be blank")
        return store.edit(pid, {"voice": dict(p["voice"], transcript=body.transcript.strip())}, body.epoch)

    @app.post("/api/projects/{pid}/uploads")
    async def uploads(pid: str, request: Request):
        store.get(pid)
        folder = root / "projects" / pid / "uploads"
        return {"paths": await receive_uploads(request, folder)}

    @app.get("/api/projects/{pid}/jobs")
    def project_jobs(pid: str):
        store.get(pid)
        return [public_job(j) for j in store.jobs(pid)]

    @app.post("/api/projects/{pid}/jobs")
    def start_job(pid: str, body: JobRequest):
        data = body.model_dump()
        kind = data.pop("kind")
        return public_job(app.state.jobs.submit(pid, kind, data))

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return public_job(store.job(jid))

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        return public_job(app.state.jobs.cancel(jid))

    @app.post("/api/jobs/{jid}/retry")
    def retry_job(jid: str):
        return public_job(app.state.jobs.retry(jid))

    def file_response(raw, download=False):
        if not raw:
            raise HTTPException(404, "This output is not ready")
        path = Path(raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "Registered file is unavailable")
        return FileResponse(path, filename=path.name if download else None)

    @app.get("/api/projects/{pid}/media/{mid}/{part}")
    def media_file(pid: str, mid: str, part: str):
        p = store.get(pid)
        m = next((m for m in p["media"] if m["id"] == mid), None)
        if not m:
            raise HTTPException(404, "Source video does not exist")
        return file_response(
            m["path"] if part == "video" else m["frames"][0]["path"] if part == "thumbnail" else None
        )

    @app.get("/api/projects/{pid}/files/{name}")
    def artifact_file(pid: str, name: str, download: bool = False):
        p = store.get(pid)
        if name == "reference":
            return file_response((p.get("voice") or {}).get("path"), download)
        return file_response(p.get("artifacts", {}).get(name), download)

    static = Path(__file__).parent / "static"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app
