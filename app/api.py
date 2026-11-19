"""FastAPI 应用：网页界面以及可直接集成的 JSON API。"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import jobs
from app.backends import BackendError
from app.config import ConfigError, Settings
from app.llm import ChatClient, LLMError
from app.models import CheckRequest, CheckResult

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Bilingual Keyword Checker", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    """即使没有开 LM Studio，网页也应能加载并给出可行动的错误消息。"""
    try:
        client = ChatClient(Settings())
    except (ConfigError, BackendError) as exc:
        # 配置错误时页面仍可加载，并明确指出是哪项配置有问题。
        return {"connected": False, "models": [], "error": f"配置错误：{exc}", "base_url": None}
    try:
        connected = None  # unknown is NOT the same as connected
        try:
            if client.settings.model_discovery != "off":
                models = client.models()
                connected, discovery_available = True, True
            else:
                models, discovery_available = [], False
        except LLMError:
            if client.settings.model_discovery == "required":
                raise
            models, discovery_available = [], False
        return {
            "connected": connected,
            "models": models,
            "model_discovery_available": discovery_available,
            "configured_model": client.settings.model,
            "configured_verifier_model": client.settings.verifier_model,
            "base_url": client.settings.base_url,
            "verifier_base_url": client.settings.verifier_base_url or client.settings.base_url,
            "backend": client.settings.backend,
            "verifier_backend": client.settings.verifier_backend or client.settings.backend,
            "schema_mode": client.settings.schema_mode,
            "json_schema": client.settings.json_schema,
            "evidence_mode": client.settings.evidence_mode,
        }
    except LLMError as exc:
        return {"connected": False, "models": [], "error": str(exc), "base_url": client.settings.base_url}
    finally:
        client.close()


def _submit_job(request: CheckRequest):
    try:
        return jobs.get_manager(Settings()).submit_with_future(request)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"配置错误：{exc}") from exc
    except jobs.QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.post("/api/jobs", status_code=202)
async def submit_job(request: CheckRequest) -> dict[str, str]:
    job_id, _ = _submit_job(request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}", response_model=jobs.JobRecord)
def get_job(job_id: str, response: Response):
    try:
        job = jobs.get_manager().get(job_id)
    except ConfigError:
        # A manager that could not be initialized has no submitted jobs.
        job = None
    if job is None:
        raise HTTPException(status_code=404, detail="检查任务不存在或已过期。")
    response.headers["Cache-Control"] = "no-store"
    return job


@app.post("/api/check", response_model=CheckResult)
async def check(request: CheckRequest):
    _, future = _submit_job(request)
    # Cancelling the HTTP waiter must leave the queued job available to the executor.
    outcome = await asyncio.shield(asyncio.wrap_future(future))
    if isinstance(outcome, jobs.JobError):
        raise HTTPException(status_code=502 if outcome.kind == "model" else 500, detail=outcome.message)
    return outcome
