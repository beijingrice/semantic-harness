"""FastAPI 应用：网页界面以及可直接集成的 JSON API。"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backends import BackendError
from app.checker import Checker
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
        return {"connected": connected, "models": models, "model_discovery_available": discovery_available,
                "configured_model": client.settings.model, "configured_verifier_model": client.settings.verifier_model,
                "base_url": client.settings.base_url, "verifier_base_url": client.settings.verifier_base_url or client.settings.base_url,
                "backend": client.settings.backend, "verifier_backend": client.settings.verifier_backend or client.settings.backend,
                "schema_mode": client.settings.schema_mode, "json_schema": client.settings.json_schema,
                "evidence_mode": client.settings.evidence_mode}
    except LLMError as exc:
        return {"connected": False, "models": [], "error": str(exc),
                "base_url": client.settings.base_url}
    finally:
        client.close()


@app.post("/api/check", response_model=CheckResult)
def check(request: CheckRequest):
    try:
        client = ChatClient(Settings())
    except (ConfigError, BackendError) as exc:
        raise HTTPException(status_code=500, detail=f"配置错误：{exc}") from exc
    try:
        return Checker(client, client.settings).check(request.english, request.chinese, request.model, request.verifier_model, request.mode)
    except ConfigError as exc:
        # 服务端配置问题，不是模型故障，也不是业务 FAIL。
        raise HTTPException(status_code=500, detail=f"配置错误：{exc}") from exc
    except LLMError as exc:
        # 失败不是 FAIL：HTTP 502 表示基础模型不可用/无效；不伪造业务判定。
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        client.close()
