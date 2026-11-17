"""Regression tests for configuration loading and error mapping."""
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from app import cli
from app.api import app
from app.backends import BackendConfig, BackendError, OpenAIChatBackend
from app.config import ConfigError, Settings
from app.llm import ChatClient, LLMError
from app.models import ReviewDraft


# --- Settings reads the environment at construction time -------------------

def test_settings_reads_env_at_construction(monkeypatch):
    assert Settings().json_schema is False
    monkeypatch.setenv("LM_JSON_SCHEMA", "true")
    monkeypatch.setenv("LM_MAX_TOKENS", "777")
    monkeypatch.setenv("LM_BASE_URL", "http://example.test/v1/")
    settings = Settings()
    assert settings.json_schema is True
    assert settings.max_tokens == 777
    assert settings.base_url == "http://example.test/v1"


def test_explicit_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("LM_EVIDENCE_MODE", "paragraph")
    assert Settings(evidence_mode="quote").evidence_mode == "quote"


def test_conftest_isolates_local_dotenv_defaults():
    settings = Settings()
    assert settings.schema_mode == "legacy"
    assert settings.json_schema is False
    assert settings.max_tokens == 3072


# --- Invalid values give a named ConfigError, not a bare crash -------------

@pytest.mark.parametrize("name,value", [
    ("LM_MAX_TOKENS", "abc"), ("LM_MAX_TOKENS", "0"),
    ("LM_TIMEOUT_SECONDS", "soon"), ("MAX_KEYWORDS", "-1"),
    ("LM_EXTRACTION_MAX_TOKENS", "1.5"),
])
def test_invalid_numeric_env_raises_named_config_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        Settings()


def test_config_error_is_still_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_api_check_bad_pipeline_is_config_error_not_crash(monkeypatch):
    monkeypatch.setenv("LM_PIPELINE", "bogus")
    result = TestClient(app).post("/api/check", json={"english": "x", "chinese": "y"})
    assert result.status_code == 500
    assert "配置错误" in result.json()["detail"]
    assert "LM_PIPELINE" in result.json()["detail"]


@pytest.mark.parametrize("name,value", [("LM_MAX_TOKENS", "abc"), ("LM_SCHEMA_MODE", "bogus"),
                                        ("LM_BACKEND", "not-a-backend")])
def test_api_check_invalid_settings_is_config_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    result = TestClient(app).post("/api/check", json={"english": "x", "chinese": "y"})
    assert result.status_code == 500
    assert "配置错误" in result.json()["detail"]


def test_api_status_reports_config_error(monkeypatch):
    monkeypatch.setenv("LM_MAX_TOKENS", "abc")
    result = TestClient(app).get("/api/status")
    assert result.status_code == 200
    body = result.json()
    assert body["connected"] is False
    assert "LM_MAX_TOKENS" in body["error"]


def test_cli_config_error_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("LM_MAX_TOKENS", "abc")
    monkeypatch.setattr(sys, "argv", ["bilingual-check", "--en", "x", "--zh", "y"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "LM_MAX_TOKENS" in capsys.readouterr().err


# --- Malformed provider responses become BackendError / LLMError -----------

def _backend(body):
    http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))
    return OpenAIChatBackend(BackendConfig("http://x/v1", "k", 5), http)


GOOD_CHOICE = {"message": {"content": "{}"}, "finish_reason": "stop"}


@pytest.mark.parametrize("body", [
    {"choices": [GOOD_CHOICE], "usage": [1]},
    {"choices": [GOOD_CHOICE], "usage": {"completion_tokens_details": [1]}},
    {"choices": [{"message": "text", "finish_reason": "stop"}]},
    {"choices": ["oops"]},
    {"choices": []},
    {"choices": {"0": GOOD_CHOICE}},
    {"nochoices": True},
])
def test_malformed_response_is_backend_error(body):
    with pytest.raises(BackendError):
        _backend(body).complete({"model": "m"})


def test_malformed_response_surfaces_as_llm_error():
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, json={"choices": [GOOD_CHOICE], "usage": [1]})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient(Settings(), http)
    with pytest.raises(LLMError, match="BackendError"):
        client.structured("m", "", "", ReviewDraft, "independent_review")
    assert client.request_trace[-1]["outcome"] == "http_or_protocol_error"
    http.close()
