"""Transport adapters; no semantic decision code belongs in this module.

A custom provider may expose *any* protocol on the wire. The adapter must
translate it into BackendResponse. See README_V0216_BACKENDS.md for a sample.
No custom adapter is imported unless explicitly configured by the operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import httpx


@dataclass(frozen=True)
class BackendConfig:
    base_url: str
    api_key: str
    timeout_seconds: float


@dataclass(frozen=True)
class BackendResponse:
    content: str | None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    http_status: int | None = None


class BackendError(RuntimeError):
    """A transport/protocol issue. Never a semantic FAIL."""


class UnsupportedNativeSchema(BackendError):
    """Provider specifically rejected native schema mode; safe auto-downgrade candidate."""


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, payload: dict[str, Any]) -> BackendResponse: ...
    def models(self) -> list[str] | None: ...


def _schema_rejected(status: int, body: str) -> bool:
    # A generic 400 is NOT proof that JSON Schema is unsupported (could be a
    # malformed request, invalid model or quota). Conservative fail closed.
    if status not in (400, 415, 422):
        return False
    import re
    signal = bool(re.search(r"response[_ -]?format|json[_ -]?schema|structured[_ -]?output", body, re.I))
    rejection = bool(re.search(r"unsupported|not supported|unrecognized|unknown|not available|not permitted|not implemented", body, re.I))
    return signal and rejection


class OpenAIChatBackend:
    """OpenAI-compatible /chat/completions adapter; /models is optional by policy."""
    def __init__(self, config: BackendConfig, http_client: httpx.Client):
        self.config = config
        self.http = http_client

    def _headers(self):
        # Some private APIs don't require an Authorization header.
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def complete(self, payload: dict[str, Any]) -> BackendResponse:
        try:
            response = self.http.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                json=payload, headers=self._headers(),
                timeout=self.config.timeout_seconds,
            )
            if response.is_error:
                # Never put response body, credentials, or prompt into user logs.
                if "response_format" in payload and _schema_rejected(response.status_code, response.text[:2048]):
                    raise UnsupportedNativeSchema(
                        f"API rejected response_format=json_schema (HTTP {response.status_code})"
                    )
                raise BackendError(f"HTTP {response.status_code} from chat/completions")
            data = response.json()
            if not isinstance(data, dict):
                raise BackendError("chat/completions: expected JSON object")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise BackendError("chat/completions: choices must be a non-empty array of objects")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise BackendError("chat/completions: choices[0].message must be an object")
            # Validate each level's type BEFORE calling .get() on it.
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                raise BackendError("chat/completions: invalid usage")
            details = usage.get("completion_tokens_details") or {}
            if not isinstance(details, dict):
                raise BackendError("chat/completions: invalid usage")
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            if content is not None and not isinstance(content, str):
                raise BackendError("chat/completions: content must be a string or null")
            if reasoning is not None and not isinstance(reasoning, str):
                reasoning = None
            return BackendResponse(
                content=content, reasoning_content=reasoning,
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                reasoning_tokens=details.get("reasoning_tokens"),
                http_status=response.status_code,
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BackendError(f"API transport/protocol failure ({type(exc).__name__})") from exc

    def models(self) -> list[str] | None:
        try:
            response = self.http.get(
                f"{self.config.base_url.rstrip('/')}/models",
                headers=self._headers(), timeout=self.config.timeout_seconds,
            )
            if response.is_error:
                raise BackendError(f"GET /models returned HTTP {response.status_code}")
            obj = response.json()
            items = obj["data"]
            if not isinstance(items, list):
                raise ValueError("data is not a list")
            return [x["id"] for x in items if isinstance(x, dict) and isinstance(x.get("id"), str)]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise BackendError(f"GET /models unavailable or invalid ({type(exc).__name__})") from exc


def build_backend(kind: str, config: BackendConfig, http_client: httpx.Client) -> LLMBackend:
    """Built-in 'openai' or explicitly trusted 'package.module:Class' adapter."""
    if kind.strip().lower() in ("openai", "openai_compatible"):
        return OpenAIChatBackend(config, http_client)
    if ":" not in kind:
        raise BackendError("LM_BACKEND must be 'openai' or 'module.path:ClassName'")
    module_name, class_name = kind.split(":", 1)
    if not module_name or not class_name or not all(part.isidentifier() for part in module_name.split(".")) or not class_name.isidentifier():
        raise BackendError("Invalid custom backend import path")
    try:
        cls = getattr(import_module(module_name), class_name)
        backend = cls(config, http_client)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise BackendError(f"Could not initialize custom backend {kind!r}: {type(exc).__name__}") from exc
    if not isinstance(backend, LLMBackend):
        raise BackendError("Custom backend must implement complete(payload) and models()")
    return backend
