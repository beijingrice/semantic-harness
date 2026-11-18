"""Example for a hypothetical NON-OpenAI provider (NOT a real production API).

Activate only after reviewing/customizing the transport contract:
  LM_BACKEND=examples.custom_backend_example:ExampleNonOpenAIBackend
  LM_MODEL_DISCOVERY=off
  LM_SCHEMA_MODE=off

Demo service wire contract: POST {base_url}/generate
  request {"model":"...","system":"...","prompt":"...","max_new_tokens":123}
  response {"output":{"text":"{...}"},"usage":{"input_tokens":10,"output_tokens":20}}

For a different real API implement the conversion accordingly. Never alter the
semantic Pipeline, only this adapter. JSON validation stays in ChatClient.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.backends import BackendConfig, BackendError, BackendResponse


class ExampleNonOpenAIBackend:
    def __init__(self, config: BackendConfig, http_client: httpx.Client):
        self.config = config
        self.http = http_client

    def models(self) -> list[str] | None:
        return None  # the hypothetical API does not expose model listing

    def complete(self, payload: dict[str, Any]) -> BackendResponse:
        messages = payload["messages"]
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system")
        headers = {"X-API-Key": self.config.api_key} if self.config.api_key else {}
        try:
            res = self.http.post(
                f"{self.config.base_url}/generate",
                headers=headers,
                json={
                    "model": payload["model"],
                    "system": system,
                    "prompt": prompt,
                    "max_new_tokens": payload["max_tokens"],
                },
                timeout=self.config.timeout_seconds,
            )
            if res.is_error:
                raise BackendError(f"Custom API returned HTTP {res.status_code}")
            data = res.json()
            text = data["output"]["text"]
            if not isinstance(text, str):
                raise BackendError("Custom provider must supply string output.text")
            usage = data.get("usage") or {}
            return BackendResponse(
                content=text,
                http_status=res.status_code,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise BackendError(f"Custom provider request failed ({type(exc).__name__})") from exc
