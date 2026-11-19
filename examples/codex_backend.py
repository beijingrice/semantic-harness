r"""Use the locally installed Codex CLI with a ChatGPT-account login.

    LM_BACKEND=examples.codex_backend:CodexExecBackend LM_MODEL_DISCOVERY=off \
    LM_MODEL=gpt-6-luna@medium LM_VERIFIER_MODEL=gpt-6-luna@high LM_TIMEOUT_SECONDS=600 \
    python -m examples.evaluate --cases examples/v2_regression_11.json --workers 4 --out eval_x.json

The checker requires distinct primary and verifier model IDs. The effort suffix
allows both roles to use the same underlying model. For evaluation only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.backends import BackendConfig, BackendError, BackendResponse

_PREAMBLE = (
    "You are being used as a plain text-completion model. Do not run any commands, do not read or write files. "
    "Answer the conversation below. Your final message must be only the requested JSON object, with no code fences."
)
_ROLE_HEADINGS = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT (your previous answer)"}
_EFFORTS = {"low", "medium", "high", "xhigh"}
_FALLBACK_BIN = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
_TOKEN_USAGE = re.compile(r"^[ \t]*tokens used[ \t]*\n[ \t]*([0-9]+(?:,[0-9]{3})*)[ \t]*$", re.MULTILINE)


def _build_prompt(payload: dict[str, Any]) -> str:
    blocks = [_PREAMBLE]
    for message in payload["messages"]:
        blocks.append(f"### {_ROLE_HEADINGS[message['role']]}\n{message['content']}")
    response_format = payload.get("response_format") or {}
    schema = (response_format.get("json_schema") or {}).get("schema")
    if schema is not None:
        blocks.append(f"The JSON must satisfy this JSON Schema: {json.dumps(schema, ensure_ascii=False)}")
    return "\n\n".join(blocks)


def _execution_error(message: str, stderr: str, prompt: str) -> BackendError:
    # Codex can echo its input to stderr.
    stderr = stderr.replace(prompt, "[prompt redacted]")
    prompt_lines = {line.strip() for line in prompt.splitlines() if line.strip()}
    tail = "\n".join(
        "[prompt redacted]" if line.strip() in prompt_lines else line for line in stderr.strip().splitlines()
    )[-300:]
    return BackendError(f"{message}: {tail}" if tail else message)


class CodexExecBackend:
    def __init__(self, config: BackendConfig, http_client: object):
        self.config = config

    def models(self) -> list[str] | None:
        return None

    def complete(self, payload: dict[str, Any]) -> BackendResponse:
        model = payload["model"]
        effort = os.environ.get("CODEX_REASONING_EFFORT", "medium")
        if "@" in model:
            model, effort = model.rsplit("@", 1)
        if effort not in _EFFORTS:
            raise BackendError("Invalid Codex reasoning effort (expected low, medium, high or xhigh)")
        if not model:
            raise BackendError("Missing Codex model name")

        codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex") or _FALLBACK_BIN
        if os.path.dirname(codex_bin):
            codex_bin = os.path.abspath(codex_bin)
        prompt = _build_prompt(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "last.txt"
            command = [
                codex_bin,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "-s",
                "read-only",
                "-m",
                model,
                "-c",
                f"model_reasoning_effort={effort}",
                "-o",
                str(output),
                "-",
            ]
            try:
                process = subprocess.run(
                    command,
                    input=prompt,
                    cwd=tmpdir,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise BackendError(f"Codex exec timed out after {self.config.timeout_seconds:g}s") from None
            except FileNotFoundError:
                raise BackendError("Codex executable not found") from None
            except OSError as exc:
                raise BackendError(f"Could not start Codex exec ({type(exc).__name__})") from None
            if process.returncode:
                raise _execution_error(f"Codex exec exited with code {process.returncode}", process.stderr, prompt)
            try:
                content = output.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                raise _execution_error("Codex exec produced no readable output file", process.stderr, prompt) from None
            if not content:
                raise _execution_error("Codex exec produced an empty output file", process.stderr, prompt)

        usage = _TOKEN_USAGE.findall(f"{process.stdout}\n{process.stderr}")
        return BackendResponse(
            content=content,
            finish_reason="stop",
            completion_tokens=int(usage[-1].replace(",", "")) if usage else None,
            http_status=200,
        )
