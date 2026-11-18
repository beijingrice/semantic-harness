"""Shared plumbing for every pipeline: model calls by stage, model selection, errors."""

from __future__ import annotations

from typing import Callable, Protocol

from pydantic import BaseModel

from app.config import Settings
from app.llm import ChatClient, LLMError
from app.models import CheckResult


class PipelineStageError(LLMError):
    """An ERROR that records which pipeline stage failed (used by batch evaluation)."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class StageRunner:
    """What a pipeline needs from the outside world: settings and stage-labelled model calls."""

    def __init__(self, client: ChatClient, settings: Settings):
        self.client = client
        self.settings = settings

    def call(
        self,
        stage: str,
        diagnostics: list[str],
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        schema_name: str,
        validate: Callable[[BaseModel], None] | None = None,
        max_tokens: int | None = None,
        max_attempts: int | None = None,
    ):
        """One structured model call. Failures become PipelineStageError(stage)."""
        kwargs = {"validate_result": validate}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if max_attempts is not None:
            kwargs["max_attempts"] = max_attempts
        try:
            # Request telemetry is labelled by pipeline stage (e.g. extract_english),
            # which is finer than the wire schema name.
            self.client.stage_label = stage
            answer = self.client.structured(model, system, user, schema, schema_name, **kwargs)
        except LLMError as exc:
            raise PipelineStageError(stage, str(exc)) from exc
        diagnostics.append(f"{stage}: OK ({model})")
        return answer

    def choose_models(self, mode: str, requested_model: str | None, requested_verifier: str | None) -> tuple[str, str]:
        """(primary, verifier); verifier is "" in primary-only mode."""
        if mode not in ("primary", "dual"):
            raise LLMError("mode 只能是 primary 或 dual")
        model = self.client.choose_model(requested_model)
        if mode == "primary":
            return model, ""
        verifier = self.client.choose_verifier(requested_verifier)
        if verifier == model:
            raise LLMError("主模型与复核模型相同：请选择两个不同的模型。")
        return model, verifier


class Pipeline(Protocol):
    def __call__(
        self,
        runner: StageRunner,
        english: str,
        chinese: str,
        requested_model: str | None,
        requested_verifier: str | None,
        mode: str,
    ) -> CheckResult: ...
