"""Public entry point: ``Checker(client, settings).check(english, chinese)``."""
from __future__ import annotations

from app.config import Settings
from app.llm import ChatClient
from app.models import CheckResult, KeyUnit
from app.pipelines import PipelineStageError, StageRunner, paragraph, quote, select

__all__ = ["Checker", "PipelineStageError"]


class Checker:
    """Runs the pipeline selected by the settings for one pair of texts.

    mode="dual" uses a primary and an independent verifier model; mode="primary"
    uses one model only. Model failures raise LLMError (PipelineStageError names
    the stage); they are never turned into a FAIL verdict.
    """

    def __init__(self, client: ChatClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.runner = StageRunner(client, settings)

    def check(self, english: str, chinese: str, requested_model: str | None = None,
              requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
        pipeline = select(self.settings)
        return pipeline(self.runner, english, chinese, requested_model, requested_verifier, mode)

    def extract(self, text: str, language: str, prefix: str, model: str,
                fallback_model: str | None = None, diagnostics: list[str] | None = None) -> list[KeyUnit]:
        """The extraction stage alone (used by examples/evaluate_extraction.py)."""
        module = paragraph if self.settings.evidence_mode == "paragraph" else quote
        return module.extract(self.runner, text, language, prefix, model, fallback_model,
                              diagnostics if diagnostics is not None else [])
