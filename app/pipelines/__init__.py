"""Pipelines: each one turns two source texts into a CheckResult.

Every pipeline has the signature of ``base.Pipeline``:
``run(runner, english, chinese, requested_model, requested_verifier, mode)``.
"""
from __future__ import annotations

from app.config import Settings
from app.pipelines import hardened, paragraph, quote
from app.pipelines.base import Pipeline, PipelineStageError, StageRunner

__all__ = ["Pipeline", "PipelineStageError", "StageRunner", "select"]


def select(settings: Settings) -> Pipeline:
    """The pipeline configured by LM_PIPELINE and LM_EVIDENCE_MODE (already validated)."""
    if settings.pipeline == "direct_hardened":
        return hardened.run
    return paragraph.run if settings.evidence_mode == "paragraph" else quote.run
