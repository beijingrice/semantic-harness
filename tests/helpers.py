"""Shared fakes and canned answers for offline pipeline tests (no model, no HTTP)."""

import json
from dataclasses import replace

from app.config import Settings
from app.models import (
    MatchDraft,
    ParagraphArbitrationDraft,
    ParagraphExtractedDraft,
    ParagraphReviewDraft,
)


def config(**kw):
    return replace(
        Settings(
            evidence_mode="paragraph",
            extraction_prompt="coarse",
            coverage_audit=False,
            atomic_refinement=False,
            extraction_fast_path=False,
            model="primary",
            verifier_model="independent",
        ),
        **kw,
    )


def extract(claim, pid, kind="constraint"):
    return ParagraphExtractedDraft.model_validate(
        {"units": [{"kind": kind, "claim": claim, "paragraph_ids": list(pid) if isinstance(pid, tuple) else [pid]}]}
    )


def match(relation="equivalent", reason="same"):
    return MatchDraft.model_validate(
        {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": relation, "reason": reason}]}
    )


def review(verdict="PASS", issues=None):
    return ParagraphReviewDraft.model_validate(
        {"verdict": verdict, "reason": "independently inspected original texts", "issues": issues or []}
    )


def arb(verdict="FAIL", kind="conflict", en=None, zh=None):
    return ParagraphArbitrationDraft.model_validate(
        {
            "verdict": verdict,
            "issue_kind": kind,
            "en_paragraph_ids": en or [],
            "zh_paragraph_ids": zh or [],
            "reason": "independently checked constraints",
        }
    )


class Fake:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def choose_model(self, requested=None):
        return requested or "primary"

    def choose_verifier(self, requested=None):
        return requested or "independent"

    def structured(self, model, system, user, schema_model, schema_name, validate_result=None):
        self.calls.append({"model": model, "system": system, "user": json.loads(user), "schema": schema_name})
        answer = self.answers.pop(0)
        assert isinstance(answer, schema_model), (type(answer), schema_model)
        if validate_result:
            validate_result(answer)
        return answer
