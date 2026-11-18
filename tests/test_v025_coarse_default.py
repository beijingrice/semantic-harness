"""v0.2.5 simplified extraction: mocked model protocol, NOT accuracy evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from app.checker import Checker, PipelineStageError
from app.config import Settings
from app.prompts import EXTRACTION_SYSTEM_COARSE, FALLBACK_EXTRACTION_SYSTEM_COARSE


def mk_unit(claim: str, source: str, kind: str = "constraint") -> dict:
    return {"kind": kind, "claim": claim, "evidence": source}


class MockModels:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def choose_model(self, requested=None):
        return requested or "qwen"

    def choose_verifier(self, requested=None):
        return requested or "ministral"

    def structured(self, model, system, user, schema, schema_name, validate_result=None):
        self.calls.append((model, schema_name, system))
        raw = self.answers.pop(0)
        answer = schema.model_validate(raw)
        if validate_result:
            validate_result(answer)
        return answer


def settings(**overrides):
    return replace(Settings(evidence_mode="quote", model="qwen", verifier_model="ministral",
                            extraction_prompt="coarse", coverage_audit=False,
                            atomic_refinement=False, json_schema=True), **overrides)


def test_default_settings_are_simplified(monkeypatch):
    # Source default contract (conftest clears LM_* env, so these are the code
    # defaults, not .env values). Explicit overrides remain possible for A/B.
    defaults = Settings()
    assert defaults.extraction_prompt == "coarse"
    assert defaults.atomic_refinement is False
    assert defaults.coverage_audit is False
    assert "DO NOT FORCE ATOMIC LIST SPLITTING" in EXTRACTION_SYSTEM_COARSE
    assert "contiguous exact" in FALLBACK_EXTRACTION_SYSTEM_COARSE


@pytest.mark.parametrize("source,claim", [
    ("The abstract must report the purpose, methods, and results.", "The abstract must report purpose, methods and results"),
    ("摘要须介绍研究目的与研究方法。", "摘要须介绍研究目的和研究方法"),
    ("提交材料应包含正文文字、插图和表格。", "提交材料应包含正文文字、插图、表格"),
])
def test_coarse_keeps_one_complete_grounded_list_and_skips_refinement(source, claim):
    fake = MockModels([{"units": [mk_unit(claim, source)]}])
    cfg = settings(atomic_refinement=True, coverage_audit=True)
    events = []
    units = Checker(fake, cfg).extract(source, "Chinese", "zh", "qwen", "ministral", events)
    assert len(units) == 1 and units[0].evidence == source
    assert len(fake.calls) == 1
    assert fake.calls[0][2] == EXTRACTION_SYSTEM_COARSE
    assert not any("atomic refinement" in e or "coverage audit triggered" in e for e in events)


def test_coarse_strictly_rejects_discontinuous_reconstructed_evidence():
    src = "摘要须介绍研究目的与研究方法。"
    fake = MockModels([{"units": [mk_unit("必须介绍研究方法", "摘要须介绍研究方法")]}])
    with pytest.raises(PipelineStageError, match="原文证据"):
        Checker(fake, settings()).extract(src, "Chinese", "zh", "qwen")
    # Fail closed: no invalid quotation is silently replaced by a made-up span.


def test_empty_first_extract_uses_independent_coarse_fallback():
    src = "No additional references are required."
    fake = MockModels([
        {"units": []},
        {"units": [mk_unit("No extra references required", src)]},
    ])
    events = []
    units = Checker(fake, settings()).extract(src, "English", "en", "qwen", "ministral", events)
    assert len(units) == 1
    assert [(m, n) for m, n, _ in fake.calls] == [
        ("qwen", "keyword_extraction"), ("ministral", "keyword_extraction_fallback")]
    assert fake.calls[1][2] == FALLBACK_EXTRACTION_SYSTEM_COARSE


@pytest.mark.parametrize("en,zh,expected,relation", [
    ("The abstract must report the purpose, methods, and results.",
     "摘要须说明研究目的、采用的方法和研究结果。", "PASS", "equivalent"),
    ("The abstract must include purpose, methods, and results.",
     "摘要须介绍研究目的与研究方法。", "FAIL", "conflict"),
    ("The submission contains text, figures, and tables.",
     "提交材料应包含正文文字、插图和表格。", "PASS", "plausible"),
])
def test_coarse_full_pipeline_list_examples(en, zh, expected, relation):
    en_unit = mk_unit("Complete English list", en)
    zh_unit = mk_unit("Complete Chinese list", zh)
    fake = MockModels([
        {"units": [en_unit]},
        {"units": [zh_unit]},
        {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"],
                    "relation": relation, "reason": "mocked content assessment"}]},
        {"verdict": expected, "reason": "mocked independent judgment", "issues": []},
    ])
    out = Checker(fake, settings()).check(en, zh, mode="dual")
    assert out.verdict == expected
    assert len(out.english_keywords) == len(out.chinese_keywords) == 1
    assert [stage for _, stage, _ in fake.calls] == [
        "keyword_extraction", "keyword_extraction", "keyword_matching", "independent_review"]
    assert not out.arbitrated


def test_legacy_modes_are_still_selectable_for_ab():
    for style in ("legacy", "proposition", "coarse"):
        Checker(MockModels([]), settings(extraction_prompt=style))
    with pytest.raises(ValueError):
        Checker(MockModels([]), settings(extraction_prompt="unknown"))
