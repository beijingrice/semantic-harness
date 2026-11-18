"""Regression simulations for the 4 remaining stable v0.2.1 failures.

Mocks imitate model output; these DO NOT prove real Qwen/Ministral recovery.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.checker import Checker, PipelineStageError
from app.config import Settings
from app.llm import ChatClient


def unit(claim, evidence, kind="concept"):
    return {"kind": kind, "claim": claim, "evidence": evidence}


def case(english, chinese, replies, mode="dual", json_schema=False):
    """Only simulated responses: assert exact model routing and stage behavior."""
    queue = iter(replies)
    calls = []

    def handler(request):
        data = json.loads(request.content)
        calls.append(data)
        model, ans = next(queue)
        assert model == data["model"], (model, data["model"])
        if json_schema:
            assert data["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [
            {"message": {"content": json.dumps(ans, ensure_ascii=False)}, "finish_reason": "stop"}
        ]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        checker = Checker(ChatClient(Settings(evidence_mode="quote", model="qwen", verifier_model="ministral",
                                              json_schema=json_schema, extraction_prompt="legacy", coverage_audit=False), http),
                          Settings(evidence_mode="quote", extraction_prompt="legacy", coverage_audit=False))
        result = checker.check(english, chinese, mode=mode)
        assert next(queue, None) is None
        return result, calls
    finally:
        http.close()


def test_bas01_empty_extraction_recovers_through_different_model():
    en = "The abstract describes the research objective and methodology."
    zh = "摘要介绍了研究目标与研究方法。"
    replies = [
        ("qwen", {"units": []}),
        ("qwen", {"units": []}),
        ("ministral", {"units": [unit("research objective", "research objective"),
                                  unit("research methodology", "methodology")]}),
        ("qwen", {"units": []}),
        ("qwen", {"units": []}),
        ("ministral", {"units": [unit("研究目标", "研究目标"), unit("研究方法", "研究方法")]}),
        ("qwen", {"links": [
            {"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "aim"},
            {"en_ids": ["en_2"], "zh_ids": ["zh_2"], "relation": "equivalent", "reason": "method"}]}),
        ("ministral", {"verdict": "PASS", "reason": "same facts", "issues": []}),
    ]
    result, calls = case(en, zh, replies)
    assert result.verdict == "PASS"
    assert [c["model"] for c in calls] == ["qwen", "qwen", "ministral"] * 2 + ["qwen", "ministral"]
    assert sum("independent fallback" in line for line in result.diagnostics) == 2


def test_mod02_negated_obligation_is_a_key_fact():
    en = "No additional references are required."
    zh = "无需额外补充参考文献。"
    replies = [
        ("qwen", {"units": []}), ("qwen", {"units": []}),
        ("ministral", {"units": [unit("No additional references required", en, "constraint")]}),
        ("qwen", {"units": []}), ("qwen", {"units": []}),
        ("ministral", {"units": [unit("无需额外补充参考文献", zh, "constraint")]}),
        ("qwen", {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"],
                             "relation": "equivalent", "reason": "negative obligation"}]}),
        ("ministral", {"verdict": "PASS", "reason": "equivalent", "issues": []}),
    ]
    result, _ = case(en, zh, replies)
    assert result.verdict == "PASS" and result.english_keywords[0].kind == "constraint"


@pytest.mark.parametrize("en,zh,phrase_en,phrase_zh", [
    ("The study outlines its research methodology.", "文章概述了研究采取的技术路线。",
     "research methodology", "技术路线"),
    ("The introduction states the research objectives.", "引言明确研究要完成的主要任务。",
     "research objectives", "主要任务"),
])
def test_ambiguity_pass_can_be_repaired_only_with_independent_original_evidence(
        en, zh, phrase_en, phrase_zh):
    replies = [
        ("qwen", {"units": [unit(phrase_en, phrase_en)]}),
        ("qwen", {"units": [unit(phrase_zh, phrase_zh)]}),
        ("qwen", {"links": []}),
        ("ministral", {"verdict": "PASS", "reason": "reasonable paraphrase", "issues": []}),
        ("qwen", {"verdict": "PASS", "issue_kind": "none", "en_evidence": "",
                   "zh_evidence": "", "reason": "same research intent"}),
        ("ministral", {"status": "supported", "en_evidence": phrase_en,
                        "zh_evidence": phrase_zh, "reason": "plausibly aligned"}),
    ]
    result, calls = case(en, zh, replies, json_schema=True)
    assert result.initial_verdict == "FAIL" and result.verdict == "PASS"
    assert result.arbitrated and len(calls) == 6
    assert result.arbitration.en_evidence == phrase_en
    assert result.arbitration.zh_evidence == phrase_zh


def test_supported_but_irrelevant_original_quotes_cannot_rescue_fail():
    en = "The study outlines its research methodology."
    zh = "文章概述了研究采取的技术路线。"
    replies = [
        ("qwen", {"units": [unit("method", "research methodology")]}),
        ("qwen", {"units": [unit("route", "技术路线")]}),
        ("qwen", {"links": []}),
        ("ministral", {"verdict": "PASS", "reason": "plausible", "issues": []}),
        ("qwen", {"verdict": "PASS", "issue_kind": "none", "en_evidence": "",
                   "zh_evidence": "", "reason": "same"}),
    ] + [
        ("ministral", {"status": "supported", "en_evidence": "The study",
                        "zh_evidence": "文章", "reason": "topic only"})
        for _ in range(3)
    ]
    with pytest.raises(PipelineStageError, match="未覆盖所有初判争议单元") as exc:
        case(en, zh, replies)
    assert exc.value.stage == "alignment_evidence_check"


def test_unsupported_alignment_does_not_fake_pass():
    replies = [
        ("qwen", {"units": [unit("purpose", "purpose")]}),
        ("qwen", {"units": [unit("方法", "方法")]}),
        ("qwen", {"links": []}),
        ("ministral", {"verdict": "PASS", "reason": "unfounded claim", "issues": []}),
        ("qwen", {"verdict": "PASS", "issue_kind": "none", "reason": "same"}),
        ("ministral", {"status": "unsupported", "reason": "purpose != method"}),
    ]
    with pytest.raises(PipelineStageError, match="无法提供足够证据"):
        case("purpose", "方法", replies)


def test_fallback_empty_twice_yields_error_not_pass():
    # Both sides: primary x2, alternate x1. No matcher/reviewer is run.
    replies = [
        (model, {"units": []}) for model in ("qwen", "qwen", "ministral",
                                             "qwen", "qwen", "ministral")
    ]
    with pytest.raises(PipelineStageError, match="两侧备用抽取后仍无关键事实") as exc:
        case("A declarative English sentence.", "一条中文陈述。", replies)
    assert exc.value.stage == "extract_both"
