"""v0.2.4 offline regression: source anchoring, atomic units and task boundary.

These mocked API replies check protocol and control flow, not real model accuracy.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.checker import Checker
from app.config import Settings
from app.evidence import resolve_evidence
from app.llm import ChatClient, LLMError
from app.models import AlignmentEvidenceDraft, ArbitrationDraft, ExtractedDraft, ReviewIssue
from app.pipelines.quote import (
    atomic_refinement_needed,
    filter_review_issues,
    validate_alignment_evidence,
    validate_arbitration,
)
from app.prompts import ARBITRATION_SYSTEM, ATOMIC_REFINEMENT_SYSTEM, MATCH_SYSTEM, REVIEW_SYSTEM


def unit(claim, evidence, kind="concept"):
    return {"kind": kind, "claim": claim, "evidence": evidence}


def mk_checker(replies, *, settings=None):
    settings = settings or Settings(evidence_mode="quote", model="qwen", verifier_model="ministral", json_schema=True,
                                    atomic_refinement=True, extraction_prompt="proposition")
    calls = []
    queue = iter(replies)

    def handler(request):
        payload = json.loads(request.content)
        model, expected_stage, result = next(queue)
        assert payload["model"] == model
        if payload.get("response_format"):
            assert payload["response_format"]["json_schema"]["name"] == expected_stage
        calls.append((model, expected_stage))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(result, ensure_ascii=False)}}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    checker = Checker(ChatClient(settings, http), settings)
    return checker, calls, queue, http


@pytest.mark.parametrize("source,quote,expected", [
    ("模型仅在MNIST上评估，并使用CIFAR-10训练。", "模型仅在 MNIST 上评估", "模型仅在MNIST上评估"),
    ("模型使用数据集B训练，并在数据集A上测试。", "模型使用数据集 B 训练，并在数据集 A 上测试。",
     "模型使用数据集B训练，并在数据集A上测试。"),
    ("报告最多20页。", "最多20 页", "最多20页"),
    ("报告最多 20 页。", "最多20页", "最多 20 页"),
    ("模型在MNIST测试。", "模型在MNIST测试", "模型在MNIST测试"),
])
def test_normalized_quotes_are_mapped_back_to_exact_original(source, quote, expected):
    assert resolve_evidence(source, quote) == expected
    assert expected in source


@pytest.mark.parametrize("source,bad_quote", [
    ("模型仅在MNIST上评估。", "模型仅在CIFAR-10上评估"),
    ("模型使用数据集B训练。", "模型使用数据集A训练"),
    ("至少20页", "至多20页"),
    ("No references are required.", "No references are optional."),
    ("not required", "notrequired"),
    ("数据集A，数据集B", "数据集A、数据集B"),
    ("The report contains text and tables", "The report contains text or tables"),
])
def test_invalid_quotes_are_never_silently_repaired(source, bad_quote):
    assert resolve_evidence(source, bad_quote) is None


def test_extraction_resolves_boundary_spaces_without_retry():
    text = "模型仅在MNIST上评估，并使用CIFAR-10训练。"
    cfg = Settings(evidence_mode="quote", model="qwen", atomic_refinement=False, coverage_audit=False)
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit("在 MNIST 上评估", "在 MNIST 上评估", "relation")]})
    ], settings=cfg)
    try:
        events = []
        got = checker.extract(text, "Chinese", "zh", "qwen", diagnostics=events)
        assert len(got) == 1 and got[0].evidence == "在MNIST上评估"
        assert "normalized quote" in " ".join(events)
        assert len(calls) == 1 and next(queue, None) is None
    finally:
        http.close()


def test_arbitration_evidence_normalized_but_never_semantically_changed():
    answer = ArbitrationDraft(verdict="FAIL", issue_kind="conflict",
                              en_evidence="trained on Dataset A",
                              zh_evidence="数据集 B 训练", reason="swapped")
    validate_arbitration(answer,
                                 "The model was trained on Dataset A.",
                                 "模型使用数据集B训练。")
    assert answer.zh_evidence == "数据集B训练"
    other = ArbitrationDraft(verdict="FAIL", issue_kind="conflict",
                             en_evidence="trained on Dataset A", zh_evidence="数据集 A 训练",
                             reason="changed data")
    with pytest.raises(LLMError, match="仲裁中文证据"):
        validate_arbitration(other, "The model was trained on Dataset A.",
                                     "模型使用数据集B训练。")


def test_review_and_alignment_quotes_receive_only_safe_rewrite():
    issue = ReviewIssue(kind="conflict", en_evidence="Dataset A",
                        zh_evidence="数据集 B 训练", reason="role")
    valid, warnings = filter_review_issues([issue], "Dataset A", "数据集B训练")
    assert valid[0].zh_evidence == "数据集B训练" and not warnings
    supported = AlignmentEvidenceDraft(status="supported", en_evidence="Dataset A",
                                       zh_evidence="数据集 B 训练", reason="claimed equivalent")
    validate_alignment_evidence(supported, "Dataset A", "数据集B训练", [], [], [], [], [])
    assert supported.zh_evidence == "数据集B训练"


def test_atomic_refinement_replaces_umbrella_units_and_skips_redundant_audit():
    source = "The abstract must include purpose, methods, and results."
    cfg = Settings(evidence_mode="quote", model="qwen", verifier_model="ministral", atomic_refinement=True,
                   coverage_audit=True, extraction_prompt="proposition", json_schema=True)
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit(source, source, "constraint")]}),
        ("qwen", "atomic_refinement", {"units": [
            unit("Abstract must include purpose", "purpose", "enumerated_item"),
            unit("Abstract must include methods", "methods", "enumerated_item"),
            unit("Abstract must include results", "results", "enumerated_item")]}),
    ], settings=cfg)
    try:
        events = []
        result = checker.extract(source, "English", "en", "qwen", diagnostics=events)
        assert [u.evidence for u in result] == ["purpose", "methods", "results"]
        assert [u.id for u in result] == ["en_1", "en_2", "en_3"]
        assert calls == [("qwen", "keyword_extraction"), ("qwen", "atomic_refinement")]
        assert any("replaced umbrella" in ev for ev in events)
        assert not any("coverage audit triggered" in ev for ev in events)
        assert next(queue, None) is None
    finally:
        http.close()


def test_atomic_refinement_preserves_initial_if_no_meaningful_split():
    source = "Abstract includes purpose and methods."
    cfg = Settings(evidence_mode="quote", model="qwen", atomic_refinement=True, coverage_audit=True,
                   extraction_prompt="proposition")
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit("includes purpose and methods", source)]}),
        ("qwen", "atomic_refinement", {"units": [unit("includes purpose and methods", source)]}),
    ], settings=cfg)
    try:
        result = checker.extract(source, "English", "en", "qwen")
        assert len(result) == 1 and result[0].evidence == source
        assert len(calls) == 2 and next(queue, None) is None
    finally:
        http.close()


def test_numerical_bounds_and_comparisons_are_not_split():
    from app.models import KeyUnit
    u = [KeyUnit(id="en_1", **unit("range", "10% and 15%"))]
    assert not atomic_refinement_needed("The limit is between 10% and 15%.", u)
    assert not atomic_refinement_needed("Model A is higher than Model B.", u)
    assert atomic_refinement_needed("Abstract includes purpose, methods, and results.", u)


def test_atomic_fallback_does_not_mask_bad_quote_from_primary():
    source = "The abstract contains purpose, methods, and results."
    cfg = Settings(evidence_mode="quote", model="qwen", atomic_refinement=True, coverage_audit=True)
    # A fake source quote cannot be rescued by the optional atomic refiner.
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit("invented", "fake text")]}),
    ] * 3, settings=cfg)
    try:
        with pytest.raises(LLMError, match="连续子串"):
            checker.extract(source, "English", "en", "qwen")
        assert len(calls) == 3
    finally:
        http.close()


def test_academic_norm_policy_explicit_in_all_decision_stages():
    for system in (MATCH_SYSTEM, REVIEW_SYSTEM, ARBITRATION_SYSTEM):
        assert '"contains A, B, C"' in system
        assert '"must" vs "may/optional"' in system
    assert 'TRAIN/TEST' not in ATOMIC_REFINEMENT_SYSTEM or 'training' in ATOMIC_REFINEMENT_SYSTEM


def test_enu03_style_difference_end_to_end_mock_pass():
    en = "The submission contains text, figures, and tables."
    zh = "提交材料应包含正文文字、插图和表格。"
    cfg = Settings(evidence_mode="quote", model="qwen", verifier_model="ministral", extraction_prompt="proposition",
                   atomic_refinement=False, coverage_audit=False, json_schema=True)
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit(en, en)]}),
        ("qwen", "keyword_extraction", {"units": [unit(zh, zh)]}),
        ("qwen", "keyword_matching", {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"],
                                                 "relation": "plausible", "reason": "content same"}]}),
        ("ministral", "independent_review", {"verdict": "PASS", "reason": "same required components", "issues": []}),
    ], settings=cfg)
    try:
        result = checker.check(en, zh)
        assert result.verdict == "PASS"
        assert len(calls) == 4 and next(queue, None) is None
    finally:
        http.close()


def test_substantive_optional_vs_required_still_fails_end_to_end_mock():
    en = "Students must submit a report."
    zh = "学生可以选择是否提交报告。"
    cfg = Settings(evidence_mode="quote", model="qwen", verifier_model="ministral", extraction_prompt="proposition",
                   atomic_refinement=False, coverage_audit=False, json_schema=True)
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit(en, en, "constraint")]}),
        ("qwen", "keyword_extraction", {"units": [unit(zh, zh, "constraint")]}),
        ("qwen", "keyword_matching", {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"],
                                                 "relation": "conflict", "reason": "must vs optional"}]}),
        ("ministral", "independent_review", {"verdict": "FAIL", "reason": "substantive force differs", "issues": []}),
    ], settings=cfg)
    try:
        result = checker.check(en, zh)
        assert result.verdict == "FAIL" and len(result.conflicts) == 1
        assert len(calls) == 4 and next(queue, None) is None
    finally:
        http.close()


def test_atomic_refinement_triggers_on_zh_conjoined_roles_and_four_item_list():
    from app.models import KeyUnit
    claim = [KeyUnit(id="zh_1", **unit("数据集角色", "模型使用数据集B训练，并在数据集A上测试。"))]
    assert atomic_refinement_needed("模型使用数据集B训练，并在数据集A上测试。", claim)
    en = "The abstract includes purpose, methods, results, and conclusions."
    units = [KeyUnit(id=f"en_{i}", **unit(x, x))
             for i, x in enumerate(("purpose", "methods", "results"), 1)]
    assert atomic_refinement_needed(en, units)


def test_invalid_optional_refinement_keeps_valid_extraction_with_warning():
    source = "The abstract includes purpose and methods."
    cfg = Settings(evidence_mode="quote", model="qwen", atomic_refinement=True, coverage_audit=True,
                   extraction_prompt="proposition")
    checker, calls, queue, http = mk_checker([
        ("qwen", "keyword_extraction", {"units": [unit("includes purpose and methods", source)]}),
        *[("qwen", "atomic_refinement", {"units": [unit("made-up", "not in original")]} ) for _ in range(3)],
    ], settings=cfg)
    try:
        diagnostics = []
        result = checker.extract(source, "English", "en", "qwen", diagnostics=diagnostics)
        assert result[0].evidence == source
        assert len(calls) == 4
        assert any("optional atomic refinement failed" in line for line in diagnostics)
    finally:
        http.close()


def test_atomic_refinement_retry_hint_is_specific_and_valid():
    messages = []
    responses = iter(['{"missing_units":[]}', '{"units":[]}'])
    def handler(req):
        payload = json.loads(req.content)
        messages.append(payload["messages"])
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)},
                                                       "finish_reason": "stop"}]})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        draft = ChatClient(Settings(evidence_mode="quote", ), http).structured(
            "qwen", "system", "input", ExtractedDraft, "atomic_refinement")
        assert not draft.units
        assert '"units"' in messages[1][-1]["content"]
        assert '"resolved_conflicts"' not in messages[1][-1]["content"]
    finally:
        http.close()
