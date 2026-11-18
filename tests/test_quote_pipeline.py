"""Synthetic model outputs test branching and failure safety, not real semantic accuracy."""

import pytest

from app.checker import Checker
from app.config import Settings
from app.llm import LLMError
from app.models import AlignmentEvidenceDraft, ArbitrationDraft, ExtractedDraft, MatchDraft, ReviewDraft


def unit(claim, evidence, kind="concept"):
    return {"kind": kind, "claim": claim, "evidence": evidence}


def link(en, zh, relation="equivalent"):
    return {"en_ids": en, "zh_ids": zh, "relation": relation, "reason": "test"}


def review(verdict="PASS", issues=None, reason="reviewed"):
    return {"verdict": verdict, "issues": issues or [], "reason": reason}


def arb(verdict="PASS", kind="none", en="", zh="", reason="arbitrated"):
    return {"verdict": verdict, "issue_kind": kind, "en_evidence": en, "zh_evidence": zh, "reason": reason}


class FakeLLM:
    def __init__(self, en, zh, links, audit=None, arbitration=None, empty_first=False, support=None):
        self.answers = [
            ExtractedDraft.model_validate({"units": en}),
            ExtractedDraft.model_validate({"units": zh}),
            MatchDraft.model_validate({"links": links}),
        ]
        if empty_first:
            self.answers.insert(0, ExtractedDraft(units=[]))
        if audit is not None:
            self.answers.append(ReviewDraft.model_validate(audit))
        if arbitration is not None:
            self.answers.append(ArbitrationDraft.model_validate(arbitration))
        if support is not None:
            self.answers.append(AlignmentEvidenceDraft.model_validate(support))
        self.calls = []

    def choose_model(self, requested=None):
        return requested or "qwen"

    def choose_verifier(self, requested=None):
        return requested or "ministral"

    def structured(self, model, system, user, schema_model, schema_name, validate_result=None):
        self.calls.append((model, schema_name, user))
        answer = self.answers.pop(0)
        assert isinstance(answer, schema_model), (type(answer), schema_model)
        if validate_result is not None:
            validate_result(answer)
        return answer


def run(en_text, zh_text, en, zh, links, audit=None, arbitration=None, mode="dual", empty_first=False, support=None):
    fake = FakeLLM(
        en,
        zh,
        links,
        audit if mode == "dual" else None,
        arbitration if mode == "dual" else None,
        empty_first=empty_first,
        support=support if mode == "dual" else None,
    )
    result = Checker(fake, Settings(evidence_mode="quote", extraction_prompt="legacy", coverage_audit=False)).check(
        en_text, zh_text, mode=mode
    )
    assert [c[0] for c in fake.calls][: 3 + int(empty_first)] == ["qwen"] * (3 + int(empty_first))
    if mode == "dual":
        assert fake.calls[3 + int(empty_first)][0] == "ministral"
    assert not fake.answers
    return result


def test_complete_list_passes():
    r = run(
        "purpose, methods, results",
        "目的、方法、结果",
        [unit("purpose", "purpose"), unit("methods", "methods"), unit("results", "results")],
        [unit("目的", "目的"), unit("方法", "方法"), unit("结果", "结果")],
        [link(["en_1"], ["zh_1"]), link(["en_2"], ["zh_2"]), link(["en_3"], ["zh_3"])],
        review(),
    )
    assert r.verdict == "PASS" and r.arbitration is None


def test_missing_enumeration_stays_fail_with_agreeing_reviewer():
    r = run(
        "purpose, methods, results",
        "目的、方法",
        [unit("purpose", "purpose"), unit("methods", "methods"), unit("results", "results")],
        [unit("目的", "目的"), unit("方法", "方法")],
        [link(["en_1"], ["zh_1"]), link(["en_2"], ["zh_2"])],
        review(
            "FAIL", [{"kind": "missing_in_chinese", "en_evidence": "results", "zh_evidence": "", "reason": "缺少结果"}]
        ),
    )
    assert r.verdict == "FAIL"
    assert len(r.missing_in_chinese) == 1


def test_reviewer_false_conflict_cannot_veto_correct_initial_pass():
    r = run(
        "Two models were compared for accuracy.",
        "研究比较了两个模型的准确率。",
        [unit("comparison", "Two models were compared for accuracy.")],
        [unit("比较", "研究比较了两个模型的准确率。")],
        [link(["en_1"], ["zh_1"])],
        review(
            "FAIL",
            [
                {
                    "kind": "conflict",
                    "en_evidence": "accuracy",
                    "zh_evidence": "准确率",
                    "reason": "双方相同却填conflict",
                }
            ],
        ),
        arb("PASS"),
    )
    assert r.initial_verdict == "PASS" and r.verdict == "PASS" and r.arbitrated


def test_review_finds_real_false_pass_and_arbitration_confirms():
    r = run(
        "purpose, methods, results",
        "研究目的、研究方法",
        [unit("purpose and methods", "purpose, methods")],
        [unit("研究目的方法", "研究目的、研究方法")],
        [link(["en_1"], ["zh_1"])],
        review(
            "FAIL",
            [{"kind": "missing_in_chinese", "en_evidence": "results", "zh_evidence": "", "reason": "缺 results"}],
        ),
        arb("FAIL", "missing_in_chinese", en="results"),
    )
    assert r.initial_verdict == "PASS" and r.verdict == "FAIL"
    assert r.arbitration.en_evidence == "results"


def test_wrong_initial_fail_can_be_rescued_after_arbitration():
    r = run(
        "research methodology",
        "技术路线",
        [unit("research methodology", "research methodology")],
        [unit("技术路线", "技术路线")],
        [],
        review("PASS", reason="语义可合理对应"),
        arb("PASS", en="research methodology", zh="技术路线"),
    )
    assert r.initial_verdict == "FAIL" and r.verdict == "PASS" and r.arbitrated
    assert not r.missing_in_chinese and not r.missing_in_english


def test_review_cannot_directly_rescue_a_missing_keyword():
    r = run(
        "purpose, results",
        "研究目的",
        [unit("purpose", "purpose"), unit("results", "results")],
        [unit("研究目的", "研究目的")],
        [link(["en_1"], ["zh_1"])],
        review("PASS", reason="错误声称两侧一致"),
        arb("FAIL", "missing_in_chinese", en="results"),
    )
    assert r.verdict == "FAIL" and r.arbitrated


def test_invalid_reviewer_quote_does_not_crash_or_override_without_arbiter():
    r = run(
        "purpose",
        "目的",
        [unit("purpose", "purpose")],
        [unit("目的", "目的")],
        [link(["en_1"], ["zh_1"])],
        review(
            "FAIL", [{"kind": "conflict", "en_evidence": "NOT IN TEXT", "zh_evidence": "目的", "reason": "fabrication"}]
        ),
        arb("PASS"),
    )
    assert r.verdict == "PASS" and r.review_warnings


def test_invalid_arbitration_quote_is_error_not_pass():
    with pytest.raises(LLMError, match="仲裁英文证据"):
        run(
            "purpose",
            "目的",
            [unit("purpose", "purpose")],
            [unit("目的", "目的")],
            [link(["en_1"], ["zh_1"])],
            review("FAIL"),
            arb("FAIL", "conflict", en="fake", zh="目的"),
        )


def test_numeric_guard_never_overridden_by_second_model():
    r = run(
        "Training accuracy reached 80%.",
        "训练准确率达到60%。",
        [unit("training accuracy 80", "reached 80%", "constraint")],
        [unit("训练准确率60", "达到60%", "constraint")],
        [link(["en_1"], ["zh_1"], "plausible")],
        review("PASS"),
    )
    assert r.verdict == "FAIL" and r.deterministic_issues and not r.arbitrated


def test_uncertain_arbitration_for_initial_pass_returns_error():
    with pytest.raises(LLMError, match="人工复核"):
        run(
            "purpose",
            "目的",
            [unit("purpose", "purpose")],
            [unit("目的", "目的")],
            [link(["en_1"], ["zh_1"])],
            review("UNSURE"),
            arb("UNSURE"),
        )


def test_unsure_arbitration_on_initial_fail_preserves_fail():
    r = run(
        "purpose", "方法", [unit("purpose", "purpose")], [unit("方法", "方法")], [], review("UNSURE"), arb("UNSURE")
    )
    assert r.verdict == "FAIL"


def test_primary_mode_never_calls_verifier():
    r = run(
        "purpose",
        "目的",
        [unit("purpose", "purpose")],
        [unit("目的", "目的")],
        [link(["en_1"], ["zh_1"])],
        mode="primary",
    )
    assert r.verdict == "PASS" and r.review is None and r.verifier_model == ""


def test_empty_first_extraction_is_retried():
    r = run(
        "purpose",
        "目的",
        [unit("purpose", "purpose")],
        [unit("目的", "目的")],
        [link(["en_1"], ["zh_1"])],
        review(),
        empty_first=True,
    )
    assert r.verdict == "PASS"


def test_unknown_link_id_retries_validate_and_never_passes():
    with pytest.raises(LLMError, match="不存在"):
        run(
            "purpose",
            "目的",
            [unit("purpose", "purpose")],
            [unit("目的", "目的")],
            [link(["en_900"], ["zh_1"])],
            mode="primary",
        )


def test_bad_evidence_never_passes():
    with pytest.raises(LLMError, match="连续子串"):
        run(
            "image, audio, text samples",
            "图像",
            [unit("image samples", "image samples")],
            [unit("图像", "图像")],
            [],
            mode="primary",
        )


def test_same_models_rejected():
    fake = FakeLLM([], [], [], review())
    with pytest.raises(LLMError, match="两个不同"):
        Checker(fake, Settings(evidence_mode="quote", extraction_prompt="legacy", coverage_audit=False)).check(
            "x", "甲", "same", "same"
        )


def test_initial_fail_cannot_be_rescued_without_evidence():
    with pytest.raises(LLMError, match="无法提供足够证据"):
        run(
            "research methodology",
            "技术路线",
            [unit("method", "research methodology")],
            [unit("route", "技术路线")],
            [],
            review("PASS"),
            arb("PASS"),
            support={"status": "uncertain", "en_evidence": "", "zh_evidence": "", "reason": "cannot confirm"},
        )


def test_missing_arbitration_evidence_is_checked_by_second_model():
    result = run(
        "research methodology",
        "技术路线",
        [unit("method", "research methodology")],
        [unit("route", "技术路线")],
        [],
        review("PASS"),
        arb("PASS"),
        support={
            "status": "supported",
            "en_evidence": "research methodology",
            "zh_evidence": "技术路线",
            "reason": "methodological approach",
        },
    )
    assert result.initial_verdict == "FAIL" and result.verdict == "PASS"
    assert any("alignment_evidence_check" in x for x in result.diagnostics)
    assert result.arbitration.en_evidence == "research methodology"
