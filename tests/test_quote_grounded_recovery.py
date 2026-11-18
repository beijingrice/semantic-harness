"""Grounded recovery regression: mocks check protocol and safety, not LLM accuracy."""
import json
from dataclasses import replace

import pytest

from app.checker import Checker
from app.config import Settings
from app.prompts import ARBITRATION_SYSTEM, EXTRACTION_SYSTEM_COARSE, MATCH_SYSTEM


class Fake:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []
    def choose_model(self, requested=None):
        return requested or "qwen"
    def choose_verifier(self, requested=None):
        return requested or "ministral"
    def structured(self, model, system, user, schema, schema_name, validate_result=None):
        self.calls.append((model, schema_name, user))
        obj = schema.model_validate(self.rows.pop(0))
        if validate_result:
            validate_result(obj)
        return obj


def cfg(**kw):
    return replace(Settings(evidence_mode="quote", model="qwen", verifier_model="ministral",
                            extraction_prompt="coarse", atomic_refinement=False,
                            coverage_audit=False), **kw)


def units(text, claim=None, kind="concept"):
    return {"units":[{"kind":kind, "claim":claim or text,"evidence":text}]}


def links(relation="equivalent"):
    return {"links":[{"en_ids":["en_1"], "zh_ids":["zh_1"],
                      "relation":relation, "reason":"mocked"}]}


def review(verdict="PASS", issues=None):
    return {"verdict":verdict,"reason":"mocked","issues":issues or []}


def arb(verdict="PASS", issue_kind="none", en="", zh=""):
    return {"verdict":verdict,"issue_kind":issue_kind,
            "en_evidence":en,"zh_evidence":zh,"reason":"mocked"}


@pytest.mark.parametrize("source,claim", [
    ("The abstract describes the research objective and methodology.",
     "The abstract must describe the research objective and methodology."),
    ("摘要介绍了研究目标与研究方法。", "摘要必须包含研究目标与研究方法。"),
    ("The paper evaluates its model.", "The paper only evaluates its model."),
])
def test_unsupported_force_or_exclusivity_is_not_carried_downstream(source,claim):
    fake=Fake([units(source,claim)])
    events=[]
    out=Checker(fake,cfg()).extract(source,"English","en","qwen",diagnostics=events)
    assert out[0].claim == source
    assert out[0].evidence == source
    assert any("replaced unsupported" in x for x in events)


def test_genuine_requirement_is_not_downgraded():
    source = "Students must submit the final report."
    fake = Fake([units(source,"Students must submit the final report")])
    out=Checker(fake,cfg()).extract(source,"English","en","qwen")
    assert out[0].claim == "Students must submit the final report"


def test_empty_primary_and_fallback_keep_short_original_without_inventing_facts():
    src = "作者分析模型的实际效果。"
    fake=Fake([{"units":[]},{"units":[]}])
    events=[]
    out=Checker(fake,cfg()).extract(src,"Chinese","zh","qwen","ministral",events)
    assert len(out)==1
    assert out[0].claim==src and out[0].evidence==src
    assert [x[0] for x in fake.calls]==["qwen","ministral"]
    assert any("source-anchored recovery" in x for x in events)


def test_large_empty_source_is_not_silently_replaced_with_truncated_claim():
    src = "Methods and results are described in the report. " * 6
    fake=Fake([{"units":[]},{"units":[]}])
    out=Checker(fake,cfg()).extract(src,"English","en","qwen","ministral")
    assert out == []


def test_legacy_behavior_still_does_not_invent_units():
    src="The study outlines its methodology."
    fake=Fake([{"units":[]},{"units":[]},{"units":[]}])
    out=Checker(fake,cfg(extraction_prompt="legacy")).extract(src,"English","en","qwen","ministral")
    assert out==[]


def test_dual_reviewer_sees_only_originals_not_generated_claims():
    en="The abstract describes research objectives."
    zh="摘要介绍研究目标。"
    fake=Fake([units(en,"The abstract MUST include research objectives"),
               units(zh,"摘要必须介绍研究目标"),links(),review()])
    result=Checker(fake,cfg()).check(en,zh)
    assert result.verdict=="PASS"
    payload=json.loads(fake.calls[3][2])
    assert payload == {"english_text":en,"chinese_text":zh}
    assert "must" not in json.dumps(payload).lower()
    assert len(fake.calls)==4


def test_inconsistent_pass_with_grounded_issue_is_arbitrated():
    en="The report must include an abstract and references."
    zh="报告应包含摘要。"
    issue={"kind":"missing_in_chinese", "en_evidence":"and references",
           "zh_evidence":"", "reason":"references missing"}
    fake=Fake([units(en),units(zh),links(),review("PASS",[issue]),
               arb("FAIL","missing_in_chinese",en="and references")])
    out=Checker(fake,cfg()).check(en,zh)
    assert out.arbitrated and out.verdict=="FAIL"
    assert any("复核结论 PASS" in x for x in out.review_warnings)
    assert any("inconsistent PASS" in x for x in out.diagnostics)


def test_no_weak_policy_bypass_on_real_modal_difference():
    assert '"must" vs "may/optional"' in MATCH_SYSTEM
    assert '"must" vs "may/optional"' in ARBITRATION_SYSTEM
    assert "Never turn an inferred" in MATCH_SYSTEM
    assert "Do not demand identical verbs" in ARBITRATION_SYSTEM
    assert "CLAIM FIDELITY" in EXTRACTION_SYSTEM_COARSE


@pytest.mark.parametrize("source,claim", [
    ("The model reports accuracy.", "The model does not report accuracy."),
    ("The report contains examples.", "The report contains exactly 9 examples."),
    ("The report contains 20 pages.", "The report contains 30 pages."),
])
def test_unsupported_negation_or_number_is_replaced_with_original(source, claim):
    fake = Fake([units(source, claim)])
    events=[]
    actual = Checker(fake, cfg()).extract(source,"English","en","qwen",diagnostics=events)
    assert actual[0].claim == source
    assert any("replaced unsupported" in e for e in events)
