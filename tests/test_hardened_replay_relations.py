"""v0.2.14: real first two LM outputs replayed; third answer adapted to new schema.

Mocked third-stage outputs are NOT live model accuracy claims. The source host
must independently prove any verdict regardless of the model's finding or
same_event opinion. Safety failures abstain.
"""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.checker import Checker
from app.config import Settings
from app.llm import ChatClient
from app.models import RelationVerificationDraft
from app.pipelines.hardened_rules import choose_questions, prove, scalar_source_scope
from app.source_risks import source_spans

REPORT = json.loads(Path("tests/fixtures/replay_v0213_questions_11.json").read_text())
CASES = {case["id"]: case for case in REPORT["cases"]}


def third_from_real(key):
    """Verbatim v0213 model operands/reason, *adapted* typed v0214 fields."""
    v = CASES[key]["detail"]["verification_questions"][0]
    return {
        "task_id": v["task_id"],
        "question_type": v["question_type"],
        "same_event": "no" if v.get("same_requirement") == "no" else "yes",
        "en_evidence_id": v["en_evidence_id"],
        "zh_evidence_id": v["zh_evidence_id"],
        "en_subject": v["en_value"],
        "zh_subject": v["zh_value"],
        "en_condition": "",
        "zh_condition": "",
        "en_value": v["en_value"],
        "zh_value": v["zh_value"],
        "finding": v["finding"],
        "reason": v["reason"],
    }


def run(key, third=None, pipeline="direct_hardened", mode="dual"):
    c = CASES[key]
    pri = c["detail"]["direct_primary"]
    review = c["detail"]["review"]
    r = {
        "verdict": review["verdict"],
        "reason": review["reason"],
        "issues": [
            {x: i[x] for x in ("kind", "en_paragraph_ids", "zh_paragraph_ids", "reason")} for i in review["issues"]
        ],
    }
    replies = [pri] + ([r] if mode == "dual" else []) + ([third] if third is not None else [])
    reqs = []

    def handler(req):
        data = json.loads(req.content)
        reqs.append(data)
        assert len(reqs) <= len(replies)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(replies[-1 if False else len(reqs) - 1], ensure_ascii=False)},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    settings = replace(
        Settings(),
        pipeline=pipeline,
        model="qwen-local",
        verifier_model="ministral-local",
        evidence_mode="paragraph",
        json_schema=True,
    )
    client = ChatClient(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    client.start_trace()
    try:
        result = Checker(client, settings).check(c["en"], c["zh"], mode=mode)
        return result, reqs, client.trace_snapshot()
    finally:
        client.close()


# v0.3.0: these replays now run through direct_hardened (direct_relations was
# removed). Hardened exception checks require an entity-specific subject; the
# v0.2.13-era V2-EXC-03 reply names a whole sentence, so the host abstains
# (fail closed). The entity-subject replay of V2-EXC-03 -> FAIL is covered in
# test_v0215_hardened.py.
@pytest.mark.parametrize(
    "key,expected",
    [
        ("V2-NUM-15", "FAIL"),
        ("V2-NUM-13", "FAIL"),
        ("V2-EXC-03", "REVIEW_REQUIRED"),
        ("V2-NUM-03", "PASS"),
        ("V2-ROL-07", "FAIL"),
    ],
)
def test_replay_actual_prior_model_explanations_with_new_relation_contract(key, expected):
    third = third_from_real(key)
    result, requests, trace = run(key, third)
    assert result.verdict == expected, (key, result.verification_questions)
    assert len(requests) == 3 and len(trace) == 3
    assert result.verification_questions[0]["host_conclusion"] == expected
    assert result.verification_questions[0]["same_event"] == third["same_event"]
    payload = json.loads(requests[-1]["messages"][1]["content"])
    assert payload["task_id"] == third["task_id"]
    assert payload["complete_english"] == CASES[key]["en"]
    assert payload["complete_chinese"] == CASES[key]["zh"]
    assert "same_event" in requests[-1]["messages"][0]["content"]


@pytest.mark.parametrize("key", ["V2-NUM-12", "V2-ENU-12", "V2-ENU-13", "V2-MIX-09", "V2-MIX-04", "V2-MIX-03"])
def test_no_risk_has_exactly_two_model_calls(key):
    result, req, trace = run(key)
    assert result.verdict == CASES[key]["predicted"]
    assert not result.verification_questions
    assert len(req) == len(trace) == 2


def verdict_direct(task_id, **changes):
    """Generate a clean signed answer for one scenario."""
    cases = {
        "Q_COUNTED_ENTITY": "V2-NUM-15",
        "Q_NUMERIC_BOUNDARY": "V2-NUM-13",
        "Q_EXCEPTION_SCOPE": "V2-EXC-03",
        "Q_SCALAR_EQUIVALENCE": "V2-NUM-03",
        "Q_COMPARISON_WINNER": "V2-ROL-07",
    }
    x = third_from_real(cases[task_id])
    x.update(changes)
    return RelationVerificationDraft(**x)


def local_prove(key, draft, en_id=None, zh_id=None, en_source=None, zh_source=None):
    c = CASES[key]
    en = en_source or c["en"]
    zh = zh_source or c["zh"]
    q = choose_questions(c["en"], c["zh"], c["detail"]["direct_primary"]["verdict"], c["detail"]["review"]["verdict"])[
        0
    ]
    ens = {x.id: x.text for x in source_spans(en, "en")}
    zhs = {x.id: x.text for x in source_spans(zh, "zh")}
    ei = en_id or draft.en_evidence_id
    zi = zh_id or draft.zh_evidence_id
    return prove(q, draft, ens[ei], zhs[zi], en, zh, en, zh)


def test_event_no_is_not_a_synonym_for_difference_in_outcome():
    # V2-EXC-03 excluded: see the hardened exception-subject note above.
    for key in ("V2-NUM-15", "V2-NUM-13", "V2-ROL-07"):
        audit = third_from_real(key)
        audit["same_event"] = "no"
        r, _, _ = run(key, audit)
        assert r.verdict == "FAIL"


def test_found_conflict_cannot_force_fail_if_mismatched_numeric_expression():
    for updates in ({"en_value": ">=10%"}, {"zh_value": ">10%"}, {"en_evidence_id": "en_all"}):
        x = third_from_real("V2-NUM-13")
        x.update(updates)
        r, _, _ = run("V2-NUM-13", x)
        assert r.verdict == "REVIEW_REQUIRED"


def test_disconnected_page_limits_must_abstain():
    a = verdict_direct("Q_COUNTED_ENTITY")
    en = "The main report has a 30-page limit. The cover page is included within the appendix limit."
    zh = "正文报告有30页限制。封面不计入30页限制。"
    assert local_prove("V2-NUM-15", a, en_source=en, zh_source=zh, en_id="en_s2", zh_id="zh_s2") == "REVIEW_REQUIRED"


def test_different_limit_values_must_abstain():
    a = verdict_direct("Q_COUNTED_ENTITY")
    en = "The main report has a 40-page limit. The cover page is included within the main-report limit."
    zh = "正文报告有30页限制。封面不计入30页限制。"
    assert local_prove("V2-NUM-15", a, en_source=en, zh_source=zh, en_id="en_s2", zh_id="zh_s2") == "REVIEW_REQUIRED"


def test_wrong_entity_operand_cannot_force_different_entity_claim():
    a = verdict_direct("Q_COUNTED_ENTITY", en_subject="reference list", zh_subject="参考文献列表")
    assert local_prove("V2-NUM-15", a) == "REVIEW_REQUIRED"


def test_different_dataset_scope_must_abstain():
    a = verdict_direct("Q_COMPARISON_WINNER")
    assert local_prove("V2-ROL-07", a, en_id="en_s1", zh_id="zh_s2") == "REVIEW_REQUIRED"
    assert local_prove("V2-ROL-07", a, en_id="en_all", zh_id="zh_all") == "REVIEW_REQUIRED"


def test_wrong_winner_operand_must_abstain():
    a = verdict_direct("Q_COMPARISON_WINNER", en_subject="baseline")
    assert local_prove("V2-ROL-07", a) == "REVIEW_REQUIRED"


def test_scalar_equivalent_only_for_single_same_activity_no_new_duty():
    a = verdict_direct("Q_SCALAR_EQUIVALENCE")
    assert local_prove("V2-NUM-03", a) == "PASS"
    assert scalar_source_scope("The survey ran for exactly six weeks.", "问卷调查持续了42天，既未延长也未缩短。")
    for en, zh in (
        ("The survey ran for approximately six weeks.", "问卷调查持续了42天。"),
        ("The survey ran for exactly six weeks.", "问卷调查持续了42天，须上传证明。"),
        ("The study ran for exactly six weeks.", "问卷调查持续了42天。"),
        ("The survey ran for exactly six weeks.", "问卷调查持续了42天，后来又收集了新资料。"),
    ):
        assert not scalar_source_scope(en, zh)


def test_false_model_equivalence_does_not_erase_source_proven_mismatch():
    # V2-EXC-03 excluded: see the hardened exception-subject note above.
    for key in ("V2-NUM-15", "V2-NUM-13", "V2-ROL-07"):
        a = third_from_real(key)
        a.update(finding="EQUIVALENT", same_event="yes")
        r, _, _ = run(key, a)
        assert r.verdict == "FAIL"


def test_invalid_task_id_one_attempt_and_fail_closed():
    a = third_from_real("V2-NUM-15")
    a["task_id"] = "Q_HALLUCINATED"
    r, req, trace = run("V2-NUM-15", a)
    assert r.verdict == "REVIEW_REQUIRED" and len(req) == 3
    assert trace[-1]["outcome"] == "schema_or_grounding_invalid"


def test_primary_only_high_risk_cannot_pass():
    r, req, _ = run("V2-NUM-15", mode="primary")
    assert r.verdict == "REVIEW_REQUIRED" and len(req) == 1
