"""Source coverage audit protocol tests; simulated LLM output only."""

from __future__ import annotations

import json

import httpx
import pytest

from app.checker import Checker
from app.config import Settings
from app.llm import ChatClient
from app.models import KeyUnit
from app.pipelines.quote import PipelineStageError, coverage_audit_reasons
from app.prompts import COVERAGE_AUDIT_SYSTEM, EXTRACTION_SYSTEM, EXTRACTION_SYSTEM_LEGACY


def unit(claim: str, evidence: str, kind: str = "concept"):
    return {"kind": kind, "claim": claim, "evidence": evidence}


def simulate_en_extraction(text, replies, settings=None):
    settings = settings or Settings(
        evidence_mode="quote",
        model="qwen",
        verifier_model="ministral",
        extraction_prompt="proposition",
        coverage_audit=True,
        atomic_refinement=False,
    )
    requested = []
    queue = iter(replies)

    def handler(req):
        payload = json.loads(req.content)
        requested.append(payload)
        expected_model, schema_output = next(queue)
        assert expected_model == payload["model"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(schema_output, ensure_ascii=False)}, "finish_reason": "stop"}
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    diagnostics = []
    try:
        result = Checker(ChatClient(settings, http), settings).extract(
            text, "English", "en", "qwen", "ministral", diagnostics
        )
        assert next(queue, None) is None
        return result, requested, diagnostics
    finally:
        http.close()


def test_prompt_reframes_short_statements_as_meaningful():
    assert "Even simple academic descriptions count" in EXTRACTION_SYSTEM
    assert "NEGATED OBLIGATION" in COVERAGE_AUDIT_SYSTEM
    assert "generic actions" in EXTRACTION_SYSTEM_LEGACY


def test_nonempty_source_without_units_triggers_audit_not_duplicate_retry():
    en = "The study outlines its research methodology."
    results, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": []}),
            ("qwen", {"missing_units": [unit("methodology is outlined", "research methodology")]}),
        ],
    )
    assert len(calls) == 2 and all(c["model"] == "qwen" for c in calls)
    assert results[0].evidence == "research methodology"
    assert "audit_triggers" in calls[1]["messages"][1]["content"]
    assert "coverage audit added 1 unique facts" in "\n".join(events)


def test_negative_obligation_can_be_rescued_by_audit():
    en = "No additional references are required."
    units, calls, _ = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": []}),
            ("qwen", {"missing_units": [unit("no additional references are required", en, "constraint")]}),
        ],
    )
    assert len(calls) == 2 and units[0].kind == "constraint"


def test_missing_parallel_fact_audited_and_appended_with_stable_ids():
    en = "The abstract must report purpose, methods, and results."
    units, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": [unit("purpose", "purpose", "enumerated_item")]}),
            (
                "qwen",
                {
                    "missing_units": [
                        unit("methods", "methods", "enumerated_item"),
                        unit("results", "results", "enumerated_item"),
                    ]
                },
            ),
        ],
    )
    assert [u.id for u in units] == ["en_1", "en_2", "en_3"]
    assert [u.evidence for u in units] == ["purpose", "methods", "results"]
    assert len(calls) == 2 and any("added 2 unique facts" in v for v in events)


def test_already_covered_dup_from_audit_is_discarded():
    en = "The study lists purpose, methods, and results."
    units, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": [unit("purpose", "purpose")]}),
            ("qwen", {"missing_units": [unit("purpose", "purpose")]}),
        ],
    )
    assert len(units) == 1 and len(calls) == 2
    assert any("added 0 unique facts" in v for v in events)


def test_grounded_single_fact_skips_audit_entirely():
    en = "The model uses PyTorch."
    units, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": [unit("uses PyTorch", "uses PyTorch")]}),
        ],
    )
    assert len(units) == 1 and len(calls) == 1
    assert not any("coverage audit" in v for v in events)


def test_source_only_audit_catches_uncovered_roles():
    reason = coverage_audit_reasons(
        "The model was trained on Dataset A and tested on Dataset B.",
        [
            KeyUnit(id="en_1", **unit("Dataset A", "Dataset A", "entity")),
            KeyUnit(id="en_2", **unit("Dataset B", "Dataset B", "entity")),
        ],
    )
    assert "possible_missing_action_or_role" in reason


def test_audit_off_does_not_send_extra_request_for_nonempty_extraction():
    cfg = Settings(
        evidence_mode="quote", extraction_prompt="proposition", coverage_audit=False, atomic_refinement=False
    )
    en = "The abstract covers purpose, methods, and results."
    units, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": [unit("covers purpose", "purpose")]}),
        ],
        cfg,
    )
    assert len(units) == 1 and len(calls) == 1
    assert any("audit disabled" in line for line in events)


def test_empty_audit_invokes_distinct_ministral_fallback():
    en = "The study outlines its methodology."
    units, calls, events = simulate_en_extraction(
        en,
        [
            ("qwen", {"units": []}),
            ("qwen", {"missing_units": []}),
            ("ministral", {"units": [unit("outlines a methodology", en)]}),
        ],
    )
    assert len(units) == 1 and [p["model"] for p in calls] == ["qwen", "qwen", "ministral"]
    assert any("fallback model=ministral" in line for line in events)


def test_bad_original_quote_from_audit_does_not_pass_validation():
    en = "The study outlines its methodology."
    cfg = Settings(
        evidence_mode="quote",
        extraction_prompt="proposition",
        coverage_audit=True,
        json_schema=True,
        atomic_refinement=False,
    )
    responses = [("qwen", {"units": []})] + [
        ("qwen", {"missing_units": [unit("invented", "THIS IS NOT IN SOURCE")]}) for _ in range(3)
    ]
    with pytest.raises(PipelineStageError, match="审计原文证据不存在") as exc:
        simulate_en_extraction(en, responses, cfg)
    assert exc.value.stage == "extract_english_coverage_audit"


def test_new_primary_pipeline_does_not_convert_missing_source_fact_into_pass():
    from app.models import ExtractedDraft, MatchDraft

    class Fake:
        def __init__(self):
            self.answers = [
                ExtractedDraft(units=[unit_draft(unit("purpose", "purpose"))]),
                ExtractedDraft(units=[unit_draft(unit("目的", "目的"))]),
                MatchDraft(links=[]),
            ]

        def choose_model(self, name=None):
            return "qwen"

        def structured(self, model, system, user, schema, name, validate_result=None):
            answer = self.answers.pop(0)
            assert isinstance(answer, schema)
            if validate_result:
                validate_result(answer)
            return answer

    result = Checker(Fake(), Settings(evidence_mode="quote", extraction_prompt="proposition")).check(
        "purpose", "目的", mode="primary"
    )
    assert result.verdict == "FAIL"


def unit_draft(mapping):
    from app.models import UnitDraft

    return UnitDraft(**mapping)


def test_bad_coverage_audit_json_retry_uses_correct_schema_hint():
    messages = []
    answers = iter(['{"units":[]}', '{"missing_units":[]}'])

    def handler(req):
        payload = json.loads(req.content)
        messages.append(payload["messages"])
        return httpx.Response(200, json={"choices": [{"message": {"content": next(answers)}, "finish_reason": "stop"}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    from app.models import CoverageAuditDraft

    try:
        got = ChatClient(
            Settings(
                evidence_mode="quote",
            ),
            http,
        ).structured("qwen", "test", "original text", CoverageAuditDraft, "coverage_audit")
        assert not got.missing_units
        assert "missing_units" in messages[1][-1]["content"]
        assert "resolved_conflicts" not in messages[1][-1]["content"]
    finally:
        http.close()


def test_diagnostics_detects_new_coverage_audit_and_proposition_prompt():
    from examples.diagnose import stage_from_request

    assert stage_from_request({"messages": [{"role": "system", "content": COVERAGE_AUDIT_SYSTEM}]}) == "coverage_audit"
    assert stage_from_request({"messages": [{"role": "system", "content": EXTRACTION_SYSTEM}]}) == "keyword_extraction"
