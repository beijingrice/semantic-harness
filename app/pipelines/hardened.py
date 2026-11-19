"""LM_PIPELINE=direct_hardened (experimental): whole-paragraph dual judgment + fixed questions.

1. Two models judge the whole original texts (pipelines/direct.py).
2. The host picks up to three fixed questions from source risks and disagreement
   (hardened_rules.choose_questions).
3. A focused model answers each question by citing source spans; the host decides
   from the originals whether the answer proves FAIL/PASS (hardened_rules.prove).
   Anything unproved is REVIEW_REQUIRED.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import create_model

from app.llm import LLMError
from app.models import CheckResult, GroundedEvidenceDraft, RelationVerificationDraft
from app.pipelines.base import StageRunner
from app.pipelines.direct import run_direct
from app.pipelines.hardened_rules import (
    Question,
    choose_questions,
    focus_exception_evidence,
    host_anchor_direction,
    prove,
)
from app.prompts import HARDENED_VERIFY_SYSTEM
from app.source_risks import source_risks, source_spans

_VERIFY_MAX_TOKENS = 1536
_MAX_CITED_CHARS = 600


def _verification_schema(ens: dict[str, str], zhs: dict[str, str]):
    """JSON Schema enums of the offered span IDs (re-checked by the host anyway)."""
    return create_model(
        "RelationVerificationV0215",
        __base__=RelationVerificationDraft,
        en_evidence_id=(Literal.__getitem__(tuple(ens)), ...),
        zh_evidence_id=(Literal.__getitem__(tuple(zhs)), ...),
    )


def _validate_answer(
    answer: RelationVerificationDraft, task: Question, ens: dict[str, str], zhs: dict[str, str]
) -> None:
    if answer.task_id != task.id or answer.question_type != task.kind:
        raise LLMError("定向复核试图回答与主程序指定任务不同的问题")
    if answer.en_evidence_id not in ens or answer.zh_evidence_id not in zhs:
        raise LLMError("复核选择了不存在的原文证据 ID")


def run(
    runner: StageRunner,
    english: str,
    chinese: str,
    requested_model: str | None = None,
    requested_verifier: str | None = None,
    mode: str = "dual",
) -> CheckResult:
    result = run_direct(runner, english, chinese, requested_model, requested_verifier, mode)
    risks = source_risks(english, chinese)
    result.grounded_risks = [f"{r.kind}: {r.detail}" for r in risks]
    if result.deterministic_issues:
        result.diagnostics.append("direct_hardened: preserve original deterministic numeric veto")
        return result
    first = result.direct_primary.verdict
    second = result.review.verdict if result.review else None
    if mode != "dual":
        if risks and result.verdict == "PASS":
            result.verdict = "REVIEW_REQUIRED"
            result.summary = "REVIEW_REQUIRED：单模型尚未独立核对原文风险。"
        return result
    allegations = (
        (result.direct_primary.issues[0].reason if result.direct_primary.issues else "")
        + " "
        + (result.review.issues[0].reason if result.review and result.review.issues else "")
    )
    tasks = choose_questions(english, chinese, first, second, allegations)
    if not tasks:
        result.diagnostics.append("direct_hardened: no source risk / disagreement; preserve direct result")
        return result

    ens = {a.id: a.text for a in source_spans(english, "en")}
    zhs = {a.id: a.text for a in source_spans(chinese, "zh")}
    conclusions = [_answer_task(runner, result, task, english, chinese, ens, zhs, allegations, first) for task in tasks]
    # One proven conflict is enough for FAIL; PASS needs every question proved PASS.
    if "FAIL" in conclusions:
        outcome = "FAIL"
    elif (
        "PASS" in conclusions
        and len(tasks) == len(result.verification_questions)
        and all(x.get("host_conclusion") == "PASS" for x in result.verification_questions)
    ):
        outcome = "PASS"
    elif _unanimous_conflict(first, second, result.verification_questions):
        # Both judges said FAIL and every focused answer also found a conflict: the
        # host rules could not prove it, but nothing contradicts the unanimous FAIL.
        # (The mirror case, unanimous PASS, stays REVIEW_REQUIRED: a false PASS costs more.)
        outcome = "FAIL"
        result.diagnostics.append("direct_hardened: unanimous FAIL kept; every focused answer found a conflict")
    else:
        outcome = "REVIEW_REQUIRED"
    result.verdict = outcome
    result.summary = (
        f"{outcome}：Hardened relation verification；"
        f"主模型 {first}；独立复核 {second}；"
        f"固定疑点 {len(tasks)} 项；逐项结果 "
        + ", ".join(f"{x['task_id']}={x.get('host_conclusion', 'INVALID')}" for x in result.verification_questions)
    )
    return result


def _unanimous_conflict(first: str, second: str | None, answers: list[dict]) -> bool:
    """Both judges FAIL and every question was answered (not skipped/invalid) with CONFLICT."""
    return (
        first == second == "FAIL"
        and bool(answers)
        and all(a.get("finding") == "CONFLICT" and a.get("host_conclusion") != "PASS" for a in answers)
    )


def _answer_task(runner, result, task: Question, english, chinese, ens, zhs, allegations, first) -> str:
    """Ask one fixed question and return the host conclusion (FAIL / PASS / REVIEW_REQUIRED)."""
    if task.kind == "undetermined":
        # prove() has no source check for an undetermined dispute, so the answer could
        # only ever give REVIEW_REQUIRED: skip the call.
        result.verification_questions.append(
            {
                "task_id": task.id,
                "question_type": task.kind,
                "host_conclusion": "REVIEW_REQUIRED",
                "outcome": "SKIPPED_NOT_SOURCE_CHECKABLE",
            }
        )
        result.diagnostics.append(
            f"direct_hardened: {task.id} skipped; an undetermined "
            "dispute cannot be source-checked, so manual review is required"
        )
        return "REVIEW_REQUIRED"
    payload = {
        "task_id": task.id,
        "question_type": task.kind,
        "FIXED_question": task.question,
        "complete_english": english,
        "complete_chinese": chinese,
        "source_warning_english": task.en_hint,
        "source_warning_chinese": task.zh_hint,
        "english_options": [{"id": k, "text": v} for k, v in ens.items()],
        "chinese_options": [{"id": k, "text": v} for k, v in zhs.items()],
        "prior_allegations_are_untrusted": allegations[:500],
    }
    # Ask the model that did not raise the FAIL, to avoid confirming its own allegation.
    target = result.verifier_model if first == "FAIL" else result.model
    try:
        answer = runner.call(
            "direct_hardened_verify",
            result.diagnostics,
            target,
            HARDENED_VERIFY_SYSTEM,
            json.dumps(payload, ensure_ascii=False),
            _verification_schema(ens, zhs),
            "relation_verification_v0215",
            validate=lambda a: _validate_answer(a, task, ens, zhs),
            max_tokens=min(runner.settings.max_tokens, _VERIFY_MAX_TOKENS),
            max_attempts=1,
        )
    except LLMError as exc:
        return _recover_invalid_answer(result, task, exc, english, chinese, ens, zhs)

    e, z = ens[answer.en_evidence_id], zhs[answer.zh_evidence_id]
    focused = None
    if task.kind == "exception_scope":
        focused = focus_exception_evidence(
            ens, zhs, answer.en_evidence_id, answer.zh_evidence_id, answer.en_subject, answer.zh_subject
        )
        if focused is not None:
            _, _, e, z = focused
    if task.kind == "exception_scope" and focused is None:
        conclusion = "REVIEW_REQUIRED"
    else:
        conclusion = prove(task, answer, e, z, english, chinese, english, chinese)
    result.verification_questions.append(
        {
            "task_id": task.id,
            "question_type": task.kind,
            "finding": answer.finding,
            "same_event": answer.same_event,
            "en_evidence_id": answer.en_evidence_id,
            "zh_evidence_id": answer.zh_evidence_id,
            "en_value": answer.en_value,
            "zh_value": answer.zh_value,
            "en_subject": answer.en_subject,
            "zh_subject": answer.zh_subject,
            "en_condition": answer.en_condition,
            "zh_condition": answer.zh_condition,
            "host_conclusion": conclusion,
            "reason": answer.reason,
            **({"host_evidence_id_en": focused[0], "host_evidence_id_zh": focused[1]} if focused else {}),
        }
    )
    result.diagnostics.append(f"direct_hardened: {task.id} => {conclusion} (finding={answer.finding})")
    if len(e) <= _MAX_CITED_CHARS and len(z) <= _MAX_CITED_CHARS:
        result.grounded_evidence_audit = GroundedEvidenceDraft(
            status=answer.finding, en_quote=e, zh_quote=z, reason=answer.reason
        )
    return conclusion


def _recover_invalid_answer(result, task, exc, english, chinese, ens, zhs) -> str:
    """No second model call. For a direction question with an invalid citation ID only,
    the originals alone may still prove a unique winner reversal."""
    id_problem = "证据 ID" in str(exc) or "en_evidence_id" in str(exc) or "zh_evidence_id" in str(exc)
    anchor = host_anchor_direction(ens, zhs) if id_problem and task.kind == "direction_reversal" else None
    if anchor is not None:
        ei, zi, e, z = anchor
        synthetic = RelationVerificationDraft(
            task_id=task.id,
            question_type=task.kind,
            same_event="unknown",
            en_evidence_id=ei,
            zh_evidence_id=zi,
            en_subject="",
            zh_subject="",
            en_condition="",
            zh_condition="",
            en_value="",
            zh_value="",
            finding="UNCERTAIN",
            reason="Independent host source anchoring after invalid model evidence ID",
        )
        if prove(task, synthetic, e, z, english, chinese, english, chinese) == "FAIL":
            result.verification_questions.append(
                {
                    "task_id": task.id,
                    "question_type": task.kind,
                    "host_conclusion": "FAIL",
                    "outcome": "HOST_SOURCE_RECOVERY",
                    "en_evidence_id": ei,
                    "zh_evidence_id": zi,
                    "model_validation_error": str(exc),
                    "reason": "Unique source-verified actor reversal at the same named dataset",
                }
            )
            result.diagnostics.append(f"direct_hardened: {task.id} recovered by source-only unique scoped proof")
            return "FAIL"
    result.diagnostics.append(f"direct_hardened: {task.id} invalid; fail closed: {exc}")
    result.verification_questions.append({"task_id": task.id, "outcome": "INVALID"})
    return "REVIEW_REQUIRED"
