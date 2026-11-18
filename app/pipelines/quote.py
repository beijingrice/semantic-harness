"""Extraction pipeline with quote evidence (LM_EVIDENCE_MODE=quote). Kept for A/B comparison.

Models retype exact source quotes as evidence; every quote is checked against the
original text (whitespace at CJK/ASCII boundaries may be normalized). The default
paragraph mode (pipelines/paragraph.py) avoids retyping altogether.
"""
from __future__ import annotations

import re

from app.evidence import resolve_evidence
from app.llm import LLMError
from app.models import (
    AlignmentEvidenceDraft,
    ArbitrationDraft,
    CheckResult,
    CoverageAuditDraft,
    ExtractedDraft,
    KeyUnit,
    Link,
    MatchDraft,
    ReviewDraft,
    ReviewIssue,
)
from app.pipelines.base import PipelineStageError, StageRunner
from app.pipelines.common import (
    clean_unsupported_claim,
    dispute_snapshot,
    extraction_summary,
    match_coverage,
)
from app.prompts import (
    ARBITRATION_SYSTEM,
    ATOMIC_REFINEMENT_SYSTEM,
    COVERAGE_AUDIT_SYSTEM,
    EVIDENCE_SUPPORT_SYSTEM,
    EXTRACTION_SYSTEM,
    EXTRACTION_SYSTEM_COARSE,
    EXTRACTION_SYSTEM_LEGACY,
    FALLBACK_EXTRACTION_SYSTEM,
    FALLBACK_EXTRACTION_SYSTEM_COARSE,
    MATCH_SYSTEM,
    REVIEW_SYSTEM,
    arbitrate_user,
    atomic_refinement_user,
    coverage_audit_user,
    evidence_support_user,
    extract_user,
    fallback_extract_user,
    match_user,
    review_user,
)
from app.validators import clear_inclusion_conflicts, clear_numeric_conflicts

# --------------------------------------------------------------------------- source-only triggers


def coverage_audit_reasons(text: str, units: list[KeyUnit]) -> list[str]:
    """Cheap triggers for a source-only omission audit; not a completeness judgment."""
    if not text.strip():
        return []
    if not units:
        return ["nonempty_source_but_zero_units"]
    reasons: list[str] = []

    # Coordination can hide list items that could be omitted independently: one
    # 'and'/'与' with a single umbrella unit, or a three-item list with fewer units.
    en_connectors = len(re.findall(r"\b(?:and|or)\b", text, re.I))
    zh_connectors = len(re.findall(r"、|以及|并且|，并|和|与|及|同时", text))
    en_commas = text.count(",")
    has_coordination = bool(en_connectors or zh_connectors or en_commas or "；" in text)
    en_min = en_commas + (1 if en_connectors else 0) + 1 if en_commas else 1 + en_connectors
    zh_min = zh_connectors + 1
    estimated_count = max(en_min, zh_min, 2 if "；" in text else 1)
    if has_coordination and len(units) < estimated_count:
        reasons.append("possible_collapsed_enumeration")

    # Train/test/evaluation roles that no unit mentions.
    role_tokens = re.findall(r"\b(?:train(?:ed|ing)?|test(?:ed|ing)?|evaluat(?:ed|ion|ing)?)\b|训练|测试|评估",
                             text, re.I)
    unit_claims = " ".join(f"{unit.claim} {unit.evidence}" for unit in units).lower()
    if role_tokens and all(token.lower() not in unit_claims for token in role_tokens):
        reasons.append("possible_missing_action_or_role")

    # A number in the source that no quote contains may be a lost bound or version.
    digits = re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?", text)
    quoted = " ".join(unit.evidence for unit in units)
    if digits and any(value not in quoted for value in digits):
        reasons.append("numeric_text_outside_extracted_evidence")

    # Short quotes from a long multi-clause source.
    if len(text) >= 170:
        covered = [False] * len(text)
        for unit in units:
            begin = text.find(unit.evidence)
            if begin >= 0:
                for j in range(begin, begin + len(unit.evidence)):
                    covered[j] = True
        if sum(covered) / len(text) < 0.35:
            reasons.append("low_evidence_coverage_for_long_source")
    return reasons


def atomic_refinement_needed(text: str, units: list[KeyUnit]) -> bool:
    """Suspected collapsed list; intervals and comparisons are indivisible relations."""
    if not units or not text.strip():
        return False
    if "possible_collapsed_enumeration" not in coverage_audit_reasons(text, units):
        return False
    if re.search(r"\bbetween\b|\bfrom\b.{1,40}\bto\b|介于|范围为|区间|之间", text, re.I):
        return False
    if re.search(r"\b(?:higher|lower|greater|less|more)\s+than\b|高于|低于|优于|劣于", text, re.I):
        return False
    return True


# --------------------------------------------------------------------------- quote validators


def filter_review_issues(issues: list[ReviewIssue], english: str, chinese: str):
    """Keep review issues whose quotes exist in the originals; others become warnings.

    An ungrounded issue is never an automatic FAIL or ERROR, and a grounded one can
    still be wrong: every disagreement goes through arbitration.
    """
    valid: list[ReviewIssue] = []
    warnings: list[str] = []
    for issue in issues:
        if issue.en_evidence:
            grounded_en = resolve_evidence(english, issue.en_evidence)
            if grounded_en is not None:
                issue.en_evidence = grounded_en
        if issue.zh_evidence:
            grounded_zh = resolve_evidence(chinese, issue.zh_evidence)
            if grounded_zh is not None:
                issue.zh_evidence = grounded_zh
        if (issue.en_evidence and issue.en_evidence not in english or
                issue.zh_evidence and issue.zh_evidence not in chinese or
                issue.kind == "missing_in_chinese" and not issue.en_evidence or
                issue.kind == "missing_in_english" and not issue.zh_evidence or
                issue.kind == "conflict" and not (issue.en_evidence and issue.zh_evidence)):
            warnings.append(f"忽略无效复核引用：{issue.reason[:100]}")
        else:
            valid.append(issue)
    return valid, warnings


def validate_arbitration(answer: ArbitrationDraft, english: str, chinese: str,
                         initial_verdict: str = "PASS",
                         allow_missing_pass_evidence: bool = False) -> None:
    if answer.en_evidence:
        grounded_en = resolve_evidence(english, answer.en_evidence)
        if grounded_en is None:
            raise LLMError(f"仲裁英文证据不在原文：{answer.en_evidence!r}")
        answer.en_evidence = grounded_en
    if answer.zh_evidence:
        grounded_zh = resolve_evidence(chinese, answer.zh_evidence)
        if grounded_zh is None:
            raise LLMError(f"仲裁中文证据不在原文：{answer.zh_evidence!r}")
        answer.zh_evidence = grounded_zh
    if answer.verdict == "FAIL":
        if answer.issue_kind == "none":
            raise LLMError("仲裁 FAIL 需要可核对的具体问题类型")
        if answer.issue_kind == "missing_in_chinese" and not answer.en_evidence:
            raise LLMError("仲裁中文缺失需要英文原文证据")
        if answer.issue_kind == "missing_in_english" and not answer.zh_evidence:
            raise LLMError("仲裁英文缺失需要中文原文证据")
        if answer.issue_kind == "conflict" and not (answer.en_evidence and answer.zh_evidence):
            raise LLMError("仲裁冲突需要两侧原文证据")
    elif answer.issue_kind != "none":
        raise LLMError("仲裁 PASS/UNSURE 不能同时声明明确问题")
    if (not allow_missing_pass_evidence and answer.verdict == "PASS"
            and initial_verdict == "FAIL" and not (answer.en_evidence and answer.zh_evidence)):
        raise LLMError("仲裁撤销初判 FAIL 时必须分别引用两侧可核对原文")


def validate_alignment_evidence(answer: AlignmentEvidenceDraft, english: str, chinese: str,
                                en: list[KeyUnit], zh: list[KeyUnit],
                                missing_zh: list[KeyUnit], missing_en: list[KeyUnit],
                                conflicts: list[Link]) -> None:
    """Support for FAIL→PASS: two real quotes that overlap every disputed unit's evidence.

    Real quotes are necessary, not sufficient: the support model must also assert
    'supported'. The overlap check stops a generic quote from hiding a gap.
    """
    if answer.status != "supported":
        return
    if not answer.en_evidence or not answer.zh_evidence:
        raise LLMError("恢复 PASS 证据需要英文和中文两个非空原文片段")
    grounded_en = resolve_evidence(english, answer.en_evidence)
    grounded_zh = resolve_evidence(chinese, answer.zh_evidence)
    if grounded_en is None or grounded_zh is None:
        raise LLMError("恢复 PASS 的证据并非双语原文中的精确连续子串")
    answer.en_evidence = grounded_en
    answer.zh_evidence = grounded_zh

    def intersects(a: str, b: str) -> bool:
        return a in b or b in a

    required_en = {u.id: u.evidence for u in missing_zh}
    required_zh = {u.id: u.evidence for u in missing_en}
    by_en, by_zh = {u.id: u.evidence for u in en}, {u.id: u.evidence for u in zh}
    for conflict in conflicts:
        for unit_id in conflict.en_ids:
            required_en[unit_id] = by_en[unit_id]
        for unit_id in conflict.zh_ids:
            required_zh[unit_id] = by_zh[unit_id]
    if any(not intersects(quote, answer.en_evidence) for quote in required_en.values()):
        raise LLMError("英文证据未覆盖所有初判争议单元，不得恢复 PASS")
    if any(not intersects(quote, answer.zh_evidence) for quote in required_zh.values()):
        raise LLMError("中文证据未覆盖所有初判争议单元，不得恢复 PASS")


# --------------------------------------------------------------------------- extraction


def _numbered(prefix: str, units) -> list[KeyUnit]:
    return [KeyUnit(id=f"{prefix}_{i}", **u.model_dump()) for i, u in enumerate(units, 1)]


def extract(runner: StageRunner, text: str, language: str, prefix: str, model: str,
            fallback_model: str | None, events: list[str]) -> list[KeyUnit]:
    """One extraction pass, an omission audit only when triggered, and a fallback for empty output.

    LM_EXTRACTION_PROMPT=legacy keeps the original retry behaviour for A/B.
    """
    settings = runner.settings
    stage = f"extract_{language.lower()}"

    def validate(answer: ExtractedDraft) -> None:
        if len(answer.units) > settings.max_keywords:
            raise LLMError(f"{language} 提取 {len(answer.units)} 项，超过 MAX_KEYWORDS={settings.max_keywords}。请拆分文本。")
        for unit in answer.units:
            grounded = resolve_evidence(text, unit.evidence)
            if grounded is None:
                raise LLMError(f"{language} 原文证据不是精确连续子串（仅允许中英数字边界空白差异）：{unit.evidence!r}")
            if grounded != unit.evidence:
                events.append(f"{stage}: normalized quote whitespace to original source")
                unit.evidence = grounded
            if settings.extraction_prompt == "coarse":
                clean_unsupported_claim(text, unit, stage, events)

    def request(system: str, target_model: str = model, fallback: bool = False) -> ExtractedDraft:
        return runner.call(
            stage + ("_fallback" if fallback else ""), events, target_model, system,
            fallback_extract_user(text, language, settings.max_keywords) if fallback
            else extract_user(text, language, settings.max_keywords),
            ExtractedDraft, "keyword_extraction_fallback" if fallback else "keyword_extraction",
            validate=validate,
        )

    if settings.extraction_prompt == "legacy":
        result = request(EXTRACTION_SYSTEM_LEGACY)
        if not result.units and text.strip():
            events.append(f"{stage}: empty first extract; retrying with explicit instructions")
            result = request(EXTRACTION_SYSTEM_LEGACY + "\nIMPORTANT: The previous extraction was EMPTY. "
                             "Re-read the original text and extract any real independently "
                             "testable key facts or obligations; do not invent facts.")
    else:
        system = EXTRACTION_SYSTEM_COARSE if settings.extraction_prompt == "coarse" else EXTRACTION_SYSTEM
        result = request(system)
        result, refined = _atomic_refinement(runner, text, language, prefix, model, stage, events,
                                             result, validate)
        result = _coverage_audit(runner, text, language, prefix, model, stage, events, result, refined)

    if not result.units and text.strip():
        result = _fallback_extraction(runner, text, model, fallback_model, stage, events, request)
    return _numbered(prefix, result.units)


def _atomic_refinement(runner, text, language, prefix, model, stage, events, result, validate):
    """Optionally split an umbrella unit into list items (proposition prompt only)."""
    settings = runner.settings
    if not (settings.extraction_prompt == "proposition" and settings.atomic_refinement
            and atomic_refinement_needed(text, _numbered(prefix, result.units))):
        return result, False
    events.append(f"{stage}: atomic refinement triggered")

    def validate_refinement(answer: ExtractedDraft) -> None:
        validate(answer)
        if len({u.evidence for u in answer.units}) != len(answer.units):
            raise LLMError("原子化提取含重复证据；请逐项提供独立引用")

    try:
        refined: ExtractedDraft = runner.call(
            stage + "_atomic_refinement", events, model, ATOMIC_REFINEMENT_SYSTEM,
            atomic_refinement_user(text, language, [u.model_dump() for u in _numbered(prefix, result.units)],
                                   settings.max_keywords),
            ExtractedDraft, "atomic_refinement", validate=validate_refinement)
        if len(refined.units) > len(result.units):
            result = refined
            events.append(f"{stage}: atomic refinement replaced umbrella with {len(result.units)} units")
        else:
            events.append(f"{stage}: atomic refinement returned no useful split; original retained")
    except PipelineStageError as exc:
        # An optional enhancement failing does not turn a grounded extraction into ERROR.
        events.append(f"{stage}: optional atomic refinement failed; original retained ({str(exc)[:180]})")
    return result, True


def _coverage_audit(runner, text, language, prefix, model, stage, events, result, refined):
    """Ask for facts the inventory missed, when a cheap trigger fires and the audit is enabled."""
    settings = runner.settings
    primary_units = _numbered(prefix, result.units)
    reasons = coverage_audit_reasons(text, primary_units)
    # A grounded full list is a valid coarse unit; refinement already looked at lists.
    if settings.extraction_prompt == "coarse" or refined:
        reasons = [r for r in reasons if r != "possible_collapsed_enumeration"]
    if not reasons:
        return result
    if not settings.coverage_audit:
        events.append(f"{stage}: audit disabled; triggers={','.join(reasons)}")
        return result
    events.append(f"{stage}: coverage audit triggered ({','.join(reasons)})")

    def validate_audit(audit: CoverageAuditDraft) -> None:
        if len(audit.missing_units) + len(primary_units) > settings.max_keywords:
            raise LLMError(f"{language} 审计补充后超过 MAX_KEYWORDS，需拆分原文")
        for extra in audit.missing_units:
            grounded = resolve_evidence(text, extra.evidence)
            if grounded is None:
                raise LLMError(f"审计原文证据不存在：{extra.evidence!r}")
            if grounded != extra.evidence:
                events.append(f"{stage}: normalized audit quote to original source")
                extra.evidence = grounded

    audit: CoverageAuditDraft = runner.call(
        stage + "_coverage_audit", events, model, COVERAGE_AUDIT_SYSTEM,
        coverage_audit_user(text, language, [u.model_dump() for u in primary_units],
                            reasons, settings.max_keywords),
        CoverageAuditDraft, "coverage_audit", validate=validate_audit,
    )
    existing = {(u.kind, u.claim.casefold(), u.evidence) for u in result.units}
    extras = []
    for extra in audit.missing_units:
        key = (extra.kind, extra.claim.casefold(), extra.evidence)
        if key not in existing:
            extras.append(extra)
            existing.add(key)
    if extras:
        result = ExtractedDraft(units=[*result.units, *extras])
    events.append(f"{stage}: coverage audit added {len(extras)} unique facts")
    return result


def _fallback_extraction(runner, text, model, fallback_model, stage, events, request):
    """A distinct second extraction for a non-empty source that yielded no facts."""
    settings = runner.settings
    target = fallback_model or model
    events.append(f"{stage}: " +
                  (f"empty after retry; independent fallback model={target}"
                   if settings.extraction_prompt == "legacy" else
                   f"no facts after extraction/audit; fallback model={target}"))
    prompt = (FALLBACK_EXTRACTION_SYSTEM_COARSE if settings.extraction_prompt == "coarse"
              else FALLBACK_EXTRACTION_SYSTEM)
    result = request(prompt, target_model=target, fallback=True)
    if result.units:
        return result
    events.append(f"{stage}: fallback produced no fact units")
    # Last resort for a short coarse source: the exact original sentence as the one
    # inventory item (grounded, not a PASS decision). Long sources fail closed.
    if (settings.extraction_prompt == "coarse" and 12 <= len(text.strip()) <= 220
            and re.search(r"[A-Za-z一-鿿]", text)):
        events.append(f"{stage}: source-anchored recovery after two empty inventories; original comparison required")
        return ExtractedDraft(units=[{"kind": "concept", "claim": text.strip(), "evidence": text.strip()}])
    return result


# --------------------------------------------------------------------------- pipeline


def run(runner: StageRunner, english: str, chinese: str, requested_model: str | None = None,
        requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
    model, verifier = runner.choose_models(mode, requested_model, requested_verifier)
    diagnostics: list[str] = []
    fallback_model = verifier if mode == "dual" else model
    en = extract(runner, english, "English", "en", model, fallback_model, diagnostics)
    zh = extract(runner, chinese, "Chinese", "zh", model, fallback_model, diagnostics)
    if not en and not zh:
        raise PipelineStageError("extract_both", "两侧备用抽取后仍无关键事实，无法可靠判定 PASS。")
    matches: MatchDraft = runner.call(
        "keyword_matching", diagnostics, model, MATCH_SYSTEM,
        match_user(english, chinese, [u.model_dump() for u in en], [u.model_dump() for u in zh]),
        MatchDraft, "keyword_matching",
        validate=lambda answer: match_coverage(en, zh, answer.links),
    )
    missing_zh, missing_en, conflicts = match_coverage(en, zh, matches.links)
    guard_issues = (clear_numeric_conflicts(en, zh, matches.links)
                    + clear_inclusion_conflicts(en, zh, matches.links))
    initial = "FAIL" if (missing_zh or missing_en or conflicts or guard_issues) else "PASS"

    verdict = initial
    review: ReviewDraft | None = None
    arbitration: ArbitrationDraft | None = None
    review_issues: list[ReviewIssue] = []
    review_warnings: list[str] = []
    if mode == "dual":
        review = runner.call(
            "independent_review", diagnostics, verifier, REVIEW_SYSTEM,
            review_user(english, chinese, [u.model_dump() for u in en], [u.model_dump() for u in zh],
                        [link.model_dump() for link in matches.links], initial),
            ReviewDraft, "independent_review",
        )
        review_issues, review_warnings = filter_review_issues(review.issues, english, chinese)
        if review.verdict == "PASS" and review.issues:
            review_warnings.append("复核结论 PASS 却同时提供问题项；以原文和争议仲裁为准")
            diagnostics.append("independent_review: inconsistent PASS with issues; adjudication required")
        if review.verdict == "FAIL" and not review_issues:
            diagnostics.append("independent_review: FAIL without grounded issues; do not treat reason as proof")
        # The reviewer can neither veto an initial PASS nor rescue an initial FAIL on its
        # own: any disagreement goes to a focused, source-grounded arbitration.
        disagreement = review.verdict != initial or initial == "PASS" and bool(review.issues)
        if disagreement and not (guard_issues and initial == "FAIL"):
            snapshot = dispute_snapshot(missing_zh, missing_en, conflicts, guard_issues)
            arbitration = _arbitrate(runner, english, chinese, model, verifier, initial, snapshot,
                                     review, en, zh, missing_zh, missing_en, conflicts, diagnostics)
            if arbitration.verdict == "UNSURE":
                if initial == "PASS":
                    raise LLMError("两模型存在实质异议且仲裁不确定，不能自动 PASS。请人工复核。")
                verdict = "FAIL"  # an unresolved initial FAIL stays flagged
            else:
                verdict = arbitration.verdict
        if guard_issues:
            verdict = "FAIL"  # a proven quantity mismatch cannot be overruled

    summary = extraction_summary(verdict, initial, mode, verifier, review.verdict if review else None,
                                 arbitration.verdict if arbitration else None,
                                 len(missing_zh) + len(missing_en), len(conflicts), len(guard_issues))
    return CheckResult(
        verdict=verdict, model=model, verifier_model=verifier, mode=mode,
        english_keywords=en, chinese_keywords=zh, links=matches.links,
        missing_in_chinese=missing_zh if verdict == "FAIL" else [],
        missing_in_english=missing_en if verdict == "FAIL" else [],
        conflicts=conflicts if verdict == "FAIL" else [],
        preliminary_missing_in_chinese=missing_zh,
        preliminary_missing_in_english=missing_en, preliminary_conflicts=conflicts,
        review=review, review_issues=review_issues,
        arbitration=arbitration, arbitrated=arbitration is not None,
        review_warnings=review_warnings, deterministic_issues=guard_issues,
        initial_verdict=initial, summary=summary, diagnostics=diagnostics,
    )


def _arbitrate(runner, english, chinese, model, verifier, initial, snapshot, review,
               en, zh, missing_zh, missing_en, conflicts, diagnostics) -> ArbitrationDraft:
    """Arbitration, plus independent quote support whenever it overturns an initial FAIL."""
    arbitration: ArbitrationDraft = runner.call(
        "dispute_arbitration", diagnostics, model, ARBITRATION_SYSTEM,
        arbitrate_user(english, chinese, initial, snapshot, review.model_dump()),
        ArbitrationDraft, "dispute_arbitration",
        # Missing PASS quotes are obtained in a separate step below.
        validate=lambda answer: validate_arbitration(answer, english, chinese, initial,
                                                     allow_missing_pass_evidence=True),
    )
    if initial != "FAIL" or arbitration.verdict != "PASS":
        return arbitration
    if not (arbitration.en_evidence and arbitration.zh_evidence):
        # The verifier supplies the two quotes; the host only checks them.
        evidence = runner.call(
            "alignment_evidence_check", diagnostics, verifier, EVIDENCE_SUPPORT_SYSTEM,
            evidence_support_user(english, chinese, snapshot, review.model_dump(), arbitration.model_dump()),
            AlignmentEvidenceDraft, "alignment_evidence_check",
            validate=lambda answer: validate_alignment_evidence(
                answer, english, chinese, en, zh, missing_zh, missing_en, conflicts),
        )
        if evidence.status != "supported":
            raise PipelineStageError(
                "alignment_evidence_check",
                "双模型无法提供足够证据恢复 PASS（" + evidence.status + "）：" + evidence.reason)
        arbitration = arbitration.model_copy(update={
            "en_evidence": evidence.en_evidence,
            "zh_evidence": evidence.zh_evidence,
            "reason": (arbitration.reason + " | 独立证据复核：" + evidence.reason)[:600],
        })
        try:
            validate_arbitration(arbitration, english, chinese, initial)
        except LLMError as exc:
            raise PipelineStageError("alignment_evidence_check", str(exc)) from exc
        diagnostics.append("FAIL→PASS proposal: independently supported by two source quotes")
    # Also applies when the arbiter supplied both quotes itself.
    try:
        validate_alignment_evidence(
            AlignmentEvidenceDraft(status="supported", en_evidence=arbitration.en_evidence,
                                   zh_evidence=arbitration.zh_evidence, reason=arbitration.reason),
            english, chinese, en, zh, missing_zh, missing_en, conflicts)
    except LLMError as exc:
        raise PipelineStageError("dispute_arbitration", str(exc)) from exc
    return arbitration
