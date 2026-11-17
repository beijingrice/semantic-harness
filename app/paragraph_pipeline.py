"""v0.2.7 paragraph-first wire protocol.

All source references in model-generated JSON are IDs, never retyped quotations.
Original immutable paragraphs become public evidence for display only.
Semantic support is NOT proved by a valid source ID.
"""
from __future__ import annotations

from app.llm import LLMError
from app.extraction_fast_path import source_anchor_pair, bounded_clause_anchor_pair
from app.models import (
    ArbitrationDraft, CheckResult, KeyUnit, Link, MatchDraft,
    ParagraphAlignmentDraft, ParagraphArbitrationDraft, ParagraphExtractedDraft,
    ParagraphReviewDraft, ParagraphDisputeRecheckDraft, NumericEquivalenceCertificate,
    ReviewDraft, ReviewIssue,
)
from app.paragraphs import build_paragraphs, compact_paragraphs, paragraph_evidence, verify_ids
from app.dispute_resolution import (numeric_equivalence_certificates,
                                    certificate_covers_issue, narrow_exactness_refutation)
from app.prompts import (
    ARBITRATION_SYSTEM_PARAGRAPH, EVIDENCE_SUPPORT_SYSTEM_PARAGRAPH,
    EXTRACTION_SYSTEM_PARAGRAPH, FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH,
    MATCH_SYSTEM, REVIEW_SYSTEM_PARAGRAPH, REVIEW_OBJECTION_CHECK_SYSTEM_PARAGRAPH, match_user,
    paragraph_arbitrate_user, paragraph_extract_user, paragraph_review_user, paragraph_recheck_user,
)
from app.validators import clear_numeric_conflicts, clear_inclusion_conflicts, source_isolated_quantity_conflicts


def extract_paragraph(checker, source: str, language: str, prefix: str,
                      model: str, fallback_model: str | None,
                      diagnostics: list[str]) -> list[KeyUnit]:
    paragraphs = build_paragraphs(source, prefix)
    stage = f"extract_{language.lower()}"
    events = diagnostics
    raw_claims_by_identity: dict[int, str] = {}

    def validate(answer: ParagraphExtractedDraft) -> None:
        if len(answer.units) > checker.settings.max_keywords:
            raise LLMError(f"超出 MAX_KEYWORDS={checker.settings.max_keywords}; 请拆分输入文本")
        for unit in answer.units:
            # Keep the actual model-produced claim for EXACT deduplication.
            # A later conservative claim rollback can map different claims to
            # the same verbatim paragraph; those must not be merged blindly.
            raw_claims_by_identity[id(unit)] = unit.claim
            try:
                verify_ids(unit.paragraph_ids, paragraphs)
            except ValueError as exc:
                raise LLMError(str(exc)) from exc
            # A claim's truth is not proved by its paragraph IDs. Apply only the
            # existing narrow hallucination guard, using CITED source context.
            cited_source = paragraph_evidence(unit.paragraph_ids, paragraphs)
            # The helper expects an evidence attribute for safe rollback.
            legacy_view = KeyUnit(id="temp", kind=unit.kind, claim=unit.claim,
                                  evidence=cited_source, paragraph_ids=unit.paragraph_ids)
            checker.clean_unsupported_claim(cited_source, legacy_view, stage, events)
            unit.claim = legacy_view.claim[:220]

    def invoke(prompt: str, target_model: str, fallback: bool = False):
        return checker._structured(
            stage + ("_fallback" if fallback else ""), events,
            target_model, prompt,
            paragraph_extract_user(source, language, compact_paragraphs(paragraphs),
                                   checker.settings.max_keywords),
            ParagraphExtractedDraft, "paragraph_extraction_fallback" if fallback else "paragraph_extraction",
            validate_result=validate,
            max_tokens=(min(checker.settings.max_tokens, checker.settings.extraction_max_tokens)
                        if checker.settings.extraction_fast_path
                        and 0 < checker.settings.extraction_max_tokens
                        and len(source.strip()) <= 220 and len(paragraphs) == 1
                        else None),
        )

    result = invoke(EXTRACTION_SYSTEM_PARAGRAPH, model)
    if not result.units and source.strip():
        target = fallback_model or model
        events.append(f"{stage}: empty inventory; fallback model={target}")
        result = invoke(FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH, target, True)
        if not result.units and 12 <= len(source.strip()) <= 220:
            # No hallucinated text and no automatic PASS: an original source
            # paragraph is an inventory placeholder requiring matching/review.
            if len(paragraphs) == 1:
                p = paragraphs[0]
                result = ParagraphExtractedDraft(units=[{
                    "kind": "concept", "claim": p.text,
                    "paragraph_ids": [p.id]
                }])
                events.append(f"{stage}: source-anchored recovery (one original paragraph)")
    # Exact duplicate semantic units add no information and may force false
    # coverage gaps / expensive pairwise matching. Drop only exact normalized
    # claim + paragraph-reference duplicates (never fuzzy semantic merges).
    units = []
    seen = set()
    for unit in result.units:
        original = raw_claims_by_identity.get(id(unit), unit.claim)
        key = (" ".join(original.casefold().split()), tuple(unit.paragraph_ids))
        if key in seen:
            events.append(f"{stage}: discarded exact duplicate claim with identical paragraph IDs")
            continue
        seen.add(key)
        units.append(KeyUnit(id=f"{prefix}_{len(units) + 1}", kind=unit.kind,
                             claim=unit.claim, paragraph_ids=unit.paragraph_ids,
                             evidence=paragraph_evidence(unit.paragraph_ids, paragraphs)))
    # Source-grounded compression of pathological single-sentence extraction.
    # Do not let dozens of synonymous model claims amplify into a quadratic
    # matching task. The ORIGINAL sentence is the lossless fallback claim:
    # semantic scrutiny remains with Matching and the original-only reviewer.
    if len(paragraphs) == 1 and len(source.strip()) <= 220:
        original = paragraphs[0].text
        if len(units) >= 9:
            events.append(f"{stage}: collapsed {len(units)} over-extracted units to one verbatim source anchor")
            return [KeyUnit(id=f"{prefix}_1", kind="constraint", claim=original,
                            paragraph_ids=[paragraphs[0].id], evidence=original)]

        # A model sometimes emits both a complete source sentence and isolated
        # nouns copied from it. Those nouns aren't independent obligations.
        # Drop ONLY literal subspans (concept/entity) if the same original is
        # already represented by a complete, source-grounded full-sentence
        # claim. A paraphrased or novel obligation is NEVER removed here.
        normalized = lambda text: "".join(text.casefold().split()).rstrip(".。！!？?")
        if any(normalized(u.claim) == normalized(original) for u in units):
            retained = []
            for u in units:
                if (u.kind in ("concept", "entity")
                        and normalized(u.claim) != normalized(original)
                        and normalized(u.claim) in normalized(original)):
                    events.append(f"{stage}: removed redundant literal entity/concept span")
                    continue
                retained.append(u)
            units = retained
            for index, u in enumerate(units, 1):
                u.id = f"{prefix}_{index}"
    return units


def _verify_review(answer: ParagraphReviewDraft, en_paragraphs, zh_paragraphs) -> None:
    if answer.verdict == "PASS" and answer.issues:
        raise LLMError("PASS 不允许同时提供问题项")
    if answer.verdict == "FAIL" and not answer.issues:
        raise LLMError("FAIL 必须提供至少一个具体问题及 Paragraph ID")
    for issue in answer.issues:
        try:
            verify_ids(issue.en_paragraph_ids, en_paragraphs, allow_empty=True)
            verify_ids(issue.zh_paragraph_ids, zh_paragraphs, allow_empty=True)
        except ValueError as exc:
            raise LLMError(str(exc)) from exc
        if issue.kind == "missing_in_chinese" and not issue.en_paragraph_ids:
            raise LLMError("中文缺失须给出英文来源段落")
        if issue.kind == "missing_in_english" and not issue.zh_paragraph_ids:
            raise LLMError("英文缺失须给出中文来源段落")
        if issue.kind == "conflict" and not (issue.en_paragraph_ids and issue.zh_paragraph_ids):
            raise LLMError("冲突须给出双语来源段落")


def _review_public(review: ParagraphReviewDraft, en_paragraphs, zh_paragraphs) -> ReviewDraft:
    return ReviewDraft(verdict=review.verdict, reason=review.reason,
                       issues=[ReviewIssue(kind=i.kind, reason=i.reason,
                               en_evidence=paragraph_evidence(i.en_paragraph_ids, en_paragraphs),
                               zh_evidence=paragraph_evidence(i.zh_paragraph_ids, zh_paragraphs),
                               en_paragraph_ids=i.en_paragraph_ids,
                               zh_paragraph_ids=i.zh_paragraph_ids) for i in review.issues])


def _verify_arb(answer: ParagraphArbitrationDraft, en_paragraphs, zh_paragraphs,
                initial: str) -> None:
    try:
        verify_ids(answer.en_paragraph_ids, en_paragraphs, allow_empty=True)
        verify_ids(answer.zh_paragraph_ids, zh_paragraphs, allow_empty=True)
    except ValueError as exc:
        raise LLMError(str(exc)) from exc
    if answer.verdict == "FAIL":
        if answer.issue_kind == "none":
            raise LLMError("FAIL 必须指出问题类型")
        if answer.issue_kind == "conflict" and not (answer.en_paragraph_ids and answer.zh_paragraph_ids):
            raise LLMError("冲突必须引用双语原文段落")
        if answer.issue_kind == "missing_in_chinese" and not answer.en_paragraph_ids:
            raise LLMError("中文缺失必须引用英文来源段落")
        if answer.issue_kind == "missing_in_english" and not answer.zh_paragraph_ids:
            raise LLMError("英文缺失必须引用中文来源段落")
    if answer.verdict == "PASS" and initial == "FAIL" and not (
            answer.en_paragraph_ids and answer.zh_paragraph_ids):
        raise LLMError("FAIL→PASS 必须分别引用双语原文段落")


def _arb_public(answer: ParagraphArbitrationDraft, en_paragraphs, zh_paragraphs) -> ArbitrationDraft:
    return ArbitrationDraft(verdict=answer.verdict, issue_kind=answer.issue_kind,
                            en_evidence=paragraph_evidence(answer.en_paragraph_ids, en_paragraphs),
                            zh_evidence=paragraph_evidence(answer.zh_paragraph_ids, zh_paragraphs),
                            en_paragraph_ids=answer.en_paragraph_ids,
                            zh_paragraph_ids=answer.zh_paragraph_ids,
                            reason=answer.reason)


def _verify_support(answer: ParagraphAlignmentDraft, en_paragraphs, zh_paragraphs,
                    en: list[KeyUnit], zh: list[KeyUnit], missing_zh, missing_en, conflicts) -> None:
    try:
        verify_ids(answer.en_paragraph_ids, en_paragraphs, allow_empty=True)
        verify_ids(answer.zh_paragraph_ids, zh_paragraphs, allow_empty=True)
    except ValueError as exc:
        raise LLMError(str(exc)) from exc
    if answer.status != "supported":
        return
    if not (answer.en_paragraph_ids and answer.zh_paragraph_ids):
        raise LLMError("独立支持 PASS 需要双语段落 ID")
    required_en = {u.id for u in missing_zh}
    required_zh = {u.id for u in missing_en}
    for conflict in conflicts:
        required_en.update(conflict.en_ids)
        required_zh.update(conflict.zh_ids)
    by_en = {u.id: u for u in en}
    by_zh = {u.id: u for u in zh}
    for uid in required_en:
        if not set(by_en[uid].paragraph_ids).intersection(answer.en_paragraph_ids):
            raise LLMError(f"独立证据未引用英文争议单元 {uid} 的来源段落")
    for uid in required_zh:
        if not set(by_zh[uid].paragraph_ids).intersection(answer.zh_paragraph_ids):
            raise LLMError(f"独立证据未引用中文争议单元 {uid} 的来源段落")
    # NB: matching ID coverage is *only a structural check*, not semantic proof.
    # The separate support model must itself affirm the contested equivalence.


def _verify_objection_recheck(answer: ParagraphDisputeRecheckDraft,
                              review_issues, en_paragraphs, zh_paragraphs) -> None:
    """One exact assessment for each reviewer issue, no ungrounded refutations."""
    if len(answer.issues) != len(review_issues):
        raise LLMError("定向复核必须逐项回应每条原复核异议")
    indexes = [issue.issue_index for issue in answer.issues]
    if sorted(indexes) != list(range(len(review_issues))):
        raise LLMError("定向复核问题索引缺失、重复或不存在")
    for item in answer.issues:
        try:
            verify_ids(item.en_paragraph_ids, en_paragraphs, allow_empty=True)
            verify_ids(item.zh_paragraph_ids, zh_paragraphs, allow_empty=True)
        except ValueError as exc:
            raise LLMError(str(exc)) from exc
        if item.status == "refuted":
            issue = review_issues[item.issue_index]
            if not (item.en_paragraph_ids and item.zh_paragraph_ids):
                raise LLMError("反驳复核异议必须引用双语来源")
            # Refutation IDs must actually overlap cited source of that issue.
            if issue.en_paragraph_ids and not set(item.en_paragraph_ids).intersection(issue.en_paragraph_ids):
                raise LLMError("反驳未引用对应英文争议段落")
            if issue.zh_paragraph_ids and not set(item.zh_paragraph_ids).intersection(issue.zh_paragraph_ids):
                raise LLMError("反驳未引用对应中文争议段落")


def check_paragraph(checker, english: str, chinese: str,
                    requested_model: str | None = None,
                    requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
    from app.pipeline import PipelineStageError
    if mode not in ("primary", "dual"):
        raise LLMError("mode 只能为 primary 或 dual")
    model = checker.client.choose_model(requested_model)
    verifier = checker.client.choose_verifier(requested_verifier) if mode == "dual" else ""
    if mode == "dual" and verifier == model:
        raise LLMError("主模型与复核模型相同，请选择不同模型")
    en_paragraphs = build_paragraphs(english, "en")
    zh_paragraphs = build_paragraphs(chinese, "zh")
    diagnostics: list[str] = ["evidence_mode: paragraph (immutable source IDs)"]
    fallback_model = verifier if mode == "dual" else model
    anchors = (source_anchor_pair(en_paragraphs, zh_paragraphs)
               if checker.settings.extraction_fast_path
               and checker.settings.extraction_prompt == "coarse"
               and not checker.settings.coverage_audit
               and not checker.settings.atomic_refinement else None)
    if anchors is not None:
        en, zh = anchors
        diagnostics.append("source_anchor_fast_path: both sources are one cohesive short clause; "
                           "skipped two extraction requests; Matching, full-source Review and Guards remain")
    elif (checker.settings.extraction_fast_path and checker.settings.extraction_prompt == "coarse"
          and not checker.settings.coverage_audit and not checker.settings.atomic_refinement
          and (anchors := bounded_clause_anchor_pair(en_paragraphs, zh_paragraphs)) is not None):
        en, zh = anchors
        diagnostics.append("bounded_clause_fast_path: two explicit clauses on each side; "
                           "verbatim anchors, no extraction requests; original-source review and guards remain")
    else:
        en = extract_paragraph(checker, english, "English", "en", model, fallback_model, diagnostics)
        zh = extract_paragraph(checker, chinese, "Chinese", "zh", model, fallback_model, diagnostics)
    if not en and not zh:
        raise PipelineStageError("extract_both", "双语两侧均未提取到事实，不能自动 PASS")
    matches: MatchDraft = checker._structured(
        "keyword_matching", diagnostics, model, MATCH_SYSTEM,
        match_user(english, chinese, [u.model_dump() for u in en],
                   [u.model_dump() for u in zh]), MatchDraft, "keyword_matching",
        validate_result=lambda answer: checker.coverage(en, zh, answer.links),
    )
    missing_zh, missing_en, conflicts = checker.coverage(en, zh, matches.links)
    # Full-paragraph display evidence may contain unrelated numbers. For the
    # narrow numeric safety net compare each extracted fact's concise claim;
    # cross-language semantics are still the model's responsibility.
    numeric_en = [u.model_copy(update={"evidence": u.claim}) for u in en]
    numeric_zh = [u.model_copy(update={"evidence": u.claim}) for u in zh]
    numeric_issues = (clear_numeric_conflicts(numeric_en, numeric_zh, matches.links)
                      + clear_inclusion_conflicts(numeric_en, numeric_zh, matches.links))
    if not numeric_issues:
        numeric_issues += source_isolated_quantity_conflicts(
            english, chinese, en, zh, matches.links, en_paragraphs, zh_paragraphs)

    initial = "FAIL" if missing_zh or missing_en or conflicts or numeric_issues else "PASS"
    verdict = initial
    review: ReviewDraft | None = None
    arb: ArbitrationDraft | None = None
    warnings: list[str] = []
    dispute_recheck: ParagraphDisputeRecheckDraft | None = None

    if mode == "dual":
        review_wire = checker._structured(
            "independent_review", diagnostics, verifier, REVIEW_SYSTEM_PARAGRAPH,
            paragraph_review_user(english, chinese, compact_paragraphs(en_paragraphs),
                                  compact_paragraphs(zh_paragraphs)),
            ParagraphReviewDraft, "paragraph_review",
            validate_result=lambda answer: _verify_review(answer, en_paragraphs, zh_paragraphs),
        )
        review = _review_public(review_wire, en_paragraphs, zh_paragraphs)
        if review.verdict != initial and not (numeric_issues and initial == "FAIL"):
            snapshot = {
                "missing_in_chinese": [u.model_dump() for u in missing_zh],
                "missing_in_english": [u.model_dump() for u in missing_en],
                "conflicts": [c.model_dump() for c in conflicts],
                "numeric_guard_issues": numeric_issues,
            }
            arb_wire = checker._structured(
                "dispute_arbitration", diagnostics, model, ARBITRATION_SYSTEM_PARAGRAPH,
                paragraph_arbitrate_user(english, chinese,
                                         compact_paragraphs(en_paragraphs),
                                         compact_paragraphs(zh_paragraphs),
                                         initial, snapshot, review_wire.model_dump()),
                ParagraphArbitrationDraft, "paragraph_arbitration",
                validate_result=lambda answer: _verify_arb(answer, en_paragraphs, zh_paragraphs, initial),
            )
            arb = _arb_public(arb_wire, en_paragraphs, zh_paragraphs)
            if arb.verdict == "PASS" and initial == "FAIL":
                # In paragraph mode broad source citations alone cannot establish
                # that *each* disputed proposition is supported. Require a
                # separate semantic support verdict even if IDs are valid.
                support_input = paragraph_arbitrate_user(
                    english, chinese, compact_paragraphs(en_paragraphs),
                    compact_paragraphs(zh_paragraphs), initial, snapshot,
                    {"independent_review": review_wire.model_dump(),
                     "proposed_arbitration": arb_wire.model_dump()},
                )
                supported = checker._structured(
                    "alignment_evidence_check", diagnostics, verifier,
                    EVIDENCE_SUPPORT_SYSTEM_PARAGRAPH, support_input,
                    ParagraphAlignmentDraft, "paragraph_alignment_support",
                    validate_result=lambda answer: _verify_support(
                        answer, en_paragraphs, zh_paragraphs, en, zh,
                        missing_zh, missing_en, conflicts),
                )
                if supported.status != "supported":
                    raise PipelineStageError("alignment_evidence_check",
                                             f"无法支持 FAIL→PASS：{supported.status}: {supported.reason}")
                diagnostics.append("FAIL→PASS: independent semantic support confirmed; ID validity alone is insufficient")
            if arb.verdict == "PASS" and initial == "PASS" and review.verdict == "FAIL" and review.issues:
                # The arbiter's repeated PASS does not invalidate a cited
                # reviewer issue. One targeted verifier call can re-examine the
                # objections, aided by SOURCE-VERIFIED numeric certificates.
                certificates = numeric_equivalence_certificates(
                    en, zh, matches.links, en_paragraphs, zh_paragraphs)
                diagnostics.append(f"dispute: {len(certificates)} source-verified numeric certificate(s)")
                dispute_recheck = checker._structured(
                    "review_objection_check", diagnostics, verifier,
                    REVIEW_OBJECTION_CHECK_SYSTEM_PARAGRAPH,
                    paragraph_recheck_user(
                        english, chinese, compact_paragraphs(en_paragraphs),
                        compact_paragraphs(zh_paragraphs),
                        [i.model_dump() for i in review_wire.issues],
                        arb_wire.model_dump(),
                        [c.model_dump() for c in certificates]),
                    ParagraphDisputeRecheckDraft, "paragraph_dispute_recheck",
                    validate_result=lambda answer: _verify_objection_recheck(
                        answer, review_wire.issues, en_paragraphs, zh_paragraphs),
                )
                # A second vote is *not* a proof. A return to PASS requires
                # every issue refuted AND a relevant independently checkable
                # formal certificate for an isolated precision objection.
                # Do not trust the model's basis label on its own; even a
                # claimed 'formal_numeric' basis cannot discharge unrelated
                # semantic conditions in the same paragraph.
                refutations = {item.issue_index: item for item in dispute_recheck.issues}
                proven = bool(certificates) and all(
                    refutations[index].status == "refuted" and
                    narrow_exactness_refutation(issue, refutations[index], certificates,
                                                en_paragraphs, zh_paragraphs)
                    for index, issue in enumerate(review_wire.issues)
                )
                if proven:
                    verdict = "PASS"
                    diagnostics.append("review_objection_check: all reviewer issues refuted with verified formal certificates")
                else:
                    verdict = "REVIEW_REQUIRED"
                    diagnostics.append("review_objection_check: unresolved or non-formal disagreement; manual review required")
            elif arb.verdict == "UNSURE":
                if initial == "PASS":
                    verdict = "REVIEW_REQUIRED"
                    diagnostics.append("dispute_arbitration: uncertain; manual review required")
                else:
                    verdict = "FAIL"
            elif (initial == "FAIL" and review.verdict == "PASS" and arb.verdict == "FAIL"):
                # Two Qwen decisions are not independent corroboration of a
                # disputed conflict, especially if Matching invents a modality
                # error by attaching 'may' from clause B to obligation A.
                # The independent source-only reviewer disagrees. Preserve
                # uncertainty rather than converting an unverified allegation
                # into a definite FAIL. A deterministic guard may still FAIL.
                verdict = "REVIEW_REQUIRED"
                diagnostics.append("dispute_arbitration: initial FAIL and arbiter FAIL contradict "
                                   "independent PASS; no verified conflict; manual review required")
            else:
                verdict = arb.verdict
        if numeric_issues:
            verdict = "FAIL"

    summary = (f"{verdict}：主模型初判 {initial}；"
               + ("未启用独立复核。" if mode == "primary" else
                  f"{verifier} 复核 {review.verdict}；"
                  f"{'已执行争议仲裁，结论 ' + arb.verdict if arb else '无需争议仲裁'}。")
               + f" 初步缺失 {len(missing_zh)+len(missing_en)} 项、"
               f"匹配冲突 {len(conflicts)} 项、确定性约束保护 {len(numeric_issues)} 项。")
    return CheckResult(
        evidence_mode="paragraph", english_paragraphs=en_paragraphs,
        chinese_paragraphs=zh_paragraphs,
        verdict=verdict, model=model, verifier_model=verifier, mode=mode,
        english_keywords=en, chinese_keywords=zh, links=matches.links,
        missing_in_chinese=missing_zh if verdict == "FAIL" else [],
        missing_in_english=missing_en if verdict == "FAIL" else [],
        conflicts=conflicts if verdict == "FAIL" else [],
        preliminary_missing_in_chinese=missing_zh,
        preliminary_missing_in_english=missing_en,
        preliminary_conflicts=conflicts,
        review=review, review_issues=review.issues if review else [],
        arbitration=arb, arbitrated=arb is not None,
        dispute_recheck=dispute_recheck,
        review_warnings=warnings, deterministic_issues=numeric_issues,
        initial_verdict=initial, summary=summary, diagnostics=diagnostics,
    )
