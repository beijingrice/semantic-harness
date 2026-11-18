"""Default extraction pipeline (LM_EVIDENCE_MODE=paragraph).

Sources are split into immutable paragraphs with IDs. Every model answer cites
paragraph IDs only; displayed evidence is copied from the original paragraphs,
never retyped by a model. A valid ID does not prove that the claim is supported.

Stages: extraction (or a source-only fast path) → matching → deterministic
numeric guards → independent review → arbitration on disagreement → support
check (FAIL→PASS) or objection recheck (PASS / review FAIL / arbitration PASS).
"""
from __future__ import annotations

from app.dispute_resolution import narrow_exactness_refutation, numeric_equivalence_certificates
from app.extraction_fast_path import bounded_clause_anchor_pair, source_anchor_pair
from app.llm import LLMError
from app.models import (
    ArbitrationDraft,
    CheckResult,
    KeyUnit,
    MatchDraft,
    ParagraphAlignmentDraft,
    ParagraphArbitrationDraft,
    ParagraphDisputeRecheckDraft,
    ParagraphExtractedDraft,
    ParagraphReviewDraft,
    ReviewDraft,
)
from app.paragraphs import build_paragraphs, compact_paragraphs, paragraph_evidence, verify_ids
from app.pipelines.base import PipelineStageError, StageRunner
from app.pipelines.common import (
    SHORT_SOURCE_CHARS,
    clean_unsupported_claim,
    dispute_snapshot,
    extraction_summary,
    match_coverage,
)
from app.pipelines.paragraph_checks import (
    arbitration_public,
    review_public,
    verify_arbitration,
    verify_objection_recheck,
    verify_review,
    verify_support,
)
from app.prompts import (
    ARBITRATION_SYSTEM_PARAGRAPH,
    EVIDENCE_SUPPORT_SYSTEM_PARAGRAPH,
    EXTRACTION_SYSTEM_PARAGRAPH,
    FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH,
    MATCH_SYSTEM,
    REVIEW_OBJECTION_CHECK_SYSTEM_PARAGRAPH,
    REVIEW_SYSTEM_PARAGRAPH,
    match_user,
    paragraph_arbitrate_user,
    paragraph_extract_user,
    paragraph_recheck_user,
    paragraph_review_user,
)
from app.validators import clear_inclusion_conflicts, clear_numeric_conflicts, source_isolated_quantity_conflicts

# A short single-sentence source extracted into this many units is over-extraction.
_OVER_EXTRACTION_UNITS = 9


# --------------------------------------------------------------------------- extraction


def extract(runner: StageRunner, source: str, language: str, prefix: str, model: str,
            fallback_model: str | None, events: list[str]) -> list[KeyUnit]:
    settings = runner.settings
    paragraphs = build_paragraphs(source, prefix)
    stage = f"extract_{language.lower()}"
    short_single = len(source.strip()) <= SHORT_SOURCE_CHARS and len(paragraphs) == 1
    # The model's own claim before any rollback, for exact deduplication: the rollback
    # can map different claims onto the same verbatim paragraph.
    raw_claims: dict[int, str] = {}

    def validate(answer: ParagraphExtractedDraft) -> None:
        if len(answer.units) > settings.max_keywords:
            raise LLMError(f"超出 MAX_KEYWORDS={settings.max_keywords}; 请拆分输入文本")
        for unit in answer.units:
            raw_claims[id(unit)] = unit.claim
            try:
                verify_ids(unit.paragraph_ids, paragraphs)
            except ValueError as exc:
                raise LLMError(str(exc)) from exc
            cited = paragraph_evidence(unit.paragraph_ids, paragraphs)
            view = KeyUnit(id="temp", kind=unit.kind, claim=unit.claim,
                           evidence=cited, paragraph_ids=unit.paragraph_ids)
            clean_unsupported_claim(cited, view, stage, events)
            unit.claim = view.claim[:220]

    max_tokens = (min(settings.max_tokens, settings.extraction_max_tokens)
                  if settings.extraction_fast_path and 0 < settings.extraction_max_tokens and short_single
                  else None)

    def invoke(prompt: str, target_model: str, fallback: bool = False) -> ParagraphExtractedDraft:
        return runner.call(
            stage + ("_fallback" if fallback else ""), events, target_model, prompt,
            paragraph_extract_user(source, language, compact_paragraphs(paragraphs), settings.max_keywords),
            ParagraphExtractedDraft, "paragraph_extraction_fallback" if fallback else "paragraph_extraction",
            validate=validate, max_tokens=max_tokens,
        )

    result = invoke(EXTRACTION_SYSTEM_PARAGRAPH, model)
    if not result.units and source.strip():
        target = fallback_model or model
        events.append(f"{stage}: empty inventory; fallback model={target}")
        result = invoke(FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH, target, True)
        if not result.units and 12 <= len(source.strip()) <= 220 and len(paragraphs) == 1:
            # The original paragraph as an inventory placeholder: grounded, and still
            # subject to matching and review (never an automatic PASS).
            p = paragraphs[0]
            result = ParagraphExtractedDraft(units=[{"kind": "concept", "claim": p.text, "paragraph_ids": [p.id]}])
            events.append(f"{stage}: source-anchored recovery (one original paragraph)")

    units = _dedupe(result.units, raw_claims, paragraphs, prefix, stage, events)
    if short_single:
        units = _compact_short_source(units, paragraphs[0], prefix, stage, events)
    return units


def _dedupe(drafts, raw_claims, paragraphs, prefix, stage, events) -> list[KeyUnit]:
    """Drop exact duplicates (normalized model claim + identical paragraph IDs) only."""
    units: list[KeyUnit] = []
    seen = set()
    for unit in drafts:
        original = raw_claims.get(id(unit), unit.claim)
        key = (" ".join(original.casefold().split()), tuple(unit.paragraph_ids))
        if key in seen:
            events.append(f"{stage}: discarded exact duplicate claim with identical paragraph IDs")
            continue
        seen.add(key)
        units.append(KeyUnit(id=f"{prefix}_{len(units) + 1}", kind=unit.kind,
                             claim=unit.claim, paragraph_ids=unit.paragraph_ids,
                             evidence=paragraph_evidence(unit.paragraph_ids, paragraphs)))
    return units


def _normalized(text: str) -> str:
    return "".join(text.casefold().split()).rstrip(".。！!？?")


def _compact_short_source(units: list[KeyUnit], paragraph, prefix, stage, events) -> list[KeyUnit]:
    """Keep a short single-sentence source from inflating into many units.

    Over-extraction collapses to one verbatim unit (matching and review still see the
    originals). When one unit already is the full sentence, literal concept/entity
    sub-spans of it are dropped; a paraphrased or new obligation is never removed.
    """
    original = paragraph.text
    if len(units) >= _OVER_EXTRACTION_UNITS:
        events.append(f"{stage}: collapsed {len(units)} over-extracted units to one verbatim source anchor")
        return [KeyUnit(id=f"{prefix}_1", kind="constraint", claim=original,
                        paragraph_ids=[paragraph.id], evidence=original)]
    full = _normalized(original)
    if not any(_normalized(u.claim) == full for u in units):
        return units
    retained = []
    for u in units:
        claim = _normalized(u.claim)
        if u.kind in ("concept", "entity") and claim != full and claim in full:
            events.append(f"{stage}: removed redundant literal entity/concept span")
            continue
        retained.append(u)
    for index, u in enumerate(retained, 1):
        u.id = f"{prefix}_{index}"
    return retained


def _units(runner, english, chinese, en_paragraphs, zh_paragraphs, model, fallback_model, diagnostics):
    """Source-only fast paths for short sources, otherwise model extraction of both sides."""
    settings = runner.settings
    fast = (settings.extraction_fast_path and settings.extraction_prompt == "coarse"
            and not settings.coverage_audit and not settings.atomic_refinement)
    if fast:
        anchors = source_anchor_pair(en_paragraphs, zh_paragraphs)
        if anchors is not None:
            diagnostics.append("source_anchor_fast_path: both sources are one cohesive short clause; "
                               "skipped two extraction requests; Matching, full-source Review and Guards remain")
            return anchors
        anchors = bounded_clause_anchor_pair(en_paragraphs, zh_paragraphs)
        if anchors is not None:
            diagnostics.append("bounded_clause_fast_path: two explicit clauses on each side; "
                               "verbatim anchors, no extraction requests; original-source review and guards remain")
            return anchors
    en = extract(runner, english, "English", "en", model, fallback_model, diagnostics)
    zh = extract(runner, chinese, "Chinese", "zh", model, fallback_model, diagnostics)
    return en, zh


def _guard_issues(english, chinese, en, zh, links, en_paragraphs, zh_paragraphs) -> list[str]:
    """Deterministic numeric/inclusion conflicts; any finding vetoes the verdict to FAIL.

    Unit claims are compared rather than whole cited paragraphs, which can contain
    unrelated numbers. Then source-isolated quantities are checked.
    """
    numeric_en = [u.model_copy(update={"evidence": u.claim}) for u in en]
    numeric_zh = [u.model_copy(update={"evidence": u.claim}) for u in zh]
    issues = (clear_numeric_conflicts(numeric_en, numeric_zh, links)
              + clear_inclusion_conflicts(numeric_en, numeric_zh, links))
    if not issues:
        issues += source_isolated_quantity_conflicts(english, chinese, en, zh, links, en_paragraphs, zh_paragraphs)
    return issues


# --------------------------------------------------------------------------- pipeline


def run(runner: StageRunner, english: str, chinese: str, requested_model: str | None = None,
        requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
    model, verifier = runner.choose_models(mode, requested_model, requested_verifier)
    en_paragraphs = build_paragraphs(english, "en")
    zh_paragraphs = build_paragraphs(chinese, "zh")
    diagnostics: list[str] = ["evidence_mode: paragraph (immutable source IDs)"]
    fallback_model = verifier if mode == "dual" else model
    en, zh = _units(runner, english, chinese, en_paragraphs, zh_paragraphs, model, fallback_model, diagnostics)
    if not en and not zh:
        raise PipelineStageError("extract_both", "双语两侧均未提取到事实，不能自动 PASS")
    matches: MatchDraft = runner.call(
        "keyword_matching", diagnostics, model, MATCH_SYSTEM,
        match_user(english, chinese, [u.model_dump() for u in en], [u.model_dump() for u in zh]),
        MatchDraft, "keyword_matching",
        validate=lambda answer: match_coverage(en, zh, answer.links),
    )
    missing_zh, missing_en, conflicts = match_coverage(en, zh, matches.links)
    guard_issues = _guard_issues(english, chinese, en, zh, matches.links, en_paragraphs, zh_paragraphs)
    initial = "FAIL" if missing_zh or missing_en or conflicts or guard_issues else "PASS"

    dispute = _Dispute(runner, english, chinese, en_paragraphs, zh_paragraphs, model, verifier,
                       en, zh, matches, missing_zh, missing_en, conflicts, guard_issues, initial, diagnostics)
    verdict = dispute.resolve() if mode == "dual" else initial

    summary = extraction_summary(verdict, initial, mode, verifier,
                                 dispute.review.verdict if dispute.review else None,
                                 dispute.arbitration.verdict if dispute.arbitration else None,
                                 len(missing_zh) + len(missing_en), len(conflicts), len(guard_issues))
    review = dispute.review
    return CheckResult(
        evidence_mode="paragraph", english_paragraphs=en_paragraphs, chinese_paragraphs=zh_paragraphs,
        verdict=verdict, model=model, verifier_model=verifier, mode=mode,
        english_keywords=en, chinese_keywords=zh, links=matches.links,
        missing_in_chinese=missing_zh if verdict == "FAIL" else [],
        missing_in_english=missing_en if verdict == "FAIL" else [],
        conflicts=conflicts if verdict == "FAIL" else [],
        preliminary_missing_in_chinese=missing_zh,
        preliminary_missing_in_english=missing_en,
        preliminary_conflicts=conflicts,
        review=review, review_issues=review.issues if review else [],
        arbitration=dispute.arbitration, arbitrated=dispute.arbitration is not None,
        dispute_recheck=dispute.recheck,
        review_warnings=[], deterministic_issues=guard_issues,
        initial_verdict=initial, summary=summary, diagnostics=diagnostics,
    )


class _Dispute:
    """Dual-mode decision: independent review, then arbitration if review disagrees."""

    def __init__(self, runner, english, chinese, en_paragraphs, zh_paragraphs, model, verifier,
                 en, zh, matches, missing_zh, missing_en, conflicts, guard_issues, initial, diagnostics):
        self.runner = runner
        self.english, self.chinese = english, chinese
        self.en_paragraphs, self.zh_paragraphs = en_paragraphs, zh_paragraphs
        self.model, self.verifier = model, verifier
        self.en, self.zh, self.matches = en, zh, matches
        self.missing_zh, self.missing_en, self.conflicts = missing_zh, missing_en, conflicts
        self.guard_issues, self.initial = guard_issues, initial
        self.diagnostics = diagnostics
        self.review: ReviewDraft | None = None
        self.arbitration: ArbitrationDraft | None = None
        self.recheck: ParagraphDisputeRecheckDraft | None = None

    def _paragraph_user(self) -> tuple:
        return (self.english, self.chinese, compact_paragraphs(self.en_paragraphs),
                compact_paragraphs(self.zh_paragraphs))

    def resolve(self) -> str:
        review_wire: ParagraphReviewDraft = self.runner.call(
            "independent_review", self.diagnostics, self.verifier, REVIEW_SYSTEM_PARAGRAPH,
            paragraph_review_user(*self._paragraph_user()),
            ParagraphReviewDraft, "paragraph_review",
            validate=lambda answer: verify_review(answer, self.en_paragraphs, self.zh_paragraphs),
        )
        self.review = review_public(review_wire, self.en_paragraphs, self.zh_paragraphs)
        verdict = self.initial
        if self.review.verdict != self.initial and not (self.guard_issues and self.initial == "FAIL"):
            verdict = self._arbitrate(review_wire)
        if self.guard_issues:
            verdict = "FAIL"
        return verdict

    def _arbitrate(self, review_wire: ParagraphReviewDraft) -> str:
        initial, review = self.initial, self.review
        snapshot = dispute_snapshot(self.missing_zh, self.missing_en, self.conflicts, self.guard_issues)
        arb_wire: ParagraphArbitrationDraft = self.runner.call(
            "dispute_arbitration", self.diagnostics, self.model, ARBITRATION_SYSTEM_PARAGRAPH,
            paragraph_arbitrate_user(*self._paragraph_user(), initial, snapshot, review_wire.model_dump()),
            ParagraphArbitrationDraft, "paragraph_arbitration",
            validate=lambda answer: verify_arbitration(answer, self.en_paragraphs, self.zh_paragraphs, initial),
        )
        arb = self.arbitration = arbitration_public(arb_wire, self.en_paragraphs, self.zh_paragraphs)
        if arb.verdict == "PASS" and initial == "FAIL":
            self._confirm_support(snapshot, review_wire, arb_wire)
        if arb.verdict == "PASS" and initial == "PASS" and review.verdict == "FAIL" and review.issues:
            return self._recheck_objections(review_wire, arb_wire)
        if arb.verdict == "UNSURE":
            if initial == "PASS":
                self.diagnostics.append("dispute_arbitration: uncertain; manual review required")
                return "REVIEW_REQUIRED"
            return "FAIL"
        if initial == "FAIL" and review.verdict == "PASS" and arb.verdict == "FAIL":
            # The arbiter is the primary model again, so it does not corroborate the
            # primary's own allegation against an independent PASS. Only a
            # deterministic guard can still make this FAIL.
            self.diagnostics.append("dispute_arbitration: initial FAIL and arbiter FAIL contradict "
                                    "independent PASS; no verified conflict; manual review required")
            return "REVIEW_REQUIRED"
        return arb.verdict

    def _confirm_support(self, snapshot, review_wire, arb_wire) -> None:
        """FAIL→PASS needs a separate semantic support verdict; valid IDs alone are not enough."""
        support_input = paragraph_arbitrate_user(
            *self._paragraph_user(), self.initial, snapshot,
            {"independent_review": review_wire.model_dump(), "proposed_arbitration": arb_wire.model_dump()})
        supported = self.runner.call(
            "alignment_evidence_check", self.diagnostics, self.verifier,
            EVIDENCE_SUPPORT_SYSTEM_PARAGRAPH, support_input,
            ParagraphAlignmentDraft, "paragraph_alignment_support",
            validate=lambda answer: verify_support(
                answer, self.en_paragraphs, self.zh_paragraphs, self.en, self.zh,
                self.missing_zh, self.missing_en, self.conflicts),
        )
        if supported.status != "supported":
            raise PipelineStageError("alignment_evidence_check",
                                     f"无法支持 FAIL→PASS：{supported.status}: {supported.reason}")
        self.diagnostics.append("FAIL→PASS: independent semantic support confirmed; ID validity alone is insufficient")

    def _recheck_objections(self, review_wire, arb_wire) -> str:
        """Initial PASS, review FAIL with issues, arbitration PASS: recheck each objection.

        Returning to PASS requires every issue refuted and each refutation backed by a
        source-verified numeric certificate for an isolated precision objection.
        """
        certificates = numeric_equivalence_certificates(
            self.en, self.zh, self.matches.links, self.en_paragraphs, self.zh_paragraphs)
        self.diagnostics.append(f"dispute: {len(certificates)} source-verified numeric certificate(s)")
        self.recheck = self.runner.call(
            "review_objection_check", self.diagnostics, self.verifier,
            REVIEW_OBJECTION_CHECK_SYSTEM_PARAGRAPH,
            paragraph_recheck_user(*self._paragraph_user(),
                                   [i.model_dump() for i in review_wire.issues],
                                   arb_wire.model_dump(), [c.model_dump() for c in certificates]),
            ParagraphDisputeRecheckDraft, "paragraph_dispute_recheck",
            validate=lambda answer: verify_objection_recheck(
                answer, review_wire.issues, self.en_paragraphs, self.zh_paragraphs),
        )
        refutations = {item.issue_index: item for item in self.recheck.issues}
        proven = bool(certificates) and all(
            refutations[index].status == "refuted"
            and narrow_exactness_refutation(issue, refutations[index], certificates,
                                            self.en_paragraphs, self.zh_paragraphs)
            for index, issue in enumerate(review_wire.issues))
        if proven:
            self.diagnostics.append("review_objection_check: all reviewer issues refuted with verified formal certificates")
            return "PASS"
        self.diagnostics.append("review_objection_check: unresolved or non-formal disagreement; manual review required")
        return "REVIEW_REQUIRED"
