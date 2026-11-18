"""Paragraph-direct judgment: two models compare the whole original texts, no extraction.

This is the first half of direct_hardened (pipelines/hardened.py). Both models see
the same numbered original paragraphs; no alignment of paragraph or sentence order
is assumed. Agreement is an LLM judgment, not proof of exhaustive alignment.
"""
from __future__ import annotations

import re

from app.llm import LLMError
from app.models import CheckResult, ParagraphReviewDraft
from app.paragraphs import build_paragraphs, compact_paragraphs
from app.pipelines.base import StageRunner
from app.pipelines.paragraph_checks import review_public, verify_review
from app.prompts import DIRECT_PRIMARY_SYSTEM, REVIEW_SYSTEM_PARAGRAPH, paragraph_review_user
from app.validators import numeric_tokens, parse_quantity, parse_range

_END = re.compile(r"[。！？!?]|(?<!\d)\.(?!\d)")
_CONDITIONAL = re.compile(r"\b(?:if|unless|except|provided|whereas|when|until)\b|如果|若|除非|除外|但|否则|当.*时", re.I)


def _simple_numeric_source(text: str) -> bool:
    """One short unconditional statement: the only grammar the source numeric veto trusts."""
    body = text.strip().rstrip(".。!?！？")
    if not body or len(body) > 220 or "\n" in body or ";" in body or "；" in body:
        return False
    return not (_END.search(body) or _CONDITIONAL.search(body))


def direct_numeric_conflicts(english: str, chinese: str, en_paragraphs, zh_paragraphs) -> list[str]:
    """A provable one-statement bound or range mismatch in the originals; otherwise [].

    Each source must be a single simple statement. Anything more complex abstains
    and is left to the models. A finding vetoes the verdict to FAIL.
    """
    if len(en_paragraphs) != 1 or len(zh_paragraphs) != 1:
        return []
    if not (_simple_numeric_source(english) and _simple_numeric_source(chinese)):
        return []
    en_range, zh_range = parse_range(english), parse_range(chinese)
    if en_range and zh_range and en_range.dimension == zh_range.dimension:
        if en_range != zh_range:
            return [f"原文数字区间不一致：{english!r} ↔ {chinese!r}"]
        return []
    if en_range or zh_range:
        return []
    if len(numeric_tokens(english)) > 1 or len(numeric_tokens(chinese)) > 1:
        return []
    x, y = parse_quantity(english), parse_quantity(chinese)
    if x and y and x.dimension == y.dimension and (x.value, x.comparator) != (y.value, y.comparator):
        return [f"原文数字边界不同：{english!r} ({x.comparator} {x.value} {x.dimension}) "
                f"↔ {chinese!r} ({y.comparator} {y.value} {y.dimension})"]
    return []


def _fuse(first: str, second: str | None, numeric_issues: list[str], diagnostics: list[str]) -> str:
    if numeric_issues:
        diagnostics.append("direct_original_numeric_guard: verified independent source-bound mismatch")
        return "FAIL"
    if first == "UNSURE" or second == "UNSURE":
        diagnostics.append("direct_fusion: uncertainty, no automatic PASS/FAIL")
        return "REVIEW_REQUIRED"
    if second is None or first == second:
        return first
    diagnostics.append("direct_fusion: two independent assessments disagree; no correlated third vote")
    return "REVIEW_REQUIRED"


def run_direct(runner: StageRunner, english: str, chinese: str, requested_model: str | None = None,
               requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
    if mode in ("primary", "dual") and (not english.strip() or not chinese.strip()):
        raise LLMError("中英文均须有非空原文；空内容无法比较")
    model, verifier = runner.choose_models(mode, requested_model, requested_verifier)
    ep, zp = build_paragraphs(english, "en"), build_paragraphs(chinese, "zh")
    diagnostics = ["pipeline: paragraph_direct (no extraction, no claim matching, no sentence alignment)"]
    user = paragraph_review_user(english, chinese, compact_paragraphs(ep), compact_paragraphs(zp))
    first: ParagraphReviewDraft = runner.call(
        "direct_primary", diagnostics, model, DIRECT_PRIMARY_SYSTEM, user,
        ParagraphReviewDraft, "direct_primary_review",
        validate=lambda a: verify_review(a, ep, zp))
    second: ParagraphReviewDraft | None = None
    review = None
    if mode == "dual":
        second = runner.call(
            "direct_independent_review", diagnostics, verifier, REVIEW_SYSTEM_PARAGRAPH, user,
            ParagraphReviewDraft, "paragraph_review",
            validate=lambda a: verify_review(a, ep, zp))
        review = review_public(second, ep, zp)

    issues = direct_numeric_conflicts(english, chinese, ep, zp)
    final = _fuse(first.verdict, second.verdict if second else None, issues, diagnostics)
    if final == "PASS":
        diagnostics.append("direct_fusion: agreement is an LLM judgment, NOT formal proof of exhaustive alignment")
    description = (f"{final}：Paragraph-Direct；主模型 {first.verdict}"
                   + (f"；独立复核 {second.verdict}" if second else "（未启用独立复核）")
                   + f"；原文数值冲突 {len(issues)} 项。")
    return CheckResult(
        evidence_mode="paragraph", english_paragraphs=ep, chinese_paragraphs=zp,
        verdict=final, model=model, verifier_model=verifier, mode=mode,
        english_keywords=[], chinese_keywords=[], links=[],
        missing_in_chinese=[], missing_in_english=[], conflicts=[],
        preliminary_missing_in_chinese=[], preliminary_missing_in_english=[],
        preliminary_conflicts=[], review=review,
        review_issues=(review.issues if review else review_public(first, ep, zp).issues),
        arbitration=None, arbitrated=False, dispute_recheck=None,
        review_warnings=[], deterministic_issues=issues,
        initial_verdict=(first.verdict if first.verdict != "UNSURE" else "FAIL"),
        summary=description, diagnostics=diagnostics, direct_primary=first,
    )
