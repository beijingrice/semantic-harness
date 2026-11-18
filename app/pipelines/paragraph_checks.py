"""Structural checks on paragraph-ID model answers, and their public (quoted) views.

A valid paragraph ID proves only that the cited paragraph exists, not that it
supports the model's claim. These checks reject impossible answers (unknown IDs,
a FAIL without a cited issue, a refutation that cites nothing) before use.
"""
from __future__ import annotations

from app.llm import LLMError
from app.models import (
    ArbitrationDraft,
    KeyUnit,
    Link,
    ParagraphAlignmentDraft,
    ParagraphArbitrationDraft,
    ParagraphDisputeRecheckDraft,
    ParagraphReviewDraft,
    ParagraphReviewIssue,
    ReviewDraft,
    ReviewIssue,
    SourceParagraph,
)
from app.paragraphs import paragraph_evidence, verify_ids


def _verify_cited(en_ids: list[str], zh_ids: list[str],
                  en_paragraphs: list[SourceParagraph], zh_paragraphs: list[SourceParagraph]) -> None:
    try:
        verify_ids(en_ids, en_paragraphs, allow_empty=True)
        verify_ids(zh_ids, zh_paragraphs, allow_empty=True)
    except ValueError as exc:
        raise LLMError(str(exc)) from exc


def verify_review(answer: ParagraphReviewDraft, en_paragraphs, zh_paragraphs) -> None:
    if answer.verdict == "PASS" and answer.issues:
        raise LLMError("PASS 不允许同时提供问题项")
    if answer.verdict == "FAIL" and not answer.issues:
        raise LLMError("FAIL 必须提供至少一个具体问题及 Paragraph ID")
    for issue in answer.issues:
        _verify_cited(issue.en_paragraph_ids, issue.zh_paragraph_ids, en_paragraphs, zh_paragraphs)
        if issue.kind == "missing_in_chinese" and not issue.en_paragraph_ids:
            raise LLMError("中文缺失须给出英文来源段落")
        if issue.kind == "missing_in_english" and not issue.zh_paragraph_ids:
            raise LLMError("英文缺失须给出中文来源段落")
        if issue.kind == "conflict" and not (issue.en_paragraph_ids and issue.zh_paragraph_ids):
            raise LLMError("冲突须给出双语来源段落")


def review_public(review: ParagraphReviewDraft, en_paragraphs, zh_paragraphs) -> ReviewDraft:
    return ReviewDraft(verdict=review.verdict, reason=review.reason,
                       issues=[ReviewIssue(kind=i.kind, reason=i.reason,
                                           en_evidence=paragraph_evidence(i.en_paragraph_ids, en_paragraphs),
                                           zh_evidence=paragraph_evidence(i.zh_paragraph_ids, zh_paragraphs),
                                           en_paragraph_ids=i.en_paragraph_ids,
                                           zh_paragraph_ids=i.zh_paragraph_ids) for i in review.issues])


def verify_arbitration(answer: ParagraphArbitrationDraft, en_paragraphs, zh_paragraphs,
                       initial: str) -> None:
    _verify_cited(answer.en_paragraph_ids, answer.zh_paragraph_ids, en_paragraphs, zh_paragraphs)
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


def arbitration_public(answer: ParagraphArbitrationDraft, en_paragraphs, zh_paragraphs) -> ArbitrationDraft:
    return ArbitrationDraft(verdict=answer.verdict, issue_kind=answer.issue_kind,
                            en_evidence=paragraph_evidence(answer.en_paragraph_ids, en_paragraphs),
                            zh_evidence=paragraph_evidence(answer.zh_paragraph_ids, zh_paragraphs),
                            en_paragraph_ids=answer.en_paragraph_ids,
                            zh_paragraph_ids=answer.zh_paragraph_ids,
                            reason=answer.reason)


def verify_support(answer: ParagraphAlignmentDraft, en_paragraphs, zh_paragraphs,
                   en: list[KeyUnit], zh: list[KeyUnit], missing_zh: list[KeyUnit],
                   missing_en: list[KeyUnit], conflicts: list[Link]) -> None:
    """A FAIL→PASS support answer must cite the source paragraph of every disputed unit.

    This is a structural check only; the support model itself must affirm the alignment.
    """
    _verify_cited(answer.en_paragraph_ids, answer.zh_paragraph_ids, en_paragraphs, zh_paragraphs)
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


def verify_objection_recheck(answer: ParagraphDisputeRecheckDraft,
                             review_issues: list[ParagraphReviewIssue],
                             en_paragraphs, zh_paragraphs) -> None:
    """Exactly one assessment per reviewer issue; a refutation must cite that issue's paragraphs."""
    if len(answer.issues) != len(review_issues):
        raise LLMError("定向复核必须逐项回应每条原复核异议")
    indexes = [issue.issue_index for issue in answer.issues]
    if sorted(indexes) != list(range(len(review_issues))):
        raise LLMError("定向复核问题索引缺失、重复或不存在")
    for item in answer.issues:
        _verify_cited(item.en_paragraph_ids, item.zh_paragraph_ids, en_paragraphs, zh_paragraphs)
        if item.status != "refuted":
            continue
        issue = review_issues[item.issue_index]
        if not (item.en_paragraph_ids and item.zh_paragraph_ids):
            raise LLMError("反驳复核异议必须引用双语来源")
        if issue.en_paragraph_ids and not set(item.en_paragraph_ids).intersection(issue.en_paragraph_ids):
            raise LLMError("反驳未引用对应英文争议段落")
        if issue.zh_paragraph_ids and not set(item.zh_paragraph_ids).intersection(issue.zh_paragraph_ids):
            raise LLMError("反驳未引用对应中文争议段落")
