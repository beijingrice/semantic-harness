"""Steps shared by the two extraction pipelines (paragraph and quote evidence)."""
from __future__ import annotations

import re

from app.llm import LLMError
from app.models import KeyUnit, Link

# A paraphrased claim must not be stronger than its source: these patterns catch an
# obligation, exclusivity, negation or number that the model introduced. The source
# patterns are deliberately broader, so only clearly new content is flagged.
_CLAIM_FORCE = re.compile(
    r"\b(?:must|shall|mandatory|required|compulsory|obligatory|have to|has to)\b|"
    r"必须|务必|强制|须|应当|应包含|应提交", re.I
)
_SOURCE_FORCE = re.compile(
    r"\b(?:must|shall|mandatory|required|compulsory|obligatory|have to|has to|"
    r"should|need|needs|necessary)\b|必须|务必|强制|须|应|需|要求|不得|不可", re.I
)
_CLAIM_EXCLUSIVE = re.compile(r"\b(?:only|solely|exclusively)\b|仅|唯一|只允许")
_SOURCE_EXCLUSIVE = re.compile(r"\b(?:only|solely|exclusively)\b|仅|只|唯一|独", re.I)
_CLAIM_NEGATION = re.compile(r"\b(?:not|never|without|cannot|prohibited|forbidden|no)\b|"
                             r"不得|禁止|没有|未|无需|不", re.I)
_SOURCE_NEGATION = re.compile(r"\b(?:not|never|without|cannot|prohibited|forbidden|no)\b|"
                              r"不得|禁止|没有|未|无需|不|无|非", re.I)
_ARABIC_NUMBER = re.compile(r"(?<![a-zA-Z0-9])\d+(?:\.\d+)?(?![a-zA-Z0-9])")

SHORT_SOURCE_CHARS = 220


def clean_unsupported_claim(source: str, unit, stage: str, events: list[str]) -> None:
    """Replace a claim that adds an obligation, exclusivity, negation or number.

    Short sources: the verbatim source becomes the claim. Otherwise the cited
    evidence does, when it is short. Long evidence is kept, with a diagnostic, so
    downstream stages compare against the originals rather than the paraphrase.
    """
    force = bool(_CLAIM_FORCE.search(unit.claim) and not _SOURCE_FORCE.search(source))
    exclusive = bool(_CLAIM_EXCLUSIVE.search(unit.claim) and not _SOURCE_EXCLUSIVE.search(source))
    negative = bool(_CLAIM_NEGATION.search(unit.claim) and not _SOURCE_NEGATION.search(source))
    numeric = bool(set(_ARABIC_NUMBER.findall(unit.claim)) - set(_ARABIC_NUMBER.findall(source)))
    if not (force or exclusive or negative or numeric):
        return
    reason = ("unsupported obligation" if force else
              "unsupported exclusivity" if exclusive else
              "unsupported negation" if negative else "unsupported numeric detail")
    if len(source) <= SHORT_SOURCE_CHARS:
        unit.claim = source
        events.append(f"{stage}: replaced {reason} in claim with verbatim source")
    elif len(unit.evidence) <= SHORT_SOURCE_CHARS:
        unit.claim = unit.evidence
        events.append(f"{stage}: replaced {reason} in claim with grounded evidence")
    else:
        events.append(f"{stage}: potential {reason}; downstream must prioritize originals")


def match_coverage(en: list[KeyUnit], zh: list[KeyUnit], links: list[Link]):
    """(missing in Chinese, missing in English, conflicts) implied by the matching links.

    A unit counts as covered when an equivalent/plausible link uses it; units in a
    conflict link are reported as conflicts, not as missing.
    """
    en_ids, zh_ids = {u.id for u in en}, {u.id for u in zh}
    linked_en: set[str] = set()
    linked_zh: set[str] = set()
    conflicts: list[Link] = []
    for link in links:
        if not set(link.en_ids) <= en_ids or not set(link.zh_ids) <= zh_ids:
            raise LLMError("匹配模型输出了不存在的关键词 ID，不能据此 PASS。")
        if link.relation in ("equivalent", "plausible"):
            linked_en.update(link.en_ids)
            linked_zh.update(link.zh_ids)
        elif link.relation == "conflict":
            conflicts.append(link)
    missing_zh = [u for u in en if u.id not in linked_en and not any(u.id in c.en_ids for c in conflicts)]
    missing_en = [u for u in zh if u.id not in linked_zh and not any(u.id in c.zh_ids for c in conflicts)]
    return missing_zh, missing_en, conflicts


def dispute_snapshot(missing_zh: list[KeyUnit], missing_en: list[KeyUnit], conflicts: list[Link],
                     guard_issues: list[str]) -> dict:
    """The preliminary findings shown to the arbitration and support stages."""
    return {
        "missing_in_chinese": [u.model_dump() for u in missing_zh],
        "missing_in_english": [u.model_dump() for u in missing_en],
        "conflicts": [c.model_dump() for c in conflicts],
        "numeric_guard_issues": guard_issues,
    }


def extraction_summary(verdict: str, initial: str, mode: str, verifier: str, review_verdict: str | None,
                       arbitration_verdict: str | None, missing_count: int, conflict_count: int,
                       guard_count: int) -> str:
    if mode == "primary":
        review_part = "未启用独立复核。"
    else:
        arbitration_part = ("已执行争议仲裁，结论 " + arbitration_verdict
                            if arbitration_verdict else "无需争议仲裁")
        review_part = f"{verifier} 复核 {review_verdict}；{arbitration_part}。"
    return (f"{verdict}：主模型初判 {initial}；{review_part}"
            f" 初步缺失 {missing_count} 项、匹配冲突 {conflict_count} 项、确定性约束保护 {guard_count} 项。")
