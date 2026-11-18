"""Conservative, bilingual *pre-generation* source-anchor fast path.

This is only an inventory shortcut. It does NOT decide bilingual equivalence:
Matching, original-text independent review and deterministic guards still run.
Abstain on structural uncertainty rather than risk hiding independently
omittable clauses, parallel duties, enumerations or multiple paragraphs.
"""

from __future__ import annotations

import re

from app.models import KeyUnit, SourceParagraph

_EN_COORD = re.compile(r"\b(?:and|or|as well as|along with|whereas|however|furthermore|moreover)\b", re.I)
_ZH_COORD = re.compile(
    r"(?:并且|以及|同时|另外|此外|分别|或者|还是|并需|还须|还应|还必须|[、；;：:]|和|(?<!参)与|(?<!涉)及)"
)
_EN_MODAL = re.compile(
    r"\b(?:must|shall|should|may|need(?:s)? to|is required to|are required to|"
    r"cannot|can't|is prohibited|are prohibited|has to|have to)\b",
    re.I,
)
_ZH_MODAL = re.compile(r"(?:必须|应当|应|须|不得|不可|禁止|无需|不必|可以|允许|只需)")
_EN_INITIAL_CONDITION = re.compile(r"^(?:if|when|after|before|once|unless|provided that|in case)\b", re.I)
_ZH_EXCEPTION = re.compile(r"(?:但|不过).{0,65}(?:除外|除非|例外)")
_ZH_INITIAL_CONDITION = re.compile(r"^(?:若|如果|当|只要|一旦|在.{1,30}(?:时|后|前))")
_EN_FAST_CUE = re.compile(
    r"\b(?:must|shall|should|may|cannot|prohibited|need|requires?|unless|except|only|\d)\b|\d", re.I
)
_ZH_FAST_CUE = re.compile(
    r"(?:必须|应当|应|须|不得|不可|禁止|无需|不必|可以|允许|除外|除非|至少|至多|超过|低于|\d|[一二三四五六七八九十百千万]+(?:天|小时|周|月|年))"
)
_ZH_NUM_BOUNDS = re.compile(
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万]+)\s*(?:至|到|—|－|~|～)\s*(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万]+)"
)


def _one_sentence(text: str, language: str) -> bool:
    # Decimal dots aren't sentence endings, but any internal end mark is.
    if language == "English":
        matches = list(re.finditer(r"(?<!\d)[.!?](?!\d)", text))
    else:
        matches = list(re.finditer(r"[。！？!?]|(?<!\d)\.(?!\d)", text))
    if len(matches) > 1:
        return False
    return not matches or matches[0].end() == len(text)


def _source_eligible(paragraph: SourceParagraph, language: str) -> bool:
    text = paragraph.text.strip()
    limit = 180 if language == "English" else 90
    min_chars = 8 if language == "English" else 4
    if not (min_chars <= len(text) <= limit):
        return False
    if any(c in text for c in ("\n", "\r", ";", "；", ":", "：", "\u2022", "\t", "…", "\\")):
        return False
    if not _one_sentence(text, language):
        return False
    if language == "English":
        if not _EN_FAST_CUE.search(text):
            return False
        if text.count(",") > 1:
            return False
        if text.count(",") == 1 and not _EN_INITIAL_CONDITION.search(text):
            return False
        # 'between 9 and 11' is one indivisible range, not a coordinated duty.
        scan = re.sub(r"\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b", "NUMERIC_RANGE", text, flags=re.I)
        if _EN_COORD.search(scan):
            return False
        if len(_EN_MODAL.findall(text)) > 1:
            return False
        if re.search(r"\b(?:first|second|third|respectively|including|such as)\b", text, re.I):
            return False
    else:
        if not _ZH_FAST_CUE.search(text):
            return False
        if text.count("，") + text.count(",") > 1:
            return False
        if text.count("，") + text.count(",") == 1 and not (
            _ZH_EXCEPTION.search(text) or _ZH_INITIAL_CONDITION.search(text)
        ):
            return False
        scan = _ZH_NUM_BOUNDS.sub("数值范围", text)
        if _ZH_COORD.search(scan) or len(_ZH_MODAL.findall(text)) > 1:
            return False
        if re.search(r"(?:例如|包括|包含以下|下列|第一|第二|第三|一方面|另一方面)", text):
            return False
    return True


def source_anchor_pair(
    english_paragraphs: list[SourceParagraph], chinese_paragraphs: list[SourceParagraph]
) -> tuple[list[KeyUnit], list[KeyUnit]] | None:
    """Both sides must independently satisfy eligibility, otherwise abstain."""
    if len(english_paragraphs) != 1 or len(chinese_paragraphs) != 1:
        return None
    en, zh = english_paragraphs[0], chinese_paragraphs[0]
    if not (_source_eligible(en, "English") and _source_eligible(zh, "Chinese")):
        return None

    def anchor(p: SourceParagraph, prefix: str) -> KeyUnit:
        marked = bool(
            (_EN_MODAL.search(p.text) if prefix == "en" else _ZH_MODAL.search(p.text)) or re.search(r"\d", p.text)
        )
        return KeyUnit(
            id=f"{prefix}_1",
            kind="constraint" if marked else "concept",
            claim=p.text,
            paragraph_ids=[p.id],
            evidence=p.text,
        )

    return [anchor(en, "en")], [anchor(zh, "zh")]


# A narrow second pre-generation route for TWO explicit adjacent
# clauses. It never uses an LLM to create or paraphrase claims; each anchor is
# a verbatim substring of the original source with the original paragraph ID.
# A single English period may correspond to a Chinese comma. Preserve both
# atomic clauses instead of forcing the whole passage into one claim.
_EN_CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_EN_INTERCLAUSE = re.compile(
    r"\b(?:if|unless|except|provided that|only if|otherwise|however|whereas|"
    r"although|but|nevertheless|because|therefore)\b",
    re.I,
)
_ZH_INTERCLAUSE = re.compile(r"(?:如果|若|除非|除外|仅当|否则|然而|但是|不过|鉴于|由于|因此|只要|前提是)")


def bounded_clause_anchor_pair(
    english_paragraphs: list[SourceParagraph], chinese_paragraphs: list[SourceParagraph]
) -> tuple[list[KeyUnit], list[KeyUnit]] | None:
    """No generative extraction for structurally clear parallel two-clause text.

    Conservative *structural* shortcut, NOT a semantic equivalence detector:
    each side must independently consist of exactly two reliably separable
    clauses, with no conditional/exception link between clauses. If unsure,
    abstain and let the original generative extraction run.
    """
    if len(english_paragraphs) != 1 or len(chinese_paragraphs) != 1:
        return None
    en, zh = english_paragraphs[0], chinese_paragraphs[0]
    english, chinese = en.text.strip(), zh.text.strip()
    if not (30 <= len(english) <= 240 and 14 <= len(chinese) <= 130):
        return None
    if any(mark in english + chinese for mark in ("\n", "\r", ";", "；", ":", "：", "\t", "…", "•")):
        return None
    # Mixed numeric / relational multi-clause passages should retain the
    # full extraction and deterministic guard's original association context.
    if re.search(r"\d", english + chinese):
        return None
    if _EN_INTERCLAUSE.search(english) or _ZH_INTERCLAUSE.search(chinese):
        return None
    # Directional contrasts often carry a material prohibition. Leave those
    # to the full extractor until individual predicate boundaries are certain.
    if re.search(r"\bnot\s+(?:in|inside|outside|above|below|before|after|under|over|within)\b", english, re.I):
        return None
    if "而非" in chinese or "而不是" in chinese:
        return None
    if not re.search(r"[.!?]$", english) or not re.search(r"[。！？]$", chinese):
        return None
    # Require exactly two complete English sentences, and exactly one Chinese
    # clause separator. Never split numeric decimals or enumeration commas.
    en_parts = [part.strip() for part in _EN_CLAUSE_BOUNDARY.split(english)]
    zh_parts = [part.strip() for part in chinese[:-1].split("，")]
    if len(en_parts) != 2 or len(zh_parts) != 2:
        return None
    if any(len(part) < 8 for part in en_parts) or any(len(part) < 5 for part in zh_parts):
        return None
    if en_parts[0][-1] not in ".!?" or en_parts[1][-1] not in ".!?":
        return None
    # Any additional Chinese full stop indicates a third clause or internal
    # sentence; a comma is allowed only as the single explicit separator.
    if any(mark in chinese[:-1] for mark in ("。", "！", "？", ",", "；")):
        return None

    def mk(parts: list[str], paragraph: SourceParagraph, prefix: str) -> list[KeyUnit]:
        out = []
        for index, clause in enumerate(parts, 1):
            if prefix == "zh" and index == 2:
                clause += chinese[-1]
            constraint = bool(
                (_EN_MODAL if prefix == "en" else _ZH_MODAL).search(clause)
                or re.search(r"\d|\bany order\b|\bno restriction\b|(?:不作限制|不限|无限制)", clause, re.I)
            )
            out.append(
                KeyUnit(
                    id=f"{prefix}_{index}",
                    kind="constraint" if constraint else "concept",
                    claim=clause,
                    paragraph_ids=[paragraph.id],
                    evidence=paragraph.text,
                )
            )
        return out

    return mk(en_parts, en, "en"), mk(zh_parts, zh, "zh")
