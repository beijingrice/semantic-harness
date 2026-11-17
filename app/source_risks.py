"""Source-only risk indicators for paragraph-direct review.

Indicators are *triage*, never translation proofs: keep full source for LLM.
No cross-language positional alignment or source text rewriting is performed.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.validators import parse_quantity, numeric_tokens, Quantity

_SENTENCE = re.compile(r"(?<!\d)\.(?!\d)|[。！？!?；;\n]+")
_EN_COUNT_POS = re.compile(r"\b(?:is|are|was|were|be|being)\s+included\b|\bcounted\s+(?:in|towards|within)\b", re.I)
_ZH_COUNT_POS = re.compile(r"(?<!不)(?<!未)(?:计入|算入|纳入|包含在|包括在)")
_ZH_COUNT_NEG = re.compile(r"(?:不|未|无需|无须)(?:予以)?(?:计入|算入|纳入)|不包括在|排除在")
_EN_COUNT_NEG = re.compile(r"\b(?:not\s+(?:be\s+)?included|not\s+counted|excluded\s+from)\b", re.I)
_EN_COUNT_DOMAIN = re.compile(r"\b(?:page|pages|limit|count|quota|total|word\s+count)\b", re.I)
_ZH_COUNT_DOMAIN = re.compile(r"页|限制|额度|字数|计入|算入")


@dataclass(frozen=True)
class SourceRisk:
    kind: str
    detail: str
    en_excerpt: str
    zh_excerpt: str


def _isolated_quantities(text: str) -> dict[str, list[tuple[Quantity, str]]]:
    """Find self-contained sentence clauses with ONE number and a known unit.

    Same-dimension pairing is only a *risk flag* unless a human/semantic audit
    verifies the referent. Two independent numeric statements in the same
    sentence are intentionally omitted from this source-only grammar.
    """
    found = defaultdict(list)
    for segment in _SENTENCE.split(text):
        clause = segment.strip()
        if not clause or len(numeric_tokens(clause)) != 1:
            continue
        quantity = parse_quantity(clause)
        if quantity is not None:
            found[quantity.dimension].append((quantity, clause))
    return found


def formal_single_quantity_equivalence(en: str, zh: str) -> tuple[Quantity, Quantity] | None:
    """Only a short, single-statement exact quantity; not semantic equivalence."""
    if len(en) > 220 or len(zh) > 220 or any(c in en+zh for c in (';', '；', '\n')):
        return None
    e, z = _isolated_quantities(en), _isolated_quantities(zh)
    if sum(len(v) for v in e.values()) != 1 or sum(len(v) for v in z.values()) != 1:
        return None
    dimension = next(iter(e))
    if dimension not in z or len(z[dimension]) != 1:
        return None
    a, b = e[dimension][0][0], z[dimension][0][0]
    if (a.value, a.comparator) != (b.value, b.comparator):
        return None
    return a, b


def source_risks(english: str, chinese: str) -> list[SourceRisk]:
    risks: list[SourceRisk] = []
    en, zh = _isolated_quantities(english), _isolated_quantities(chinese)
    # UNIQUE dimension on both sides. If the numbers are unequal, they may be
    # different referents; never infer a conflict solely from their positions.
    for dimension in sorted(en.keys() & zh.keys()):
        if len(en[dimension]) != 1 or len(zh[dimension]) != 1:
            continue
        (a, a_clause), (b, b_clause) = en[dimension][0], zh[dimension][0]
        if a.value == b.value and a.comparator != b.comparator:
            risks.append(SourceRisk('numeric_boundary',
                f'唯一可解析的 {dimension} 约束：{a.comparator} {a.value} vs {b.comparator} {b.value}。'
                '需核查描述的是否为同一要求，不可仅凭单位自动判 FAIL。', a_clause, b_clause))
    # Inclusion inventories catch the V2-NUM-15 failure without guessing
    # which Chinese clause corresponds to which English clause.
    en_pos = len(_EN_COUNT_POS.findall(english))
    zh_pos = len(_ZH_COUNT_POS.findall(chinese))
    if (_EN_COUNT_DOMAIN.search(english) and _ZH_COUNT_DOMAIN.search(chinese)
            and en_pos != zh_pos and
            (_EN_COUNT_NEG.search(english) or _ZH_COUNT_NEG.search(chinese))):
        risks.append(SourceRisk('inclusion_inventory',
            f'原文计入关系数量不一致（英文肯定 {en_pos}，中文肯定 {zh_pos}）；'
            '这只是漏检风险提示，并非跨语言实体配对证明。', english, chinese))
    return risks


# --- Shared by the direct_hardened pipeline (moved from removed v0.2.11/v0.2.12 modes) ---

@dataclass(frozen=True)
class Span:
    id: str
    text: str


def source_spans(text: str, prefix: str) -> list[Span]:
    """Give selectable citations. No cross-language alignment or semantic cuts.

    A full-text option is always offered. Punctuation-separated subspans are
    OPTIONAL short cites, preserving literal bytes and the full text separately.
    They are not provided to the primary model or interpreted as paired facts.
    """
    options = [Span(f'{prefix}_all', text)]
    if len(text) > 6000:
        return options
    # Semicolons are citation boundaries, never required logic boundaries.
    # Preserve punctuation as part of the verbatim span for unambiguous tracing.
    cuts = list(re.finditer(r'(?<!\d)\.(?!\d)|[。！？!?；;\n]', text))
    start = 0
    for cut in cuts:
        value = text[start:cut.end()]
        if value.strip():
            options.append(Span(f'{prefix}_s{len(options)}', value))
        start = cut.end()
    if text[start:].strip():
        options.append(Span(f'{prefix}_s{len(options)}', text[start:]))
    # Avoid huge evidence menus for long documents; in that case the whole
    # paragraph is still always available for precise source-grounding.
    if len(options) > 18:
        return [options[0]]
    return options


def _source_quantitative_issue_only(english: str, chinese: str):
    """Very narrow proof scope: one unambiguous reported quantity each side.

    Existence of a certificate does NOT prove entire passage equivalence.
    A separate focused semantic recheck must refute any alleged extra duty.
    """
    certificate = formal_single_quantity_equivalence(english, chinese)
    if certificate is None:
        return None
    # Even a numeral equality is no proof for a second independent sentence.
    # Trailing sentence punctuation is ignored; decimals are not sentence stops.
    for source in (english, chinese):
        body = source.strip().rstrip('.。!?！？')
        if re.search(r'(?<!\d)\.(?!\d)|[。!?！？；;\n]', body):
            return None
    # Do not erase additional explicit obligations merely because a quantity
    # happens to be equal on both sides. This is a *strict* safe subset.
    extra_obligation = re.compile(
        r"\b(?:must|shall|should|required|mandatory|may|prohibited|forbidden)\b|"
        r"必须|须|应当|应提交|不得|不允许|禁止|除非|仅当|才可|上传|另行|同时|而且|另外|还要", re.I)
    return certificate if not (extra_obligation.search(english) or extra_obligation.search(chinese)) else None
