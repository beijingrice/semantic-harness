"""Conservative, source-verified formal certificates for disputed equivalence.

A certificate proves one numeric predicate only, never general semantic alignment.
For 1:N/N:1 links, *every* participant in the link must independently pass
source-grounded checks before any certificates from that link are emitted.
"""
from __future__ import annotations

import re

from app.models import KeyUnit, Link, NumericEquivalenceCertificate, ParagraphReviewIssue
from app.paragraphs import paragraph_evidence
from app.validators import parse_quantity


def _pair_certificate(en: KeyUnit, zh: KeyUnit, english_paragraphs, chinese_paragraphs):
    # Verify model-extracted claims AND immutable originals independently.
    eq = parse_quantity(en.claim)
    zq = parse_quantity(zh.claim)
    ep = parse_quantity(paragraph_evidence(en.paragraph_ids, english_paragraphs))
    zp = parse_quantity(paragraph_evidence(zh.paragraph_ids, chinese_paragraphs))
    if not all((eq, zq, ep, zp)):
        return None
    if len({q.dimension for q in (eq, zq, ep, zp)}) != 1:
        return None
    if len({(q.value, q.comparator) for q in (eq, zq, ep, zp)}) != 1:
        return None
    return NumericEquivalenceCertificate(
        en_unit_id=en.id, zh_unit_id=zh.id,
        en_paragraph_ids=en.paragraph_ids, zh_paragraph_ids=zh.paragraph_ids,
        dimension=eq.dimension, canonical_value=str(eq.value),
        comparator=eq.comparator, en_claim=en.claim, zh_claim=zh.claim,
    )


def numeric_equivalence_certificates(
    english: list[KeyUnit], chinese: list[KeyUnit], links: list[Link],
    english_paragraphs, chinese_paragraphs,
) -> list[NumericEquivalenceCertificate]:
    """Proofs for 1:1 and independently verified 1:N/N:1 matched quantities.

    Many-to-many mappings are intentionally not expanded into a Cartesian
    product. One nonnumeric or inconsistent child vetoes the *whole* linked
    group: a number present in part of a mixed link cannot prove it all.
    """
    by_en = {u.id: u for u in english}
    by_zh = {u.id: u for u in chinese}
    output: list[NumericEquivalenceCertificate] = []
    seen: set[tuple[str, str]] = set()
    for link in links:
        if link.relation not in ("equivalent", "plausible"):
            continue
        if len(link.en_ids) > 1 and len(link.zh_ids) > 1:
            continue
        grouped = []
        for en_id in link.en_ids:
            for zh_id in link.zh_ids:
                en, zh = by_en.get(en_id), by_zh.get(zh_id)
                cert = _pair_certificate(en, zh, english_paragraphs, chinese_paragraphs) if en and zh else None
                if cert is None:
                    grouped = []
                    break
                grouped.append(cert)
            else:
                continue
            break
        # Ignore partial results if *any* pair was unprovable.
        if len(grouped) != len(link.en_ids) * len(link.zh_ids):
            continue
        for cert in grouped:
            key = (cert.en_unit_id, cert.zh_unit_id)
            if key not in seen:
                output.append(cert)
                seen.add(key)
    return output


def certificate_covers_issue(certificates: list[NumericEquivalenceCertificate],
                             issue: ParagraphReviewIssue) -> bool:
    """Paragraph overlap is a structural precondition, never semantic proof."""
    return bool(issue.en_paragraph_ids and issue.zh_paragraph_ids and any(
        set(c.en_paragraph_ids).intersection(issue.en_paragraph_ids)
        and set(c.zh_paragraph_ids).intersection(issue.zh_paragraph_ids)
        for c in certificates
    ))


# These markers identify a *review objection about numerical precision* rather
# than a second, unrelated duty merely co-located with a numeric fact. The
# narrow fallback below is NOT a general semantic classifier.
_PRECISION_OBJECTION = re.compile(
    r"\b(?:exact(?:ly|ness)?|precis(?:e|ely|ion)|approximat(?:e|ely|ion)|"
    r"round(?:ed|ing)?|strict(?:ly|ness)?|accurac(?:y|ies)|numerical precision)\b|"
    r"精确|精度|准确|严格|约数|近似|取整|四舍五入|恰好|正好", re.I
)
_EXTRA_SEMANTIC_TOPIC = re.compile(
    r"\b(?:additionally|also|furthermore|moreover|as well as|in addition|"
    r"another (?:issue|requirement)|unrelated|separate (?:issue|requirement))\b|"
    r"另(?:外|一)|此外|以及其他|同时还", re.I
)
# Non-period sentence separators are meaningful: a formal quantity in one
# sentence must never be allowed to discharge a duty in another sentence.
_MULTIPLE_STATEMENTS_EN = re.compile(r"[;\n]|[.!?]\s+\S")
_MULTIPLE_STATEMENTS_ZH = re.compile(r"[；;\n]|[。！？]\s*[^\s。！？]")
# Even a single sentence can contain two independent actions or scopes.
_COMPOUND_SCOPE_EN = re.compile(r"\b(?:and|but|whereas|unless|although|however|provided that)\b", re.I)
_COMPOUND_SCOPE_ZH = re.compile(r"(?:同时|并且|但|而且|除非|另外|以及|应当|必须|不得|可以删除|须归档)")

# A certificate for equal duration does not prove that monitoring continuity,
# duty, permission, or prohibition are equal. Explicit nonnumeric qualifiers
# are veto checks, not a comprehensive bilingual semantic classifier.
_MODAL_EN = (
    ("prohibited", re.compile(r"\b(?:must not|shall not|prohibited|forbidden|cannot)\b", re.I)),
    ("not_required", re.compile(r"\b(?:not required|not mandatory|need not|does not have to)\b", re.I)),
    ("required", re.compile(r"\b(?:must|shall|required|mandatory|have to)\b", re.I)),
    ("permitted", re.compile(r"\b(?:may|permitted|allowed|optional)\b", re.I)),
)
_MODAL_ZH = (
    ("prohibited", re.compile(r"(?:不得|禁止|不允许|不可|不能)")),
    ("not_required", re.compile(r"(?:无需|无须|不必|不需要)")),
    ("required", re.compile(r"(?:必须|(?<!无)(?<!不)须|应当|(?<!不)需要|务必)")),
    ("permitted", re.compile(r"(?:可以|允许|可选|可自由)")),
)
_CONTINUITY_EN = (
    ("negative", re.compile(r"\b(?:intermittent|interrupted|with breaks|not continuous)\b", re.I)),
    ("positive", re.compile(r"\b(?:without interruption|uninterrupted|continuously|continuous|nonstop|non-stop)\b", re.I)),
)
_CONTINUITY_ZH = (
    ("negative", re.compile(r"(?:间歇|(?<!不)中断|断断续续|不连续|非连续)")),
    ("positive", re.compile(r"(?:连续|不间断|不中断)")),
)


def _profile(text: str, patterns) -> str | None:
    for name, regex in patterns:
        if regex.search(text):
            return name
    return None


def _unmatched_nonnumeric_qualifiers(en_source: str, zh_source: str) -> bool:
    for en_patterns, zh_patterns in ((_MODAL_EN, _MODAL_ZH),
                                      (_CONTINUITY_EN, _CONTINUITY_ZH)):
        if _profile(en_source, en_patterns) != _profile(zh_source, zh_patterns):
            return True
    return False


def narrow_exactness_refutation(issue: ParagraphReviewIssue,
                                recheck_item,
                                certificates: list[NumericEquivalenceCertificate],
                                english_paragraphs, chinese_paragraphs) -> bool:
    """Permit basis='other' ONLY for isolated, formally exact quantity concerns.

    This is a conservative escape hatch for a verifier mislabelling an
    *exactness* objection as 'other', NOT an override of its status or a proof
    that two complicated paragraphs mean the same thing. Must still have an
    independent verifier status='refuted' and original-grounded certificates.
    """
    if issue.kind != "conflict" or recheck_item.status != "refuted":
        return False
    if not certificate_covers_issue(certificates, issue):
        return False
    if not _PRECISION_OBJECTION.search(issue.reason):
        return False
    if _EXTRA_SEMANTIC_TOPIC.search(issue.reason) or _EXTRA_SEMANTIC_TOPIC.search(recheck_item.reason):
        return False
    relevant = [c for c in certificates if
                set(c.en_paragraph_ids).intersection(issue.en_paragraph_ids)
                and set(c.zh_paragraph_ids).intersection(issue.zh_paragraph_ids)]
    # Only equality, not inequalities or a numerical limit in a larger clause.
    if not relevant or any(c.comparator != "=" for c in relevant):
        return False
    # Each source must contain exactly one quantity and one statement; a
    # numeric equivalence cannot hide an independent nonnumeric requirement.
    for c in relevant:
        en_source = paragraph_evidence(c.en_paragraph_ids, english_paragraphs)
        zh_source = paragraph_evidence(c.zh_paragraph_ids, chinese_paragraphs)
        if (_MULTIPLE_STATEMENTS_EN.search(en_source.strip().rstrip('.'))
                or _MULTIPLE_STATEMENTS_ZH.search(zh_source.strip().rstrip('。'))
                or _COMPOUND_SCOPE_EN.search(en_source)
                or _COMPOUND_SCOPE_ZH.search(zh_source)
                or _unmatched_nonnumeric_qualifiers(en_source, zh_source)):
            return False
    return True
