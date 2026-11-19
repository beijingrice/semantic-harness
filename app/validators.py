"""Conservative, model-independent bilingual constraint sanity checks.

LLMs supply *semantic* links. These checks validate only small, explicitly
parseable formal properties of a 1:1 linked claim: numeric magnitude, units,
bound openness, and unambiguous inclusion/exclusion polarity.

Unrecognized/ambiguous syntax -> no deterministic assertion, NOT a PASS proof.
Never compare raw paragraph-level evidence: it may contain many unrelated facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.models import KeyUnit, Link

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_EN_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def _cn_to_int(s: str) -> int | None:
    if all(ch in _CN_DIGITS for ch in s):
        return _CN_DIGITS[s] if len(s) == 1 else None
    if s == "十":
        return 10
    if "十" in s and s.count("十") == 1:
        left, right = s.split("十")
        if (not left or left in _CN_DIGITS) and (not right or right in _CN_DIGITS):
            return (1 if not left else _CN_DIGITS[left]) * 10 + (0 if not right else _CN_DIGITS[right])
    return None


_NUMBER_RX = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9.])|"
    r"[零〇一二两三四五六七八九十]+|"
    r"\b(?:" + "|".join(_EN_NUM) + r")\b",
    re.I,
)


def _value(token: str) -> Decimal | None:
    try:
        if token[0].isdigit():
            return Decimal(token)
        if token.lower() in _EN_NUM:
            return Decimal(_EN_NUM[token.lower()])
        n = _cn_to_int(token)
        return Decimal(n) if n is not None else None
    except (ValueError, TypeError):
        return None


def numeric_tokens(text: str) -> list[float]:
    """Legacy public helper retained for compatibility and diagnostics."""
    return [float(n) for match in _NUMBER_RX.finditer(text) if (n := _value(match.group())) is not None]


@dataclass(frozen=True)
class Quantity:
    value: Decimal  # canonicalized to a common unit
    dimension: str  # e.g. duration_days, pages, percent
    comparator: str  # <, <=, >, >=, =
    literal: str


@dataclass(frozen=True)
class NumericRange:
    lower: Decimal
    upper: Decimal
    dimension: str
    lower_inclusive: bool
    upper_inclusive: bool


def parse_range(text: str) -> NumericRange | None:
    """A single explicit bounded interval; ambiguous open/closed limits abstain.

    Both bounds must belong to the SAME range and share an explicit following
    unit. Do not turn two unrelated numbers in a paragraph into a range.
    """
    nums = [(m, _value(m.group())) for m in _NUMBER_RX.finditer(text)]
    nums = [
        (m, n)
        for m, n in nums
        if n is not None and not (m.group() in ("两", "二") and text[m.end() :].startswith("端"))
    ]
    if len(nums) != 2:
        return None
    (m1, n1), (m2, n2) = nums
    prefix = text[: m1.start()]
    between = text[m1.end() : m2.start()]
    suffix = text[m2.end() :]
    en = bool(re.search(r"\bbetween\s*$", prefix, re.I) and re.fullmatch(r"\s+and\s+", between, re.I))
    cn = bool(re.fullmatch(r"\s*(?:至|到|[-–—~～])\s*", between))
    if not (en or cn):
        return None
    unit = _unit(suffix)
    if unit is None or n1 > n2:
        return None
    inclusive = bool(re.search(r"\binclusive\b|包含两端|含两端|包括两端|含端点", text, re.I))
    exclusive = bool(re.search(r"\bexclusive\b|不含两端|不包括两端|不含端点", text, re.I))
    if inclusive == exclusive:
        return None
    dimension, factor = unit
    return NumericRange(n1 * factor, n2 * factor, dimension, inclusive, inclusive)


# Small transparent *grammars*, not model-name patches or per-test strings.
# Extensible dimensions can be registered here without changing the LLM prompt.
_UNIT_PATTERNS: tuple[tuple[re.Pattern, str, Decimal], ...] = tuple(
    (re.compile(pattern, re.I), dimension, Decimal(factor))
    for pattern, dimension, factor in (
        (r"^(?:\s*%|\s*percent\b|\s*percentage\b|\s*百分比)", "percent", "1"),
        (r"^\s*(?:weeks?\b|周(?!岁)|星期)", "duration_days", "7"),
        (r"^\s*(?:days?\b|天|日)", "duration_days", "1"),
        (r"^\s*(?:hours?\b|hrs?\b|小时|小時|个小时|小時)", "duration_days", "0.04166666666666666666666666667"),
        (r"^\s*(?:minutes?\b|mins?\b|分钟|分鐘)", "duration_days", "0.0006944444444444444444444444444"),
        (r"^\s*(?:years?\b|年)", "duration_years", "1"),
        (r"^\s*(?:pages?\b|页|頁)", "pages", "1"),
        (r"^\s*(?:points?\b|磅|pt\b)", "typographic_points", "1"),
        (r"^\s*(?:millimet(?:re|er)s?\b|mm\b|毫米|公釐)", "length_mm", "1"),
        (r"^\s*(?:centimet(?:re|er)s?\b|cm\b|厘米|公分)", "length_mm", "10"),
        (r"^\s*(?:met(?:re|er)s?\b|m\b|米|公尺)", "length_mm", "1000"),
        (r"^\s*(?:participants?\b|respondents?\b|people\b|persons?\b|名(?:参与者|受访者|评审)|人)", "persons", "1"),
        (r"^\s*(?:responses?\b|答卷|份(?:有效答卷)?)", "responses", "1"),
    )
)

# Comparator grammar. Every pattern must end at the number, so the bound attached
# to "more than 10%" is parsed while an unrelated "more" elsewhere is ignored.
# A negation before a strict comparator flips it into the opposite inclusive
# bound: "不小于 / no less than / must not fall below" -> >=, "不超过 / no more
# than / may not exceed" -> <=. Listing negated forms one by one missed many of
# them ("不小于" was parsed as "<", a false FAIL vetoed by the guard).
_EN_GT = r"(?:(?:more|greater|longer|larger|higher)\s+than|over|exceed(?:s|ed|ing)?|in\s+excess\s+of|above)"
_EN_LT = r"(?:(?:less|fewer|shorter|smaller|lower)\s+than|under|below)"
_EN_NEG = r"(?:\bno|\bnot|\bnever|\bcannot|n't)(?:\s+(?:be|fall|go|drop|rise|run|last|get)(?:s|ing)?)?\s+(?:to\s+)?"
_ZH_GT = r"(?:超过|超出|高于|大于|多于|长于)"
_ZH_LT = r"(?:少于|低于|小于|短于)"
# Not 不必/无需 ("不必超过" is not a bound).
_ZH_NEG = r"不(?:得|应|能|可|会|准|要)?"

_GE = re.compile(
    rf"(?:\b(?:at least|greater than or equal to|more than or equal to|minimum of)\s*|{_EN_NEG}{_EN_LT}\s*|"
    rf"(?:至少|大于等于|大于或等于|高于或等于|{_ZH_NEG}{_ZH_LT})\s*)$",
    re.I,
)
_GT = re.compile(rf"(?:\b{_EN_GT}\s*|{_ZH_GT}\s*)$", re.I)
_LE = re.compile(
    rf"(?:\b(?:at most|up to|maximum of|limited to|less than or equal to)\s*|{_EN_NEG}{_EN_GT}\s*|"
    rf"(?:最多|至多|上限为|小于等于|小于或等于|低于或等于|{_ZH_NEG}{_ZH_GT})\s*)$",
    re.I,
)
_LT = re.compile(rf"(?:\b{_EN_LT}\s*|(?:{_ZH_LT}|不到|不足)\s*)$", re.I)
# "不必超过 / need not exceed" lifts a requirement; it states no bound at all.
_NOT_A_BOUND = re.compile(
    rf"(?:(?:不必|无需|无须|不需要|不用){_ZH_GT}|(?:不必|无需|无须|不需要|不用){_ZH_LT}|"
    rf"\b(?:need\s+not|needn't|does\s+not\s+have\s+to|do\s+not\s+have\s+to|doesn't\s+have\s+to|"
    rf"don't\s+have\s+to)(?:\s+be)?\s+(?:{_EN_GT}|{_EN_LT}))\s*$",
    re.I,
)
_EQ = re.compile(r"(?:\b(?:exactly|precisely|equal to)\s*|(?:恰好|正好|刚好|等于|固定为)\s*)$", re.I)
_GE_SUFFIX = re.compile(r"^\s*(?:%\s*)?(?:或以上|及以上|以上|or\s+(?:higher|more)|and\s+above)", re.I)
_LE_SUFFIX = re.compile(r"^\s*(?:%\s*)?(?:或以下|及以下|以下|or\s+(?:lower|less)|and\s+below)", re.I)
# Bound words AFTER the unit ("10 页以上", "30 pages or more", "七天内").
# Lowest priority in parse_quantity: any comparator before the number wins.
_GE_AFTER_UNIT = re.compile(r"^\s*(?:或以上|及以上|以上|or\s+(?:higher|more|above|greater)|and\s+(?:above|more))", re.I)
_LE_AFTER_UNIT = re.compile(r"^\s*(?:或以下|及以下|以下|or\s+(?:lower|less|fewer|below)|and\s+(?:below|less))", re.I)
# "within 6 weeks" / "七天内" are upper bounds only for TIME spans. For other
# units they often mean inclusion ("the cover page is counted within 30 pages"
# / "封面计入30页内"), so they are not treated as bounds there.
_TIME_DIMENSIONS = frozenset({"duration_days", "duration_years"})
_LE_TIME = re.compile(r"\b(?:within|no later than|not later than)\s*$", re.I)
_LE_TIME_AFTER_UNIT = re.compile(r"^\s*(?:以内|之内|内(?![容部]))")
# Chinese may place a verb between the quantifier and number: "至少保留五天".
# Stay within the same punctuation-delimited clause; do not use arbitrary far
# upstream words. If multiple different operators appear, abstain.
_NOT_NEGATED = "(?<!不)(?<!不得)(?<!不应)(?<!不能)(?<!不可)(?<!不会)(?<!不准)(?<!不要)"
_REMOTE_CN = (
    (">=", re.compile(rf"(?:至少|{_ZH_NEG}{_ZH_LT})[^，。；,;]{{0,10}}$")),
    ("<=", re.compile(rf"(?:最多|至多|{_ZH_NEG}{_ZH_GT})[^，。；,;]{{0,10}}$")),
    (">", re.compile(rf"{_NOT_NEGATED}{_ZH_GT}[^，。；,;]{{0,10}}$")),
    ("<", re.compile(rf"(?:{_NOT_NEGATED}{_ZH_LT}|不到|不足)[^，。；,;]{{0,10}}$")),
)


def _unit_match(after: str) -> tuple[str, Decimal, int] | None:
    """(dimension, factor, end offset of the unit within ``after``)."""
    for rx, dimension, factor in _UNIT_PATTERNS:
        found = rx.search(after)
        if found:
            return dimension, factor, found.end()
    return None


def _unit(after: str) -> tuple[str, Decimal] | None:
    found = _unit_match(after)
    return None if found is None else found[:2]


def parse_quantity(text: str) -> Quantity | None:
    """One number + recognizable unit = comparable constraint.

    A multi-number clause is intentionally out of scope, rather than comparing
    the wrong numbers. A bare number has unknown dimension and is not guarded.
    """
    candidates = [(match, _value(match.group())) for match in _NUMBER_RX.finditer(text)]
    candidates = [(match, value) for match, value in candidates if value is not None]
    if len(candidates) != 1:
        return None
    match, value = candidates[0]
    unit_found = _unit_match(text[match.end() :])
    if unit_found is None:
        return None
    unit = unit_found[:2]
    prefix = text[max(0, match.start() - 80) : match.start()].strip()
    suffix = text[match.end() : match.end() + 45]
    # Bound words usually follow the unit ("10 页以上", "30 pages or more").
    after_unit = text[match.end() + unit_found[2] : match.end() + unit_found[2] + 45]
    # Approximate quantities must not be elevated into formal EXACT equivalence.
    # Keep these out of deterministic equality certificates and guard verdicts.
    if re.search(
        r"\b(?:about|around|approximately|roughly|nearly|circa)\s*$|"
        r"(?:大约|大概|约|近|将近|差不多)\s*$|[≈~～]\s*$",
        prefix,
        re.I,
    ) or re.search(r"^\s*(?:weeks?|days?|周|天|日|页|pages?)\s*(?:左右|上下|余|or so)", suffix, re.I):
        return None
    if _NOT_A_BOUND.search(prefix):
        return None
    # Longest / negated forms first to avoid "no more" -> "more".
    if _GE.search(prefix) or _GE_SUFFIX.search(suffix):
        comp = ">="
    elif _LE.search(prefix) or _LE_SUFFIX.search(suffix):
        comp = "<="
    elif unit[0] in _TIME_DIMENSIONS and _LE_TIME.search(prefix):
        comp = "<="
    elif _GT.search(prefix):
        comp = ">"
    elif _LT.search(prefix):
        comp = "<"
    elif _EQ.search(prefix):
        comp = "="
    else:
        remote = {name for name, rx in _REMOTE_CN if rx.search(prefix)}
        if len(remote) > 1:
            return None
        if remote:
            comp = remote.pop()
        # Lowest priority: a bound word after the unit ("10 页以上",
        # "30 pages or more", "七天内"). An explicit comparator before the
        # number always wins, e.g. "少于五天内" stays "<".
        elif _GE_AFTER_UNIT.search(after_unit):
            comp = ">="
        elif _LE_AFTER_UNIT.search(after_unit) or (
            unit[0] in _TIME_DIMENSIONS and _LE_TIME_AFTER_UNIT.search(after_unit)
        ):
            comp = "<="
        else:
            # Declarative values ("ran for six weeks" / "持续了42天")
            # are exact reported facts, not approximate inequality bounds.
            comp = "="
    dimension, factor = unit
    canonical = value * factor
    if dimension == "duration_days":
        # 1/24 and 1/1440 are recurring decimals. Canonical precision here is
        # 1e-12 day (sub-microsecond), avoiding false inequality from Decimal
        # approximation of those unit ratios.
        canonical = canonical.quantize(Decimal("0.000000000001")).normalize()
    return Quantity(value=canonical, dimension=dimension, comparator=comp, literal=match.group())


def _one_to_one_links(en: list[KeyUnit], zh: list[KeyUnit], links: list[Link]):
    by_en, by_zh = {u.id: u for u in en}, {u.id: u for u in zh}
    for link in links:
        if link.relation not in ("equivalent", "plausible"):
            continue
        if len(link.en_ids) != 1 or len(link.zh_ids) != 1:
            continue
        a = by_en.get(link.en_ids[0])
        b = by_zh.get(link.zh_ids[0])
        if a is not None and b is not None:
            yield a, b


def _quantity_for_unit(unit: KeyUnit) -> Quantity | None:
    quantity = parse_quantity(unit.claim)
    if quantity is not None:
        return quantity
    # Legacy model claims sometimes omit the unit ("accuracy 80") while their
    # short source quote retains it ("reached 80%"). Recover only if both have
    # exactly ONE equal numeric value. Never use a multi-number paragraph as
    # a free-form source for selecting an arbitrary quantity.
    from_claim, from_evidence = numeric_tokens(unit.claim), numeric_tokens(unit.evidence)
    if len(from_claim) == len(from_evidence) == 1 and from_claim[0] == from_evidence[0]:
        return parse_quantity(unit.evidence)
    return None


def clear_numeric_conflicts(en: list[KeyUnit], zh: list[KeyUnit], links: list[Link]) -> list[str]:
    """Compare typed quantities only; never raw digits or full paragraphs."""
    issues = []
    for a, b in _one_to_one_links(en, zh, links):
        xr, yr = parse_range(a.claim), parse_range(b.claim)
        if xr is not None and yr is not None and xr.dimension == yr.dimension:
            if xr != yr:
                issues.append(
                    f"数值区间不同：{a.claim!r} [{xr.lower}, {xr.upper}] ↔ {b.claim!r} [{yr.lower}, {yr.upper}]"
                )
            continue
        x, y = _quantity_for_unit(a), _quantity_for_unit(b)
        if x is None or y is None or x.dimension != y.dimension:
            continue
        if (x.value, x.comparator) != (y.value, y.comparator):
            issues.append(
                f"数值约束不同：{a.claim!r} ({x.comparator} {x.value:g} {x.dimension})"
                f" ↔ {b.claim!r} ({y.comparator} {y.value:g} {y.dimension})"
            )
    return issues


# A narrowly scoped inclusion/exclusion *polarity* sanity check. Only veto
# if both sides explicitly discuss counting/inclusion AND one side affirms
# inclusion while the other expresses only exclusions (or vice versa). No
# entity dictionary, no assumption that paragraph sentence order is aligned.
_EN_INCLUSION = re.compile(
    r"\b(?:not\s+included|excluded|not\s+counted|(?:does|do)\s+not\s+count|"
    r"included|counts?\s+towards?|counted\s+towards?|counted\s+in)\b",
    re.I,
)
_ZH_INCLUSION = re.compile(r"不计入|未计入|不算入|未算入|不占|计入|算入")


def _inclusion_signs(text: str, language: str) -> list[int]:
    regex = _EN_INCLUSION if language == "en" else _ZH_INCLUSION
    return [
        (-1 if re.match(r"(?:not\b|excluded|does\s+not|do\s+not)", match.group(), re.I) else 1)
        if language == "en"
        else (-1 if match.group().startswith(("不", "未")) else 1)
        for match in regex.finditer(text)
    ]


def clear_inclusion_conflicts(en: list[KeyUnit], zh: list[KeyUnit], links: list[Link]) -> list[str]:
    """Find explicit, high-confidence counting/inclusion polarity contradictions.

    The caller already supplies a semantically linked 1:1 claim pair. This
    guard does NOT attempt cross-language entity translation or ordering.
    """
    issues = []
    for a, b in _one_to_one_links(en, zh, links):
        x, y = _inclusion_signs(a.claim, "en"), _inclusion_signs(b.claim, "zh")
        if x and y and ((1 in x and all(sign == -1 for sign in y)) or (1 in y and all(sign == -1 for sign in x))):
            issues.append(f"计入/排除关系疑似反转：{a.claim!r} ↔ {b.claim!r}")
    return issues


# Original-first backstop for a ONE-PREDICATE bilingual source pair. This is
# intentionally narrow: no paragraph-wise numbers from different requirements,
# no multi-paragraph text, no inferred entity translation.
def source_isolated_quantity_conflicts(
    english: str, chinese: str, en: list[KeyUnit], zh: list[KeyUnit], links: list[Link], en_paragraphs, zh_paragraphs
) -> list[str]:
    if len(en_paragraphs) != 1 or len(zh_paragraphs) != 1:
        return []
    if len(english.strip()) > 220 or len(chinese.strip()) > 220:
        return []
    # The source shortcut cannot associate one number with the right subject
    # across multiple standalone statements. Decimals are NOT sentence stops.
    for source in (english, chinese):
        body = source.strip().rstrip("。.!?！？")
        if ";" in body or "；" in body or "\n" in body or re.search(r"[。！？!?]|(?<!\d)\.(?!\d)", body):
            return []
    # A model must explicitly have linked source-grounded facts as equivalent;
    # otherwise the original-only independent review handles semantics.
    en_by_id, zh_by_id = {u.id: u for u in en}, {u.id: u for u in zh}
    linked = any(
        link.relation in ("equivalent", "plausible")
        and any(en_by_id.get(uid) and en_by_id[uid].paragraph_ids == [en_paragraphs[0].id] for uid in link.en_ids)
        and any(zh_by_id.get(uid) and zh_by_id[uid].paragraph_ids == [zh_paragraphs[0].id] for uid in link.zh_ids)
        for link in links
    )
    if not linked:
        return []
    xr, yr = parse_range(english), parse_range(chinese)
    if xr and yr and xr.dimension == yr.dimension:
        if xr != yr:
            return [f"原文数值区间不同：{english!r} ↔ {chinese!r}"]
        return []
    x, y = parse_quantity(english), parse_quantity(chinese)
    if x and y and x.dimension == y.dimension and (x.value, x.comparator) != (y.value, y.comparator):
        return [
            f"原文数值约束不同：{english!r} ({x.comparator} {x.value:g} {x.dimension}) "
            f"↔ {chinese!r} ({y.comparator} {y.value:g} {y.dimension})"
        ]
    return []
