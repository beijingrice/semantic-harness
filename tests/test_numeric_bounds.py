"""Comparator parsing for the deterministic numeric guard.

Regression for a false-FAIL bug: bound words after the unit ("10 页以上",
"30 pages or more"), "exceed" and "within" were parsed as exact equality, so
two equivalent sentences produced a guard conflict, which vetoes to FAIL.
"""

import pytest

from app.models import KeyUnit, Link
from app.validators import clear_numeric_conflicts, parse_quantity


@pytest.mark.parametrize(
    "text,comparator",
    [
        # Bound word after the unit
        ("10 页以上", ">="),
        ("至少五天以上", ">="),
        ("30 pages or more", ">="),
        ("30 pages or above", ">="),
        ("10 pages or fewer", "<="),
        ("20 pages or less", "<="),
        ("6 周内", "<="),
        ("42 天以内", "<="),
        ("42 天之内", "<="),
        # exceed / above / within
        ("must exceed 10%", ">"),
        ("exceeds 10%", ">"),
        ("in excess of 10%", ">"),
        ("above 10%", ">"),
        ("within 6 weeks", "<="),
        ("no later than 6 weeks", "<="),
        # Negated exceed must stay "at most", never fall through to ">"
        ("must not exceed 10%", "<="),
        ("does not exceed 10%", "<="),
        ("doesn't exceed 10%", "<="),
        ("should never exceed 10%", "<="),
        ("cannot exceed 10%", "<="),
        ("not to exceed 10%", "<="),
        ("not exceeding 20 pages", "<="),
        ("not above 10%", "<="),
    ],
)
def test_bound_phrasings(text, comparator):
    assert parse_quantity(text).comparator == comparator


@pytest.mark.parametrize(
    "text,comparator",
    [
        ("at least 10 pages", ">="),
        ("不少于 30 页", ">="),
        ("10%以上", ">="),
        ("至少保留五天", ">="),
        ("more than 10%", ">"),
        ("须超过 10%", ">"),
        ("no more than 3 hours", "<="),
        ("不超过 42 天", "<="),
        ("less than 5 pages", "<"),
        ("below 5%", "<"),
        ("not below 10%", ">="),
        ("exactly 10 pages", "="),
        ("ran for six weeks", "="),
        ("调查持续 42 天。", "="),
        # A comparator before the number outranks a bound word after the unit.
        ("少于五天内", "<"),
        # "内" in an unrelated word is not a bound.
        ("5 年内容", "="),
        # "within" / 内 are bounds only for time spans; for pages they usually
        # mean inclusion in a limit, so they keep the old (exact) reading.
        ("The cover page is included within 30 pages.", "="),
        ("封面计入30页内。", "="),
    ],
)
def test_existing_phrasings_unchanged(text, comparator):
    assert parse_quantity(text).comparator == comparator


def test_age_in_years_is_not_weeks():
    # "周岁" is age in years; it must not match the 周 (week) unit.
    assert parse_quantity("年满18周岁的参与者") is None
    assert parse_quantity("6 周内").value == 42


def test_approximate_quantities_still_abstain():
    assert parse_quantity("about 6 weeks") is None
    assert parse_quantity("6 周左右") is None


def _guard(en_claim, zh_claim):
    en = [KeyUnit(id="en_1", kind="constraint", claim=en_claim, evidence=en_claim)]
    zh = [KeyUnit(id="zh_1", kind="constraint", claim=zh_claim, evidence=zh_claim)]
    links = [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation="equivalent", reason="same requirement")]
    return clear_numeric_conflicts(en, zh, links)


@pytest.mark.parametrize(
    "en,zh",
    [
        ("The paper must be at least 10 pages.", "论文须 10 页以上。"),
        ("The paper must be 30 pages or more.", "论文至少 30 页。"),
        ("The response rate must exceed 10%.", "回复率须超过 10%。"),
        ("Reports must be submitted within 6 weeks.", "报告须在 42 天内提交。"),
        ("Reports must be submitted within 6 weeks.", "报告提交时间不超过 42 天。"),
        ("The survey ran for six weeks.", "调查持续 42 天。"),
        ("The cover page is included within 30 pages.", "封面计入30页。"),
    ],
)
def test_equivalent_pairs_no_longer_vetoed(en, zh):
    assert _guard(en, zh) == []


@pytest.mark.parametrize(
    "en,zh",
    [
        ("The response rate must exceed 10%.", "回复率须不低于 10%。"),  # > vs >=
        ("The paper must be at least 10 pages.", "论文须 12 页以上。"),  # value differs
        ("Reports must be submitted within 6 weeks.", "报告须在 30 天内提交。"),
    ],
)
def test_real_differences_still_caught(en, zh):
    assert _guard(en, zh)


# A negation before a strict comparator flips it into the opposite inclusive bound.
# Before this was a general rule, 不小于 / 不大于 / no less than / not fewer than were
# parsed as the strict comparator itself, and the guard vetoed correct PASS verdicts.
@pytest.mark.parametrize(
    "text,comparator",
    [
        ("字号不小于24磅", ">="),
        ("字号不得小于24磅", ">="),
        ("字号不应少于24磅", ">="),
        ("时长不短于2小时", ">="),
        ("字号不大于24磅", "<="),
        ("报告不得多于二十页", "<="),
        ("时长不长于2小时", "<="),
        ("no less than 24 pt", ">="),
        ("not fewer than 30 participants", ">="),
        ("must not fall below 24 pt", ">="),
        ("must be no longer than 12 pages", "<="),
        ("must not be more than 30 pages", "<="),
        ("样本不足30人", "<"),
        ("字号小于24磅", "<"),
        ("时长超过2小时", ">"),
    ],
)
def test_negated_comparators_flip_to_inclusive_bounds(text, comparator):
    assert parse_quantity(text).comparator == comparator


@pytest.mark.parametrize(
    "text",
    [
        "时长不必超过2小时",
        "页数无需少于10页",
        "it need not exceed 30 pages",
        "it does not have to be longer than 2 hours",
    ],
)
def test_lifted_requirements_are_not_bounds(text):
    assert parse_quantity(text) is None
