"""Offline regression for *real* 11-case failures: no LLM API access.

These tests exercise explicit numeric semantics and the paragraph-mode control
flow with mocked semantic outputs; they don't claim a live-model accuracy rate.
"""

import pytest
from helpers import Fake, config, extract, match, review

from app.checker import Checker
from app.models import KeyUnit, Link
from app.validators import (
    clear_inclusion_conflicts,
    clear_numeric_conflicts,
)


def paired(en, zh, relation="equivalent"):
    left = [KeyUnit(id="en_1", kind="constraint", claim=en, evidence=en)]
    right = [KeyUnit(id="zh_1", kind="constraint", claim=zh, evidence=zh)]
    links = [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation=relation, reason="mock model judged equivalent")]
    return left, right, links


@pytest.mark.parametrize(
    "en,zh",
    [
        ("The survey ran for exactly six weeks.", "问卷调查持续了42天，既未延长也未缩短。"),
        ("The survey duration is exactly six weeks.", "问卷调查的持续时间固定为42天，不允许延长或缩短。"),
        ("If more than 10% of a respondent's fields are blank, invalidate.", "若必填字段缺失比例超过10%，该答卷无效。"),
        ("No fewer than 120 responses are needed.", "不少于120份答卷。"),
        ("The report must not exceed 20 pages.", "报告不得超过二十页。"),
        ("The appendix may have up to 6 pages.", "附录最多6页。"),
        ("Keep samples for fewer than five days.", "样本保留少于5天。"),
        ("Use a 12-point font.", "使用12磅字号。"),
    ],
)
def test_equivalent_quantities_do_not_trigger_guard(en, zh):
    assert not clear_numeric_conflicts(*paired(en, zh))


@pytest.mark.parametrize(
    "en,zh",
    [
        ("If more than 10% of mandatory fields are blank, reject it.", "若必填字段缺失比例达到10%或以上，该答卷无效。"),
        ("The survey ran for exactly six weeks.", "问卷调查持续了43天。"),
        ("The report may be up to 20 pages.", "报告最多30页。"),
        ("Training accuracy reached 80%.", "训练准确率达到60%。"),
        ("A value must be less than 10%.", "该数值不超过10%。"),
        ("The report is limited to 30 pages.", "报告至少30页。"),
        ("At most 12 pages are allowed.", "至少12页。"),
    ],
)
def test_numeric_mismatch_detected_even_if_llm_marks_equivalent(en, zh):
    issues = clear_numeric_conflicts(*paired(en, zh))
    assert len(issues) == 1, (en, zh)


def test_numeric_guard_never_reads_unrelated_numbers_from_whole_paragraph():
    en, zh, links = paired("The cover page is included in the main-report limit.", "封面不计入30页限制。")
    assert clear_numeric_conflicts(en, zh, links) == []
    en, zh, links = paired("The limit is 6 pages, separate from a 30-page report.", "附录上限6页，不占30页正文额度。")
    assert clear_numeric_conflicts(en, zh, links) == []


def test_dimension_mismatch_is_abstention_not_false_conflict():
    assert clear_numeric_conflicts(*paired("The study took 6 weeks.", "研究持续6天。"))
    assert not clear_numeric_conflicts(*paired("The report has 20 pages.", "研究持续20天。"))
    assert not clear_numeric_conflicts(*paired("The count is 20.", "总量为30。"))


def test_inclusion_contradiction_v2_num15():
    en = "The cover page is included within the main-report limit, whereas the reference list is not."
    zh = "封面和参考文献列表不计入30页的正文报告限制。"
    issues = clear_inclusion_conflicts(*paired(en, zh))
    assert len(issues) == 1
    assert not clear_inclusion_conflicts(*paired(en, "封面计入30页正文报告限制，参考文献列表不计入。"))
    assert not clear_inclusion_conflicts(
        *paired("References are excluded from the page count.", "参考文献不计入总页数。")
    )


def test_inclusion_guard_never_guesses_without_explicit_bilingual_cues():
    assert not clear_inclusion_conflicts(*paired("The appendix is available.", "附录不计入页数。"))
    assert not clear_inclusion_conflicts(*paired("The cover page is included.", "封面已提交。"))
    assert not clear_inclusion_conflicts(*paired("The cover page is included.", "封面不计入。", relation="conflict"))


@pytest.mark.parametrize(
    "en,zh,should_fail",
    [
        ("The survey ran for exactly six weeks.", "问卷调查持续了42天，既未延长也未缩短。", False),
        (
            "If more than 10% of a respondent's mandatory fields are blank, that response is invalid.",
            "若某份答卷的必填字段缺失比例达到10%或以上，该答卷无效。",
            True,
        ),
        (
            "The cover page is included within the main-report limit, whereas the reference list is not.",
            "封面和参考文献列表不计入30页的正文报告限制。",
            True,
        ),
    ],
)
def test_paragraph_pipeline_numeric_and_inclusion_guard(en, zh, should_fail):
    fake = Fake(extract(en, "en_p1"), extract(zh, "zh_p1"), match(), review("PASS"))
    out = Checker(fake, config()).check(en, zh, mode="dual")
    assert (out.verdict == "FAIL") == should_fail
    assert bool(out.deterministic_issues) == should_fail
    assert not out.arbitrated
    assert len(fake.calls) == 4  # guards are Python only; no extra model calls


def test_duplicate_paragraph_units_are_dropped_without_fuzzy_semantic_merges():
    from app.models import ParagraphExtractedDraft

    def many(*claims):
        return ParagraphExtractedDraft.model_validate(
            {"units": [{"kind": "constraint", "claim": claim, "paragraph_ids": ["zh_p1"]} for claim in claims]}
        )

    text = "报告最多30页。"
    fake = Fake(many("报告最多30页。", "报告最多30页。", "报告的页数上限为30页。"))
    events = []
    result = Checker(fake, config()).extract(text, "Chinese", "zh", "primary", None, events)
    assert len(result) == 2
    assert [u.id for u in result] == ["zh_1", "zh_2"]
    assert any("duplicate" in event for event in events)
