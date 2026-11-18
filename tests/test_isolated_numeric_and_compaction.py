"""Isolated numeric predicates and lossless single-source compaction.

Some fixtures are taken from actual v0.2.8 holdout model outputs. All tests
are offline; successful replay is not a promise of identical live inference.
"""

import json
from pathlib import Path

import pytest
from helpers import Fake, arb, config, extract

from app.checker import Checker
from app.dispute_resolution import narrow_exactness_refutation, numeric_equivalence_certificates
from app.models import (
    KeyUnit,
    Link,
    MatchDraft,
    ParagraphDisputeRecheckDraft,
    ParagraphExtractedDraft,
    ParagraphReviewDraft,
)
from app.paragraphs import build_paragraphs
from app.validators import clear_numeric_conflicts, parse_range, source_isolated_quantity_conflicts

SAMPLES = json.loads((Path(__file__).parent / "fixtures/v028_failures_v029.json").read_text())


def paired(en, zh):
    left = [KeyUnit(id="en_1", kind="constraint", claim=en, evidence=en)]
    right = [KeyUnit(id="zh_1", kind="constraint", claim=zh, evidence=zh)]
    links = [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation="equivalent", reason="model says equal")]
    return left, right, links


@pytest.mark.parametrize(
    "en,zh,expected",
    [
        ("The calibration error must stay below 0.8 millimetres.", "校准误差不得超过0.8毫米。", True),
        ("The calibration error must stay below 0.8 millimetres.", "校准误差应低于0.8毫米。", False),
        ("The width must be at least 1.2 cm.", "宽度不得低于12毫米。", False),
        ("A sample must be more than 10 mm long.", "样本长度须超过1厘米。", False),
        ("The report may have at most 0.8 metres of margins.", "边距不得超过80厘米。", False),
        ("The process lasted 72 hours.", "该过程持续三天。", False),
        ("The process lasted 90 minutes.", "该过程持续1.5小时。", False),
        ("The process lasted 72 hours.", "该过程持续四天。", True),
        ("The precision must not be below 0.8 mm.", "精度至少为0.8毫米。", False),
        ("The precision must not be below 0.8 mm.", "精度必须低于0.8毫米。", True),
    ],
)
def test_units_dimension_boundary_and_exact_conversion(en, zh, expected):
    assert bool(clear_numeric_conflicts(*paired(en, zh))) is expected


def test_v028_real_error_replay_recovers_strict_vs_nonstrict_bound():
    sample = SAMPLES["H28-02"]
    fake = Fake(
        ParagraphExtractedDraft.model_validate(
            {
                "units": [
                    {"kind": u["kind"], "claim": u["claim"], "paragraph_ids": u["paragraph_ids"]}
                    for u in sample["english_keywords"]
                ]
            }
        ),
        ParagraphExtractedDraft.model_validate(
            {
                "units": [
                    {"kind": u["kind"], "claim": u["claim"], "paragraph_ids": u["paragraph_ids"]}
                    for u in sample["chinese_keywords"]
                ]
            }
        ),
        MatchDraft.model_validate({"links": sample["links"]}),
        ParagraphReviewDraft.model_validate(sample["review"]),
    )
    out = Checker(fake, config()).check(sample["en"], sample["zh"], mode="dual")
    assert out.verdict == "FAIL"
    assert any("0.8" in issue and "length_mm" in issue for issue in out.deterministic_issues)
    assert len(fake.calls) == 4  # never spend money arbitrating a formal contradiction


def test_intervals_parse_ambiguous_inclusivity_without_hallucinating_it():
    a = parse_range("Tables must use a font size between 9 and 11 points, inclusive.")
    b = parse_range("表格字体须在9至12磅之间（包含两端）。")
    c = parse_range("表格字体须在9至11磅之间（包含两端）。")
    assert a and b and c
    assert a.dimension == b.dimension == c.dimension == "typographic_points"
    assert a == c and a != b
    assert bool(
        clear_numeric_conflicts(
            *paired(
                "Tables must use a font size between 9 and 11 points, inclusive.",
                "表格字体须在9至12磅之间（包含两端）。",
            )
        )
    )
    assert not parse_range("Tables must use between 9 and 11 points.")
    assert not parse_range("The report may use 9 pages and the appendix may use 11 pages, inclusive.")
    assert parse_range("The font should be between 9 and 11 points, exclusive.").lower_inclusive is False


def test_original_first_backstop_catches_bad_match_claim_and_abstains_on_multiple_numbers():
    en = "The calibration error must stay below 0.8 millimetres."
    zh = "校准误差不得超过0.8毫米。"
    en_units, zh_units, links = paired("Calibration error is 0.8 mm.", "校准误差为0.8毫米。")
    en_units[0].paragraph_ids = ["en_p1"]
    zh_units[0].paragraph_ids = ["zh_p1"]
    issues = source_isolated_quantity_conflicts(
        en, zh, en_units, zh_units, links, build_paragraphs(en, "en"), build_paragraphs(zh, "zh")
    )
    assert issues and "原文" in issues[0]
    multi = "The report has 6 pages and the appendix has 30 pages."
    assert not source_isolated_quantity_conflicts(
        multi,
        "报告有6页、附录有31页。",
        en_units,
        zh_units,
        links,
        build_paragraphs(multi, "en"),
        build_paragraphs("报告有6页、附录有31页。", "zh"),
    )
    assert not source_isolated_quantity_conflicts(
        en,
        zh,
        en_units,
        zh_units,
        [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation="conflict", reason="yes")],
        build_paragraphs(en, "en"),
        build_paragraphs(zh, "zh"),
    )


def test_real_v028_04_extraction_bloat_compressed_to_original_verbatim():
    sample = SAMPLES["H28-04"]
    draft = ParagraphExtractedDraft.model_validate(
        {
            "units": [
                {"kind": u["kind"], "claim": u["claim"], "paragraph_ids": u["paragraph_ids"]}
                for u in sample["chinese_keywords"]
            ]
        }
    )
    assert len(draft.units) == 27
    fake = Fake(draft)
    events = []
    units = Checker(fake, config()).extract(sample["zh"], "Chinese", "zh", "primary", None, events)
    assert len(units) == 1
    assert units[0].claim == sample["zh"] and units[0].evidence == sample["zh"]
    assert "collapsed 22" in " ".join(events)


def test_short_source_nonindependent_noun_fragments_removed_but_real_constraints_retained():
    source = "参与者接受连续三天的监测。"
    draft = ParagraphExtractedDraft.model_validate(
        {
            "units": [
                {"kind": "constraint", "claim": source, "paragraph_ids": ["zh_p1"]},
                {"kind": "entity", "claim": "参与者", "paragraph_ids": ["zh_p1"]},
                {"kind": "entity", "claim": "连续三天的监测", "paragraph_ids": ["zh_p1"]},
                {"kind": "constraint", "claim": "监测持续三天", "paragraph_ids": ["zh_p1"]},
            ]
        }
    )
    fake = Fake(draft)
    events = []
    units = Checker(fake, config()).extract(source, "Chinese", "zh", "primary", None, events)
    assert len(units) == 2
    assert [u.id for u in units] == ["zh_1", "zh_2"]
    assert all(u.kind == "constraint" for u in units)
    assert "literal entity" in " ".join(events)


def test_short_source_output_budget_signaled_to_model_without_weaker_schema():
    src = "表格字体大小须在9至12磅之间（包含两端）。"
    fake = Fake(extract(src, "zh_p1"))
    Checker(fake, config()).extract(src, "Chinese", "zh", "primary", None, [])
    assert fake.calls[0]["user"]["target_units_for_short_source"] == 1
    assert fake.calls[0]["user"]["maximum_units"] == 30


def test_never_compact_multi_paragraph_evidence_or_split_independent_requirements():
    source = "Maintain the original log.\n\nDelete only derived caches."
    draft = ParagraphExtractedDraft.model_validate(
        {
            "units": [
                {"kind": "constraint", "claim": "Maintain the original log.", "paragraph_ids": ["en_p1"]},
                {"kind": "constraint", "claim": "Delete only derived caches.", "paragraph_ids": ["en_p2"]},
            ]
        }
    )
    fake = Fake(draft)
    units = Checker(fake, config()).extract(source, "English", "en", "primary", None, [])
    assert len(units) == 2


def test_72_hours_three_days_certificate_does_not_override_unrelated_duty():
    en = "Participants were monitored for 72 hours without interruption."
    zh = "参与者接受连续三天的监测。"
    e = [KeyUnit(id="en_1", kind="constraint", claim=en, evidence=en, paragraph_ids=["en_p1"])]
    z = [KeyUnit(id="zh_1", kind="constraint", claim=zh, evidence=zh, paragraph_ids=["zh_p1"])]
    link = [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation="equivalent", reason="same duration")]
    certs = numeric_equivalence_certificates(e, z, link, build_paragraphs(en, "en"), build_paragraphs(zh, "zh"))
    assert len(certs) == 1 and certs[0].canonical_value == "3"
    # A true formal equality can't prove unrelated archival requirements.
    different_en = en + " The recordings must be archived."
    different_zh = zh + " 录音可以删除。"
    certs = numeric_equivalence_certificates(
        e, z, link, build_paragraphs(different_en, "en"), build_paragraphs(different_zh, "zh")
    )
    from app.models import ParagraphIssueRecheck, ParagraphReviewIssue

    issue = ParagraphReviewIssue(
        kind="conflict",
        en_paragraph_ids=["en_p1"],
        zh_paragraph_ids=["zh_p1"],
        reason="The numeric duration matches, but archival permissions differ.",
    )
    answer = ParagraphIssueRecheck(
        issue_index=0,
        status="refuted",
        basis="formal_numeric",
        en_paragraph_ids=["en_p1"],
        zh_paragraph_ids=["zh_p1"],
        reason="refuted",
    )
    assert not narrow_exactness_refutation(
        issue, answer, certs, build_paragraphs(different_en, "en"), build_paragraphs(different_zh, "zh")
    )


def test_real_h28_03_model_units_plus_targeted_objection_path():
    sample = SAMPLES["H28-03"]
    en_draft = ParagraphExtractedDraft.model_validate(
        {
            "units": [
                {"kind": u["kind"], "claim": u["claim"], "paragraph_ids": u["paragraph_ids"]}
                for u in sample["english_keywords"]
            ]
        }
    )
    zh_draft = ParagraphExtractedDraft.model_validate(
        {
            "units": [
                {"kind": u["kind"], "claim": u["claim"], "paragraph_ids": u["paragraph_ids"]}
                for u in sample["chinese_keywords"]
            ]
        }
    )
    # Real source fragments zh_2 and zh_3 disappear, while original zh_4
    # remains as zh_2. The live model would see and use the new IDs.
    links = MatchDraft.model_validate(
        {
            "links": [
                {"en_ids": ["en_1"], "zh_ids": ["zh_1", "zh_2"], "relation": "equivalent", "reason": "72h = 3 days"}
            ]
        }
    )
    recheck = ParagraphDisputeRecheckDraft.model_validate(
        {
            "issues": [
                {
                    "issue_index": 0,
                    "status": "refuted",
                    "basis": "formal_numeric",
                    "en_paragraph_ids": ["en_p1"],
                    "zh_paragraph_ids": ["zh_p1"],
                    "reason": "continuous monitoring for 72 hours equals three continuous days",
                }
            ],
            "reason": "formal units and continuity align",
        }
    )
    fake = Fake(
        en_draft,
        zh_draft,
        links,
        ParagraphReviewDraft.model_validate(
            {
                "verdict": sample["review"]["verdict"],
                "reason": sample["review"]["reason"],
                "issues": [
                    {k: v for k, v in issue.items() if k not in ("en_evidence", "zh_evidence")}
                    for issue in sample["review"]["issues"]
                ],
            }
        ),
        arb("PASS", "none", ["en_p1"], ["zh_p1"]),
        recheck,
    )
    result = Checker(fake, config()).check(sample["en"], sample["zh"], mode="dual")
    assert result.verdict == "PASS"
    assert result.initial_verdict == "PASS"
    assert len(result.chinese_keywords) == 2
    assert len(fake.calls) == 6
    assert len(fake.calls[-1]["user"]["verified_formal_equivalences"]) == 2


def test_numeric_proof_cannot_override_continuity_or_modal_disagreement():
    from app.dispute_resolution import _unmatched_nonnumeric_qualifiers

    assert not _unmatched_nonnumeric_qualifiers(
        "Participants were monitored for 72 hours without interruption.", "参与者接受连续三天的监测。"
    )
    assert not _unmatched_nonnumeric_qualifiers("Uninterrupted monitoring lasted 72 hours.", "连续监测三天。")
    assert not _unmatched_nonnumeric_qualifiers(
        "Monitoring continued for 72 hours without interruption.", "监测持续三天且不中断。"
    )
    assert _unmatched_nonnumeric_qualifiers(
        "Monitoring continued for 72 hours without interruption.", "监测持续三天，但有间歇。"
    )
    assert _unmatched_nonnumeric_qualifiers("Monitoring continued for 72 hours.", "监测连续进行了三天。")
    assert _unmatched_nonnumeric_qualifiers("The recordings must be archived after 72 hours.", "录音在三天后可以归档。")
    assert _unmatched_nonnumeric_qualifiers("The recordings must be archived after 72 hours.", "录音在三天后无须归档。")
