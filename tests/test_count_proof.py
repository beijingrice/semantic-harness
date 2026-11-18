"""direct_hardened counted-entity proof (count_proof).

The proof used to require the literal wording of one dev case ("main report
limit", "N页限制"), so any other phrasing of the same conflict fell back to
REVIEW_REQUIRED (live: NUM30-09F). It now accepts any wording that points at
the same identified limit, and still abstains when the cited rule may belong
to a different limit.
"""
import pytest

from app.pipelines.hardened_rules import count_proof


def prove(english, chinese, en_text=None, zh_text=None, en_subject="cover page", zh_subject="封面"):
    return count_proof(en_text or english, zh_text or chinese, english, chinese, en_subject, zh_subject)


@pytest.mark.parametrize("english,chinese,en_text,zh_text,subjects", [
    # Live case NUM30-09F: "counted within" + "篇幅限制"
    ("The cover page is counted within the 25-page limit.", "封面不计入25页的篇幅限制。", None, None, None),
    # Deictic "this limit" / "该限制", cited sentences only
    ("The report has a 30-page limit. The title page counts toward this limit.",
     "报告有30页限制。标题页不计入该限制。",
     "The title page counts toward this limit.", "标题页不计入该限制。", ("title page", "标题页")),
    # "limited to N pages" + "does not count toward"
    ("The main report is limited to 30 pages; the cover page does not count toward this limit.",
     "封面计入正文报告30页的篇幅限制。", None, None, None),
    # Word limit
    ("The essay has an 8000-word limit; references are included.", "论文有8000字限制，参考文献不计入。",
     None, None, ("references", "参考文献")),
    # Original dev wording still works
    ("The appendix is outside the 30-page limit for the main report. The cover page is included within the main-report limit.",
     "附录不占正文报告30页的额度。封面不计入30页限制。",
     "The cover page is included within the main-report limit.", "封面不计入30页限制。", None),
])
def test_conflict_with_other_wordings_is_proven(english, chinese, en_text, zh_text, subjects):
    en_subject, zh_subject = subjects or ("cover page", "封面")
    assert prove(english, chinese, en_text, zh_text, en_subject, zh_subject) == "FAIL"


@pytest.mark.parametrize("english,chinese,en_text,zh_text", [
    # The cited EN rule names a different (appendix) limit never tied to 25 pages
    ("The report has a 25-page limit. The cover page is included within the appendix limit.",
     "报告有25页的篇幅限制。封面不计入25页的篇幅限制。",
     "The cover page is included within the appendix limit.", "封面不计入25页的篇幅限制。"),
    # Two different limit values on one side: the entity could belong to either
    ("The appendix has a 6-page limit and the report has a 30-page limit. The cover page is counted within the 30-page limit.",
     "封面不计入30页的篇幅限制。", None, None),
    # Different limit values across languages
    ("The cover page is counted within the 25-page limit.", "封面不计入30页的篇幅限制。", None, None),
    # "this limit" while the source also names an untied appendix limit
    ("The report has a 25-page limit and the appendix limit is separate. The cover page counts toward the limit.",
     "报告有25页的篇幅限制。封面不计入该限制。",
     "The cover page counts toward the limit.", "封面不计入该限制。"),
    # "counted separately" is not inclusion
    ("The cover page is counted separately from the 25-page limit.", "封面计入25页的篇幅限制。", None, None),
    # Same state on both sides: nothing to prove
    ("The cover page is counted within the 25-page limit.", "封面计入25页的篇幅限制。", None, None),
    # No limit stated at all
    ("The cover page is counted.", "封面不计入。", None, None),
])
def test_ambiguous_or_consistent_cases_still_abstain(english, chinese, en_text, zh_text):
    assert prove(english, chinese, en_text, zh_text) == "REVIEW_REQUIRED"


def test_model_naming_a_different_known_entity_still_abstains():
    english = "The cover page is counted within the 25-page limit; the appendix is not counted."
    chinese = "封面不计入25页的篇幅限制，附录也不计入。"
    assert prove(english, chinese, en_subject="appendix", zh_subject="附录") == "REVIEW_REQUIRED"
    assert prove(english, chinese) == "FAIL"


def test_undetermined_dispute_skips_the_model_call():
    """Disagreement with no source-checkable risk -> Q_DISPUTED_CLAIM.

    prove can only return REVIEW_REQUIRED for it, so no third model call is
    made (live, that call spent ~3 minutes truncating) and the verdict is the same.
    """
    import json

    import httpx

    from app.checker import Checker
    from app.config import Settings
    from app.llm import ChatClient

    replies = [
        {"verdict": "PASS", "reason": "same duty", "issues": []},
        {"verdict": "FAIL", "reason": "alleged difference", "issues": [
            {"kind": "conflict", "en_paragraph_ids": ["en_p1"], "zh_paragraph_ids": ["zh_p1"],
             "reason": "alleged wording difference"}]},
    ]
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert len(calls) <= len(replies), "unexpected extra model call"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(replies[len(calls) - 1])},
                                                      "finish_reason": "stop"}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(model="primary", verifier_model="verifier", evidence_mode="paragraph",
                   pipeline="direct_hardened", json_schema=True)
    client = ChatClient(cfg, http)
    result = Checker(client, cfg).check("Authors must upload the dataset.", "作者须上传数据集。", mode="dual")
    http.close()
    assert len(calls) == 2
    assert result.verdict == "REVIEW_REQUIRED"
    assert result.verification_questions == [{
        "task_id": "Q_DISPUTED_CLAIM", "question_type": "undetermined",
        "host_conclusion": "REVIEW_REQUIRED", "outcome": "SKIPPED_NOT_SOURCE_CHECKABLE"}]
