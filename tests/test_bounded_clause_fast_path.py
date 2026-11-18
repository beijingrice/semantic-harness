"""Bounded-clause fast path: pre-generation safeguards and disagreement safety gates.

These are offline protocol assertions, not claims of real Qwen accuracy.
"""

import json

import httpx
import pytest
from helpers import Fake, arb, config, match, review

from app.checker import Checker
from app.config import Settings
from app.extraction_fast_path import bounded_clause_anchor_pair
from app.llm import ChatClient
from app.models import MatchDraft
from app.paragraphs import build_paragraphs

EN = (
    "The review addresses data quality, annotation consistency, and evaluation "
    "reproducibility. These topics may be discussed in any order."
)
ZH = "综述需要涉及数据质量、标注一致性以及评测可复现性，三者的叙述顺序不作限制。"


def multiple_links(links):
    return MatchDraft.model_validate({"links": links})


def anchors(en=EN, zh=ZH):
    return bounded_clause_anchor_pair(build_paragraphs(en, "en"), build_paragraphs(zh, "zh"))


def test_real_v0291_mismatch_uses_two_verbatim_source_anchors():
    en, zh = anchors()
    assert [u.claim for u in en] == [
        "The review addresses data quality, annotation consistency, and evaluation reproducibility.",
        "These topics may be discussed in any order.",
    ]
    assert [u.claim for u in zh] == [
        "综述需要涉及数据质量、标注一致性以及评测可复现性",
        "三者的叙述顺序不作限制。",
    ]
    for unit in en + zh:
        assert unit.claim in unit.evidence
        assert unit.paragraph_ids == (["en_p1"] if unit.id.startswith("en_") else ["zh_p1"])


def test_no_extraction_requests_for_two_structurally_aligned_clauses():
    fake = Fake(
        multiple_links(
            [
                {"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "same topics"},
                {"en_ids": ["en_2"], "zh_ids": ["zh_2"], "relation": "equivalent", "reason": "same order freedom"},
            ]
        ),
        review("PASS"),
    )
    result = Checker(fake, config(extraction_fast_path=True)).check(EN, ZH, mode="dual")
    assert result.verdict == "PASS"
    assert [call["schema"] for call in fake.calls] == ["keyword_matching", "paragraph_review"]
    assert len(result.chinese_keywords) == 2
    assert any("bounded_clause_fast_path" in line for line in result.diagnostics)


def test_http_protocol_original_text_still_goes_to_both_models():
    requests = []
    answers = [
        {
            "links": [
                {"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "same components"},
                {"en_ids": ["en_2"], "zh_ids": ["zh_2"], "relation": "equivalent", "reason": "order is free"},
            ]
        },
        {"verdict": "PASS", "reason": "the originals contain identical list and order conditions", "issues": []},
    ]

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(answers[len(requests) - 1])}, "finish_reason": "stop"}]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(
        evidence_mode="paragraph", model="main", verifier_model="verifier", extraction_fast_path=True, json_schema=True
    )
    client = ChatClient(cfg, http)
    client.start_trace()
    out = Checker(client, cfg).check(EN, ZH, mode="dual")
    assert out.verdict == "PASS"
    assert len(requests) == 2
    assert [r["stage"] for r in client.trace_snapshot()] == ["keyword_matching", "independent_review"]
    assert EN in requests[0]["messages"][1]["content"]
    assert ZH in requests[1]["messages"][1]["content"]
    http.close()


@pytest.mark.parametrize(
    "en,zh",
    [
        (
            "The abstract lists the motivation, method, key result, and contribution. "
            "Keywords appear immediately below it, not inside the abstract.",
            "摘要覆盖研究动机、方法、核心结果与贡献，关键词写入摘要正文而非放在其下方。",
        ),
        ("The file must be archived. The file may be deleted.", "文件必须归档；文件可以删除。"),
        (
            "If the author withdraws consent, the files must be deleted. The archive must remain private.",
            "如果作者撤回同意，必须删除文件，存档必须保密。",
        ),
        ("The paper must include two sections. The tables must be editable.", "论文必须包含两个部分。表格必须可编辑。"),
        (
            "The paper must list the 0.8 mm bound. The limit must remain strict.",
            "论文必须列出0.8毫米的边界，限制必须严格。",
        ),
        ("One sentence. Two sentences. Three sentences.", "一个句子，第二个句子。"),
    ],
)
def test_structurally_ambiguous_sources_keep_llm_extraction(en, zh):
    assert anchors(en, zh) is None


def test_two_clauses_mismatching_obligation_do_not_automatically_pass():
    en = "The report must include tables. The original files must be archived."
    zh = "报告必须包含表格，原始文件可以删除。"
    assert anchors(en, zh) is not None
    # Even if the matcher wrongly declares equivalence, independent review
    # grounds a contradiction. The arbitral PASS cannot override it without
    # a narrow formal numeric certificate.
    from app.models import ParagraphDisputeRecheckDraft

    recheck = ParagraphDisputeRecheckDraft.model_validate(
        {
            "issues": [
                {
                    "issue_index": 0,
                    "status": "refuted",
                    "basis": "other",
                    "en_paragraph_ids": ["en_p1"],
                    "zh_paragraph_ids": ["zh_p1"],
                    "reason": "false refutation",
                }
            ],
            "reason": "unsubstantiated",
        }
    )
    issue = {
        "kind": "conflict",
        "reason": "archive required versus delete allowed",
        "en_paragraph_ids": ["en_p1"],
        "zh_paragraph_ids": ["zh_p1"],
    }
    fake = Fake(
        multiple_links(
            [
                {"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "tables"},
                {"en_ids": ["en_2"], "zh_ids": ["zh_2"], "relation": "equivalent", "reason": "wrong"},
            ]
        ),
        review("FAIL", [issue]),
        arb("PASS", "none", ["en_p1"], ["zh_p1"]),
        recheck,
    )
    out = Checker(fake, config(extraction_fast_path=True)).check(en, zh, mode="dual")
    assert out.verdict == "REVIEW_REQUIRED"


def test_initial_and_arbitration_fail_against_reviewer_pass_is_not_verified_fail():
    # Actual v0.2.9.1 was initial FAIL, independent reviewer PASS and Qwen
    # arbitration FAIL, with invented 'may' scope / citation explanations.
    fake = Fake(
        multiple_links(
            [
                {
                    "en_ids": ["en_1"],
                    "zh_ids": ["zh_1"],
                    "relation": "conflict",
                    "reason": "invents a modality mismatch",
                },
                {"en_ids": ["en_2"], "zh_ids": ["zh_2"], "relation": "equivalent", "reason": "same order"},
            ]
        ),
        review("PASS"),
        arb("FAIL", "conflict", ["en_p1"], ["zh_p1"]),
    )
    out = Checker(fake, config(extraction_fast_path=True)).check(EN, ZH, mode="dual")
    assert out.initial_verdict == "FAIL"
    assert out.review.verdict == "PASS"
    assert out.arbitration.verdict == "FAIL"
    assert out.verdict == "REVIEW_REQUIRED"
    assert any("no verified conflict" in x for x in out.diagnostics)


def test_no_numeric_guard_or_review_safety_regression_on_declared_inequality():
    en = "The error must remain less than 0.8 mm."
    zh = "误差不得超过0.8毫米。"
    # Single-clause Fast Path should still enforce the numerical comparison.
    from app.extraction_fast_path import source_anchor_pair

    assert source_anchor_pair(build_paragraphs(en, "en"), build_paragraphs(zh, "zh")) is not None
    fake = Fake(match(), review("PASS"))
    out = Checker(fake, config(extraction_fast_path=True)).check(en, zh, mode="dual")
    assert out.verdict == "FAIL"
    assert out.deterministic_issues
