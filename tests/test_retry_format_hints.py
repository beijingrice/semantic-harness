"""Regression: a failed model turn must receive a hint for ITS actual wire schema.

These mocked tests reproduce the protocol bug fixed in v0.2.7.1.
They don't pretend to measure real Qwen / Ministral accuracy.
"""

import json

import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient
from app.models import (
    ParagraphAlignmentDraft,
    ParagraphArbitrationDraft,
    ParagraphExtractedDraft,
    ParagraphReviewDraft,
)
from app.prompts import (
    EXTRACTION_SYSTEM_PARAGRAPH,
    FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH,
    paragraph_extract_user,
)


@pytest.mark.parametrize(
    "stage,schema,result,expected_words",
    [
        (
            "paragraph_extraction",
            ParagraphExtractedDraft,
            {"units": [{"kind": "constraint", "claim": "需要报告结果。", "paragraph_ids": ["zh_p1"]}]},
            ["allowed", "zh_p1", "paragraph_ids"],
        ),
        (
            "paragraph_extraction_fallback",
            ParagraphExtractedDraft,
            {"units": [{"kind": "constraint", "claim": "需要报告结果。", "paragraph_ids": ["zh_p1"]}]},
            ["allowed", "zh_p1", "paragraph_ids"],
        ),
        (
            "paragraph_review",
            ParagraphReviewDraft,
            {"verdict": "PASS", "reason": "same facts", "issues": []},
            ["issues", "en_paragraph_ids", "zh_paragraph_ids"],
        ),
        (
            "paragraph_arbitration",
            ParagraphArbitrationDraft,
            {
                "verdict": "FAIL",
                "issue_kind": "conflict",
                "en_paragraph_ids": ["en_p1"],
                "zh_paragraph_ids": ["zh_p1"],
                "reason": "opposite polarity",
            },
            ["issue_kind", "en_paragraph_ids", "zh_paragraph_ids"],
        ),
        (
            "paragraph_alignment_support",
            ParagraphAlignmentDraft,
            {"status": "uncertain", "en_paragraph_ids": [], "zh_paragraph_ids": [], "reason": "cannot verify"},
            ["status", "en_paragraph_ids", "zh_paragraph_ids"],
        ),
    ],
)
def test_retry_uses_paragraph_stage_format(stage, schema, result, expected_words):
    hints = []
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
        hint = payload["messages"][-1]["content"]
        hints.append(hint)
        assert all(word in hint for word in expected_words)
        assert "ALL THREE ARRAYS" not in hint
        assert "resolved_missing" not in hint or "Never output" in hint
        assert "SOURCE QUOTE validation" not in hint
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}, "finish_reason": "stop"}]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    user = paragraph_extract_user("需要报告结果。", "Chinese", [{"id": "zh_p1", "text": "需要报告结果。"}], 30)
    out = ChatClient(Settings(json_schema=True), http).structured("arbitrary-model", "", user, schema, stage)
    assert out is not None and len(calls) == 2 and hints
    http.close()


def test_primary_extraction_prompt_has_no_english_only_id_example():
    for prompt in (EXTRACTION_SYSTEM_PARAGRAPH, FALLBACK_EXTRACTION_SYSTEM_PARAGRAPH):
        assert '"en_p1"' not in prompt
    user = json.loads(
        paragraph_extract_user(
            "第一段。\n\n第二段。",
            "Chinese",
            [{"id": "zh_p1", "text": "第一段。"}, {"id": "zh_p2", "text": "第二段。"}],
            30,
        )
    )
    assert user["allowed_paragraph_ids"] == ["zh_p1", "zh_p2"]


def test_invalid_chinese_id_retry_reports_valid_ids_in_full_pipeline():
    """The core v0.2.7 production failure pattern: en_p1 inside Chinese extraction.

    A wrong but otherwise correctly structured response must not cause
    downstream silent remapping or an irrelevant old-schema repair hint.
    """
    from app.checker import Checker

    posted = []
    stages = iter(
        [
            {"units": [{"kind": "constraint", "claim": "References are included.", "paragraph_ids": ["en_p1"]}]},
            {"units": [{"kind": "constraint", "claim": "参考文献计入。", "paragraph_ids": ["en_p1"]}]},
            {"units": [{"kind": "constraint", "claim": "参考文献计入。", "paragraph_ids": ["zh_p1"]}]},
            {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "same"}]},
        ]
    )

    def handler(request):
        payload = json.loads(request.content)
        posted.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(next(stages), ensure_ascii=False)}, "finish_reason": "stop"}
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(extraction_fast_path=False, model="test-model", evidence_mode="paragraph", json_schema=True)
    checker = Checker(ChatClient(cfg, http), cfg)
    result = checker.check("References are included.", "参考文献计入。", mode="primary")
    assert result.verdict == "PASS"
    assert result.chinese_keywords[0].paragraph_ids == ["zh_p1"]
    assert len(posted) == 4
    assert "zh_p1" in posted[2]["messages"][-1]["content"]
    assert "引用不存在的 Paragraph ID" in posted[2]["messages"][-1]["content"]
    http.close()
