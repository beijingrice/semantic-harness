import json
import httpx
import pytest
from app.config import Settings
from app.llm import ChatClient, LLMError, parse_json_object
from app.models import ExtractedDraft, MatchDraft, ReviewDraft


def response(content, reason="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": reason}]})


def test_parse_fenced_json_and_thinking():
    assert parse_json_object('<think>x</think>\n```json\n{"units":[]}\n```') == {"units": []}


def test_invalid_json_fails():
    with pytest.raises(LLMError, match="有效 JSON"):
        parse_json_object("not JSON")


def test_two_models_sent_on_same_compatible_endpoint():
    seen = []
    def handler(req):
        data = json.loads(req.content)
        seen.append(data)
        if data["model"] == "qwen":
            return response('{"units":[]}')
        return response('{"verdict":"PASS","reason":"same meaning","issues":[]}')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    llm = ChatClient(Settings(evidence_mode="quote", ), http)
    llm.structured("qwen", "", "", ExtractedDraft, "keyword_extraction")
    llm.structured("ministral", "", "", ReviewDraft, "independent_review")
    assert [x["model"] for x in seen] == ["qwen", "ministral"]
    assert all("response_format" not in x for x in seen)
    http.close()


def test_json_schema_support_on_both_models():
    def handler(req):
        data = json.loads(req.content)
        assert data["response_format"]["type"] == "json_schema"
        return response('{"verdict":"PASS","reason":"same meaning","issues":[]}')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    ChatClient(Settings(evidence_mode="quote", json_schema=True), http).structured("ministral", "", "", ReviewDraft, "independent_review")
    http.close()


def test_evidence_validation_failure_retries_with_feedback():
    outputs = iter(['{"units":[{"kind":"concept","claim":"x","evidence":"image samples"}]}',
                    '{"units":[{"kind":"concept","claim":"image","evidence":"image"}]}'])
    http = httpx.Client(transport=httpx.MockTransport(lambda req: response(next(outputs))))
    llm = ChatClient(Settings(evidence_mode="quote", ), http)
    def validate(ans):
        for u in ans.units:
            if u.evidence not in "image, audio, text samples":
                raise LLMError("noncontiguous evidence")
    result = llm.structured("qwen", "", "", ExtractedDraft, "keyword_extraction", validate_result=validate)
    assert result.units[0].evidence == "image"
    http.close()


def test_persistent_flat_match_response_errors_after_retry():
    http = httpx.Client(transport=httpx.MockTransport(lambda req: response('{"en_ids":["en_1"]}')))
    with pytest.raises(LLMError, match="连续三次"):
        ChatClient(Settings(evidence_mode="quote", ), http).structured("qwen", "", "", MatchDraft, "keyword_matching")
    http.close()


def test_truncated_completion_error():
    http = httpx.Client(transport=httpx.MockTransport(lambda req: response('{"units":[]}', "length")))
    with pytest.raises(LLMError, match="截断"):
        ChatClient(Settings(evidence_mode="quote", generation_recovery=False), http).structured("qwen", "", "", ExtractedDraft, "keyword_extraction")
    http.close()
