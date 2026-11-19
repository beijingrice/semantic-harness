"""Mock LM Studio protocol; assert dispatch to the right model for each stage."""

import json

import httpx
from fastapi.testclient import TestClient

from app import api, jobs
from app.config import Settings
from app.llm import ChatClient


def test_two_model_disagreement_calls_adjudicator(monkeypatch):
    calls = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "qwen"}, {"id": "ministral"}]})
        data = json.loads(request.content)
        calls.append(data)
        i = len(calls)
        if i == 1:
            ans = {"units": [{"kind": "concept", "claim": "purpose and methods", "evidence": "purpose, methods"}]}
        elif i == 2:
            ans = {"units": [{"kind": "concept", "claim": "目的和方法", "evidence": "目的、方法"}]}
        elif i == 3:
            ans = {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "same"}]}
        elif i == 4:
            ans = {
                "verdict": "FAIL",
                "reason": "missing results",
                "issues": [
                    {"kind": "missing_in_chinese", "en_evidence": "results", "zh_evidence": "", "reason": "missing"}
                ],
            }
        else:
            ans = {
                "verdict": "FAIL",
                "issue_kind": "missing_in_chinese",
                "en_evidence": "results",
                "zh_evidence": "",
                "reason": "confirmed",
            }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(ans)}, "finish_reason": "stop"}]}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        jobs,
        "ChatClient",
        lambda cfg: ChatClient(
            Settings(
                evidence_mode="quote",
                model="qwen",
                verifier_model="ministral",
                extraction_prompt="legacy",
                coverage_audit=False,
            ),
            http,
        ),
    )
    result = TestClient(api.app).post(
        "/api/check", json={"english": "purpose, methods, results", "chinese": "目的、方法", "mode": "dual"}
    )
    assert result.status_code == 200
    assert result.json()["verdict"] == "FAIL"
    assert result.json()["arbitrated"] is True
    assert [x["model"] for x in calls] == ["qwen", "qwen", "qwen", "ministral", "qwen"]
    http.close()


def test_primary_api_uses_only_three_requests(monkeypatch):
    count = []

    def handler(req):
        count.append(json.loads(req.content))
        i = len(count)
        if i <= 2:
            word = "purpose" if i == 1 else "目的"
            ans = {"units": [{"kind": "concept", "claim": word, "evidence": word}]}
        else:
            ans = {"links": [{"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "equivalent", "reason": "same"}]}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(ans)}, "finish_reason": "stop"}]}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        jobs,
        "ChatClient",
        lambda cfg: ChatClient(
            Settings(
                evidence_mode="quote",
                model="qwen",
                verifier_model="ministral",
                extraction_prompt="legacy",
                coverage_audit=False,
            ),
            http,
        ),
    )
    result = TestClient(api.app).post("/api/check", json={"english": "purpose", "chinese": "目的", "mode": "primary"})
    assert result.status_code == 200 and result.json()["verdict"] == "PASS"
    assert [x["model"] for x in count] == ["qwen"] * 3
    http.close()
