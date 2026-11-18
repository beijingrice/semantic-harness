"""LM Studio #1773: JSON Schema completion arrives only as reasoning_content.

Never treat unconstrained chain of thought as a valid schema response.
"""
import json
import sys

import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient, LLMError
from app.models import ParagraphExtractedDraft
from examples import evaluate


def _reply(content, reasoning):
    return httpx.Response(200, json={"choices": [{
        "message": {"role": "assistant", "content": content,
                    "reasoning_content": reasoning, "tool_calls": []},
        "finish_reason": "stop",
    }], "usage": {"completion_tokens": 33, "completion_tokens_details": {"reasoning_tokens": 32}}})


def test_structured_json_is_recovered_from_reasoning_only_in_schema_mode():
    calls = []
    result = {'units': [{'kind': 'constraint', 'claim': 'The survey ran six weeks.',
                         'paragraph_ids': ['en_p1']}]}
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body['response_format']['type'] == 'json_schema'
        return _reply('', json.dumps(result))
    http = httpx.Client(transport=httpx.MockTransport(handler))
    def check_source_refs(draft):
        assert set(draft.units[0].paragraph_ids) == {'en_p1'}
    answer = ChatClient(Settings(json_schema=True, evidence_mode='paragraph'), http).structured(
        'arbitrary-reasoning-model', 'Return JSON', '{}', ParagraphExtractedDraft,
        'paragraph_extraction', validate_result=check_source_refs,
    )
    assert answer.units[0].claim == 'The survey ran six weeks.'
    assert len(calls) == 1
    http.close()


def test_normal_content_wins_over_reasoning():
    calls = []
    def handler(request):
        calls.append(request)
        return _reply('{"units": []}', '{"units":[{"kind":"concept","claim":"wrong","paragraph_ids":["en_p1"]}]}')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    answer = ChatClient(Settings(json_schema=True, generation_recovery=False), http).structured(
        'any-model', '', '', ParagraphExtractedDraft, 'paragraph_extraction'
    )
    assert not answer.units and len(calls) == 1
    http.close()


@pytest.mark.parametrize('reasoning', [
    'Thinking Process: I should return JSON',
    '<think>{"units":[]}</think>',
    '{"units":[]}{"units":[]}',
    'I think {"units": []} is right',
    '',
])
def test_freeform_thinking_is_never_promoted_to_model_answer(reasoning):
    calls = []
    def handler(request):
        calls.append(request)
        return _reply('', reasoning)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match='LM_GENERATION_RECOVERY=false'):
        ChatClient(Settings(json_schema=True, generation_recovery=False), http).structured(
            'model-x', '', '', ParagraphExtractedDraft, 'paragraph_extraction'
        )
    assert len(calls) == 1, 'empty response must fail fast, not burn retries'
    http.close()


def test_recovered_json_must_still_pass_pydantic_schema_and_source_grounding():
    seen = []
    malformed = json.dumps({'units':[{'kind':'constraint', 'claim':'fake', 'paragraph_ids':['zh_p9']}]})
    correct = json.dumps({'units':[{'kind':'constraint', 'claim':'source', 'paragraph_ids':['zh_p1']}]})
    def handler(request):
        seen.append(json.loads(request.content))
        return _reply('', malformed if len(seen) == 1 else correct)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    def grounding(draft):
        if draft.units[0].paragraph_ids != ['zh_p1']:
            raise LLMError('引用不存在的 Paragraph ID')
    answer = ChatClient(Settings(json_schema=True, generation_recovery=False), http).structured(
        'model-x', '', '{}', ParagraphExtractedDraft, 'paragraph_extraction',
        validate_result=grounding)
    assert answer.units[0].paragraph_ids == ['zh_p1']
    assert len(seen) == 2
    http.close()


def test_recovery_not_enabled_when_json_schema_is_off():
    calls = []
    def handler(request):
        calls.append(request)
        return _reply('', '{"units": []}')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match='空 content'):
        ChatClient(Settings(json_schema=False, generation_recovery=False), http).structured('model-x', '', '',
                                       ParagraphExtractedDraft, 'paragraph_extraction')
    assert len(calls) == 1
    http.close()


def test_evaluator_stops_on_first_execution_error_and_saves_partial_report(monkeypatch, tmp_path, capsys):
    source = tmp_path / 'cases.json'
    out = tmp_path / 'partial.json'
    source.write_text(json.dumps([
        {'id': 'one', 'en': 'A', 'zh':'甲', 'expected':'PASS'},
        {'id': 'two', 'en': 'B', 'zh':'乙', 'expected':'PASS'},
    ]), encoding='utf8')
    class MockClient:
        def __init__(self, settings):
            self.settings = settings
        def choose_model(self, name):
            return 'm'
        def choose_verifier(self, name):
            return 'v'
        def models(self):
            return ['m','v']
        def close(self):
            pass
    calls=[]
    class MockChecker:
        def __init__(self, *_):
            pass
        def check(self, *args):
            calls.append(args)
            raise LLMError('model returned empty content')
    monkeypatch.setattr(evaluate, 'ChatClient', MockClient)
    monkeypatch.setattr(evaluate, 'Checker', MockChecker)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--cases', str(source), '--out', str(out), '--stop-after-errors', '1'])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1
    assert len(calls) == 1
    data = json.loads(out.read_text(encoding='utf8'))
    assert data['summary']['total'] == 1
    assert data['summary']['configured_total'] == 2
    assert data['summary']['stopped_early'] is True
    assert 'model returned empty content' in capsys.readouterr().out


def test_full_paragraph_dual_pipeline_recovers_qwen_json_from_reasoning(monkeypatch):
    """Exact user path: en extraction, zh extraction, matcher Qwen JSON in reasoning;
    independent reviewer returns normal content. All grounding still applies.
    """
    from fastapi.testclient import TestClient

    from app import api
    calls = []
    outputs = [
        {'units': [{'kind': 'constraint', 'claim': 'A report must include a date.',
                    'paragraph_ids': ['en_p1']}]},
        {'units': [{'kind': 'constraint', 'claim': '报告必须包含日期。',
                    'paragraph_ids': ['zh_p1']}]},
        {'links': [{'en_ids': ['en_1'], 'zh_ids': ['zh_1'],
                    'relation': 'equivalent', 'reason': 'date requirement is same'}]},
        {'verdict': 'PASS', 'reason': 'equivalent', 'issues': []},
    ]
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'id':'qwen'},{'id':'reviewer'}]})
        obj = json.loads(request.content)
        calls.append(obj)
        assert obj['response_format']['type'] == 'json_schema'
        reply = json.dumps(outputs[len(calls) - 1], ensure_ascii=False)
        return _reply('', reply) if obj['model'] == 'qwen' else _reply(reply, '')
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(extraction_fast_path=False, model='qwen', verifier_model='reviewer',
                   json_schema=True, evidence_mode='paragraph')
    monkeypatch.setattr(api, 'ChatClient', lambda settings: ChatClient(cfg, transport))
    result = TestClient(api.app).post('/api/check', json={
        'english': 'A report must include a date.',
        'chinese': '报告必须包含日期。', 'mode': 'dual',
    })
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['verdict'] == 'PASS'
    assert body['evidence_mode'] == 'paragraph'
    assert body['english_keywords'][0]['paragraph_ids'] == ['en_p1']
    assert body['chinese_keywords'][0]['evidence'] == '报告必须包含日期。'
    assert len(calls) == 4
    transport.close()
