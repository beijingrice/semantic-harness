"""v0.2.8: no real LLM calls; inspect actual HTTP payloads and evidence validators."""
import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient, LLMError
from app.models import ParagraphExtractedDraft
from app.telemetry import evidence_integrity, percentile, reliability_summary
from examples.evaluate import load_cases


def reply(content, *, reason='stop', reasoning=None, tokens=50):
    return httpx.Response(200, json={
        'choices': [{'finish_reason': reason, 'message': {
            'content': content, 'reasoning_content': reasoning}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': tokens,
                  'completion_tokens_details': {'reasoning_tokens': tokens-5}},
    })


def test_length_recovery_uses_plain_json_once_and_keeps_same_token_budget():
    seen = []
    def handler(req):
        p = json.loads(req.content)
        seen.append(p)
        return reply('', reason='length', tokens=4096) if len(seen) == 1 else reply('{"units":[]}')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient(Settings(json_schema=True, max_tokens=4096, generation_recovery=True), http)
    client.start_trace()
    result = client.structured('qwen', 'system', 'user', ParagraphExtractedDraft, 'paragraph_extraction')
    assert result.units == [] and len(seen) == 2
    assert 'response_format' in seen[0] and 'response_format' not in seen[1]
    assert seen[1]['max_tokens'] == 4096
    assert seen[1]['messages'][-1]['role'] == 'user'
    assert [x['outcome'] for x in client.trace_snapshot()] == ['truncated', 'recovered_validated']
    assert all('reasoning_content' not in x for x in client.trace_snapshot())
    http.close()


def test_empty_stop_recovery_success_and_metadata():
    seen=[]
    def handler(req):
        seen.append(req)
        return reply('', reasoning='Thinking should be private') if len(seen)==1 else reply('{"units":[]}')
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=ChatClient(Settings(json_schema=True, generation_recovery=True), http)
    client.structured('m','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(seen)==2
    assert client.trace_snapshot()[0]['outcome']=='empty_content'
    http.close()


def test_recovered_json_must_pass_grounding_and_fails_no_extra_retries():
    seen=[]
    def handler(req):
        seen.append(req)
        return reply('',reason='length') if len(seen)==1 else reply(
            '{"units":[{"kind":"constraint","claim":"fake","paragraph_ids":["en_p999"]}]}')
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=ChatClient(Settings(json_schema=True, generation_recovery=True),http)
    def validator(d):
        if d.units and d.units[0].paragraph_ids != ['en_p1']:
            raise LLMError('unknown paragraph ID')
    with pytest.raises(LLMError,match='未通过原 Schema'):
        client.structured('m','','',ParagraphExtractedDraft,'paragraph_extraction',validator)
    assert len(seen)==2
    assert client.trace_snapshot()[-1]['outcome']=='recovery_validation_failed'
    http.close()


def test_no_runaway_on_second_generation_failure():
    seen=[]
    def handler(req):
        seen.append(req)
        return reply('',reason='length',tokens=4096)
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=ChatClient(Settings(json_schema=True, generation_recovery=True),http)
    with pytest.raises(LLMError,match='生成异常恢复失败'):
        client.structured('m','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(seen)==2
    assert all(q['outcome']=='truncated' for q in client.trace_snapshot())
    http.close()


def test_empty_reasoning_json_compatibility_path_not_unnecessarily_retried():
    calls=[]
    def handler(req):
        calls.append(req)
        return reply('',reasoning='{"units":[]}')
    http=httpx.Client(transport=httpx.MockTransport(handler))
    c=ChatClient(Settings(json_schema=True,generation_recovery=True),http)
    c.structured('m','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(calls)==1
    assert c.trace_snapshot()[0]['response_source']=='reasoning_json'
    http.close()


def test_http_error_does_not_trigger_alt_generation():
    seen=[]
    def handler(req):
        seen.append(req)
        return httpx.Response(503,text='server busy')
    http=httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError,match='503'):
        ChatClient(Settings(generation_recovery=True),http).structured(
            'm','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(seen)==1
    http.close()


def test_source_evidence_ids_and_links_are_structural_not_semantic():
    detail={'evidence_mode':'paragraph',
        'english_paragraphs':[{'id':'en_p1','text':'A.'}],
        'chinese_paragraphs':[{'id':'zh_p1','text':'甲。'}],
        'english_keywords':[{'id':'en_1','paragraph_ids':['en_p1'],'evidence':'A.'}],
        'chinese_keywords':[{'id':'zh_1','paragraph_ids':['zh_p1'],'evidence':'甲。'}],
        'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],'relation':'equivalent'}]}
    valid=evidence_integrity(detail)
    assert valid['checked'] and valid['structural_ok']
    assert valid['semantic_correctness_verified'] is False
    detail['links'][0]['en_ids']=['en_999']
    assert 'link_0:invalid_english_unit' in evidence_integrity(detail)['violations']


def test_telemetry_rates_latency_and_outcomes():
    results=[{'predicted':'PASS','correct':True,'elapsed_seconds':1,
              'requests':[{'stage':'paragraph_extraction','duration_ms':20,'outcome':'truncated',
                          'completion_tokens':10,'reasoning_tokens':9},
                          {'stage':'paragraph_extraction','duration_ms':30,'outcome':'recovered_validated',
                          'recovery':True,'completion_tokens':11,'reasoning_tokens':0}]},
             {'predicted':'REVIEW_REQUIRED','correct':False,'elapsed_seconds':3,
              'requests':[]},
             {'predicted':'ERROR','correct':False,'elapsed_seconds':10,'requests':[]}]
    s=reliability_summary(results)
    assert s['auto_decision_coverage']==0.3333
    assert s['end_to_end_correct_rate']==0.3333
    assert s['execution_error_rate']==0.3333
    assert s['review_required_rate']==0.3333
    assert s['latency_seconds']['p50']==3 and s['latency_seconds']['p95']==10
    assert s['recoveries_attempted']==1 and s['recoveries_validated']==1
    assert s['token_usage_reported']['reasoning_tokens']==9
    assert percentile([],0.95) is None


def test_holdout_36_balanced_and_frozen():
    from collections import Counter
    import hashlib
    p=Path(__file__).resolve().parents[1]/'examples'/'validation_v028_holdout_36.json'
    cases=load_cases(p)
    assert len(cases)==36 and len({x['id'] for x in cases})==36
    assert Counter(x['expected'] for x in cases)=={'PASS':18,'FAIL':18}
    assert len(set(x['category'] for x in cases))==6
    assert hashlib.sha256(p.read_bytes()).hexdigest() in (p.with_suffix('.sha256').read_text())


def test_evaluate_reports_trace_even_on_failure_and_exports_audit(monkeypatch,tmp_path):
    import sys
    from examples import evaluate
    input_file=tmp_path/'sample.json'
    input_file.write_text(json.dumps([{'id':'case','en':'A','zh':'甲',
                                        'expected':'PASS'}]),encoding='utf-8')
    out=tmp_path/'report.json'
    audit=tmp_path/'manual.csv'
    class FakeClient:
        def __init__(self,s): self.settings=s; self.records=[]
        def choose_model(self,*_): return 'm'
        def choose_verifier(self,*_): return 'v'
        def models(self): return ['m','v']
        def close(self): pass
        def start_trace(self): self.records=[]
        def trace_snapshot(self): return self.records
    class FakeChecker:
        def __init__(self,client,settings): self.client=client
        def check(self,*args):
            self.client.records.append({'stage':'paragraph_extraction','model':'m',
                'duration_ms':12,'outcome':'truncated',
                'completion_tokens':4096,'reasoning_tokens':4096})
            raise LLMError('truncated')
    monkeypatch.setattr(evaluate,'ChatClient',FakeClient)
    monkeypatch.setattr(evaluate,'Checker',FakeChecker)
    monkeypatch.setattr(sys,'argv',['evaluate','--cases',str(input_file),
        '--out',str(out),'--stop-after-errors','1','--audit-csv',str(audit)])
    # A one-case run has no remaining cases, so there is no early stop exit.
    evaluate.main()
    report=json.loads(out.read_text(encoding='utf-8'))
    assert report['summary']['reliability']['requests']==1
    assert report['summary']['reliability']['request_outcomes']=={'truncated':1}
    assert report['cases'][0]['requests'][0]['reasoning_tokens']==4096
    assert 'manual_reason_correct' in audit.read_text(encoding='utf-8-sig')


def test_full_paragraph_dual_pipeline_recovers_one_chinese_truncation():
    from app.pipeline import Checker
    en='The protocol requires a signed receipt.'
    zh='该方案要求签署收据。'
    seen=[]
    outputs=[
        {'units':[{'kind':'constraint','claim':en,'paragraph_ids':['en_p1']}]},
        None,
        {'units':[{'kind':'constraint','claim':zh,'paragraph_ids':['zh_p1']}]},
        {'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],
                   'relation':'equivalent','reason':'same signed receipt requirement'}]},
        {'verdict':'PASS','reason':'both require signed receipt','issues':[]},
    ]
    def handler(req):
        body=json.loads(req.content)
        seen.append(body)
        idx=len(seen)-1
        if idx==1:
            return reply('',reason='length',tokens=4096)
        assert idx<len(outputs)
        return reply(json.dumps(outputs[idx],ensure_ascii=False))
    http=httpx.Client(transport=httpx.MockTransport(handler))
    settings=Settings(extraction_fast_path=False, evidence_mode='paragraph',extraction_prompt='coarse',
                      json_schema=True,generation_recovery=True,model='main',verifier_model='verify')
    client=ChatClient(settings,http)
    client.start_trace()
    response=Checker(client,settings).check(en,zh,'main','verify','dual')
    assert response.verdict=='PASS'
    assert len(seen)==5
    assert 'response_format' not in seen[2]
    assert [r['stage'] for r in client.trace_snapshot()]==[
        'extract_english','extract_chinese','extract_chinese',
        'keyword_matching','independent_review']
    assert client.trace_snapshot()[2]['outcome']=='recovered_validated'
    http.close()
