"""Exercise the actual JSON Schema HTTP adapter with paragraph wire schemas."""
import json
import httpx
from fastapi.testclient import TestClient
from app import api
from app.config import Settings
from app.llm import ChatClient


def test_v027_http_api_paragraph_mode(monkeypatch):
    calls=[]
    def handler(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':[{'id':'main-model'},{'id':'review-model'}]})
        payload=json.loads(request.content)
        calls.append(payload)
        n=len(calls)
        assert payload['response_format']['type']=='json_schema'
        if n==1:
            response={'units':[{'kind':'constraint','claim':"The respondent’s file is confidential.",
                'paragraph_ids':['en_p1']}]}
        elif n==2:
            response={'units':[{'kind':'constraint','claim':'受访者文件保密。',
                'paragraph_ids':['zh_p1']}]}
        elif n==3:
            response={'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],
                                 'relation':'equivalent','reason':'same confidentiality requirement'}]}
        elif n==4:
            response={'verdict':'PASS','reason':'same critical facts','issues':[]}
        else:
            raise AssertionError('Unexpected call')
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(response)},
                                                   'finish_reason':'stop'}]})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    cfg=Settings(extraction_fast_path=False, evidence_mode='paragraph', model='main-model',verifier_model='review-model',
                 json_schema=True, extraction_prompt='coarse')
    monkeypatch.setattr(api,'ChatClient',lambda settings:ChatClient(cfg,http))
    response=TestClient(api.app).post('/api/check',json={
        'english':'The respondent’s file is confidential.',
        'chinese':'受访者文件保密。','mode':'dual'})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['verdict']=='PASS' and result['evidence_mode']=='paragraph'
    assert result['english_keywords'][0]['evidence']=='The respondent’s file is confidential.'
    assert result['english_keywords'][0]['paragraph_ids']==['en_p1']
    assert len(calls)==4
    assert calls[0]['response_format']['json_schema']['schema']['properties']['units']
    assert 'paragraph_ids' in calls[0]['response_format']['json_schema']['schema']['$defs']['ParagraphUnitDraft']['properties']
    assert 'evidence' not in calls[0]['response_format']['json_schema']['schema']['$defs']['ParagraphUnitDraft']['properties']


def test_v0274_http_real_schema_disagreement_recheck_and_reasoning_recovery(monkeypatch):
    """6 actual HTTP JSON-schema calls: evidence, match, review, arb, recheck."""
    en = 'The survey ran for exactly six weeks.'
    zh = '问卷调查持续了42天，既未延长也未缩短。'
    outputs = [
        {'units': [{'kind': 'constraint', 'claim': en, 'paragraph_ids': ['en_p1']}]},
        {'units': [{'kind': 'constraint', 'claim': zh, 'paragraph_ids': ['zh_p1']}]},
        {'links': [{'en_ids': ['en_1'], 'zh_ids': ['zh_1'],
                    'relation': 'equivalent', 'reason': 'six weeks equals 42 days'}]},
        {'verdict': 'FAIL', 'reason': 'question whether exactly maps to 42 days',
         'issues': [{'kind': 'conflict', 'en_paragraph_ids': ['en_p1'],
                     'zh_paragraph_ids': ['zh_p1'], 'reason': 'precision disputed'}]},
        {'verdict': 'PASS', 'issue_kind': 'none', 'en_paragraph_ids': ['en_p1'],
         'zh_paragraph_ids': ['zh_p1'], 'reason': 'exact duration is equal'},
        {'issues': [{'issue_index': 0, 'status': 'refuted',
                     'basis': 'formal_numeric', 'en_paragraph_ids': ['en_p1'],
                     'zh_paragraph_ids': ['zh_p1'],
                     'reason': '6*7=42 days; both refer to a fixed duration'}],
         'reason': 'specific quantity objection resolved'},
    ]
    calls = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'id': 'main-model'}, {'id': 'review-model'}]})
        payload = json.loads(request.content)
        calls.append(payload)
        idx = len(calls)-1
        assert idx < len(outputs)
        assert payload['response_format']['json_schema']['strict'] is True
        reply = json.dumps(outputs[idx], ensure_ascii=False)
        # Reflect LM Studio's observed behavior: JSON in reasoning, empty content
        # when using the primary model in JSON schema mode.
        if payload['model'] == 'main-model':
            msg = {'role': 'assistant', 'content': '', 'reasoning_content': reply}
        else:
            msg = {'role': 'assistant', 'content': reply}
        return httpx.Response(200, json={'choices': [{'message': msg, 'finish_reason': 'stop'}]})
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(extraction_fast_path=False, model='main-model', verifier_model='review-model',
                   evidence_mode='paragraph', json_schema=True)
    monkeypatch.setattr(api, 'ChatClient', lambda _: ChatClient(cfg, transport))
    response = TestClient(api.app).post('/api/check', json={
        'english': en, 'chinese': zh, 'mode': 'dual'})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['verdict'] == 'PASS'
    assert body['dispute_recheck']['issues'][0]['basis'] == 'formal_numeric'
    assert len(calls) == 6
    assert calls[-1]['response_format']['json_schema']['name'] == 'paragraph_dispute_recheck'
    assert calls[-1]['model'] == 'review-model'
    inputs = json.loads(calls[-1]['messages'][1]['content'])
    assert inputs['verified_formal_equivalences'][0]['canonical_value'] == '42'
    transport.close()


def test_v0274_http_manual_review_is_200_not_fake_semantic_fail_or_502(monkeypatch):
    en = 'The original file must be archived.'
    zh = '原始文件可以删除。'
    outputs = [
        {'units': [{'kind': 'constraint', 'claim': en, 'paragraph_ids': ['en_p1']}]},
        {'units': [{'kind': 'constraint', 'claim': zh, 'paragraph_ids': ['zh_p1']}]},
        {'links': [{'en_ids': ['en_1'], 'zh_ids': ['zh_1'],
                    'relation': 'equivalent', 'reason': 'mistaken equivalence'}]},
        {'verdict': 'FAIL', 'reason': 'opposing rules',
         'issues': [{'kind': 'conflict', 'en_paragraph_ids': ['en_p1'],
                     'zh_paragraph_ids': ['zh_p1'], 'reason': 'delete vs archive'}]},
        {'verdict': 'PASS', 'issue_kind': 'none', 'en_paragraph_ids': [],
         'zh_paragraph_ids': [], 'reason': 'mistaken arbitration'},
        {'issues': [{'issue_index': 0, 'status': 'sustained',
                     'basis': 'other', 'en_paragraph_ids': ['en_p1'],
                     'zh_paragraph_ids': ['zh_p1'],
                     'reason': 'cannot delete a file that must remain archived'}],
         'reason': 'semantic contradiction remains'},
    ]
    calls = []
    def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        idx = len(calls)-1
        return httpx.Response(200, json={
            'choices': [{'message': {'content': json.dumps(outputs[idx], ensure_ascii=False)},
                         'finish_reason': 'stop'}]})
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings(extraction_fast_path=False, model='main-model', verifier_model='review-model',
                   evidence_mode='paragraph', json_schema=True)
    monkeypatch.setattr(api, 'ChatClient', lambda _: ChatClient(cfg, transport))
    response = TestClient(api.app).post('/api/check', json={
        'english': en, 'chinese': zh, 'mode': 'dual'})
    assert response.status_code == 200, response.text
    assert response.json()['verdict'] == 'REVIEW_REQUIRED'
    assert len(calls) == 6
    transport.close()
