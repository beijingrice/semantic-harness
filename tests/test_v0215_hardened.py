"""v0.2.15: historical real LM traces are replayed, NOT real new LM runs."""
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient
from app.pipeline import Checker
from app.relation_direct import (
    _focus_exception_evidence,
    _host_anchor_direction,
    _numeric_symbol_unicode,
)
from app.source_risks import source_spans

HISTORY = json.loads(Path('tests/fixtures/replay_v0214_relations_11.json').read_text())
CASES = {r['id']: r for r in HISTORY['cases']}


def old_reply(case_id):
    c = CASES[case_id]
    q = c['detail']['verification_questions'][0]
    if q.get('outcome') == 'INVALID':
        # Original v0.2.14 model returned an invalid citation ID, but the
        # content was not logged. This is a *synthetic* invalid ID fault.
        return dict(task_id=q['task_id'], question_type='direction_reversal',
                    same_event='yes', en_evidence_id='en_HALLUCINATED',
                    zh_evidence_id='zh_s2', en_subject='proposed system',
                    zh_subject='基线方法', en_condition='second dataset',
                    zh_condition='第二组数据', en_value='proposed system',
                    zh_value='基线方法', finding='CONFLICT', reason='Invalid ID fault')
    return {k:q[k] for k in ('task_id','question_type','same_event',
        'en_evidence_id','zh_evidence_id','en_subject','zh_subject',
        'en_condition','zh_condition','en_value','zh_value','finding','reason')}


def execute(case_id, third=None, *, english=None, chinese=None):
    c = CASES[case_id]
    primary = c['detail']['direct_primary']
    raw_review = c['detail']['review']
    review = dict(verdict=raw_review['verdict'],reason=raw_review['reason'],issues=[
        {key:item[key] for key in ('kind','en_paragraph_ids','zh_paragraph_ids','reason')}
        for item in raw_review['issues']])
    third = third or old_reply(case_id)
    outputs=[primary,review,third]
    requests=[]
    def mock(request):
        wire=json.loads(request.content)
        requests.append(wire)
        assert len(requests)<=3
        response=outputs[len(requests)-1]
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(response,ensure_ascii=False)},'finish_reason':'stop'}]})
    settings=replace(Settings(),pipeline='direct_hardened', evidence_mode='paragraph',
                    model='primary',verifier_model='independent',json_schema=True)
    client=ChatClient(settings,httpx.Client(transport=httpx.MockTransport(mock)))
    client.start_trace()
    try:
        result=Checker(client,settings).check(english or c['en'],chinese or c['zh'],mode='dual')
        return result,requests,client.trace_snapshot()
    finally:
        client.close()


@pytest.mark.parametrize('case_id,expected', [
    ('V2-NUM-15','FAIL'), ('V2-NUM-13','FAIL'), ('V2-EXC-03','FAIL'),
    ('V2-NUM-03','PASS'),('V2-ROL-07','FAIL')])
def test_real_prior_output_replayed_through_hardened_host(case_id,expected):
    result,requests,trace=execute(case_id)
    assert result.verdict==expected,(case_id,result.verification_questions)
    assert len(requests)==len(trace)==3
    last=requests[-1]
    schema=last['response_format']['json_schema']['schema']
    evidence_source=CASES[case_id]
    en={x.id:x.text for x in source_spans(evidence_source['en'],'en')}
    zh={x.id:x.text for x in source_spans(evidence_source['zh'],'zh')}
    assert schema['properties']['en_evidence_id']['enum']==list(en)
    assert schema['properties']['zh_evidence_id']['enum']==list(zh)
    assert result.verification_questions[0]['host_conclusion']==expected
    assert trace[-1]['stage']=='direct_hardened_verify'


def test_unicode_operators_are_normalized_without_changing_strictness():
    from decimal import Decimal
    assert _numeric_symbol_unicode('≥１０％')==('>=',Decimal('10'),'%')
    assert _numeric_symbol_unicode('≤ 3.5mm')==('<=',Decimal('3.5'),'mm')
    assert _numeric_symbol_unicode('≧ 10%')==('>=',Decimal('10'),'%')
    for invalid in ('>10% or <12%', 'roughly ≥10%', '>=10-20%', '>= 10%%'):
        assert _numeric_symbol_unicode(invalid) is None


def test_wrong_model_comparator_cannot_override_source():
    answer=old_reply('V2-NUM-13')
    answer['zh_value']='>10%'
    r,_,_=execute('V2-NUM-13',answer)
    assert r.verdict=='REVIEW_REQUIRED'


def test_exception_cites_are_host_focused_but_original_choices_preserved():
    r,_,_=execute('V2-EXC-03')
    q=r.verification_questions[0]
    assert q['en_evidence_id']=='en_all' and q['zh_evidence_id']=='zh_all'
    assert q['host_evidence_id_en']=='en_s2' and q['host_evidence_id_zh']=='zh_s2'
    assert r.grounded_evidence_audit.en_quote.strip().startswith('An additional Chinese abstract')
    assert '无论' in r.grounded_evidence_audit.zh_quote


def test_exception_no_unique_source_target_abstains():
    source=CASES['V2-EXC-03']
    en={s.id:s.text for s in source_spans(source['en'],'en')}
    zh={s.id:s.text for s in source_spans(source['zh'],'zh')}
    assert _focus_exception_evidence(en,zh,'en_all','zh_all','Chinese abstract','中文摘要')
    # Two independent exceptions about the same Chinese abstract: do not guess.
    en_twice=source['en']+' An additional Chinese abstract is optional unless another degree program requires it.'
    en={s.id:s.text for s in source_spans(en_twice,'en')}
    assert _focus_exception_evidence(en,zh,'en_all','zh_all','Chinese abstract','中文摘要') is None
    assert _focus_exception_evidence(en,zh,'en_all','zh_all','English abstract','英文摘要') is None


def test_invalid_citation_has_single_call_host_recovery_only_for_unique_direction():
    r,req,trace=execute('V2-ROL-07')
    assert r.verdict=='FAIL' and len(req)==3
    assert trace[-1]['outcome']=='schema_or_grounding_invalid'
    assert r.verification_questions[0]['outcome']=='HOST_SOURCE_RECOVERY'
    assert 'en_s2' == r.verification_questions[0]['en_evidence_id']
    assert 'zh_s2' == r.verification_questions[0]['zh_evidence_id']
    assert '连续三次' not in r.verification_questions[0]['model_validation_error']
    assert '1 次尝试' in r.verification_questions[0]['model_validation_error']


def test_ambiguous_direction_scopes_never_auto_recover():
    c=CASES['V2-ROL-07']
    en={s.id:s.text for s in source_spans(c['en'],'en')}
    zh={s.id:s.text for s in source_spans(c['zh'],'zh')}
    assert _host_anchor_direction(en,zh) is not None
    # Several contrary datasets => not a unique conflict.
    et=c['en']+' On the third dataset, the proposed system achieved the higher score.'
    zt=c['zh']+' 在第三组数据上，基线方法得分更高。'
    en={s.id:s.text for s in source_spans(et,'en')}
    zh={s.id:s.text for s in source_spans(zt,'zh')}
    assert _host_anchor_direction(en,zh) is None
    # Missing label / duplicate label also abstains.
    en={s.id:s.text for s in source_spans(c['en']+' On the second dataset, baseline scored higher.','en')}
    zh={s.id:s.text for s in source_spans(c['zh'],'zh')}
    assert _host_anchor_direction(en,zh) is None
    en={s.id:s.text for s in source_spans(c['en'],'en')}
    zh={s.id:s.text for s in source_spans(c['zh'].replace('第二组数据','第三组数据'),'zh')}
    assert _host_anchor_direction(en,zh) is None


def test_ambiguous_or_unknown_evidence_cannot_force_other_conflict():
    # Even when the model insists on CONFLICT, invalid citation ID in a
    # NON-direction question has no host fallback and never forces FAIL.
    answer=old_reply('V2-NUM-15')
    answer['en_evidence_id']='en_unknown'
    r,req,trace=execute('V2-NUM-15',answer)
    assert r.verdict=='REVIEW_REQUIRED' and len(req)==3
    assert r.verification_questions[0]['outcome']=='INVALID'

