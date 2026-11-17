"""v0.2.13 structured-question workflow, grounded in prior REAL direct output.

Only synthetic THIRD-step answers. Prior primary and independent model outputs
are replayed verbatim, so none of these tests claim live v0.2.13 accuracy.
"""
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from app.config import Settings
from app.llm import ChatClient
from app.pipeline import Checker
from app.question_direct import choose_questions, _prove, _same_condition

CASES={c['id']:c for c in json.loads(Path('examples/reference_v0212_verified_11.json').read_text())['cases']}


def response(case):
    a=case['detail']['direct_primary']
    r=case['detail']['review']
    b={'verdict':r['verdict'],'reason':r['reason'],
       'issues':[{k:i[k] for k in ('kind','en_paragraph_ids','zh_paragraph_ids','reason')}
                 for i in r['issues']]}
    return a,b


def answer(task, typ, en_id, zh_id, en_val='', zh_val='', same='yes', finding='CONFLICT'):
    return {'task_id':task,'question_type':typ,'same_requirement':same,
            'en_evidence_id':en_id,'zh_evidence_id':zh_id,
            'en_value':en_val,'zh_value':zh_val, 'finding':finding,
            'reason':'Checked source operands and their scope without changing the question.'}


KNOWN={
 'V2-NUM-15':answer('Q_COUNTED_ENTITY','inclusion_exclusion','en_s3','zh_s3','cover page','封面'),
 'V2-NUM-13':answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s2','zh_s2','>10%','>=10%'),
 'V2-EXC-03':answer('Q_EXCEPTION_SCOPE','exception_scope','en_s2','zh_s2','Chinese abstract','中文摘要'),
 'V2-NUM-03':answer('Q_SCALAR_EQUIVALENCE','scalar_equivalence','en_s1','zh_s1',finding='EQUIVALENT'),
 'V2-ROL-07':answer('Q_COMPARISON_WINNER','direction_reversal','en_s2','zh_s2','proposed system','基线方法'),
}


def run(key, third=None, mode='dual'):
    c=CASES[key]
    replies=list(response(c)) if mode=='dual' else [response(c)[0]]
    if third is not None:
        replies.append(third)
    requests=[]
    def handler(req):
        payload=json.loads(req.content)
        requests.append(payload)
        assert len(requests)<=len(replies), 'Unexpected request/retry'
        val=replies[len(requests)-1]
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(val,ensure_ascii=False)},
                                                        'finish_reason':'stop'}]})
    settings=replace(Settings(),model='qwen-local',verifier_model='ministral-local',
                     pipeline='direct_questions',evidence_mode='paragraph',json_schema=True)
    client=ChatClient(settings,httpx.Client(transport=httpx.MockTransport(handler)))
    client.start_trace()
    try:
        result=Checker(client,settings).check(c['en'],c['zh'],mode=mode)
        return result,requests,client.trace_snapshot()
    finally:
        client.close()


@pytest.mark.parametrize('key,expected',[
 ('V2-NUM-15','FAIL'),('V2-NUM-13','FAIL'),('V2-EXC-03','FAIL'),
 ('V2-NUM-03','PASS'),('V2-ROL-07','FAIL')])
def test_synthetic_typed_answers_repair_five_historical_cases(key,expected):
    r,req,tr=run(key,KNOWN[key])
    assert r.verdict==expected, (key,r.summary,r.verification_questions)
    assert len(req)==3 and len(tr)==3
    assert r.verification_questions[0]['host_conclusion']==expected
    assert req[-1]['messages'][1]['content']
    task=json.loads(req[-1]['messages'][1]['content'])
    assert task['task_id']==KNOWN[key]['task_id']
    assert task['question_type']==KNOWN[key]['question_type']
    assert task['complete_english']==CASES[key]['en']
    assert task['complete_chinese']==CASES[key]['zh']
    assert 'FIXED_question' in task
    assert 'task_id' in req[-1]['messages'][0]['content']


@pytest.mark.parametrize('key', ['V2-NUM-12','V2-ENU-12','V2-ENU-13',
                                      'V2-MIX-09','V2-MIX-04','V2-MIX-03'])
def test_no_risk_two_calls_preserves_historical_judgments(key):
    r,req,tr=run(key)
    assert len(req)==2 and not r.verification_questions
    assert r.verdict==CASES[key]['predicted']


def test_numeric_question_cannot_be_replaced_by_denominator_omission():
    wrong=answer('Q_OTHER','undetermined','en_s3','zh_s3',finding='EQUIVALENT')
    r,req,tr=run('V2-NUM-13',wrong)
    assert r.verdict=='REVIEW_REQUIRED' and len(req)==3
    assert tr[-1]['outcome']=='schema_or_grounding_invalid'
    assert r.verification_questions[0]['outcome']=='INVALID'


def test_numeric_model_cannot_invent_comparator_or_pivot_to_other_clause():
    for wrong in (
      answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s3','zh_s3','>10%','>=10%'),
      answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s2','zh_s2','>=10%','>=10%'),
      answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s2','zh_s2','>10%','>10%',finding='EQUIVALENT'),
      answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s2','zh_s2','>10%','>=10%',same='unknown')
    ):
        r,_,_=run('V2-NUM-13',wrong)
        assert r.verdict=='REVIEW_REQUIRED',wrong


def test_inclusion_different_entity_cannot_force_fail():
    r,_,_=run('V2-NUM-15',answer('Q_COUNTED_ENTITY','inclusion_exclusion','en_s3','zh_s3',
                                'cover page','参考文献列表'))
    assert r.verdict=='REVIEW_REQUIRED'


def test_audit_all_paragraph_direction_scope_mismatch_cannot_force_fail():
    r,_,_=run('V2-ROL-07',answer('Q_COMPARISON_WINNER','direction_reversal','en_all','zh_all',
                                'proposed system','基线方法'))
    assert r.verdict=='REVIEW_REQUIRED'
    assert not _same_condition(CASES['V2-ROL-07']['en'],CASES['V2-ROL-07']['zh'])


def test_false_equivalence_label_does_not_override_source_checked_boundary():
    audit=dict(KNOWN['V2-NUM-13'],finding='EQUIVALENT')
    r,_,_=run('V2-NUM-13',audit)
    assert r.verdict=='FAIL'


def test_false_equivalence_label_does_not_override_source_checked_direction():
    audit=dict(KNOWN['V2-ROL-07'],finding='EQUIVALENT')
    r,_,_=run('V2-ROL-07',audit)
    assert r.verdict=='FAIL'


def test_wrong_or_broad_risk_cannot_auto_pass():
    for key,third in [('V2-NUM-15',dict(KNOWN['V2-NUM-15'],same_requirement='unknown')),
                      ('V2-NUM-13',dict(KNOWN['V2-NUM-13'],en_evidence_id='en_all'))]:
        r,_,_=run(key,third)
        assert r.verdict=='REVIEW_REQUIRED'


def test_primary_only_warning_never_auto_pass():
    r,req,_=run('V2-NUM-15',mode='primary')
    assert r.verdict=='REVIEW_REQUIRED' and len(req)==1


def test_no_forced_chinese_english_sentence_order_pairing():
    en='The first dataset had the baseline winner. The second dataset had the proposed system winner.'
    zh='在第二组数据上新系统更高。在第一组数据上基线方法更高。'
    qs=choose_questions(en,zh,'PASS','FAIL')
    assert isinstance(qs,list)


def test_all_prior_pass_cases_without_extra_risk():
    from app.source_risks import source_risks
    for f in ('v2_regression_11.json','validation_v028_holdout_36.json'):
        cases=json.loads(Path('examples',f).read_text())
        if isinstance(cases,dict):cases=cases['cases']
        assert all(not source_risks(c['en'],c['zh']) for c in cases if c['expected']=='PASS')


def test_cross_topic_percentage_cannot_be_disguised_as_the_same_requirement():
    from app.question_direct import Question, _prove
    a=Question('Q_NUMERIC_BOUNDARY','numeric_boundary','fixed question',
                'The accuracy threshold must be greater than 10%.',
                '逾期比例达到10%或以上即不合格。')
    b=answer('Q_NUMERIC_BOUNDARY','numeric_boundary','en_s1','zh_s1','>10%','>=10%')
    from app.models import QuestionVerificationDraft
    x=QuestionVerificationDraft(**b)
    assert _prove(a,x,a.en_hint,a.zh_hint,a.en_hint,a.zh_hint,a.en_hint,a.zh_hint)=='REVIEW_REQUIRED'


def test_percent_numeric_requires_independent_context_anchors():
    from app.question_direct import _same_numeric_context
    en=CASES['V2-NUM-13']['en'].split('.')[1]
    zh='若某份答卷的必填字段缺失比例达到10%或以上，该答卷无效'
    assert _same_numeric_context(en,zh)
    assert not _same_numeric_context('A 10% price increase', '论文大于等于10%')


def test_question_types_are_specific_not_old_generic_verdict_vote():
    scenarios={'V2-NUM-15':'inclusion_exclusion','V2-NUM-13':'numeric_boundary',
               'V2-EXC-03':'exception_scope','V2-NUM-03':'scalar_equivalence',
               'V2-ROL-07':'direction_reversal'}
    for key, kind in scenarios.items():
        c=CASES[key]
        qs=choose_questions(c['en'],c['zh'], c['detail']['direct_primary']['verdict'],
                            c['detail']['review']['verdict'])
        assert qs and qs[0].kind==kind
