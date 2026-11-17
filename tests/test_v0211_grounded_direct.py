"""Offline-only HTTP mocks. The prior LM Studio outputs are NEVER rerun here."""
import json
from dataclasses import replace
from pathlib import Path
import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient
from app.pipeline import Checker
from app.source_risks import source_risks, formal_single_quantity_equivalence

EN_EQ = 'The survey ran for exactly six weeks.'
ZH_EQ = '问卷调查持续了42天，既未延长也未缩短。'


def label(verdict, reason='source comparison', issues=None):
    return {'verdict': verdict, 'reason': reason, 'issues': issues or []}


def issue(reason='different'):
    return {'kind': 'conflict', 'en_paragraph_ids': ['en_p1'],
            'zh_paragraph_ids': ['zh_p1'], 'reason': reason}


def audit(status, en, zh, reason='focused conclusion'):
    return {'status': status, 'en_quote': en, 'zh_quote': zh, 'reason': reason}


def run(en, zh, responses, mode='dual'):
    calls = []
    def handler(req):
        payload = json.loads(req.content)
        calls.append(payload)
        assert len(calls) <= len(responses), 'unexpected LLM request'
        return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps(
            responses[len(calls)-1],ensure_ascii=False)}, 'finish_reason':'stop'}]})
    cfg = replace(Settings(), model='qwen-local', verifier_model='ministral-local',
                  pipeline='direct_grounded', evidence_mode='paragraph',json_schema=True)
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=ChatClient(cfg,http)
    client.start_trace()
    try:
        result=Checker(client,cfg).check(en,zh,mode=mode)
        return result,calls,client.trace_snapshot()
    finally:
        client.close()


def cases():
    obj=json.loads(Path('examples/reference_v0210_direct_11.json').read_text())
    return {c['id']:c for c in obj['cases']}


def responses_of(case):
    primary=case['detail']['direct_primary']
    sec=case['detail']['review']
    secondary={'verdict':sec['verdict'],'reason':sec['reason'],'issues':[
        {k:i[k] for k in ('kind','en_paragraph_ids','zh_paragraph_ids','reason')}
        for i in sec['issues']]}
    return primary,secondary


def test_replay_unique_percentage_bound_detected_from_multi_statement_original():
    c=cases()['V2-NUM-13']
    risks=source_risks(c['en'],c['zh'])
    assert [r.kind for r in risks]==['numeric_boundary']
    a,b=responses_of(c)
    en_excerpt='more than 10%'
    zh_excerpt='达到10%或以上'
    result, calls,trace=run(c['en'],c['zh'],[a,b,audit('CONFLICT',en_excerpt,zh_excerpt)])
    assert result.verdict=='FAIL'
    assert len(calls)==3
    assert [r['stage'] for r in trace]==['direct_primary','direct_independent_review','direct_evidence_audit']
    assert result.grounded_evidence_audit.en_quote in c['en']


def test_replay_both_false_passes_counting_inventory_forces_review():
    c=cases()['V2-NUM-15']
    a,b=responses_of(c)
    r,calls,_=run(c['en'],c['zh'],[a,b,audit('CONFLICT',
        'The cover page is included within the main-report limit', '封面不计入30页限制')])
    assert r.verdict=='REVIEW_REQUIRED' and len(calls)==3
    assert 'inclusion_inventory' in r.grounded_risks[0]


def test_replay_both_false_failures_single_duration_becomes_pass_with_proof():
    c=cases()['V2-NUM-03']
    a,b=responses_of(c)
    assert formal_single_quantity_equivalence(c['en'],c['zh'])
    r,calls,_=run(c['en'],c['zh'],[a,b,audit('EQUIVALENT','exactly six weeks','42天')])
    assert r.verdict=='PASS' and len(calls)==3


def test_replay_secondary_failure_rechecked_by_initially_dissenting_primary():
    for key, enq, zhq in (
        ('V2-EXC-03','An additional Chinese abstract is optional unless the degree program requires it',
            '无论项目是否另有规定，中文摘要都只能作为可选材料'),
        ('V2-ROL-07','On the second dataset, the proposed system achieved the higher score',
            '在第二组数据上，仍然是基线方法得分更高')):
        c=cases()[key]
        a,b=responses_of(c)
        r,calls,_=run(c['en'],c['zh'],[a,b,audit('CONFLICT',enq,zhq)])
        assert r.verdict=='FAIL' and len(calls)==3
        assert calls[-1]['model']=='qwen-local'


def test_no_trigger_cost_remains_two_and_originals_untouched():
    for key in ('V2-NUM-12','V2-ENU-12','V2-ENU-13','V2-MIX-09','V2-MIX-04','V2-MIX-03'):
        c=cases()[key]
        a,b=responses_of(c)
        r, calls, trace=run(c['en'],c['zh'],[a,b])
        assert len(calls)==2 and len(trace)==2
        assert r.verdict==c['predicted']
        assert not r.english_keywords and not r.links
        for call in calls:
            data=json.loads(call['messages'][1]['content'])
            assert data['english_text']==c['en']
            assert data['chinese_text']==c['zh']


def test_unverified_quote_cannot_flip_label_even_if_model_says_conflict():
    c=cases()['V2-NUM-15'];a,b=responses_of(c)
    # Up to 3 schema/grounding correction attempts, still unverified:
    r,calls,trace=run(c['en'],c['zh'],[a,b]+[audit('CONFLICT','invented substring','封面不计入30页限制')]*4)
    assert r.verdict=='REVIEW_REQUIRED'
    assert r.grounded_evidence_audit is None
    assert len(calls)>2


def test_only_formally_supported_quantity_can_restore_automatic_pass():
    a=label('FAIL',issues=[issue()])
    b=label('FAIL',issues=[issue()])
    en='The paper must be at least 30 pages.'
    zh='论文至少30页，而且必须匿名提交。'
    # Same quantity, but another explicit obligation: audit cannot erase.
    r,calls,_=run(en,zh,[a,b])
    assert r.verdict=='FAIL' and len(calls)==2


def test_boundary_citation_requires_real_comparators_not_just_narrative():
    c=cases()['V2-NUM-13']; a,b=responses_of(c)
    r,calls,_=run(c['en'],c['zh'],[a,b,audit('CONFLICT',
        'The study must obtain no fewer than 120 valid responses', '研究须获得至少120份有效答卷')])
    assert r.verdict=='REVIEW_REQUIRED' and len(calls)==3


def test_quote_status_uncertain_fails_closed():
    c=cases()['V2-NUM-15']; a,b=responses_of(c)
    r,_,_=run(c['en'],c['zh'],[a,b,audit('UNCERTAIN','cover page','封面')])
    assert r.verdict=='REVIEW_REQUIRED'


def test_disagreement_no_reason_grounded_cannot_assert_fail():
    en='The source is required.';zh='来源必须提供。'
    r,calls,_=run(en,zh,[label('PASS'),label('FAIL',issues=[issue()]),
       audit('EQUIVALENT','The source is required','来源必须提供')])
    assert r.verdict=='REVIEW_REQUIRED' and len(calls)==3


def test_source_risk_triage_never_pairs_by_position_or_raw_number_order():
    en='The appendix may be 6 pages. The main text may be 30 pages.'
    zh='正文可为30页。附录可为6页。'
    assert source_risks(en,zh)==[]


def test_primary_only_source_risk_abstains_without_unavailable_verifier():
    c=cases()['V2-NUM-15']
    a,_=responses_of(c)
    r,calls,_=run(c['en'],c['zh'],[a],mode='primary')
    assert r.verdict=='REVIEW_REQUIRED' and len(calls)==1


def test_equivalent_number_does_not_overwrite_additional_sentence_duty():
    en='The survey ran for exactly six weeks. A report was submitted.'
    zh='调查持续42天。另须公布个人姓名。'
    r,calls,_=run(en,zh,[label('FAIL',issues=[issue()]),label('FAIL',issues=[issue()])])
    assert r.verdict=='FAIL' and len(calls)==2


def test_risk_scanner_does_not_warn_on_validated_positive_cases():
    for fn in ('v2_regression_11.json','validation_v028_holdout_36.json'):
        data=json.loads(Path('examples',fn).read_text())
        if isinstance(data,dict):data=data['cases']
        assert not [(r['id'],source_risks(r['en'],r['zh'])) for r in data
                    if r['expected']=='PASS' and source_risks(r['en'],r['zh'])]
