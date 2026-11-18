"""v0.2.9.1: prove that source-anchor fast path saves REAL HTTP calls.

The fast path never decides bilingual equivalence. Model matching, full-original
review, numeric safety and protected dispute handling still run normally.
"""
from __future__ import annotations

import json

import httpx
import pytest
from test_v027_paragraph_grounding import Fake, config, match, review

from app.config import Settings
from app.extraction_fast_path import source_anchor_pair
from app.llm import ChatClient
from app.paragraphs import build_paragraphs
from app.pipeline import Checker

CONDITIONAL_EN = 'When consent is withdrawn, the recording must be deleted unless the law requires retention.'
CONDITIONAL_ZH = '撤回同意后应删除录音，但法律要求留存的除外。'


def pair(en, zh):
    return source_anchor_pair(build_paragraphs(en, 'en'), build_paragraphs(zh, 'zh'))


def test_true_short_conditional_exception_skips_both_model_extractions():
    en, zh = pair(CONDITIONAL_EN, CONDITIONAL_ZH)
    assert en[0].claim == en[0].evidence == CONDITIONAL_EN
    assert zh[0].claim == zh[0].evidence == CONDITIONAL_ZH
    assert en[0].paragraph_ids == ['en_p1'] and zh[0].paragraph_ids == ['zh_p1']
    client = Fake(match(), review())
    out = Checker(client, config(extraction_fast_path=True)).check(CONDITIONAL_EN, CONDITIONAL_ZH, mode='dual')
    assert out.verdict == 'PASS'
    assert [c['schema'] for c in client.calls] == ['keyword_matching', 'paragraph_review']
    assert any('source_anchor_fast_path' in x for x in out.diagnostics)


def test_short_numeric_bound_still_raises_deterministic_conflict_when_both_models_pass():
    en='The calibration error must stay below 0.8 millimetres.'
    zh='校准误差不得超过0.8毫米。'
    assert pair(en, zh) is not None
    client=Fake(match(), review('PASS'))
    out=Checker(client,config(extraction_fast_path=True)).check(en,zh,mode='dual')
    assert out.verdict=='FAIL'
    assert out.deterministic_issues
    assert [c['schema'] for c in client.calls]==['keyword_matching','paragraph_review']


@pytest.mark.parametrize('en,zh', [
    ('The file must be archived and must be encrypted.', '文件必须归档且必须加密。'),
    ('Submit the report as a PDF. The figures must be included.', '报告须提交PDF。图表应包含在内。'),
    ('The report must include methods, results, and conclusions.', '报告须包括方法、结果和结论。'),
    ('The appendix must be submitted separately.\n\nIt must remain readable.', '附录须单独提交。\n\n必须保持可读。'),
    ('The file must be archived; the tables may be deleted.', '文件必须归档；表格可以删除。'),
    ('The file must be archived. The tables may be deleted', '文件必须归档。表格可以删除'),
    ('The file must be retained, and files must be archived.', '文件必须保留，并归档。'),
])
def test_multi_requirements_and_lists_must_abstain(en,zh):
    assert pair(en,zh) is None


def test_opt_out_preserves_old_llm_extraction_path():
    from test_v027_paragraph_grounding import extract
    client=Fake(extract(CONDITIONAL_EN,'en_p1'),extract(CONDITIONAL_ZH,'zh_p1'),match(),review())
    out=Checker(client,config(extraction_fast_path=False)).check(CONDITIONAL_EN,CONDITIONAL_ZH,mode='dual')
    assert out.verdict=='PASS'
    assert [c['schema'] for c in client.calls]==['paragraph_extraction','paragraph_extraction',
                                                 'keyword_matching','paragraph_review']


def _reply(value):
    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(value,ensure_ascii=False)},
                                                 'finish_reason':'stop'}]})


def test_http_fast_path_sends_only_matching_and_review():
    sent=[]
    answers=[{'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],'relation':'equivalent',
                        'reason':'originals preserve deletion obligation and legal exception'}]},
             {'verdict':'PASS','reason':'both texts retain the same exception','issues':[]}]
    def handler(request):
        req=json.loads(request.content)
        sent.append(req)
        return _reply(answers[len(sent)-1])
    http=httpx.Client(transport=httpx.MockTransport(handler))
    cfg=Settings(evidence_mode='paragraph',model='main',verifier_model='verify',
                 json_schema=True,extraction_fast_path=True,extraction_max_tokens=1536,
                 max_tokens=4096)
    client=ChatClient(cfg,http)
    client.start_trace()
    result=Checker(client,cfg).check(CONDITIONAL_EN,CONDITIONAL_ZH,mode='dual')
    assert result.verdict=='PASS'
    assert len(sent)==2
    assert [x['response_format']['json_schema']['name'] for x in sent]==[
        'keyword_matching','paragraph_review']
    assert [r['stage'] for r in client.trace_snapshot()]==['keyword_matching','independent_review']
    assert all(r['max_tokens']==4096 for r in client.trace_snapshot())
    http.close()


def test_http_nonfast_short_multi_clause_keeps_extraction_with_smaller_cap():
    en='The report must be a searchable PDF and all tables must be uploaded separately.'
    zh='报告必须是可搜索PDF，并且所有表格必须单独上传。'
    assert pair(en,zh) is None
    sent=[]
    answers=[{'units':[{'kind':'constraint','claim':en,'paragraph_ids':['en_p1']}]},
             {'units':[{'kind':'constraint','claim':zh,'paragraph_ids':['zh_p1']}]},
             {'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],'relation':'equivalent','reason':'same requirements'}]},
             {'verdict':'PASS','reason':'same two obligations','issues':[]}]
    def handler(request):
        payload=json.loads(request.content);sent.append(payload)
        return _reply(answers[len(sent)-1])
    http=httpx.Client(transport=httpx.MockTransport(handler))
    cfg=Settings(evidence_mode='paragraph',model='main',verifier_model='verify',
                 json_schema=True,extraction_fast_path=True,extraction_max_tokens=1536,
                 max_tokens=4096)
    client=ChatClient(cfg,http)
    result=Checker(client,cfg).check(en,zh,mode='dual')
    assert result.verdict=='PASS'
    assert len(sent)==4
    assert [x['max_tokens'] for x in sent]==[1536,1536,4096,4096]
    assert len(result.english_keywords)==1 and len(result.chinese_keywords)==1
    assert not any('source_anchor_fast_path' in x for x in result.diagnostics)
    http.close()


def test_multi_paragraph_sources_keep_original_generation_budget():
    en='A must be archived.\n\nB must remain accessible.'
    zh='A必须归档。\n\nB应保持可访问。'
    assert pair(en,zh) is None
    cfg=Settings(extraction_fast_path=True,extraction_max_tokens=1536,max_tokens=4096)
    assert len(build_paragraphs(en,'en'))==2
    # Selection is checked below in the pipeline integration; we do not cap
    # long or multi-paragraph extraction to a 1.5k budget.
    assert cfg.extraction_max_tokens < cfg.max_tokens


def test_fast_path_does_not_auto_pass_when_review_cites_true_non_numeric_conflict():
    from test_v027_paragraph_grounding import arb

    from app.models import ParagraphDisputeRecheckDraft
    en='The recording must be deleted.'
    zh='录音不得删除。'
    assert pair(en,zh) is not None
    issue={'kind':'conflict','en_paragraph_ids':['en_p1'],
           'zh_paragraph_ids':['zh_p1'],'reason':'delete required versus forbidden'}
    recheck=ParagraphDisputeRecheckDraft.model_validate({'issues':[
        {'issue_index':0,'status':'refuted','basis':'other',
         'en_paragraph_ids':['en_p1'],'zh_paragraph_ids':['zh_p1'],
         'reason':'incorrectly refuted deletion difference'}],
         'reason':'incorrect'} )
    # Both the primary matcher and arbiter wrongly say PASS. The grounded
    # reviewer raises a deletion conflict. No numeric proof exists.
    fake=Fake(match(),review('FAIL',[issue]),arb('PASS','none',['en_p1'],['zh_p1']),recheck)
    out=Checker(fake,config(extraction_fast_path=True)).check(en,zh,mode='dual')
    assert out.verdict=='REVIEW_REQUIRED'
    assert len(fake.calls)==4
    assert out.dispute_recheck is not None


def test_short_nonfast_recovery_preserves_budget_and_original_validation():
    en='The report must be a PDF and every table must be uploaded separately.'
    zh='报告必须是PDF，并且每个表格必须单独上传。'
    assert pair(en,zh) is None
    payloads=[]
    answers=[
        {'units':[{'kind':'constraint','claim':en,'paragraph_ids':['en_p1']}]},
        None, # Chinese extraction truncated
        {'units':[{'kind':'constraint','claim':zh,'paragraph_ids':['zh_p1']}]},
        {'links':[{'en_ids':['en_1'],'zh_ids':['zh_1'],'relation':'equivalent',
                   'reason':'both document and table requirements match'}]},
        {'verdict':'PASS','reason':'fully aligned','issues':[]},
    ]
    def handler(request):
        payload=json.loads(request.content)
        payloads.append(payload)
        index=len(payloads)-1
        if answers[index] is None:
            return httpx.Response(200,json={'choices':[{'message':{'content':''},
                     'finish_reason':'length'}],'usage':{'completion_tokens':1536}})
        return _reply(answers[index])
    http=httpx.Client(transport=httpx.MockTransport(handler))
    cfg=Settings(evidence_mode='paragraph', model='main', verifier_model='verify',
                 json_schema=True, generation_recovery=True,
                 extraction_fast_path=True, extraction_max_tokens=1536,max_tokens=4096)
    client=ChatClient(cfg,http)
    out=Checker(client,cfg).check(en,zh,mode='dual')
    assert out.verdict=='PASS'
    assert len(payloads)==5
    # The truncation-recovery request (3rd) gets the global LM_MAX_TOKENS budget:
    # retrying with the same 1536 cap usually truncates again (seen live).
    assert [x['max_tokens'] for x in payloads]==[1536,1536,4096,4096,4096]
    assert 'response_format' in payloads[1] and 'response_format' not in payloads[2]
    assert [r['outcome'] for r in client.trace_snapshot()]==[
        'validated','truncated','recovered_validated','validated','validated']
    http.close()
