"""Offline tests for the alternative *whole-original* pipeline.

No real LM Studio invocation or accuracy claim is made by these mocks.
"""
import json
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.direct_pipeline import check_direct, direct_numeric_conflicts
from app.llm import ChatClient
from app.paragraphs import build_paragraphs
from app.pipeline import Checker
from app.prompts import paragraph_review_user

EN = ('The review addresses data quality, annotation consistency, and evaluation '
      'reproducibility. These topics may be discussed in any order.')
ZH = '综述需要涉及数据质量、标注一致性以及评测可复现性，三者的叙述顺序不作限制。'


def run_http(en, zh, responses, mode='dual'):
    calls = []
    def handler(req):
        payload = json.loads(req.content)
        calls.append(payload)
        index = len(calls) - 1
        assert index < len(responses), 'direct pipeline unexpectedly made additional HTTP calls'
        return httpx.Response(200, json={
            'choices': [{'message': {'content': json.dumps(responses[index], ensure_ascii=False)},
                         'finish_reason': 'stop'}]})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    # LM_PIPELINE=direct was removed in v0.3.0; check_direct is still the
    # first stage of direct_hardened, so it is exercised directly here.
    cfg = Settings(model='qwen-local', verifier_model='ministral-local',
                   evidence_mode='paragraph', json_schema=True)
    client = ChatClient(cfg, http)
    client.start_trace()
    try:
        result = check_direct(Checker(client, cfg), en, zh, mode=mode)
        return result, calls, client.trace_snapshot()
    finally:
        client.close()


def label(verdict, reason='checked originals', issues=None):
    return {'verdict': verdict, 'reason': reason, 'issues': issues or []}


def conflict():
    return {'kind':'conflict','en_paragraph_ids':['en_p1'],
            'zh_paragraph_ids':['zh_p1'],'reason':'one requirement differs'}


def test_direct_enumeration_two_calls_no_extraction_or_matching_and_source_only():
    out, calls, trace = run_http(EN, ZH, [label('PASS'),label('PASS')])
    assert out.verdict == 'PASS'
    assert len(calls) == len(trace) == 2
    assert [r['stage'] for r in trace] == ['direct_primary','direct_independent_review']
    assert not out.english_keywords and not out.chinese_keywords and not out.links
    assert out.direct_primary.verdict == out.review.verdict == 'PASS'
    for req in calls:
        user = json.loads(req['messages'][1]['content'])
        assert user['english_text'] == EN and user['chinese_text'] == ZH
        assert user['english_paragraphs'][0]['text'] == EN
        assert user['chinese_paragraphs'][0]['text'] == ZH
    assert out.arbitrated is False


def test_independent_disagreement_abstains_without_correlated_third_vote():
    out,calls,_ = run_http(EN, ZH, [label('FAIL',issues=[conflict()]),label('PASS')])
    assert out.verdict == 'REVIEW_REQUIRED'
    assert len(calls) == 2
    assert out.initial_verdict == 'FAIL'
    assert out.direct_primary.issues


def test_reverse_disagreement_abstains():
    out, calls, _ = run_http(EN, ZH, [label('PASS'),label('FAIL',issues=[conflict()])])
    assert out.verdict == 'REVIEW_REQUIRED'
    assert len(calls) == 2
    assert len(out.review_issues) == 1


def test_both_fail_with_source_ids_remains_fail():
    out,calls,_ = run_http('The file must be archived.', '文件不得归档。',
                            [label('FAIL',issues=[conflict()]),label('FAIL',issues=[conflict()])])
    assert out.verdict == 'FAIL'
    assert len(calls) == 2


def test_unknown_is_not_auto_pass():
    out,_,_ = run_http(EN,ZH,[label('PASS'),label('UNSURE')])
    assert out.verdict == 'REVIEW_REQUIRED'


def test_bound_conflict_prevents_both_false_passes():
    en='The calibration error must stay below 0.8 millimetres.'
    zh='校准误差不得超过0.8毫米。'
    out,calls,_ = run_http(en,zh,[label('PASS'),label('PASS')])
    assert out.verdict=='FAIL'
    assert out.deterministic_issues
    assert len(calls)==2


def test_duration_paraphrase_is_not_a_guard_mismatch():
    en='Participants were monitored for 72 hours without interruption.'
    zh='参与者接受连续三天的监测。'
    out,_,_ = run_http(en,zh,[label('PASS'),label('PASS')])
    assert out.verdict=='PASS'
    assert not out.deterministic_issues


def test_complex_multinumeric_source_abstains_from_forced_guard():
    en='If fewer than five participants attend, the appendix may be 6 pages.'
    zh='少于五名参与者时，附录最多6页。'
    assert direct_numeric_conflicts(en,zh,build_paragraphs(en,'en'),build_paragraphs(zh,'zh'))==[]


def test_input_sentence_order_is_not_alignment_constraint():
    en='The document must have a title. The abstract must follow the title.'
    zh='摘要须置于标题之后。文档必须有标题。'
    out, calls, _ = run_http(en,zh,[label('PASS'),label('PASS')])
    assert out.verdict=='PASS' and len(calls)==2
    assert out.english_paragraphs[0].text==en
    assert out.chinese_paragraphs[0].text==zh


def test_multiple_paragraphs_pass_through_whole_source_unchanged():
    en='An abstract is mandatory.\n\nThe bibliography is optional.'
    zh='参考文献列表可选。\n\n摘要是必需的。'
    out, calls, _ = run_http(en,zh,[label('PASS'),label('PASS')])
    assert out.verdict=='PASS'
    assert len(out.english_paragraphs)==len(out.chinese_paragraphs)==2
    for req in calls:
        user=json.loads(req['messages'][1]['content'])
        assert user['english_text']==en and user['chinese_text']==zh


def test_primary_only_one_request_and_no_reviewer():
    out,calls,_=run_http(EN,ZH,[label('PASS')], mode='primary')
    assert out.verdict=='PASS' and out.review is None and len(calls)==1


def test_invalid_immutable_paragraph_id_blocks_result():
    from app.llm import LLMError
    broken={'kind':'conflict','en_paragraph_ids':['en_p999'],
            'zh_paragraph_ids':['zh_p1'],'reason':'invalid source'}
    # Structural validation should fail; LM retries may occur, but cannot
    # accidentally return a valid FAIL without valid evidence.
    with pytest.raises(LLMError):
        run_http(EN,ZH,[label('FAIL',issues=[broken])]*4)


def test_legacy_settings_still_default_to_extraction():
    assert Settings().pipeline=='extraction'


def test_matched_reference_comparison_does_not_call_models():
    from examples.evaluate import paired_reference_comparison, load_cases
    from pathlib import Path
    path=Path('examples/reference_v028_regression_11.json')
    ref=json.loads(path.read_text())
    assert len(load_cases(Path('examples/v2_regression_11.json')))==11
    expected=[{**case, 'predicted':case['predicted'], 'elapsed_seconds':10, 'requests':[{},{}]}
              for case in ref['cases']]
    summary=paired_reference_comparison(expected,path)
    assert summary['baseline_correct']==summary['direct_correct']==11
    assert summary['baseline_requests']>summary['direct_requests']
    assert not summary['changed']
    assert 'not_contemporaneous' in summary['kind']


def test_reference_mismatch_rejected():
    from examples.evaluate import paired_reference_comparison
    from pathlib import Path
    path=Path('examples/reference_v028_regression_11.json')
    example=dict(json.loads(path.read_text())['cases'][0])
    example['zh']='stale and invalid data'
    with pytest.raises(ValueError, match='不一致'):
        paired_reference_comparison([example],path)
