"""v0.2.12 offline full-HTTP replay of actual v0.2.10 model outputs.

These tests are synthetic audit replies and do NOT establish live LLM accuracy.
The original model results are replayed unchanged, and no network is used.
"""
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.llm import ChatClient
from app.pipeline import Checker
from app.verified_direct import source_spans, _kind_supported, _proof_bound
from app.models import FocusedVerificationDraft


CASES = {c['id']: c for c in json.loads(
    Path('examples/reference_v0210_direct_11.json').read_text(encoding='utf-8'))['cases']}


def original_responses(c):
    first = c['detail']['direct_primary']
    review = c['detail']['review']
    return first, {'verdict': review['verdict'], 'reason': review['reason'],
                   'issues': [{k: i[k] for k in ('kind','en_paragraph_ids','zh_paragraph_ids','reason')}
                              for i in review['issues']]}


def structured(status, difference_type, en_id, zh_id, same='yes'):
    return dict(status=status, difference_type=difference_type,
                same_subject=same, en_evidence_id=en_id, zh_evidence_id=zh_id,
                reason='Focused relation of the selected original-language source spans.')


def check_case(id_or_pair, audit_reply=None, override_responses=None, mode='dual'):
    if isinstance(id_or_pair, str):
        case = CASES[id_or_pair]
        en, zh = case['en'], case['zh']
        base = list(original_responses(case))
    else:
        en, zh = id_or_pair
        base = []
    responses = list(override_responses) if override_responses is not None else base
    if audit_reply is not None:
        responses.append(audit_reply)
    records = []
    def handler(request):
        payload = json.loads(request.content)
        records.append(payload)
        assert len(records) <= len(responses), 'Unexpected or repeated LLM request'
        content = json.dumps(responses[len(records) - 1], ensure_ascii=False)
        return httpx.Response(200, json={'choices': [{'message': {'content': content},
                                                       'finish_reason': 'stop'}]})
    settings = replace(Settings(), model='qwen-local', verifier_model='ministral-local',
                       pipeline='direct_verified', evidence_mode='paragraph', json_schema=True)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient(settings, http)
    client.start_trace()
    try:
        res = Checker(client, settings).check(en, zh, mode=mode)
        return res, records, client.trace_snapshot()
    finally:
        client.close()


def test_source_span_ids_always_refer_to_unmodified_source():
    sample = 'A 3.5% threshold applies. A different clause follows; but keep the context.'
    options = source_spans(sample, 'en')
    assert options[0].id == 'en_all' and options[0].text == sample
    assert all(span.text in sample for span in options)
    assert '3.5%' in options[1].text
    assert options[1].text.startswith('A 3.5%')  # Do not split decimal 3.5


def test_no_risk_still_exactly_two_model_calls_and_full_originals():
    for key in ('V2-NUM-12', 'V2-ENU-12', 'V2-ENU-13',
                'V2-MIX-09','V2-MIX-04', 'V2-MIX-03'):
        res, calls, trace = check_case(key)
        assert len(calls) == len(trace) == 2
        assert res.verdict == CASES[key]['predicted']
        assert res.focused_verification is None
        assert not res.english_keywords and not res.chinese_keywords and not res.links
        for payload in calls:
            original = json.loads(payload['messages'][1]['content'])
            assert original['english_text'] == CASES[key]['en']
            assert original['chinese_text'] == CASES[key]['zh']


def test_numeric_boundary_requires_source_selectors_and_normalized_comparators():
    ans = structured('CONFLICT', 'numeric_boundary', 'en_s2', 'zh_s2')
    res, calls, trace = check_case('V2-NUM-13', ans)
    assert res.verdict == 'FAIL'
    assert len(calls) == 3
    assert trace[-1]['stage'] == 'direct_proposition_verify'
    assert res.grounded_evidence_audit.en_quote in CASES['V2-NUM-13']['en']
    assert res.grounded_evidence_audit.zh_quote in CASES['V2-NUM-13']['zh']
    assert res.focused_verification.difference_type == 'numeric_boundary'
    assert _proof_bound(res.focused_verification,
                        res.grounded_evidence_audit.en_quote,
                        res.grounded_evidence_audit.zh_quote)
    source_prompt = json.loads(calls[-1]['messages'][1]['content'])
    assert source_prompt['complete_english'] == CASES['V2-NUM-13']['en']
    assert source_prompt['complete_chinese'] == CASES['V2-NUM-13']['zh']
    assert calls[-1]['max_tokens'] == 1536


def test_same_value_different_referent_does_not_force_conflict():
    ans = structured('UNCERTAIN', 'undetermined', 'en_s2', 'zh_s2', same='unknown')
    res, calls, _ = check_case('V2-NUM-13', ans)
    assert res.verdict == 'REVIEW_REQUIRED' and len(calls) == 3


def test_shared_false_pass_inclusion_cannot_be_silently_accepted():
    # Even a declared semantic mismatch must not automatically assert a
    # bilingual subject mapping without cross-language proof.
    for reply in (structured('CONFLICT', 'inclusion_exclusion', 'en_s3', 'zh_s3'),
                  structured('EQUIVALENT', 'equivalent_restatement', 'en_s3', 'zh_s3')):
        res, calls, _ = check_case('V2-NUM-15', reply)
        assert res.verdict == 'REVIEW_REQUIRED' and len(calls) == 3
        assert 'inclusion_inventory' in res.grounded_risks[0]


def test_equal_duration_speculative_process_difference_is_never_auto_fail():
    # Replays v0.2.11's known wrong CONFLICT explanation, but structured.
    reply = structured('CONFLICT', 'extra_requirement', 'en_s1', 'zh_s1')
    res, calls, _ = check_case('V2-NUM-03', reply)
    assert res.verdict == 'REVIEW_REQUIRED' and len(calls) == 3
    assert any('speculative FAIL' in d for d in res.diagnostics)


def test_equal_duration_equivalent_but_not_if_extra_real_obligation():
    reply = structured('EQUIVALENT', 'equivalent_restatement', 'en_s1', 'zh_s1')
    res, calls, _ = check_case('V2-NUM-03', reply)
    assert res.verdict == 'PASS' and len(calls) == 3
    en = 'The survey ran for exactly six weeks.'
    zh = '问卷调查持续42天，而且必须公布每位参与者的姓名。'
    first = {'verdict':'FAIL','reason':'Contains additional requirement',
             'issues':[{'kind':'missing_in_english','en_paragraph_ids':[],
                        'zh_paragraph_ids':['zh_p1'],'reason':'Chinese adds disclosure'}]}
    secondary = {'verdict':'FAIL','reason':'Contains additional requirement',
                 'issues':[{'kind':'conflict','en_paragraph_ids':['en_p1'],
                            'zh_paragraph_ids':['zh_p1'],'reason':'Extra duty'}]}
    r, calls, _ = check_case((en,zh), override_responses=[first,secondary])
    assert r.verdict == 'FAIL' and len(calls) == 2


def test_exception_scope_equivalent_claim_does_not_overturn_dissent():
    # A model can cite a genuine substring and still omit '无论 ...'.
    reply = structured('EQUIVALENT', 'equivalent_restatement', 'en_s2','zh_s2')
    res, calls, _ = check_case('V2-EXC-03', reply)
    assert res.verdict == 'REVIEW_REQUIRED' and len(calls) == 3


def test_role_direction_conflict_can_be_verified_with_cited_source_spans():
    reply = structured('CONFLICT','direction_reversal','en_s2','zh_s2')
    res, calls, _ = check_case('V2-ROL-07', reply)
    assert res.verdict == 'FAIL' and len(calls) == 3
    assert _kind_supported(res.focused_verification, res.grounded_evidence_audit.en_quote,
                           res.grounded_evidence_audit.zh_quote)
    assert '第二组数据' in res.grounded_evidence_audit.zh_quote


def test_unrecognized_difference_cannot_force_fail():
    reply=structured('CONFLICT','other','en_s2','zh_s2')
    r, calls, _ = check_case('V2-ROL-07', reply)
    assert r.verdict == 'REVIEW_REQUIRED' and len(calls)==3


def test_audit_invalid_id_only_one_request_and_safe_abstention():
    reply=structured('CONFLICT','numeric_boundary','en_fake','zh_s2')
    res, calls, trace = check_case('V2-NUM-13', reply)
    assert res.verdict == 'REVIEW_REQUIRED'
    assert res.focused_verification is None
    assert len(calls) == 3 and trace[-1]['outcome'] == 'schema_or_grounding_invalid'
    assert trace[-1]['attempt'] == 1
    assert 'attempts=1' in res.diagnostics[-1]


def test_conflict_without_verified_same_subject_only_one_attempt():
    reply=structured('CONFLICT','direction_reversal','en_s2','zh_s2', same='no')
    res, calls, _ = check_case('V2-ROL-07', reply)
    assert res.verdict == 'REVIEW_REQUIRED' and len(calls)==3


def test_primary_only_risk_is_deferred():
    a, _ = original_responses(CASES['V2-NUM-15'])
    r,calls,_=check_case('V2-NUM-15', override_responses=[a],mode='primary')
    assert r.verdict == 'REVIEW_REQUIRED' and len(calls)==1


def test_v0211_pipeline_stays_selectable_and_v028_pass_warnings_unchanged():
    # Risk detector remains conservative on earlier known-positive pairs.
    from app.source_risks import source_risks
    for file in ('v2_regression_11.json', 'validation_v028_holdout_36.json'):
        data = json.loads(Path('examples',file).read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = data['cases']
        for case in data:
            if case['expected']=='PASS':
                assert not source_risks(case['en'], case['zh']), case['id']


def test_source_grounded_evidence_lists_do_not_assume_equal_clause_positions():
    en='The report may contain figures. The appendix must be uploaded.'
    zh='附录必须上传。报告可以包含插图。'
    e,z=source_spans(en,'en'),source_spans(zh,'zh')
    assert e[1].text != z[1].text
    assert e[0].text == en and z[0].text == zh


def test_long_evidence_selector_cannot_hide_unsupported_differences():
    en = 'The paper discusses an ordinary requirement. ' * 35
    zh = '论文讨论一项普通要求。' * 35
    a = {'verdict':'PASS','reason':'aligned','issues':[]}
    b = {'verdict':'FAIL','reason':'possible omission','issues':[
        {'kind':'conflict','en_paragraph_ids':['en_p1'],
         'zh_paragraph_ids':['zh_p1'],'reason':'alleged omission'}]}
    verification = structured('CONFLICT','direction_reversal','en_all','zh_all')
    result, calls, _ = check_case((en,zh), override_responses=[a,b,verification])
    assert result.verdict=='REVIEW_REQUIRED' and len(calls)==3
    assert result.focused_verification is not None
    assert result.grounded_evidence_audit is None
