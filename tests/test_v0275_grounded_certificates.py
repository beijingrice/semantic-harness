"""Recorded v0.2.7.4 model replies + deliberately adversarial synthetic cases.

NO live model requests. Recorded replies are test fixtures, not training data.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.dispute_resolution import (numeric_equivalence_certificates,
                                    narrow_exactness_refutation)
from app.llm import ChatClient, LLMError
from app.models import (
    KeyUnit, Link, ParagraphExtractedDraft, MatchDraft,
    ParagraphReviewDraft, ParagraphArbitrationDraft,
    ParagraphDisputeRecheckDraft, ParagraphReviewIssue, ParagraphIssueRecheck,
)
from app.paragraphs import build_paragraphs
from app.pipeline import Checker
from app.validators import parse_quantity
from test_v027_paragraph_grounding import Fake, config

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/v0274_dispute_replay.json').read_text())


def to_draft(rows):
    return ParagraphExtractedDraft.model_validate({'units': [
        {'kind': r['kind'], 'claim': r['claim'], 'paragraph_ids': r['paragraph_ids']}
        for r in rows
    ]})


@pytest.mark.parametrize('name', ['V2-NUM-03', 'C27-01'])
def test_true_model_outputs_with_1_to_2_link_replay_as_pass(name):
    record = FIXTURES[name]
    fake = Fake(
        to_draft(record['english_keywords']),
        to_draft(record['chinese_keywords']),
        MatchDraft.model_validate({'links': record['links']}),
        ParagraphReviewDraft.model_validate(record['review']),
        ParagraphArbitrationDraft.model_validate(record['arbitration']),
        ParagraphDisputeRecheckDraft.model_validate(record['dispute_recheck']),
    )
    result = Checker(fake, config()).check(record['en'], record['zh'], mode='dual')
    assert result.verdict == 'PASS'
    assert result.initial_verdict == 'PASS'
    assert result.dispute_recheck.issues[0].basis == 'other'
    assert result.review.verdict == 'FAIL'
    assert len(fake.calls) == 6
    certificates = fake.calls[-1]['user']['verified_formal_equivalences']
    assert {(c['en_unit_id'], c['zh_unit_id']) for c in certificates} == {
        ('en_1','zh_1'), ('en_1','zh_2')}
    assert all((c['canonical_value'], c['dimension'], c['comparator']) ==
               ('42', 'duration_days', '=') for c in certificates)
    assert not fake.answers


def units(claims, prefix):
    return [KeyUnit(id=f'{prefix}_{i}', kind='constraint', claim=c, evidence=c,
                    paragraph_ids=[f'{prefix}_p1']) for i,c in enumerate(claims,1)]


def certificates(en, zh, en_claims, zh_claims, link=None):
    link = link or Link(en_ids=[f'en_{i}' for i in range(1,len(en_claims)+1)],
                        zh_ids=[f'zh_{i}' for i in range(1,len(zh_claims)+1)],
                        relation='equivalent', reason='test')
    return numeric_equivalence_certificates(
        units(en_claims,'en'), units(zh_claims,'zh'), [link],
        build_paragraphs(en,'en'), build_paragraphs(zh,'zh'))


def test_one_to_many_both_children_must_be_individually_provable():
    en = 'The survey ran for exactly six weeks.'
    zh = '调查持续了42天，既未延长也未缩短。'
    assert len(certificates(en,zh,[en],['调查持续了42天。','调查时间固定为42天。'])) == 2
    assert not certificates(en,zh,[en],['调查持续了42天。','源文件可以删除。'])
    assert not certificates(en,zh,[en],['调查持续了42天。','调查持续了43天。'])


def test_many_to_one_supported_without_blind_cross_product():
    en = 'The survey ran for exactly six weeks.'
    zh = '调查持续了42天。'
    assert len(certificates(en, zh, [en, 'The survey duration is precisely six weeks.'], [zh])) == 2
    assert not certificates(en, zh, [en, 'The file must be archived.'], [zh])
    # M:N not independently decomposed by an LLM: cannot fabricate mappings.
    assert not certificates(en, zh, [en, en], [zh, zh])


def test_never_prove_claim_that_conflicts_with_original_source():
    en = 'The survey ran for exactly six weeks.'
    zh = '调查持续了43天。'
    assert not certificates(en, zh, [en], ['调查持续了42天。'])


def test_ambiguous_approximate_quantities_do_not_become_exact_proofs():
    assert parse_quantity('The survey ran for approximately six weeks.') is None
    assert parse_quantity('调查持续约42天。') is None
    assert parse_quantity('调查持续42天左右。') is None
    assert not certificates('The survey ran for approximately six weeks.',
                            '调查持续了42天。',
                            ['The survey ran for approximately six weeks.'],['调查持续了42天。'])


@pytest.mark.parametrize('en,zh', [
    ('The survey ran for exactly six weeks. Its data must be archived.',
     '调查持续了42天。其数据可删除。'),
    ('The survey ran for exactly six weeks and the files must be archived.',
     '调查持续了42天，文件可以删除。'),
])
@pytest.mark.parametrize('basis', ['other','formal_numeric'])
def test_numeric_certificate_cannot_overrule_unrelated_archival_duty(en, zh, basis):
    en_u,zh_u = [en.split('.')[0]+'.'],['调查持续了42天。']
    # A candidate claim may correctly express the numeric part, but the full
    # source still has a separate obligation: never auto-authorize PASS.
    certs = certificates(en,zh,en_u,zh_u)
    issue=ParagraphReviewIssue(kind='conflict',
        en_paragraph_ids=['en_p1'],zh_paragraph_ids=['zh_p1'],
        reason='The exactly six week duration matches, but the file archiving duty conflicts with deletion.')
    recheck=ParagraphIssueRecheck(issue_index=0,status='refuted',basis=basis,
        en_paragraph_ids=['en_p1'], zh_paragraph_ids=['zh_p1'],
        reason='The six-week duration is exact, but the file permissions differ.')
    assert not narrow_exactness_refutation(issue,recheck,certs,
                                           build_paragraphs(en,'en'),build_paragraphs(zh,'zh'))


def test_spoofed_basis_formal_numeric_cannot_refute_nonnumeric_issue():
    en = 'The survey ran for exactly six weeks.'
    zh = '调查持续42天。'
    certs=certificates(en,zh,[en],[zh])
    issue=ParagraphReviewIssue(kind='conflict',
        en_paragraph_ids=['en_p1'],zh_paragraph_ids=['zh_p1'],
        reason='The journal policy about archiving data was omitted.')
    recheck=ParagraphIssueRecheck(issue_index=0,status='refuted',basis='formal_numeric',
        en_paragraph_ids=['en_p1'], zh_paragraph_ids=['zh_p1'],reason='refuted')
    assert not narrow_exactness_refutation(issue,recheck,certs,
                                           build_paragraphs(en,'en'),build_paragraphs(zh,'zh'))


def test_length_failure_reports_actual_reasoning_budget_without_echoing_content():
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'length',
            'message':{'content':'incomplete', 'reasoning_content':'do not print this private reasoning'}}],
            'usage': {'completion_tokens':4096,'completion_tokens_details':{'reasoning_tokens':4010}}})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError,match='reasoning_tokens=4010') as exc:
            ChatClient(Settings(json_schema=True, generation_recovery=False),http).structured(
                'qwen-local','system','user',ParagraphExtractedDraft,'paragraph_extraction')
        msg=str(exc.value)
        assert 'content_chars=10' in msg
        assert 'reasoning_chars=35' in msg
        assert 'finish_reason=length' in msg
        assert 'private reasoning' not in msg
        assert len(seen)==1
    finally:
        http.close()
