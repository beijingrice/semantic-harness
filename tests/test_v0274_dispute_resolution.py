"""Offline state-machine and certificate safety checks; no LM Studio calls."""
import pytest

from app.models import (
    ParagraphDisputeRecheckDraft, ParagraphReviewIssue, KeyUnit, Link,
)
from app.dispute_resolution import numeric_equivalence_certificates, certificate_covers_issue
from app.paragraphs import build_paragraphs
from app.pipeline import Checker, PipelineStageError
from app.paragraph_pipeline import _verify_objection_recheck
from app.llm import LLMError
from examples.evaluate import summarize
from test_v027_paragraph_grounding import Fake, config, extract, match, review, arb


def issue(en=('en_p1',), zh=('zh_p1',), reason='exactly six weeks versus 42 days'):
    return {'kind': 'conflict', 'en_paragraph_ids': list(en),
            'zh_paragraph_ids': list(zh), 'reason': reason}


def recheck(status, en=('en_p1',), zh=('zh_p1',), basis='formal_numeric'):
    return ParagraphDisputeRecheckDraft.model_validate({
        'issues': [{'issue_index': 0, 'status': status,
                    'basis': basis, 'en_paragraph_ids': list(en), 'zh_paragraph_ids': list(zh),
                    'reason': 'Inspected both originals and the formal quantity'}],
        'reason': 'Focused objection review complete'})


EN = 'The survey ran for exactly six weeks.'
ZH = '问卷调查持续了42天，既未延长也未缩短。'


def fake_numeric(status='refuted'):
    return Fake(extract(EN, 'en_p1'), extract(ZH, 'zh_p1'), match(),
                review('FAIL', [issue()]),
                arb('PASS', kind='none', en=['en_p1'], zh=['zh_p1']),
                recheck(status))


def test_v2_num03_only_passes_with_proven_equivalence_and_issue_refutation():
    fake = fake_numeric()
    out = Checker(fake, config()).check(EN, ZH, mode='dual')
    assert out.initial_verdict == 'PASS'
    assert out.review.verdict == 'FAIL'
    assert out.arbitration.verdict == 'PASS'
    assert out.verdict == 'PASS'
    assert out.deterministic_issues == []
    assert out.dispute_recheck.issues[0].status == 'refuted'
    assert len(fake.calls) == 6
    last = fake.calls[-1]
    assert last['schema'] == 'paragraph_dispute_recheck'
    assert last['model'] == 'independent'
    assert last['user']['review_issues'][0]['issue_index'] == 0
    cert = last['user']['verified_formal_equivalences'][0]
    assert cert['canonical_value'] == '42' and cert['dimension'] == 'duration_days'
    assert cert['comparator'] == '='


@pytest.mark.parametrize('status', ['sustained', 'uncertain'])
def test_unresolved_numeric_objection_is_review_required_not_execution_error(status):
    out = Checker(fake_numeric(status), config()).check(EN, ZH, mode='dual')
    assert out.verdict == 'REVIEW_REQUIRED'
    assert out.dispute_recheck.issues[0].status == status
    assert out.initial_verdict == 'PASS'


def test_model_refuting_archive_vs_delete_cannot_auto_pass_without_proof():
    en, zh = 'The original file must be archived.', '原始文件可以删除。'
    fake = Fake(extract(en, 'en_p1'), extract(zh, 'zh_p1'), match(),
                review('FAIL', [issue(reason='archiving contradicted by deletion')]),
                arb('PASS', kind='none'), recheck('refuted', basis='other'))
    out = Checker(fake, config()).check(en, zh, mode='dual')
    assert out.verdict == 'REVIEW_REQUIRED'
    assert fake.calls[-1]['user']['verified_formal_equivalences'] == []


def test_paragraph_certificate_requires_same_value_and_source_comparator():
    en, zh = 'The survey ran for exactly six weeks.', '问卷调查持续了43天。'
    en_units = [KeyUnit(id='en_1', kind='constraint', evidence=en, claim=en, paragraph_ids=['en_p1'])]
    zh_units = [KeyUnit(id='zh_1', kind='constraint', evidence=zh, claim=zh, paragraph_ids=['zh_p1'])]
    links = [Link(en_ids=['en_1'], zh_ids=['zh_1'], relation='equivalent', reason='mock')]
    assert not numeric_equivalence_certificates(en_units, zh_units, links,
                                                build_paragraphs(en, 'en'), build_paragraphs(zh, 'zh'))
    # Even if model claim changes only the numeric value to hide the mismatch,
    # independently parsed original paragraph vetoes the certificate.
    zh_units[0].claim = '问卷调查持续了42天。'
    assert not numeric_equivalence_certificates(en_units, zh_units, links,
                                                build_paragraphs(en, 'en'), build_paragraphs(zh, 'zh'))


def test_certificate_does_not_infer_equivalence_from_different_bounds():
    en, zh = 'More than 10% missing.', '缺失率达到10%或以上。'
    en_units = [KeyUnit(id='en_1', kind='constraint', evidence=en, claim=en, paragraph_ids=['en_p1'])]
    zh_units = [KeyUnit(id='zh_1', kind='constraint', evidence=zh, claim=zh, paragraph_ids=['zh_p1'])]
    links = [Link(en_ids=['en_1'], zh_ids=['zh_1'], relation='equivalent', reason='mock')]
    assert not numeric_equivalence_certificates(en_units, zh_units, links,
                                                build_paragraphs(en, 'en'), build_paragraphs(zh, 'zh'))


def test_refutation_must_cover_both_sides_and_assessment_indexes_exact():
    issues = [ParagraphReviewIssue.model_validate(issue())]
    good = recheck('refuted')
    _verify_objection_recheck(good, issues, build_paragraphs(EN, 'en'), build_paragraphs(ZH, 'zh'))
    with pytest.raises(LLMError, match='双语'):
        _verify_objection_recheck(recheck('refuted', zh=[]), issues,
                                  build_paragraphs(EN, 'en'), build_paragraphs(ZH, 'zh'))
    with pytest.raises(LLMError, match='索引'):
        _verify_objection_recheck(ParagraphDisputeRecheckDraft.model_validate({
            'issues': [{'issue_index': 1, 'status': 'sustained', 'basis':'other', 'reason': 'invalid'}],
            'reason': 'wrong index'}), issues,
            build_paragraphs(EN, 'en'), build_paragraphs(ZH, 'zh'))


def test_arbiter_unsure_returns_review_required_without_extra_calls():
    fake = Fake(extract(EN, 'en_p1'), extract(ZH, 'zh_p1'), match(),
                review('FAIL', [issue()]), arb('UNSURE'))
    out = Checker(fake, config()).check(EN, ZH, mode='dual')
    assert out.verdict == 'REVIEW_REQUIRED'
    assert out.dispute_recheck is None
    assert len(fake.calls) == 5


def test_evaluation_review_required_not_counted_as_error_or_false_fail():
    rows = [
        {'id': 'manual', 'category': 'x', 'expected': 'PASS',
         'predicted': 'REVIEW_REQUIRED', 'correct': False},
        {'id': 'correct', 'category': 'x', 'expected': 'PASS',
         'predicted': 'PASS', 'correct': True},
        {'id': 'execution', 'category': 'x', 'expected': 'FAIL',
         'predicted': 'ERROR', 'correct': False, 'error_stage': 'schema'},
    ]
    s = summarize(rows)
    assert (s['total'], s['evaluated'], s['review_required'], s['errors'], s['correct']) == (3, 1, 1, 1, 1)
    assert s['false_fail'] == 0 and s['false_pass'] == 0
    assert s['review_required_ids'] == ['manual']
    assert s['error_ids'] == ['execution']
    assert s['by_category']['x']['review_required'] == 1


def test_numerical_proof_cannot_suppress_non_numerical_disagreement():
    # Even an exact 6 weeks / 42 days fact must NOT discharge an unrelated
    # archiving-versus-deleting objection in the same cited paragraph.
    en = 'The survey ran for exactly six weeks. Its source file must be archived.'
    zh = '调查持续42天。调查源文件可以删除。'
    fake = Fake(extract(en, 'en_p1'), extract(zh, 'zh_p1'), match(),
                review('FAIL', [issue(reason='archiving versus deletion')]),
                arb('PASS', kind='none'), recheck('refuted', basis='other'))
    out = Checker(fake, config()).check(en, zh, mode='dual')
    assert out.verdict == 'REVIEW_REQUIRED'


def test_numeric_recheck_refuted_other_basis_only_on_isolated_precision_dispute():
    # v0.2.7.4 LM Studio reviewer consistently called this basis='other'.
    fake = Fake(extract(EN, 'en_p1'), extract(ZH, 'zh_p1'), match(),
                review('FAIL', [issue()]), arb('PASS', kind='none'),
                recheck('refuted', basis='other'))
    assert Checker(fake, config()).check(EN, ZH, mode='dual').verdict == 'PASS'


def test_recheck_schema_retry_uses_correct_protocol_not_legacy_issue_arrays():
    """Regression for the v0.2.7 extraction-specific retry mixup."""
    import json
    import httpx
    from app.config import Settings
    from app.llm import ChatClient

    calls = []
    first = {
        'issues': [{'issue_index': 0, 'status': 'refuted',
                    'en_paragraph_ids': ['en_p1'], 'zh_paragraph_ids': ['zh_p1'],
                    'reason': 'missing basis field'}],
        'reason': 'first response does not conform',
    }
    second = recheck('refuted').model_dump()

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'choices': [{
            'message': {'content': json.dumps(first if len(calls) == 1 else second)},
            'finish_reason': 'stop'}]})

    transport = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient(Settings(json_schema=True), transport)
    result = client.structured('arbitrary-model', 'system', '{}',
                               ParagraphDisputeRecheckDraft, 'paragraph_dispute_recheck')
    assert result.issues[0].basis == 'formal_numeric'
    assert len(calls) == 2
    hint = calls[1]['messages'][-1]['content']
    assert 'basis' in hint and 'issue_index' in hint
    assert 'resolved_missing' not in hint
    transport.close()
