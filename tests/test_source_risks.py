"""Source-only helpers used by the direct_hardened pipeline.

Kept from the removed v0.2.11 (direct_grounded) and v0.2.12 (direct_verified)
test suites: these helpers are still on the direct_hardened code path.
"""
import json
from pathlib import Path

from app.source_risks import source_risks, source_spans


def test_source_risk_triage_never_pairs_by_position_or_raw_number_order():
    en = 'The appendix may be 6 pages. The main text may be 30 pages.'
    zh = '正文可为30页。附录可为6页。'
    assert source_risks(en, zh) == []


def test_risk_scanner_does_not_warn_on_validated_positive_cases():
    for fn in ('v2_regression_11.json', 'validation_v028_holdout_36.json'):
        data = json.loads(Path('examples', fn).read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = data['cases']
        assert not [(r['id'], source_risks(r['en'], r['zh'])) for r in data
                    if r['expected'] == 'PASS' and source_risks(r['en'], r['zh'])]


def test_source_span_ids_always_refer_to_unmodified_source():
    sample = 'A 3.5% threshold applies. A different clause follows; but keep the context.'
    options = source_spans(sample, 'en')
    assert options[0].id == 'en_all' and options[0].text == sample
    assert all(span.text in sample for span in options)
    assert '3.5%' in options[1].text
    assert options[1].text.startswith('A 3.5%')  # Do not split decimal 3.5


def test_source_grounded_evidence_lists_do_not_assume_equal_clause_positions():
    en = 'The report may contain figures. The appendix must be uploaded.'
    zh = '附录必须上传。报告可以包含插图。'
    e, z = source_spans(en, 'en'), source_spans(zh, 'zh')
    assert e[1].text != z[1].text
    assert e[0].text == en and z[0].text == zh
