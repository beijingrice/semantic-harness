"""只检查数据集格式和统计逻辑，不调用 LM Studio。"""
from collections import Counter
from pathlib import Path

import pytest

from examples.evaluate import load_cases, summarize

CASES_DIR = Path(__file__).resolve().parents[1] / 'examples'


def test_both_formats_same_data():
    data_json = load_cases(CASES_DIR / 'validation_60.json')
    data_jsonl = load_cases(CASES_DIR / 'validation_60.jsonl')
    assert data_json == data_jsonl
    assert len(data_json) == 60
    assert Counter(item['expected'] for item in data_json) == {'PASS': 30, 'FAIL': 30}
    assert len({item['category'] for item in data_json}) == 7
    assert all(len(item['en']) <= 6000 and len(item['zh']) <= 6000 for item in data_json)


def test_evaluate_statistics():
    rows = [
        {'id':'a', 'category':'cat', 'expected':'FAIL','predicted':'PASS','correct':False},
        {'id':'b', 'category':'cat', 'expected':'PASS','predicted':'FAIL','correct':False},
        {'id':'c', 'category':'cat', 'expected':'PASS','predicted':'PASS','correct':True},
        {'id':'d', 'category':'cat', 'expected':'FAIL','predicted':'ERROR','correct':False},
    ]
    s = summarize(rows)
    assert s['errors'] == 1
    assert s['accuracy'] == 0.3333
    assert s['false_pass'] == 1
    assert s['false_pass_rate'] == 1.0
    assert s['false_fail'] == 1
    assert s['false_fail_rate'] == 0.5
    assert s['mismatched_ids'] == ['a','b']


def test_invalid_cases(tmp_path):
    p=tmp_path/'bad.json'
    p.write_text('[{"id":"1","en":"x","zh":"甲","expected":"MAYBE"}]', encoding='utf-8')
    with pytest.raises(ValueError, match='PASS/FAIL'):
        load_cases(p)


def test_five_error_cases_and_error_stage_stats():
    failures = load_cases(CASES_DIR / "errors_5.json")
    assert {r["id"] for r in failures} == {
        "BAS-01", "AMB-01", "TEC-02", "MOD-02", "AMB-05"
    }
    assert {r["expected"] for r in failures} == {"PASS"}
    report = summarize([
        {"id": "a", "category": "cat", "expected": "PASS", "predicted": "ERROR",
         "correct": False, "error_stage": "extract_english_fallback"},
        {"id": "b", "category": "cat", "expected": "PASS", "predicted": "ERROR",
         "correct": False, "error_stage": "alignment_evidence_check"},
    ])
    assert report["errors_by_stage"] == {
        "alignment_evidence_check": 1, "extract_english_fallback": 1,
    }
