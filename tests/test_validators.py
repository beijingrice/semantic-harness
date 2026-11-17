from app.models import KeyUnit, Link
from app.validators import clear_numeric_conflicts, numeric_tokens


def test_digits_and_chinese_numbers():
    assert numeric_tokens("at most 20 pages") == [20]
    assert numeric_tokens("不得超过二十页") == [20]
    assert numeric_tokens("至少五天") == [5]
    assert numeric_tokens("达到60%") == [60]


def test_no_false_conflict_if_identical():
    en = [KeyUnit(id="en_1", kind="constraint", claim="twenty pages", evidence="20 pages")]
    zh = [KeyUnit(id="zh_1", kind="constraint", claim="不超过20", evidence="二十页")]
    links = [Link(en_ids=["en_1"], zh_ids=["zh_1"], relation="equivalent", reason="same")]
    assert not clear_numeric_conflicts(en, zh, links)
