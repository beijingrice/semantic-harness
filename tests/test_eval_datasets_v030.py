"""The v0.3.0 generalization sets: format, balance, and the frozen blind file."""

import hashlib
from collections import Counter
from pathlib import Path

from examples.evaluate import load_cases

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_blind_set_is_frozen():
    blind = EXAMPLES / "validation_v030_blind_99.json"
    expected = (EXAMPLES / "validation_v030_blind_99.sha256").read_text().split()[0]
    assert hashlib.sha256(blind.read_bytes()).hexdigest() == expected


def test_v030_sets_are_well_formed_and_balanced():
    blind = load_cases(EXAMPLES / "validation_v030_blind_99.json")
    dev = load_cases(EXAMPLES / "validation_v030_gen_dev_60.json")
    assert len(blind) == 99 and Counter(c["expected"] for c in blind) == {"PASS": 50, "FAIL": 49}
    assert len(dev) == 60 and Counter(c["expected"] for c in dev) == {"PASS": 30, "FAIL": 30}
    assert not {c["id"] for c in blind} & {c["id"] for c in dev}
