"""Compare two evaluation JSON reports on their COMMON successful case IDs.

Usage:
 python -m examples.compare_reports --old /path/to/eval_60_v02.json --new eval_60_v021_dual.json
The intersection is essential: comparing 51-case accuracy against 57-case accuracy
without controlling for ERROR exclusions is misleading.
"""

import argparse
import json
from pathlib import Path

from examples.evaluate import summarize


def parse(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("cases"), list):
        raise ValueError(f"{path}: missing cases list")
    return {item["id"]: item for item in data["cases"]}


def compare(old: dict, new: dict) -> dict:
    same = sorted(set(old) & set(new))
    both = [
        key
        for key in same
        if old[key].get("predicted") in ("PASS", "FAIL") and new[key].get("predicted") in ("PASS", "FAIL")
    ]
    for key in both:
        if old[key]["expected"] != new[key]["expected"]:
            raise ValueError(f"Dataset labels changed for {key}; comparison invalid")
    old_group = [old[key] for key in both]
    new_group = [new[key] for key in both]

    def brief(items):
        return {
            k: v
            for k, v in summarize(items).items()
            if k in ("total", "accuracy", "correct", "false_pass", "false_pass_rate", "false_fail", "false_fail_rate")
        }

    changed = [
        {"id": key, "expected": old[key]["expected"], "old": old[key]["predicted"], "new": new[key]["predicted"]}
        for key in both
        if old[key]["predicted"] != new[key]["predicted"]
    ]
    return {
        "old_total": len(old),
        "new_total": len(new),
        "shared_case_ids": len(same),
        "both_evaluated": len(both),
        "old": brief(old_group),
        "new": brief(new_group),
        "improvements": [x for x in changed if x["new"] == x["expected"]],
        "regressions": [x for x in changed if x["old"] == x["expected"]],
    }


def main():
    parser = argparse.ArgumentParser(description="按相同且都成功执行的样例，对比新旧模型评测")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    report = compare(parse(args.old), parse(args.new))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
