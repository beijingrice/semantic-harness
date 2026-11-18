"""Compare matched extraction-only reports without pretending count changes prove recall."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="按 ID 对比两份提取报告（不计算语义准确率）")
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--new", required=True, type=Path)
    args = p.parse_args()
    a = json.loads(args.old.read_text(encoding="utf-8"))
    b = json.loads(args.new.read_text(encoding="utf-8"))
    old = {c["id"]: c for c in a["cases"]}
    new = {c["id"]: c for c in b["cases"]}
    common = list(dict.fromkeys([c["id"] for c in a["cases"] if c["id"] in new]))
    print(f"共有样例 {len(common)}，旧版独有 {len(old) - len(common)}，新版独有 {len(new) - len(common)}")
    print(f"{'ID':<12} {'Old EN/ZH':>12} {'New EN/ZH':>12} {'Old/New':>12}")
    for case_id in common:
        o, n = old[case_id], new[case_id]
        oc = f"{len(o.get('english_units', []))}/{len(o.get('chinese_units', []))}"
        nc = f"{len(n.get('english_units', []))}/{len(n.get('chinese_units', []))}"
        print(f"{case_id:<12} {oc:>12} {nc:>12} {o['status']}/{n['status']:>12}")
    for label, source in [("旧", a), ("新", b)]:
        print(label, "config:", source.get("configuration"), "summary:", source.get("summary"))
    print("注意：关键词数量差异并不直接代表正确率或事实召回率；请人工检查证据和遗漏。")


if __name__ == "__main__":
    main()
