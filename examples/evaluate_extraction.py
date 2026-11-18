"""Extraction-only A/B comparison: unit counts and source evidence, NOT accuracy.

Examples:
    python -m examples.evaluate_extraction --cases examples/errors_5.json --prompt legacy --out extraction_legacy.json
    python -m examples.evaluate_extraction --cases examples/errors_5.json --prompt proposition --out extraction_new.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from app.checker import Checker
from app.config import Settings
from app.llm import ChatClient, LLMError
from examples.evaluate import load_cases


def run():
    parser = argparse.ArgumentParser(description="仅测试双语事实提取，不执行匹配与仲裁")
    parser.add_argument("--cases", type=Path, default=Path("examples/errors_5.json"))
    parser.add_argument("--prompt", choices=("legacy", "proposition", "coarse"), default="coarse")
    parser.add_argument("--audit", choices=("on", "off"), default="off")
    parser.add_argument("--atomic-refinement", choices=("on", "off"),
                        help="on/off; default from .env")
    parser.add_argument("--mode", choices=("primary", "dual"), default="dual",
                        help="dual 可使用 Ministral 做空提取 fallback")
    parser.add_argument("--model", help="主模型 ID")
    parser.add_argument("--verifier-model", help="备用模型 ID")
    parser.add_argument("--category", help="只取指定类别")
    parser.add_argument("--id", action="append", dest="ids", help="仅测试指定 id，可重复")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("extraction_report.json"))
    args = parser.parse_args()
    rows = load_cases(args.cases)
    if args.category:
        rows = [r for r in rows if r.get("category") == args.category]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("筛选后无样例")
    original_settings = Settings()
    settings = replace(original_settings, extraction_prompt=args.prompt, coverage_audit=args.audit == "on",
                       atomic_refinement=(args.atomic_refinement == "on") if args.atomic_refinement
                       else original_settings.atomic_refinement)
    client = ChatClient(settings)
    results = []
    try:
        primary = client.choose_model(args.model)
        fallback = client.choose_verifier(args.verifier_model) if args.mode == "dual" else primary
        if hasattr(client, "validate_model_access"):
            client.validate_model_access(primary, fallback if args.mode == "dual" else "")
        else:
            available = client.models()
            if primary not in available or fallback not in available:
                parser.error(f"模型未加载。需要 {primary!r}, {fallback!r}；实际 {available}")
        checker = Checker(client, settings)
        print(f"提取测试：{len(rows)} 条；prompt={args.prompt}；audit={args.audit}；atomic={settings.atomic_refinement}；"
              f"primary={primary}；fallback={fallback}", flush=True)
        for i, row in enumerate(rows, 1):
            started = time.monotonic()
            details = []
            result = {k: row.get(k) for k in ("id", "category", "en", "zh", "expected")}
            try:
                en = checker.extract(row["en"], "English", "en", primary, fallback, details)
                zh = checker.extract(row["zh"], "Chinese", "zh", primary, fallback, details)
                result.update(status="OK", english_units=[u.model_dump() for u in en],
                              chinese_units=[u.model_dump() for u in zh],
                              english_empty=not bool(en), chinese_empty=not bool(zh))
            except LLMError as exc:
                result.update(status="ERROR", error_stage=getattr(exc, "stage", "unknown"),
                              error=str(exc))
            result["diagnostics"] = details
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            results.append(result)
            print(f"[{i:02}/{len(rows):02}] {row['id']} {result['status']} "
                  f"units={len(result.get('english_units', []))}/{len(result.get('chinese_units', []))} "
                  f"{result['elapsed_seconds']:.1f}s", flush=True)
    finally:
        client.close()
    summary = {
        "total": len(results), "errors": sum(r["status"] == "ERROR" for r in results),
        "english_empty": sum(r.get("english_empty") is True for r in results),
        "chinese_empty": sum(r.get("chinese_empty") is True for r in results),
        "audit_triggers": sum("coverage audit triggered" in event
                              for r in results for event in r["diagnostics"]),
        "added_units": sum(int(event.split("coverage audit added ", 1)[1].split()[0])
                           for r in results for event in r["diagnostics"]
                           if "coverage audit added " in event),
        "fallback_calls": sum("fallback model=" in event or "independent fallback model=" in event
                              for r in results for event in r["diagnostics"]),
        "atomic_refinement_triggers": sum("atomic refinement triggered" in event
                                           for r in results for event in r["diagnostics"]),
        "atomic_refinement_successes": sum("atomic refinement replaced umbrella" in event
                                            for r in results for event in r["diagnostics"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"configuration": {"prompt": args.prompt,
                             "audit": args.audit, "atomic_refinement": settings.atomic_refinement,
                             "mode": args.mode},
                             "summary": summary, "cases": results}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("Summary:", json.dumps(summary, ensure_ascii=False))
    print("Only extraction coverage and protocol were measured; NO semantic accuracy label is available.")
    print("Report:", args.out.resolve())


if __name__ == "__main__":
    run()
