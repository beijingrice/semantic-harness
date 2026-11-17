"""用 LM Studio 批量评测人工标注集；兼容标准 JSON 数组及 JSONL。

运行：
    python -m examples.evaluate --cases examples/validation_60.jsonl
    python -m examples.evaluate --cases examples/validation_60.json

primary: 三阶段初判；dual: 追加独立复核，意见不同时额外进行定向仲裁。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any


def paired_reference_comparison(results: list[dict[str, Any]], reference_file: Path) -> dict[str, Any]:
    """Compare identical source cases with a PRIOR run; no extra model calls.

    Historical baselines are not same-time randomized A/B controls.
    """
    ref = json.loads(reference_file.read_text(encoding="utf-8-sig"))
    previous = {r["id"]: r for r in ref["cases"]}
    compared = []
    for current in results:
        other = previous.get(current["id"])
        if other is None:
            raise ValueError(f"基线缺少样例 {current['id']}")
        if (current.get("en"), current.get("zh"), current.get("expected")) != (
                other.get("en"), other.get("zh"), other.get("expected")):
            raise ValueError(f"基线与新版本的原文/标签不一致：{current['id']}")
        compared.append({
            "id": current["id"], "expected": current["expected"],
            "baseline": other.get("predicted"), "direct": current.get("predicted"),
            "baseline_seconds": other.get("elapsed_seconds"),
            "direct_seconds": current.get("elapsed_seconds"),
            "baseline_requests": len(other.get("requests", [])),
            "direct_requests": len(current.get("requests", [])),
        })
    return {
        "kind": "historical_matched_case_comparison_not_contemporaneous_ab",
        "baseline_file": str(reference_file), "cases": len(compared),
        "baseline_correct": sum(x["baseline"] == x["expected"] for x in compared),
        "direct_correct": sum(x["direct"] == x["expected"] for x in compared),
        "baseline_auto_decisions": sum(x["baseline"] in ("PASS", "FAIL") for x in compared),
        "direct_auto_decisions": sum(x["direct"] in ("PASS", "FAIL") for x in compared),
        "baseline_requests": sum(x["baseline_requests"] for x in compared),
        "direct_requests": sum(x["direct_requests"] for x in compared),
        "baseline_total_seconds": round(sum(x["baseline_seconds"] or 0 for x in compared), 2),
        "direct_total_seconds": round(sum(x["direct_seconds"] or 0 for x in compared), 2),
        "changed": [x for x in compared if x["baseline"] != x["direct"]],
        "per_case": compared,
    }


from app.config import Settings
from app.llm import ChatClient, LLMError
from app.pipeline import Checker
from app.telemetry import evidence_integrity, reliability_summary

VALID_LABELS = frozenset({"PASS", "FAIL"})


def load_cases(path: Path) -> list[dict[str, Any]]:
    """.json 接受数组或 {"cases":[...]}；.jsonl 接受每行一个 JSON 对象。"""
    try:
        raw = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            data = json.loads(raw)
            rows = data["cases"] if isinstance(data, dict) and "cases" in data else data
        else:
            raise ValueError("只接受 .json 或 .jsonl 文件")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取样例文件 {path}: {exc}") from exc

    if not isinstance(rows, list) or not rows:
        raise ValueError("样例集必须是非空 JSON 数组或非空 JSONL 数据")

    used_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"样例 #{index} 不是 JSON object")
        for field in ("id", "en", "zh", "expected"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"样例 #{index} 的 {field!r} 必须是非空字符串")
        if row["expected"] not in VALID_LABELS:
            raise ValueError(f"样例 {row['id']} 的 expected 只能是 PASS/FAIL")
        if row["id"] in used_ids:
            raise ValueError(f"重复的样例 id: {row['id']}")
        used_ids.add(row["id"])
    return rows


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Execution ERROR and semantic REVIEW_REQUIRED are separate from PASS/FAIL."""
    evaluated = [r for r in results if r["predicted"] in VALID_LABELS]
    correct = sum(r["correct"] for r in evaluated)
    false_pass = sum(r["expected"] == "FAIL" and r["predicted"] == "PASS" for r in evaluated)
    false_fail = sum(r["expected"] == "PASS" and r["predicted"] == "FAIL" for r in evaluated)
    expected_fail = sum(r["expected"] == "FAIL" for r in evaluated)
    expected_pass = sum(r["expected"] == "PASS" for r in evaluated)
    summary: dict[str, Any] = {
        "total": len(results),
        "evaluated": len(evaluated),
        "errors": sum(r["predicted"] == "ERROR" for r in results),
        "review_required": sum(r["predicted"] == "REVIEW_REQUIRED" for r in results),
        "correct": correct,
        "accuracy": round(correct / len(evaluated), 4) if evaluated else None,
        "false_pass": false_pass,
        "false_pass_rate": round(false_pass / expected_fail, 4) if expected_fail else None,
        "false_fail": false_fail,
        "false_fail_rate": round(false_fail / expected_pass, 4) if expected_pass else None,
    }
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        categories[r.get("category") or "uncategorized"].append(r)
    summary["by_category"] = {}
    for category, group in sorted(categories.items()):
        checked = [r for r in group if r["predicted"] in VALID_LABELS]
        fail_population = sum(r["expected"] == "FAIL" for r in checked)
        fp = sum(r["expected"] == "FAIL" and r["predicted"] == "PASS" for r in checked)
        summary["by_category"][category] = {
            "total": len(group),
            "evaluated": len(checked),
            "errors": sum(r["predicted"] == "ERROR" for r in group),
            "review_required": sum(r["predicted"] == "REVIEW_REQUIRED" for r in group),
            "accuracy": round(sum(r["correct"] for r in checked) / len(checked), 4) if checked else None,
            "false_pass": fp,
            "false_pass_rate": round(fp / fail_population, 4) if fail_population else None,
        }
    summary["errors_by_kind"] = dict(sorted(Counter(
        r.get("error_kind") or "unclassified" for r in results
        if r["predicted"] == "ERROR"
    ).items()))
    summary["mismatched_ids"] = [r["id"] for r in evaluated if not r["correct"]]
    summary["error_ids"] = [r["id"] for r in results if r["predicted"] == "ERROR"]
    summary["review_required_ids"] = [r["id"] for r in results if r["predicted"] == "REVIEW_REQUIRED"]
    summary["errors_by_stage"] = dict(sorted(Counter(
        r.get("error_stage") or "unspecified" for r in results
        if r["predicted"] == "ERROR"
    ).items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="评测双语关键词检测器（JSON/JSONL 数据集）")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("validation_60.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--stop-after-errors", type=int, default=0,
                        help="累计 N 个执行 ERROR 后立即停止，保留部分报告；0 表示不限制")
    parser.add_argument("--model", help="主模型 ID（覆盖 LM_MODEL）")
    parser.add_argument("--verifier-model", help="独立复核模型 ID（覆盖 LM_VERIFIER_MODEL）")
    parser.add_argument("--pipeline", choices=("extraction", "direct_hardened"),
                        help="extraction=默认的提取+匹配流程；direct_hardened=整段双模型比较+固定疑点核查（实验）")
    parser.add_argument("--mode", choices=("primary", "dual"), default="dual",
                        help="primary 只跑主模型；dual 追加独立复核和必要仲裁")
    parser.add_argument("--category", help="只运行指定类型，例：enumeration_completeness")
    parser.add_argument("--extraction-prompt", choices=("legacy", "proposition", "coarse"),
                        help="对照试验：coarse=v0.2.6（默认），legacy=v0.2.2，proposition=v0.2.4")
    parser.add_argument("--extraction-fast-path", choices=("on", "off"),
                        help="v0.2.9.2：符合保守原文锚点条件时跳过生成式提取（默认从 .env）")
    parser.add_argument("--coverage-audit", choices=("on", "off"),
                        help="仅对可疑提取执行原文覆盖审计（默认从 .env）")
    parser.add_argument("--atomic-refinement", choices=("on", "off"),
                        help="历史实验：仅 proposition 模式可选原子化重写（默认关闭）")
    parser.add_argument("--baseline-report", type=Path, default=None,
                        help="可选：仅与已存在的同一批原文历史报告比较，不重复调用旧流程模型")
    parser.add_argument("--out", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--audit-csv", type=Path, default=None,
                        help="可选：导出人工核查证据/链接理由的 CSV，人工填写质量判断")
    args = parser.parse_args()
    try:
        rows = load_cases(args.cases)
    except ValueError as exc:
        print(f"读取样例失败：{exc}", file=sys.stderr)
        sys.exit(2)
    if args.category:
        rows = [r for r in rows if r.get("category") == args.category]
        if not rows:
            print(f"找不到类别：{args.category}", file=sys.stderr)
            sys.exit(2)
    if args.limit > 0:
        rows = rows[:args.limit]
    if args.baseline_report:
        # Validate file and identical source samples BEFORE any expensive model calls.
        try:
            reference = json.loads(args.baseline_report.read_text(encoding="utf-8-sig"))
            by_id = {r["id"]: r for r in reference["cases"]}
            for row in rows:
                old = by_id.get(row["id"])
                if old is None or any(old.get(k) != row.get(k) for k in ("en", "zh", "expected")):
                    raise ValueError(f"基线样例缺失/原文或标签不一致：{row['id']}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"基线报告检查失败：{exc}", file=sys.stderr)
            sys.exit(2)
    results: list[dict[str, Any]] = []
    configured_total = len(rows)
    stopped_early = False
    config = Settings()
    config = replace(config,
        pipeline=args.pipeline or config.pipeline,
        extraction_prompt=args.extraction_prompt or config.extraction_prompt,
        extraction_fast_path=(args.extraction_fast_path == "on") if args.extraction_fast_path
                             else config.extraction_fast_path,
        coverage_audit=(args.coverage_audit == "on") if args.coverage_audit else config.coverage_audit,
        atomic_refinement=(args.atomic_refinement == "on") if args.atomic_refinement else config.atomic_refinement)
    if config.pipeline not in ("extraction", "direct_hardened"):
        parser.error("LM_PIPELINE / --pipeline 只能为 extraction 或 direct_hardened")
    n_calls = (1 if args.mode == "primary" else 2) if config.pipeline == "direct_hardened" else (3 if args.mode == "primary" else 4)
    print(f"准备评测 {len(rows)} 条样例，Pipeline={config.pipeline}，模式={args.mode}；"
          f"基础约 {len(rows)*n_calls} 次模型请求（Direct 实验模式触发定向证据复核时另计）。", flush=True)
    print(f"提取模式={config.extraction_prompt}；覆盖审计={'on' if config.coverage_audit else 'off'}；"
          f"原子化={config.atomic_refinement}；JSON Schema={config.json_schema}；"
          f"Fast Path={config.extraction_fast_path}；短句提取预算={config.extraction_max_tokens}", flush=True)
    client = ChatClient(config)
    try:
        # Fail fast before attempting a potentially expensive 60-case evaluation.
        try:
            primary = client.choose_model(args.model)
            verifier = client.choose_verifier(args.verifier_model) if args.mode == "dual" else ""
            if args.mode == "dual" and primary == verifier:
                raise LLMError("主模型与复核模型不能相同")
            if hasattr(client, "validate_model_access"):
                discovery = client.validate_model_access(primary, verifier)
            else:  # older mocked evaluation clients
                available = client.models()
                for name in ([primary, verifier] if args.mode == "dual" else [primary]):
                    if name not in available:
                        raise LLMError(f"未找到模型 {name!r}，可用模型：{available}")
                discovery = "legacy_required"
            print(f"预检成功：主模型 {primary}" + (f"；复核模型 {verifier}" if verifier else "（不启用复核）"), flush=True)
        except LLMError as exc:
            print(f"模型预检失败：{exc}", file=sys.stderr)
            sys.exit(2)
        checker = Checker(client, client.settings)
        for index, row in enumerate(rows, 1):
            start = time.monotonic()
            if hasattr(client, 'start_trace'):
                client.start_trace()
            base: dict[str, Any] = {
                field: row.get(field) for field in
                ("id", "category", "difficulty", "en", "zh", "expected", "why")
            }
            try:
                check = checker.check(row["en"], row["zh"], primary, verifier, args.mode)
                result = {
                    **base, "predicted": check.verdict,
                    "correct": check.verdict == row["expected"],
                    "detail": check.model_dump(),
                }
            except LLMError as exc:
                result = {**base, "predicted": "ERROR", "correct": False,
                          "error_stage": getattr(exc, "stage", "client_or_pipeline"),
                          "error": str(exc)}
            result["elapsed_seconds"] = round(time.monotonic() - start, 2)
            result["requests"] = client.trace_snapshot() if hasattr(client, 'trace_snapshot') else []
            if result["predicted"] == "ERROR":
                last = result["requests"][-1] if result["requests"] else {}
                result["error_kind"] = last.get("outcome", "client_or_pipeline_error")
            result["evidence_audit"] = evidence_integrity(result.get("detail"))
            results.append(result)
            marker = ("OK" if result["correct"] else
                      "ERROR" if result["predicted"] == "ERROR" else
                      "REVIEW" if result["predicted"] == "REVIEW_REQUIRED" else "MISMATCH")
            print(f"[{index:02d}/{len(rows):02d}] {row['id']} "
                  f"{row.get('category', '')} expected={row['expected']} "
                  f"predicted={result['predicted']}  {marker}  {result['elapsed_seconds']:.1f}s", flush=True)
            if result["predicted"] == "REVIEW_REQUIRED":
                print("    需要人工复核（非执行 ERROR；本例不计入自动裁决准确率）", flush=True)
            if result["predicted"] == "ERROR":
                print(f"    Stage: {result.get('error_stage', 'unknown')}"
                      f"\n    Error: {result.get('error', 'unknown')}", flush=True)
                if args.stop_after_errors > 0 and sum(
                    r["predicted"] == "ERROR" for r in results
                ) >= args.stop_after_errors:
                    stopped_early = index < configured_total
                    print(f"已达到 ERROR 停止阈值 ({args.stop_after_errors})，停止后续模型调用。", flush=True)
                    break
    finally:
        client.close()

    summary = summarize(results)
    summary["reliability"] = reliability_summary(results)
    if args.baseline_report:
        summary["paired_reference"] = paired_reference_comparison(results, args.baseline_report)
    summary["configured_total"] = configured_total
    summary["stopped_early"] = stopped_early
    # True same-cohort A/B: every successfully executed dual case contains a Qwen
    # initial verdict from THAT SAME RUN. Do not compare 51 vs 57 different cases.
    primary_on_same = [
        {**r, "predicted": (r["detail"].get("direct_primary") or {}).get("verdict", r["detail"]["initial_verdict"]),
         "correct": (r["detail"].get("direct_primary") or {}).get("verdict", r["detail"]["initial_verdict"]) == r["expected"]}
        for r in results if r.get("detail") is not None
    ]
    if args.mode == "dual":
        summary["initial_on_same_cases"] = {
            k: v for k, v in summarize(primary_on_same).items()
            if k not in ("by_category", "mismatched_ids", "error_ids", "review_required_ids")
        }
        summary["arbitration_count"] = sum(bool(r.get("detail", {}).get("arbitrated")) for r in results)
        summary["flip_to_pass"] = sum(
            r.get("detail", {}).get("initial_verdict") == "FAIL" and r["predicted"] == "PASS"
            for r in results
        )
        summary["flip_to_fail"] = sum(
            r.get("detail", {}).get("initial_verdict") == "PASS" and r["predicted"] == "FAIL"
            for r in results
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary["coverage_audit_count"] = sum(
        sum("coverage audit triggered" in event for event in r.get("detail", {}).get("diagnostics", []))
        for r in results)
    summary["coverage_audit_added_units"] = sum(
        int(event.split("coverage audit added ", 1)[1].split()[0])
        for r in results
        for event in r.get("detail", {}).get("diagnostics", [])
        if "coverage audit added " in event)
    args.out.write_text(
        json.dumps({"configuration": {"extraction_prompt": config.extraction_prompt,
                                       "coverage_audit": config.coverage_audit,
                                       "json_schema": config.json_schema,
                                       "mode": args.mode,
                                       "pipeline": config.pipeline},
                    "summary": summary, "cases": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_csv:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_csv.open('w', newline='', encoding='utf-8-sig') as output_file:
            fieldnames = ['id', 'expected', 'predicted', 'structural_integrity',
                          'en', 'zh', 'link_reasons', 'review_reasons',
                          'manual_evidence_correct', 'manual_reason_correct', 'manual_notes']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                d = item.get('detail') or {}
                writer.writerow({
                    'id': item['id'], 'expected': item['expected'], 'predicted': item['predicted'],
                    'structural_integrity': item['evidence_audit'].get('structural_ok'),
                    'en': item.get('en', ''), 'zh': item.get('zh', ''),
                    'link_reasons': ' | '.join(x.get('reason', '') for x in d.get('links', [])),
                    'review_reasons': ' | '.join(x.get('reason', '') for x in d.get('review_issues', [])),
                    'manual_evidence_correct': '', 'manual_reason_correct': '', 'manual_notes': '',
                })
        print('人工证据审核表：' + str(args.audit_csv.resolve()))
    if args.baseline_report:
        paired = summary["paired_reference"]
        print("\n历史基线（非同期随机 A/B）对照：" + json.dumps({
            k: v for k, v in paired.items() if k not in ("per_case", "changed")}, ensure_ascii=False))
        print("判决变化：" + json.dumps(paired["changed"], ensure_ascii=False))
    print("\n可靠性指标：" + json.dumps(summary['reliability'], ensure_ascii=False))
    print("\n总览：" + json.dumps({k: v for k, v in summary.items() if k != "by_category"}, ensure_ascii=False))
    print("\n按类别统计：")
    for category, stats in summary["by_category"].items():
        print(f"  {category:<26} accuracy={str(stats['accuracy']):>6} "
              f"false_pass={stats['false_pass']} errors={stats['errors']} review={stats['review_required']}")
    if args.mode == "dual":
        print("\n同批成功样例的 Qwen 初判：" + json.dumps(summary["initial_on_same_cases"], ensure_ascii=False))
        print(f"争议仲裁 {summary['arbitration_count']} 次；FAIL→PASS {summary['flip_to_pass']}，PASS→FAIL {summary['flip_to_fail']}")
    print("\n完整结果：" + str(args.out.resolve()))
    if stopped_early:
        sys.exit(1)


if __name__ == "__main__":
    main()
