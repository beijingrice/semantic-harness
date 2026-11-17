"""Inspect raw LM Studio model responses for the v0.2.6 pipeline.

Does not modify pipeline logic, prompts, models, or validation logic.
Never writes outbound user prompts, HTTP authorization headers, or API keys.
The model's reply may quote confidential input. Trace files stay on your machine.

python -m examples.diagnose --cases examples/errors_5.json --out debug_v026.jsonl
LM_JSON_SCHEMA=true python -m examples.diagnose --cases examples/errors_5.json --out debug_v026_schema.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.llm import ChatClient, LLMError, parse_json_object
from app.pipeline import Checker
from examples.evaluate import load_cases

STAGE_MARKERS = (
    ("alignment_evidence_check", "focused BILINGUAL EVIDENCE CHECKER"),
    ("dispute_arbitration", "bilingual factual ADJUDICATOR"),
    ("independent_review", "INDEPENDENT bilingual key-FACT reviewer"),
    ("keyword_matching", "Compare atomic ENGLISH and CHINESE"),
    ("keyword_extraction_fallback", "targeted SECOND-PASS FACT INVENTORY"),
    ("keyword_extraction_fallback", "independently inventory the KEY FACTS"),
    ("coverage_audit", "auditing a SINGLE source-language proposition inventory"),
    ("atomic_refinement", "refine an existing SOURCE-ONLY fact inventory"),
    ("keyword_extraction", "SOURCE-ONLY SEMANTIC PROPOSITION EXTRACTOR"),
    ("keyword_extraction", "SOURCE-ONLY KEY FACTS"),
    ("keyword_extraction", "ATOMIC KEY INFORMATION"),
)


def stage_from_request(payload: dict[str, Any]) -> str:
    schema = payload.get("response_format", {}).get("json_schema", {})
    if schema.get("name"):
        return str(schema["name"])
    messages = payload.get("messages") or []
    text = messages[0].get("content", "") if messages else ""
    for stage, marker in STAGE_MARKERS:
        if marker in text:
            return stage
    return "unknown"


def parse_summary(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        return {"parse_ok": False, "parse_error": f"content is {type(content).__name__}"}
    try:
        parsed = parse_json_object(content)
        return {"parse_ok": True, "top_level_keys": sorted(parsed.keys())}
    except LLMError as exc:
        return {"parse_ok": False, "parse_error": str(exc)}


class TracedHttp:
    """Transparent HTTP wrapper: records LOCAL model output, never changes responses."""

    def __init__(self, inner: httpx.Client, path: Path, preview_chars: int = 1200):
        self.inner = inner
        self.path = path
        self.preview_chars = preview_chars
        self.case_id = "unknown"
        self.sequence = 0

    def get(self, *args, **kwargs):
        return self.inner.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        self.sequence += 1
        payload = kwargs.get("json") or {}
        event: dict[str, Any] = {
            "case_id": self.case_id,
            "sequence": self.sequence,
            "stage": stage_from_request(payload),
            "model": payload.get("model"),
            "json_schema_enabled": bool(payload.get("response_format")),
            "max_tokens": payload.get("max_tokens"),
            "message_count": len(payload.get("messages") or []),
        }
        try:
            response = self.inner.post(*args, **kwargs)
            event["http_status"] = response.status_code
            try:
                data = response.json()
                choices = data.get("choices") or []
                choice = choices[0] if choices else {}
                message = choice.get("message") or {}
                content = message.get("content")
                event["finish_reason"] = choice.get("finish_reason")
                event["message_keys"] = sorted(message)
                event["content_type"] = type(content).__name__
                event["content_length"] = len(content) if isinstance(content, str) else None
                event["content_preview"] = content[:self.preview_chars] if isinstance(content, str) else None
                reasoning_content = message.get("reasoning_content")
                event["reasoning_content_length"] = len(str(reasoning_content or ""))
                event["reasoning_length"] = len(str(message.get("reasoning") or ""))
                if isinstance(reasoning_content, str) and reasoning_content:
                    event["reasoning_json_parse"] = parse_summary(reasoning_content)
                else:
                    event["reasoning_json_parse"] = None
                event["usage"] = data.get("usage")
                event.update(parse_summary(content))
                if response.is_error:
                    event["error_body_preview"] = response.text[:self.preview_chars]
            except (ValueError, TypeError) as exc:
                event["http_parse_error"] = str(exc)
                event["body_preview"] = response.text[:self.preview_chars]
            return response
        except Exception as exc:
            event["transport_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.2.6 model output diagnostic trace (does NOT change results)")
    parser.add_argument("--cases", type=Path, default=Path("examples/errors_5.json"))
    parser.add_argument("--out", type=Path, default=Path("debug_v026.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--id", action="append", dest="case_ids", help="run only matching case ID; repeatable")
    parser.add_argument("--mode", choices=("dual", "primary"), default="dual")
    parser.add_argument("--extraction-prompt", choices=("legacy", "proposition"))
    parser.add_argument("--coverage-audit", choices=("on", "off"))
    parser.add_argument("--atomic-refinement", choices=("on", "off"))
    parser.add_argument("--preview-chars", type=int, default=1200,
                        help="number of characters of raw model reply to save per request")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.case_ids:
        cases = [case for case in cases if case["id"] in set(args.case_ids)]
    if not cases:
        print("No matching cases", file=sys.stderr)
        return 2
    if args.limit > 0:
        cases = cases[:args.limit]
    settings = Settings()
    settings = replace(settings,
        extraction_prompt=args.extraction_prompt or settings.extraction_prompt,
        coverage_audit=(args.coverage_audit == "on") if args.coverage_audit else settings.coverage_audit,
        atomic_refinement=(args.atomic_refinement == "on") if args.atomic_refinement else settings.atomic_refinement)
    args.out.unlink(missing_ok=True)
    print(f"Trace: {args.out.resolve()}")
    print(f"Effective LM_JSON_SCHEMA={settings.json_schema}; max_tokens={settings.max_tokens}; "
          f"mode={args.mode}; extraction={settings.extraction_prompt}; audit={settings.coverage_audit}; "
          f"atomic={settings.atomic_refinement}")
    print("Trace includes excerpts of MODEL OUTPUT; review before sharing if your source texts are private.")
    inner = httpx.Client(timeout=settings.timeout_seconds)
    trace = TracedHttp(inner, args.out, max(0, args.preview_chars))
    client = ChatClient(settings, http_client=trace)
    try:
        checker = Checker(client, settings)
        for item in cases:
            trace.case_id = item["id"]
            start = time.monotonic()
            try:
                result = checker.check(item["en"], item["zh"], mode=args.mode)
                print(f"{item['id']}: {result.verdict} (expected {item['expected']}) "
                      f"{time.monotonic()-start:.1f}s")
            except LLMError as exc:
                print(f"{item['id']}: ERROR stage={getattr(exc, 'stage', 'unknown')} "
                      f"{time.monotonic()-start:.1f}s: {exc}")
    finally:
        client.close()
        inner.close()
    print("\nSummary of malformed responses:")
    for line in args.out.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not item.get("parse_ok", True) or item.get("finish_reason") == "length":
            print(f"  {item['case_id']} #{item['sequence']} {item['stage']} "
                  f"model={item['model']} finish={item.get('finish_reason')} "
                  f"content_len={item.get('content_length')} "
                  f"reasoning_len={item.get('reasoning_content_length')} "
                  f"parse={item.get('parse_error', 'OK')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
