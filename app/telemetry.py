"""Evaluation-only reliability metrics. Never inspect or persist model reasoning text."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import Any


def percentile(values: list[float], rank: float) -> float | None:
    """Nearest-rank sample percentile, meaningful even for a one-item smoke run."""
    if not values:
        return None
    sorted_values = sorted(values)
    return round(sorted_values[max(0, ceil(rank * len(sorted_values)) - 1)], 3)


def evidence_integrity(detail: dict[str, Any] | None) -> dict[str, Any]:
    """Validate ID/reference structure only. This CANNOT validate semantic claims/reasons."""
    if not detail or detail.get('evidence_mode') != 'paragraph':
        return {'checked': False, 'structural_ok': None, 'violations': []}
    en_paras = {p['id']: p['text'] for p in detail.get('english_paragraphs', [])}
    zh_paras = {p['id']: p['text'] for p in detail.get('chinese_paragraphs', [])}
    en_units = {u['id']: u for u in detail.get('english_keywords', [])}
    zh_units = {u['id']: u for u in detail.get('chinese_keywords', [])}
    errors = []
    for side, units, paras in [('en', en_units, en_paras), ('zh', zh_units, zh_paras)]:
        for unit in units.values():
            ids = unit.get('paragraph_ids', [])
            if not ids or any(ref not in paras for ref in ids):
                errors.append(f'{side}:invalid_source_id:{unit["id"]}')
            elif unit.get('evidence') not in ('\n\n'.join(paras[ref] for ref in ids),
                                               '\n'.join(paras[ref] for ref in ids),
                                               *[paras[ref] for ref in ids]):
                # Multi-paragraph evidence rendering may vary; verify each
                # original paragraph is present rather than relying on joins.
                evidence = unit.get('evidence', '')
                if not all(paras[ref] in evidence for ref in ids):
                    errors.append(f'{side}:evidence_mismatch:{unit["id"]}')
    for idx, link in enumerate(detail.get('links', [])):
        if not link.get('en_ids') or any(ref not in en_units for ref in link['en_ids']):
            errors.append(f'link_{idx}:invalid_english_unit')
        if not link.get('zh_ids') or any(ref not in zh_units for ref in link['zh_ids']):
            errors.append(f'link_{idx}:invalid_chinese_unit')
    for idx, issue in enumerate(detail.get('review_issues', [])):
        for side, paras in [('en', en_paras), ('zh', zh_paras)]:
            if any(ref not in paras for ref in issue.get(f'{side}_paragraph_ids', [])):
                errors.append(f'issue_{idx}:invalid_{side}_source_id')
    return {'checked': True, 'structural_ok': not errors, 'violations': errors,
            'semantic_correctness_verified': False}


def reliability_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    decided = [r for r in results if r.get('predicted') in ('PASS', 'FAIL')]
    requests = [q for r in results for q in r.get('requests', [])]
    stage = defaultdict(list)
    for q in requests:
        if isinstance(q.get('duration_ms'), (int, float)):
            stage[q['stage']].append(q['duration_ms'])
    latencies = [float(r['elapsed_seconds']) for r in results if 'elapsed_seconds' in r]
    recoveries = [q for q in requests if q.get('recovery')]
    token_total = lambda field: sum(q[field] for q in requests if isinstance(q.get(field), (int, float)))
    integrity = [r.get('evidence_audit', {}) for r in results]
    checked = [i for i in integrity if i.get('checked')]
    return {
        'auto_decision_coverage': round(len(decided) / n, 4) if n else None,
        'end_to_end_correct_rate': round(sum(r.get('correct', False) for r in decided) / n, 4) if n else None,
        'execution_error_rate': round(sum(r.get('predicted') == 'ERROR' for r in results) / n, 4) if n else None,
        'review_required_rate': round(sum(r.get('predicted') == 'REVIEW_REQUIRED' for r in results) / n, 4) if n else None,
        'latency_seconds': {'p50': percentile(latencies, .50), 'p95': percentile(latencies, .95),
                            'max': round(max(latencies), 3) if latencies else None},
        'requests': len(requests),
        'request_outcomes': dict(sorted(Counter(q.get('outcome', 'unspecified') for q in requests).items())),
        'response_sources': dict(sorted(Counter(q.get('response_source', 'unavailable') for q in requests).items())),
        'recoveries_attempted': len(recoveries),
        'recoveries_validated': sum(q.get('outcome') == 'recovered_validated' for q in recoveries),
        'token_usage_reported': {'prompt_tokens': token_total('prompt_tokens'),
                                 'completion_tokens': token_total('completion_tokens'),
                                 'reasoning_tokens': token_total('reasoning_tokens')},
        'stage_duration_ms': {
            name: {'requests': len(times), 'p50': percentile(times, .5), 'p95': percentile(times, .95)}
            for name, times in sorted(stage.items())
        },
        'source_integrity': {
            'checked_cases': len(checked),
            'structurally_valid_cases': sum(i.get('structural_ok') is True for i in checked),
            'violations': [v for i in checked for v in i.get('violations', [])],
            'semantic_reason_quality': 'NOT AUTOMATICALLY VERIFIED; see optional audit CSV',
        },
    }
