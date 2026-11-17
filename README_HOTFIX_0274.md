# v0.2.7.4 — Verified Dispute Recheck and REVIEW_REQUIRED

## Why

The real v0.2.7.3 regression stopped at `V2-NUM-03` after four correct
cases. Both original phrases represent the same **exact 42-day duration**,
but the independent reviewer reported a non-substantive precision objection.
The existing paragraph safety veto returned an execution ERROR after the
arbiter correctly proposed PASS. That conflated a *semantic dispute* with an
*API/runtime failure*.

## New, model-agnostic behavior

For initial PASS / reviewer FAIL / arbiter PASS, and only in the default
`LM_EVIDENCE_MODE=paragraph` pipeline:

1. Build a strictly narrow formal numeric certificate from the original
   paragraphs AND their respective extracted claims. Certificates exist only
   for one-to-one semantically linked facts containing exactly one interpretable
   quantity on each side; all four representations must have identical
   normalized value, dimension and comparator. For example,
   `exactly six weeks` and `42天` both normalize to `= 42 duration_days`.
   A certificate does **not** establish equivalence of other facts in those
   paragraphs.
2. Send the **original** sources, each numbered reviewer objection, the
   proposed arbitration and any numeric certificates to the independently
   configured verifier for one targeted, structured recheck. The verifier
   must return an assessment for each issue index with `status`, `basis`,
   source IDs, and an explanation.
3. Return PASS only when EVERY original reviewer issue is explicitly
   `refuted` with a `formal_numeric` basis, real matching source IDs,
   AND a Python-verified certificate covering those sources. This is still
   a conservative heuristic, not a formal proof of semantic entailment.
   Semantic-only refutations (`basis=other`), outstanding disputes, or
   uncertain verdicts return `REVIEW_REQUIRED`.
4. A review/arbitration disagreement that cannot be decided is a distinct
   public verdict, **not** an execution ERROR or a fabricated semantic FAIL.
   True HTTP, Schema, parsing, or grounding failures remain ERROR.

No extra model calls for ordinary cases that do not trigger this precise
three-way disagreement. One additional verifier call is required when it
occurs. Prompts never name Qwen, Ministral, or LM Studio. The old `quote`
mode is intentionally left unchanged for A/B comparison.

## User-facing interfaces and evaluation

- `CheckResult.verdict`: PASS | FAIL | REVIEW_REQUIRED.
- `CheckResult.dispute_recheck`: optional per-issue record; absent when no
  targeted recheck was run.
- `/api/check`: returns HTTP 200 for a completed REVIEW_REQUIRED judgment.
- Web UI: REVIEW_REQUIRED shown separately in amber rather than incorrectly
  displaying a red FAIL.
- CLI exit codes: 0 PASS, 1 FAIL, 2 runtime ERROR, 3 REVIEW_REQUIRED.
- `examples.evaluate`: `review_required` and `review_required_ids` separate
  from `errors` / `error_ids`. Correctness statistics include only automatic
  PASS/FAIL outcomes; also inspect **correct / total** for end-to-end utility.
- `scripts/run_local_regressions.sh`: evaluates 11 known anomalies and 12
  constraint cases, stops on FIRST runtime ERROR, saves report. A manual
  review does not trip the execution-error kill switch.

## Offline validation

Uses 171 pytest tests including two end-to-end HTTP JSON Schema MockTransport
scenarios: (a) Qwen-like empty content / JSON in reasoning plus a formally
resolved numeric dispute; (b) real archive-vs-delete conflict requiring
manual adjudication. No claim of running real local model inference here.

For a low-cost local smoke test after setup:

```bash
python -m pytest -q
python -m examples.evaluate --cases examples/v2_regression_11.json \
  --limit 5 --mode dual --stop-after-errors 1 \
  --out eval_v0274_first5.json
```

If PASS is not recovered for `V2-NUM-03`, expect REVIEW_REQUIRED rather
than ERROR; inspect `detail.dispute_recheck` and the review objection before
modifying model behavior or broadening the gate. This is a deliberate
fail-closed safety rule, not a promised 11/11 perfect model accuracy.
