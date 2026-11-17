# v0.2.9 — Numeric Semantics & Bounded Extraction

This is an incremental update of v0.2.8, NOT a redesigned harness.
It preserves paragraph grounding, schema validation, independent dual-model
review, guarded arbitration, structured-output recovery and request telemetry.

## Implementation

- Numeric Guard recognizes millimetres/centimetres/metres (including common
  English and Chinese spellings), and minutes/hours/days/weeks. Duration is
  normalized to days with deterministic 12-decimal precision; length to mm.
- Strict vs non-strict comparisons are distinguished, including negative
  phrases such as `must not be below` and `不得低于`. Identical numeric literals
  are **not** automatically considered equivalent bounds.
- Fully specified inclusive/exclusive ranges are parsed as intervals; bare
  ambiguous `between` / `至` intervals remain unverified.
- For one short, single-paragraph source with an explicit linked quantity,
  a conservative original-source numeric check supplements claim-based checks.
  Multi-quantity and multi-paragraph sources abstain from this shortcut.
- Single-source extraction now requests one cohesive claim rather than 20+
  paraphrases of a short sentence. A pathological result (>=9 distinct units,
  source <=220 characters, one source paragraph) becomes one lossless, verbatim
  source-backed claim. This preserves the full original wording and all duties,
  but changes matching granularity; the independent reviewer continues reading
  the full original text. Literal entity/concept subspans are removed only
  when the same source sentence is already fully represented as a claim.
- Existing numerical equivalence certificates now cover hour/day and length
  conversions without automatically dismissing other duties. The existing
  objection/recheck mechanism remains guarded; we did NOT add unconditional
  PASS overrides or default extra LLM calls.

## Empirical scope

- The v0.2.8 known-case results were 11/11 historical regression and 34/36
  holdout (two numeric mistakes). Those **historical figures are not results
  of v0.2.9** and the 36-case suite has become a development set.
- Offline tests include replay of H28-02 model outputs, H28-04 claim bloat,
  and an H28-03 model-output-based pipeline simulation with adjusted IDs.
  The other cases are formal parser/guard safety tests. No v0.2.9 live LM
  Studio execution is claimed.
- Initial regression smoke set: `examples/validation_v029_smoke_6.json`.
  Run 6 before the legacy 11. Do not automatically rerun all 36, unless you
  want an exploratory regression rather than a fresh unbiased estimate.
- Generation fallback remains bounded as in v0.2.8; the live 47-case run did
  not exercise fallback, so its real-world efficacy is still unmeasured.

## Install (from Downloads, after unpacking old v0.2.8)

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.9.zip
cd bilingual-keyword-checker-v0.2.9
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp ../bilingual-keyword-checker-v0.2.8/.env .env
```

## Exec — independent commands (no automatic pytest)

```bash
python -m examples.evaluate \
  --cases examples/validation_v029_smoke_6.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v029_smoke_6.json
```

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v029_regression_11.json
```

Do not interpret 6/6 or 11/11 development cases as independent accuracy.
