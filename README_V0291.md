# v0.2.9.1 — Pre-generation Short Clause Fast Path

Targeted fix for the **150.57 s** `H28-13` / `V29-SMOKE-04` real LM Studio run.
The provided logs showed `extract_english` 24.4 s, `extract_chinese` 88.6 s
truncated at 4095 generated tokens, followed by a 30.8 s prompt-only recovery.
The existing post-generation deduplication and compaction could not recover this time.

## Change summary

1. **Pre-generation source anchors** (`app/extraction_fast_path.py`): when BOTH
   English and Chinese are each exactly one short original source paragraph
   containing a single cohesive clause, construct one verbatim grounded source
   `KeyUnit` on each side **before making an extraction LLM request**. Skip the
   two extraction requests; keep the original bilingual text for `keyword_matching`
   and `independent_review`, and retain all deterministic guard and arbitration
   logic unchanged. A valid paragraph ID is still NOT proof of semantic truth.
2. **Fail conservative on uncertainty**: do NOT fast-path multi-sentence,
   multi-paragraph, lists, coordinated independent duties, repeated modal verbs,
   or other complicated structures. Such inputs retain model extraction.
   The heuristic is deliberately narrower than all short text. This avoids
   silently losing an omitted clause; it is not a complete grammar parser.
3. **Short-source extraction budget**: when the Fast Path is enabled but the
   input fails eligibility and is ONE source paragraph of <=220 chars,
   extraction gets a 1536-token cap (configurable). This applies also to the
   one prompt-only recovery request. Longer / multi-paragraph extractions,
   Matching, independent review and arbitration still use `LM_MAX_TOKENS`.
4. **Opt-out and A/B**: `.env` `LM_EXTRACTION_FAST_PATH=false` or CLI
   `--extraction-fast-path off` disable both the optimization and short-source
   token budget to reproduce old v0.2.9 behavior.
5. **Trace integrity**: an optimized case records `source_anchor_fast_path`
   in `diagnostics` and has no `extract_english` / `extract_chinese` HTTP traces.
   No made-up model requests appear in telemetry.

## Safety and test coverage

- v0.2.9 numeric boundary and original-source checks remain enabled, including
  `below 0.8 mm` vs `不得超过0.8毫米` (strict vs non-strict).
- True non-numeric disagreement is **not** automatically PASS even when both
  the matcher and arbiter mistakenly vote PASS: cited reviewer objections with
  no numeric certificate still require human review.
- HTTP integration confirms `H28-13` sends **two** requests (`keyword_matching`,
  `independent_review`), not four. Another integration confirms that ineligible
  parallel duties still send four requests, with 1536/1536/4096/4096 budgets.
- HTTP simulation also tests truncation + recovery with the 1536 cap.
- Offline/simulated tests do not establish latency or accuracy of live LM Studio.
  **Do not claim the 150 s problem is resolved until live validation.**

## Phase 1 — install

```bash
cd ~/PycharmProjects

unzip ~/Downloads/bilingual-keyword-checker-v0.2.9.1.zip
cd bilingual-keyword-checker-v0.2.9.1

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e '.[dev]'
cp ../bilingual-keyword-checker-v0.2.9/.env .env
```

No `pytest` or LLM evaluations are run automatically during installation.
The new flags default ON / 1536 even when the copied old `.env` has no entries.

## Exec — minimal live verification, one command per dataset

```bash
python -m examples.evaluate \
  --cases examples/validation_v0291_smoke_2.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0291_smoke_2.json
```

The two cases are the original `H28-13` exception PASS (150.57 s baseline)
and `H28-02` strict boundary FAIL. After the two-case smoke passes:

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0291_regression_11.json
```

For optional A/B on a particular case, use the `--extraction-fast-path off`
switch instead of modifying code; running A/B will cost extra LM Studio time.
