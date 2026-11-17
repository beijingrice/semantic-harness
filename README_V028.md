# v0.2.8 — Reliability & Generalization

## What changed

The previously verified v0.2.7.5 paragraph-first semantic pipeline, dual-model reviewer, Numeric Guard and dispute resolution remain intact. No new category-specific PASS/FAIL rules have been added.

### Protocol resilience

- Existing `json_schema` initial request and complete JSON recovered from `reasoning_content` remain supported.
- Only for `finish_reason=length`, missing content or empty content without a safe JSON object: one **alternate prompt-only completion** (original `response_format=json_schema` removed, instructions demand concise complete JSON). No automatic increase in `LM_MAX_TOKENS`.
- The fallback MUST satisfy the *same Pydantic schema* and original paragraph/source-ID validation. A failure returns an execution `ERROR`, never semantic `FAIL` or `PASS`.
- Ordinary schema/grounding correction remains the original up-to-three-attempt mechanism. Network errors and HTTP failures are not retried automatically. No uncontrolled retry loops.
- `LM_GENERATION_RECOVERY=true` default; set `false` in `.env` to restore prior immediate generation-failure behavior. The alternative prompt-only mode is an experiment, **not a claim of LM Studio support**. It could also exhaust the model's reasoning budget, and it may consume one extra max-token generation. If frequent, evaluate/model settings need re-examination rather than further retries.

### Observability and evaluation

- Each `examples.evaluate` case adds `requests` with stage, schema, model, attempt, schema mode, recovery flag, HTTP status, finish reason, token usage (including reasoning tokens if provider reports them), response length, duration, source (`content` or `reasoning_json`), outcome/error type. **Never stores raw reasoning or complete prompts.**
- Summary `reliability` contains auto-decision coverage, end-to-end correct rate, execution ERROR rate, REVIEW_REQUIRED rate, P50/P95 case latency, per-stage request latency, total provider-reported tokens, recovery outcomes, and source reference integrity.
- `error_kind` and `errors_by_kind` distinguish failed-stage symptoms. No guesswork if provider omits usage.
- Source integrity checks prove only that IDs point to actual original paragraphs and link IDs exist. They **do not** establish semantic entailment or correctness of quoted reasons.
- Opt-in `--audit-csv` exports a row per case with links/reasons and blank manual quality fields for review.

### New datasets

- `examples/validation_v028_reliability_smoke_4.json`: four inexpensive real-model end-to-end examples.
- `examples/validation_v028_holdout_36.json`: **new and unrun** 36 examples, 18 PASS and 18 FAIL, six categories. It was fixed before running any LLM test and never used to tune the code. SHA-256 stored alongside in `.sha256`. This is a one-time fresh evaluation set, not a guarantee of statistical or real-world representativeness. **Freeze version 0.2.8, run it once, report all outcomes honestly.** Do not tune on holdout and then quote the same score as an unbiased estimate.
- `examples/probe_generation.py`: opt-in **four API calls** (two short source texts × Schema ON/OFF) to compare finish reasons and token usage. It runs no long validation set; n=1 per condition, so only diagnostic, not a statistically valid A/B claim.

## Phase 1 — Install (no automatic tests)

Download ZIP into `~/Downloads`:

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.8.zip
cd bilingual-keyword-checker-v0.2.8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp ../bilingual-keyword-checker-v0.2.7.5/.env .env
```

If v0.2.7.5 config is unavailable, use `cp .env.example .env` and configure model IDs. When inheriting an old `.env`, the new `LM_GENERATION_RECOVERY` is enabled by default, even if its line is missing.

## Exec — one dataset at a time

Low-cost smoke:

```bash
python -m examples.evaluate \
  --cases examples/validation_v028_reliability_smoke_4.json \
  --mode dual --stop-after-errors 1 \
  --out eval_v028_smoke_4.json
```

Historical regression, keep v0.2.7.5 baseline:

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual --stop-after-errors 1 \
  --out eval_v028_regression_11.json
```

New unrun evaluation **after freezing all code/configuration**:

```bash
python -m examples.evaluate \
  --cases examples/validation_v028_holdout_36.json \
  --mode dual --stop-after-errors 1 \
  --audit-csv eval_v028_holdout_36_audit.csv \
  --out eval_v028_holdout_36.json
```

Optional diagnostic: four short completion calls, separate from dataset testing:

```bash
python -m examples.probe_generation --out probe_v028_generation.json
```

If needed, the developer's offline suite is `python -m pytest -q`, but it is intentionally **not** part of the installation or execution commands.

## Known limitations

1. Model's thinking-token allocation is controlled by its server/runtime. Prompt-only alternate output does not guarantee shorter reasoning; it trades one bounded extra call for possible recovery. Each request retains its configured timeout and max token budget. Backend-side compute might continue even if the client times out.
2. `max_tokens=4096` remains unchanged from the user's previous run. v0.2.8 is not validated on the user's machine until their actual tests return.
3. Metrics from 4 or 36 synthetic examples are small-sample diagnostics; confidence intervals and independent real-document evaluation remain future work.
4. Structural evidence is not a logical entailment proof. A final correct verdict can still contain flawed match explanations; manual CSV review is essential for measuring reason-level precision.
5. Numeric and polarity guards are conservative, intentionally incomplete, and may not handle every phrasing. Their correctness must be evaluated without turning fresh holdout failures into special-case patches.
