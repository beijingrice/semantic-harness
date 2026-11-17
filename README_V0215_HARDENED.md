# v0.2.15 — Hardened Relation Verification (experimental)

## Scope

The new `direct_hardened` pipeline is a surgical continuation of v0.2.14. It preserves complete English and Chinese originals through the independent Qwen primary and Ministral review. It preserves previous `extraction`, `direct`, `direct_grounded`, `direct_verified`, `direct_questions` and `direct_relations` options; **`extraction` remains the default**. There is **no mandatory Claim Extraction, no positional sentence matching**, and no additional normal LLM stage.

## Concrete fixes from the user's v0.2.14 live report

1. **Unicode numeric comparators**. When verifying a narrow number-boundary discrepancy, a model reply such as `≥10%`, `≤10%`, `≧10％`, or full-width `１０％` is normalized to a canonical typed comparator (`>=10%`, `<=10%`) before comparison with **independently parsed source operands**. Inexact expressions, broad ranges, mismatched quantities and spurious changed comparators are *not* accepted. Old v0.2.14 `direct_relations` behavior remains available.
2. **Exception evidence scope**. For a specified target component (currently conservative known bilingual mapping `Chinese abstract` / `中文摘要`), if a model cites `en_all`/`zh_all`, the host can identify **exactly one qualifying, complete exception clause per original language** and perform proof on those original verbatim clauses. Host-selected cite IDs are recorded **separately** from raw model cite IDs in `verification_questions`. Missing or multiple eligible clauses, unknown terminology, or unclear conditions cause `REVIEW_REQUIRED`; the model's conflict claim is not proof.
3. **Citation-ID safety for direction reversals**. The JSON Schema sent to the model now contains actual enumerated allowed IDs for *this request*, with identical host-side validation. If a model still emits an invalid ID, the host does **not** retry the LLM. Only for an explicitly fixed **comparison winner** question, it tries a source-only proof: English and Chinese must contain identical, uniquely referenced dataset scopes; one clause per dataset per language; exactly one scope must contain a demonstrable winner reversal; all role names must be within the restrictive lexicon. Otherwise it abstains. `HOST_SOURCE_RECOVERY` is recorded to distinguish this outcome from a validated LLM reply. The original model reply is not fabricated or shown as validated.
4. **Chinese comparison grammar**. The new source-only verification distinguishes `基线方法比新系统表现更好` (baseline wins) from a separate assertion `新系统表现更好` (proposed system wins). A substring inside the `比` object cannot be treated as a winner. Unknown and compound constructions abstain.
5. **Truthful retry diagnostics**. One invalid request now says `1 次尝试`, not the obsolete `连续三次` message. Three-attempt legacy callers still receive the historical phrase for backwards-compatible tests.

### Source proof vs model text

Even if the reply says `finding=EQUIVALENT`, the host can prove a narrow `FAIL` from original source relationships. Conversely, even an apparently compelling `finding=CONFLICT` yields `REVIEW_REQUIRED` without a narrowly verified original-source contradiction. This is a bounded relation checker, **not a full semantic theorem prover** and not a proof of all paragraph requirements.

### Verification limits

- Host citation focus currently recognizes *specific known forms*, not every possible paraphrase of a bilingual exception.
- Source-only direction recovery requires clear ordered **named dataset conditions** and a unique inversion. It must abstain for ambiguous scopes, duplicate scopes, unknown actors, conditional/negated or complex grammatical comparison, or multiple inversions.
- Incorrect overall PASS decisions not detected by source risk rules can still survive two-model agreement. The prior model behavior notably misread several explicit negation relationships; passing this small regression suite will not prove universal reliability.
- The prior 11 cases have been repeatedly used to develop and debug the Harness. They should **not** be treated as an independent, untouched holdout dataset.

## Phase 1 — install (macOS, LM Studio)

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.15.zip
cd bilingual-keyword-checker-v0.2.15
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

The bundled `.env.example` sets `LM_MODEL=qwen/qwen3.5-9b`, `LM_VERIFIER_MODEL=ministral-3-8b-instruct-2512`, `LM_JSON_SCHEMA=true` and uses the local OpenAI-compatible LM Studio server. Both exact IDs need to be loaded in LM Studio. No web services or purchased inference tokens are required.

## Exec — one complete real-model 11-case regression

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct_hardened \
  --mode dual \
  --baseline-report examples/reference_v0214_relations_11.json \
  --stop-after-errors 1 \
  --out eval_v0215_hardened_11.json
```

The included `reference_v0214_relations_11.json` is an **unaltered copy** of the user's actual v0.2.14 live report, SHA-256 `a8287af4879af0da4fc5d0aa8313babdeac1945ffd7eb0f1cb68c6570bb64f00`. Matched-case baseline fields `id`, `en`, `zh`, `expected` are checked before starting inference; the old model is **not** rerun. Minimum ~22 model requests for 11 dual decisions, plus one for each triggered fixed verification (historically five, but not guaranteed), no additional retries for invalid evidence IDs. Compare correct automated verdicts **out of all 11**, false PASS/FAIL, manual review coverage, requests/latency, and per-case evidence trace.

## What was actually tested locally

The offline suite includes prior tests, five selected real-model JSON *trace* replays (not new live inferencing), dynamically enumerated schema checking, edge-case Unicode values, ambiguous exception scopes, deliberately invalid evidence IDs, duplicated and mismatched dataset conditions, and an explicit subject/object reversal adversary. The v0.2.14 comparison-winner third reply was **not retained** in the historical JSON because it failed schema validation; that particular replay uses a *synthetic malformed ID* to test the recovery path. Other five-case model answers are taken from prior real report where available. Do not interpret mock-based regression success as a new v0.2.15 model accuracy measurement.
