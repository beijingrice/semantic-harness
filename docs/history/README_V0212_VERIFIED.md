# v0.2.12 — Paragraph-Direct with Focused Proposition Verification

**Status:** research preview, not production default. Historical extraction and v0.2.10/v0.2.11 direct modes remain independently selectable. The `extraction` pipeline remains the default.

## Motivation from REAL v0.2.11 local LM Studio results

The 11-case `direct_grounded` run had 7 correct automatic verdicts, one false FAIL (`V2-NUM-03`), three `REVIEW_REQUIRED`, 32 requests and 119.94 seconds. Of the ten evidence-audit requests, seven failed schema/source checking. `V2-NUM-15` and `V2-NUM-13` made three failing attempts each. A *verbatim correct quote* did not entail a *valid interpretation* (`V2-NUM-03`, `V2-EXC-03`). See bundled `examples/reference_v0211_grounded_11.json` for the complete report. This is **NOT** an assertion of new v0.2.12 live-model performance.

## New pipeline: `direct_verified`

1. Primary Qwen and independent Ministral independently read the **full original** English/Chinese paragraphs and emit their direct assessments. No mandatory Extraction or 1:1 sentence, clause or paragraph positional alignment.
2. Conservative source-only risk indicators (numeric strictness; inclusion-count inventory) and a very narrow numeric equality certificate decide whether there is a specific dispute worth examining.
3. A focused request is sent **only if** two judges disagree, both PASS in the face of a source warning, or both FAIL in the face of a valid single-quantity equality certificate. Other cases use exactly two model calls.
4. Python provides each language's complete original passage (`en_all`, `zh_all`) plus optional *same-language* source-spans split at obvious sentence punctuation. These are **evidence selectors only**; they are not claim extraction or bilingual position pairing. The model selects IDs, never hand-copies quoted text. Source bytes are resolved by Python. English and Chinese spans may be in wholly different positions or may each use the full original.
5. The focused verifier outputs explicit `status`, `difference_type`, `same_subject`, `en_evidence_id`, `zh_evidence_id`, `reason` in strict structured JSON. Python rejects invented evidence IDs, inconsistent structured declarations, and unsupported numeric-comparator claims. Numeric comparisons require a source-verifiable bound in the selected spans. For scope/direction checks, lexical sanity checks supplement—but **cannot prove**—cross-language semantics.
6. Unlike v0.2.11, focused verification makes **only ONE request**: if it fails schema/ID validation, returns `REVIEW_REQUIRED` instead of trying identical suggestions three times. The old 3-attempt behavior remains unchanged for normal pipeline stages. Generation anomaly recovery remains available under its previous validated policy.
7. Two models agreeing on FAIL despite a strictly isolated matching quantity cannot sustain an extra-process-duty FAIL **solely because the third model asserted one**. If the focused audit verifies equivalence and the strict source-only single-quantity guard finds no additional explicit obligation, PASS is permitted. Otherwise the case is `REVIEW_REQUIRED` (not forced FAIL). This is not a general semantic-equivalence proof.

### Structural evidence is not semantic proof

Evidence selectors prove only that quoted bytes originated from the original source. Checking the same referent and whether a direction/scope conflict is genuine still involves an LLM judgment. The deterministic numeric checker only proves normalized quantities/boundaries, not whole-passage equivalence. A risky double PASS (e.g. inclusion inventory mismatch) is never upgraded to automatic FAIL on a mere third judgment; it is held for review. No engineering test substitutes for live-model evaluation or manual auditing of rationale quality.

### Historical pipeline modes

- `--pipeline extraction`: existing multi-stage Extraction + Matching + independent review + guards.
- `--pipeline direct`: two independent whole-paragraph decisions (v0.2.10).
- `--pipeline direct_grounded`: original v0.2.11 quoted-evidence path (kept for reproducing history).
- `--pipeline direct_verified`: new v0.2.12 focused span-ID + structured verification experiment.

## Phase 1 — Installation (Mac / LM Studio)

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.12.zip
cd bilingual-keyword-checker-v0.2.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Check LM Studio is serving both `qwen/qwen3.5-9b` and `ministral-3-8b-instruct-2512` at `http://127.0.0.1:1234`. Confirm model IDs and `LM_JSON_SCHEMA=true` in `.env`; leave `LM_PIPELINE=extraction` unless intentionally overriding it with `--pipeline`.

## Exec — single 11-case regression experiment

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct_verified \
  --mode dual \
  --baseline-report examples/reference_v0211_grounded_11.json \
  --stop-after-errors 1 \
  --out eval_v0212_verified_11.json
```

This runs only the new pipeline (at least ~22 model calls, plus up to one focused request per risky case). The bundled prior report is a historical comparison, **not contemporaneous randomized A/B**. No smoke/preliminary eval or pytest is required of the user. Submit the resulting JSON for full diagnosis. In addition to accuracy, compare false PASS, false FAIL, review-required rate, number of focused-verify failures, latency and source-ID integrity.

## Local engineering verification

`pytest` from source and from the fresh extracted ZIP include HTTP MockTransport (no LM Studio needed) for all earlier pipelines and new high-risk cases, invalid evidence IDs, bounded retry and source-only risk abstention. These are developer checks only. Real v0.2.12 correctness is not known until run on actual models.
