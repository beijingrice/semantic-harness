# v0.2.7.5 — source-grounded numeric certificates for 1:N / N:1 links

## What changed

1. `app/dispute_resolution.py`: matching links of cardinality 1:N or N:1 can yield multiple certificates, **only if every participating claim independently matches the original source and the same canonicalized numeric value, unit and comparator**. No unvalidated Cartesian expansion of M:N links; a nonnumeric or contradictory child vetoes the entire group.
2. In the `PASS` / reviewer `FAIL` / arbiter `PASS` disagreement state, a recheck marked `status=refuted` can now be accepted even if the recheck model assigns `basis=other`, **but only for isolated exactness objections with source-verified equalities**. The `basis` label by itself never authorizes PASS, even if it says `formal_numeric`. Multiple statements or combined clauses abstain and return `REVIEW_REQUIRED`.
3. Approximations (`approximately`, `约`, `左右`, etc.) no longer produce formal exact-equality certificates.
4. `app/llm.py`: if an LM Studio completion ends due to token length, ERROR now reports stage/model/max_tokens, completion/reasoning token usage, and content/reasoning character counts. It does **not** echo model reasoning and **does not** automatically retry a truncated response.
5. `examples/validation_v0275_smoke_3.json`: the 6-week / 42-day PASS pair in two wordings plus a 43-day FAIL control. This is a short real-model smoke test, not an independent holdout.
6. `scripts/run_local_regressions.sh` kept as an optional convenience; installation never auto-runs it.

## Why conservative?

Proof that `6 weeks = 42 days` only establishes **duration equality**, not that other obligations in a paragraph match. If an independent reviewer disputes another requirement (archiving, deletion, permission, exception, etc.), numeric certificates cannot overturn it. Unresolved disagreements remain `REVIEW_REQUIRED`.

A model may still output `REVIEW_REQUIRED`, or a real local model may still fail to deliver complete extraction JSON. This release **does not solve** generation truncation. It makes truncation diagnosable rather than blindly increasing the token budget.

## Install (Mac)

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.7.5.zip
cd bilingual-keyword-checker-v0.2.7.5
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp ../bilingual-keyword-checker-v0.2.7.4/.env .env
```

If the previous `.env` does not exist, use `cp .env.example .env` and edit model names / endpoint settings.

## Execution (no pytest as part of installation)

```bash
python -m examples.evaluate \
  --cases examples/validation_v0275_smoke_3.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0275_smoke_3.json
```

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0275_regression_11.json
```

Only when you wish to retry the known truncation-prone set:

```bash
python -m examples.evaluate \
  --cases examples/validation_v0272_constraints_12.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0275_constraints_12.json
```

## Offline verification

Tests contain **recorded real model responses** from `V2-NUM-03` and `C27-01`, plus adversarial examples where identical numbers must NOT override unrelated mismatches. HTTP mock tests cover `finish_reason=length` diagnostics. `pytest -q` may be run separately; never hits LM Studio.
