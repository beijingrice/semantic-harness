# v0.2.7.2: Constraint Guard and Local Regression

This is an incremental maintenance release built on v0.2.7.1, preserving the
**paragraph-first immutable source ID protocol**, FastAPI, CLI, web UI,
independent model review, and model-agnostic prompts.

## Why this release exists

The real v0.2.7.1 11-case regression completed **11/11 requests, 0 ERROR** but
scored only **8/11**: two missed FAIL cases and a deterministic false FAIL:

- `V2-NUM-03`: six weeks = 42 days; legacy numeric guard compared 6 != 42.
- `V2-NUM-13`: `> 10%` vs `>= 10%`; same literal number masked open/closed bound.
- `V2-NUM-15`: cover *included* vs cover *excluded* from the report limit;
  both language models called the linked facts equivalent.

## Implementation

`app/validators.py` implements a **small conservative constraint grammar**.
It reads **one-to-one linked claim units** and extracts comparable typed values
(number, unit/dimension, comparator). Supported numeric dimensions include
percent, days/weeks (1 week = 7 days), years (only years), pages, typographic
points, participants, and responses. Operators are `>`, `>=`, `<`, `<=`, `=`.
An operator mismatch at the same bound is meaningful even if the raw numerals
match. Full paragraphs and multi-number claims are NOT blindly compared.

`clear_inclusion_conflicts()` also inspects explicitly linked inclusion/counting
claims for high-confidence opposite *polarity* such as `included` vs `不计入`.
It does not use a bilingual noun lookup table or assume cross-language clause
order. Unrecognized phrasings are **abstentions**, not proofs of consistency.
These guards are a **partial safety net**, not comprehensive natural-language
constraint understanding or a formal proof engine.

Both the legacy quote path (`pipeline.py`) and the default paragraph path
(`paragraph_pipeline.py`) invoke the same guards. They do not make any extra
LLM request, do not inspect model brand, and do not change generation parameters.

`paragraph_pipeline.py` additionally removes **exact** duplicated model claims
with identical paragraph ID lists, before assigning IDs. This reduces redundant
matches while leaving paraphrases and distinct claims untouched. The original
unmodified claims are used to identify duplicates even if a claim-rollback
rewrites the public claim.

## Verification

After installing dependencies and setting `.env`, run both short real-model
validation sets with `bash scripts/run_local_regressions.sh`. This produces
separate JSON reports and concise final summaries. It **does not** rerun all
120 examples automatically.


- `python -m pytest -q` — offline unit/mock HTTP tests.
- `python -m examples.evaluate --cases examples/v2_regression_11.json --mode dual --out eval_v0272_regression_11.json`
- `python -m examples.evaluate --cases examples/validation_v0272_constraints_12.json --mode dual --out eval_v0272_constraints_12.json`
- `python -m examples.evaluate --cases examples/validation_v027_paragraph_16.json --mode dual --out eval_v0272_paragraph_16.json`
- `python -m examples.evaluate --cases examples/validation_v2_120.json --mode dual --out eval_v0272_v2_120.json`

The 12-case new constraint suite is **development/regression**, not an independent
held-out test. Other datasets shipped unmodified: original 60, v2 120, pilot 20,
paragraph 16, 11-case regression, 4 false-PASS cases.

As a no-LLM sanity check, the new guards were replayed over recorded **old**
model decisions: they raised no incorrect veto on 60 previously correct v0.2.6
samples and flagged both false PASS cases from the v0.2.7.1 11-case run.
This does **not** establish accuracy for future live model generations.

## Known limitations

- Explicit lexical cues are required, and not all numeric/temporal units are
  supported. In particular, months and years are not converted to days.
- Evidence IDs establish source fidelity but not semantic entailment.
- A language model can still fail to extract a fact, or create a faulty link;
  these guards do not address every false PASS.
- Models/backend thinking controls are unchanged; adjust the LM Studio model
  configuration to manage latency if needed.
- Do not present dev-set accuracy as held-out generalization.
