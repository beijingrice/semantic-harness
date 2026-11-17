# v0.2.13 — Question-driven Paragraph Verification (experimental)

> Experimental pipeline: `direct_questions`. `extraction` stays the default. This is **not a claim of production accuracy** and does not replace independent holdout evaluation.

## Why this change?

The user's actual v0.2.12 LM Studio run, bundled in `examples/reference_v0212_verified_11.json`, obtained **7/11 correct automated verdicts**, **4/11 `REVIEW_REQUIRED`**, **0 false verdicts on the seven automated cases**, 27 LLM requests and 101.45 seconds. It is **not** 11/11 accuracy. The focused verifier repeatedly answered the wrong question:

- `V2-NUM-13`: the source risk was `>10%` versus `≥10%`; the focused model instead checked whether the denominator excluded optional items.
- `V2-NUM-15`: all models claimed the cover-page inclusion/exclusion rules were equivalent despite opposite source words.
- `V2-ROL-07`: even after selecting the correct winner sentences, a model said “proposed system wins” was equivalent to “baseline method wins.”
- `V2-NUM-03`: a narrow equivalence recheck correctly restored `six weeks` = `42 days` to PASS.

## Design

1. **Whole originals remain intact.** Qwen and Ministral each read the original English and Chinese paragraphs independently. There is no mandatory sentence pairing, original-substring “alignment,” or claim extraction.
2. **Host chooses fixed question(s).** The Python controller registers separate immutable tasks for independently detected `numeric_boundary` and `inclusion_exclusion` source risks; for two-model disagreements it can register a `direction_reversal`, `exception_scope` or unresolved-dispute task. On isolated, equal numerical durations with two model FAIL votes, it registers `scalar_equivalence`. Tasks have IDs such as `Q_NUMERIC_BOUNDARY`; the model must echo both ID and type. An unrelated response fails validation or produces `REVIEW_REQUIRED`.
3. **Narrow typed operands.** Instead of requesting just `EQUIVALENT`/`CONFLICT`, the verifier must supply source-side comparison quantities or named document components / winning actors, two source-span IDs, and `same_requirement`. The original sentences in those spans need not occupy corresponding positions in the two languages. The complete original remains available during inference.
4. **Python checks real relations.** For a numeric boundary, numeric comparators and values must agree with source parsing, the cited span may contain only one scalar, and bilingual predicate anchors must support an identical context. For inclusion, the same glossary-backed document component must be included in one language and excluded in the other. For direction, named actors must be explicitly winning in matching *dataset scopes*, with a glossary-backed unequal role. For exception scope, the original wording must support a limited conditional-vs-unconditional contrast. Any unmatched subject, unknown scope or unrecognized wording **abstains** instead of inventing a FAIL.
5. **All questions must be accounted for.** A proven specific mismatch suffices for FAIL. A PASS correction is allowed only for the former very narrow single-quantity equivalence case; any unresolved task blocks PASS. Ordinary samples without disputes remain exactly two model requests. Every task is at most one focused request (no repeated ineffective retries).
6. **Full observability.** `detail.verification_questions` holds the task ID, selected evidence IDs, source-side operands, model finding and independently checked host conclusion. All original source text and the prior judgments remain in the report.

### Safety / limitations

- Source substring membership validates provenance, **not universal semantic identity**. Bilingual lexical anchors and a small academic-domain glossary are *intentionally narrow engineering heuristics*. Novel synonyms or unfamiliar named actors often result in `REVIEW_REQUIRED`. `same_requirement=yes` remains a model assertion supplemented by lexical guardrails, **not a formal theorem**. This implementation must be evaluated on genuinely unseen examples before production use.
- The source risk detector is not a complete semantic verifier. Two models can still agree incorrectly on a complex passage without triggering a risk. The new mode does not guarantee a complete proof of alignment.
- The bundled 11 historical cases were used during development, so 11/11 if achieved would demonstrate regression fit, **not generalization**. The 36 historical extra cases also have influenced earlier fixes and are not pristine held-out data. Do not report them as independent holdouts.
- All production-hardened extraction, original `direct`, `direct_grounded` and `direct_verified` modes are still selectable independently, and the default remains `extraction`.

## Phase 1 — Mac / LM Studio installation

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.13.zip
cd bilingual-keyword-checker-v0.2.13
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

The default `.env.example` includes `LM_MODEL=qwen/qwen3.5-9b`, `LM_VERIFIER_MODEL=ministral-3-8b-instruct-2512`, and `LM_JSON_SCHEMA=true`. LM Studio must serve both at `127.0.0.1:1234/v1`. Confirm exact model IDs if the server names differ.

## Exec — one complete regression command, no preliminary smoke

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct_questions \
  --mode dual \
  --baseline-report examples/reference_v0212_verified_11.json \
  --stop-after-errors 1 \
  --out eval_v0213_questions_11.json
```

Only new `direct_questions` inference runs; the reference is the user's real previous JSON. Expect 22 core calls plus up to one per triggered question (five expected on these previously observed two-model judgments, but fresh judgments may differ). Monitor correct AUTO decisions, FALSE PASS/FAIL, REVIEW_REQUIRED, total requests/latency and `verification_questions`, especially any unknown referents. There is no mandatory smoke run or user-facing pytest step.

## Developer-only local validation

Suite includes all prior tests plus HTTP MockTransport replay of v0.2.12 first- and second-stage **real** model outputs with **synthetic** typed focused answers, malformed/misdirected questions, two numeric predicates in one citation, unmatched referents, and changed compare scopes. Offline/mocked passing results **do not guarantee the model will actually answer a typed task correctly**.
