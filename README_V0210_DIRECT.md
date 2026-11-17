# v0.2.10 — Paragraph-Direct validation preview

This version is **an architecture experiment**, not a replacement verdict benchmark. The `extraction` pipeline from v0.2.9.2 remains the default and its implementation is unchanged. `direct` deliberately skips **both** LLM Extraction calls and Claim-to-Claim Matching. It never splits sentences, attempts cross-language chunk-by-position alignment, or invents fact-level correspondences.

## New direct data path

1. `build_paragraphs` creates immutable paragraph IDs (only blank-line paragraph boundaries). The entire original `english_text` and `chinese_text` are passed **unmodified**, together with the numbered paragraph list, to each model.
2. `qwen/qwen3.5-9b` gives its standalone source-based judgment in `direct_primary`; `ministral-3-8b-instruct-2512` independently assesses the **same original text** in `direct_independent_review`. The second model is not shown the first model's prediction.
3. Both return a strict `ParagraphReviewDraft` with `PASS|FAIL|UNSURE`, reasons, issues, and ORIGINAL paragraph IDs. Missing or fabricated ID references are rejected; note that a valid ID does not prove semantic correctness.
4. If both models agree and no source-backed narrow numeric contradiction is found, that decision is used. If they disagree or one abstains, the system returns `REVIEW_REQUIRED` (not ERROR, not fabricated PASS). No third correlated Qwen vote.
5. A conservative original-text numeric guard recognizes single-predicate bounds and simple numeric ranges including `<` vs `<=`. It abstains for multi-condition or multi-number scope ambiguity. Existing original-aware parsing is reused. This cannot replace full semantic interpretation.

Both predictions are visible in `cases[].detail.direct_primary` and `.review`; `cases[].requests` documents token usage and generation times; `cases[].detail.deterministic_issues` reports narrow hard contradictions. `english_keywords`, `chinese_keywords`, and `links` are **empty by design**: no hidden Extraction or Matching is claimed. Existing dashboard consumers expecting Claim lists will need a different direct-view presentation.

## Validation philosophy — one full dataset, not multiple smoke tests

Start with the established **11-case complete regression suite**. The command below only runs the new Direct approach; it does not repeat the expensive old-model evaluation. It compares predictions with an **archived, real v0.2.8 extraction-based evaluation on exactly the same texts** (baseline 11/11) using `--baseline-report`. Results are summarized under `summary.paired_reference` with paired-case verdict changes, completed requests and elapsed time. The comparison is **historical**, not a simultaneous randomized A/B; LLM stochastic variation, differing software versions, and hardware load can confound causal claims.

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct \
  --mode dual \
  --baseline-report examples/reference_v028_regression_11.json \
  --stop-after-errors 1 \
  --out eval_v0210_direct_11.json
```

Optional later evaluation with the known 36-case set can use `examples/validation_v028_holdout_36.json` and `examples/reference_v028_holdout_36.json`. It is no longer blind because it has previously been analyzed during development. Do not claim generalization from the 11 cases or from model/mock tests.

**Acceptance metrics**: exact PASS/FAIL, False PASS, False FAIL, `REVIEW_REQUIRED` (abstention), ERROR, auto-decision coverage, per-case p50/p95 and request/token count. Raw accuracy should be read alongside decision coverage: a system that defers every hard case is not automatically superior. Audit descriptions manually, especially on long lists, conditional exceptions, multi-sentence paraphrases, cross-paragraph anaphora, and reversed English/Chinese clause order.

## Configuration

`LM_PIPELINE=extraction` is the default, preserving the old path. One CLI flag `--pipeline direct` overrides the setting for an evaluation. To enable Direct in the API or app, set `LM_PIPELINE=direct` in `.env` (experimental: downstream UI expects the direct output to contain no Claim links). `LM_JSON_SCHEMA`, the JSON-in-`reasoning_content` compatibility fallback, and bounded retry instrumentation still apply to both direct requests. For a future actual blinded head-to-head study, collect a *new* untouched annotation set, freeze both implementations, and run both under matched conditions.
