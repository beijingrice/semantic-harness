# v0.2.14 — Relation-aware Question Verification (experimental)

**Purpose.** Repair v0.2.13's decision contract: the model interpreted `same_requirement=no` as "the requirement differs," but the host interpreted it as "these passages are about unrelated subjects," aborting before verifying already discovered contradictions. A model's typed field cannot take precedence over stronger independent source evidence.

## Real baseline and performance expectations

The user's real v0.2.13 `direct_questions` run is included as `examples/reference_v0213_questions_11.json`: **7 of 11 cases automatically decided correctly**, four `REVIEW_REQUIRED`, zero wrong decisions *among the seven automated*, **27 LM requests**, **105.07 seconds**. A prior v0.2.12 run also automatically decided 7/11. Do not conflate 7/7 auto accuracy with 11/11 coverage; these datasets have already been used repeatedly for tuning and are **not fresh holdout measurements**.

No local live Qwen/Ministral run is claimed for v0.2.14. Offline/mock tests replay real primary and reviewer outputs from v0.2.13, with third-stage replies *adapted to the new JSON contract*; those tests demonstrate host logic, not how the model will actually answer the new schema.

## Pipeline architecture

1. **Preserve whole paragraphs.** As before, Qwen3.5-9B and Ministral-3-8B independently see complete source text. No pre-LLM Claim Extraction, artificial sentence-to-sentence alignment, positional matching or compulsory source fragmentation.
2. **Register fixed verification tasks** triggered by source-only numeric / count indicators, model disagreement, or narrow quantity equivalence. A numeric-boundary task cannot switch into discussing optional survey fields. A program-specific exception task cannot silently become an abstract enumeration task.
3. **Dedicated v0.2.14 typed answer.** Fields include `same_event` (same real-world item and condition, **not same rule or verdict**), selected immutable original span IDs, `en_subject`, `zh_subject`, `en_condition`, `zh_condition`, `en_value`, `zh_value`, `finding`, and a short reason. These are *model hypotheses*, never sufficient proof on their own.
4. **Source relation checker.** The host resolves spans from original text, checks a bounded domain glossary and tight language-specific syntax. A model's `same_event=no` cannot hide a source-proven opposite rule, nor can `finding=EQUIVALENT` erase a demonstrated strict comparator difference. A source with multiple conflicting referents or unrecognized scope remains `REVIEW_REQUIRED`.
5. **Narrow proof types.** Numeric comparator claims must exactly reproduce host-parsed quantitative operands in unambiguous selected clauses and share predicate anchors. Count inclusion/exclusion requires the *same glossary-backed component*, direct opposite polarity, and a narrow shared report/page-limit context. Comparison direction requires a single matched condition (e.g., second dataset), exactly one source-identified winner on each side, and canonical unequal roles. Exception scope requires an identified same component and specific program-dependent `optional unless required` versus `regardless-of-program always optional` syntax. A scalar PASS proof requires **one exact, same-activity quantity**, same canonical value/unit/comparator, no independent duty, and a strictly limited no-deviation restatement.
6. **Conservative finalization.** One *proved* inconsistency gives FAIL. A narrow fully proved isolated scalar statement may repair a speculative shared FAIL into PASS. A mere model-confident assertion of DIFFERENT or EQUIVALENT cannot override source evidence. An unresolved task returns REVIEW_REQUIRED. Earlier modes are unchanged.

### Example: why `same_event` differs from agreement

EN: `On the second dataset, the proposed system achieved the higher score.`
ZH: `在第二组数据上，仍然是基线方法得分更高。`

These source spans discuss the **same event/dataset**, despite opposing winners; therefore `same_event=yes`, `en_subject=proposed system`, `zh_subject=基线方法`, and `finding=CONFLICT` is the correct typed interpretation. The host independently verifies source winners and dataset scope. Even if a model incorrectly reports `same_event=no`, it cannot veto an independently proven mismatch. Conversely, if model-chosen spans belong to different datasets, the host abstains.

### Explicit limitations and safety

- The terminology mapping is a **small, explicit glossary**, not a translator or universal semantic formalizer. Other ways of expressing a count inclusion rule, complex coreference, compound conditions and unknown comparison actors often abstain. No passage's positional correspondence is assumed.
- Each proof checks a **specific local relation**, not exhaustive paragraph equivalence. Direct PASS on other complex cases still rests on two LLM judgments. Real production needs unseen bilingual examples, especially many-to-many sentence structure and adversarial counterexamples.
- Inclusion must refer to the same known page-limit context; conflicting page-limit numbers, unrelated appendix rules, and broad spans with multiple winner datasets do not yield automatic FAIL. The scalar PASS certificate rejects approximate quantities, unrelated subjects and separate administrative obligations.
- For every focused question, the original entire paragraph remains visible to the model. Citation subspans merely *locate evidence* after the question is fixed. They are not pre-extraction or mandatory sentence alignment.
- All source text is untrusted task data, not instructions. The host checks returned task IDs and citation IDs; schema failures fail closed with at most one focused request.

## Phase 1 — Local install (macOS / LM Studio)

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.14.zip
cd bilingual-keyword-checker-v0.2.14
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

`.env.example` provides the user's historical model IDs `LM_MODEL=qwen/qwen3.5-9b` and `LM_VERIFIER_MODEL=ministral-3-8b-instruct-2512`, `LM_JSON_SCHEMA=true` and `LM_EVIDENCE_MODE=paragraph`. Confirm that both exact IDs are loaded in LM Studio if local naming differs. `LM_PIPELINE=extraction` remains the default, and the CLI flag switches only this evaluation to `direct_relations`.

## Exec — One complete 11-case regression

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct_relations \
  --mode dual \
  --baseline-report examples/reference_v0213_questions_11.json \
  --stop-after-errors 1 \
  --out eval_v0214_relations_11.json
```

**No separate smoke/pytest user step.** Only new-mode inference runs; the baseline JSON is read from disk. Minimum 22 requests, with an additional question request whenever a risk/disagreement is triggered (historically five on this group, not guaranteed on fresh runs). Compare correct auto verdicts / all 11, False PASS, False FAIL, REVIEW_REQUIRED, total requests/time, and `detail.verification_questions`. The previous JSON data plus the new report should be shared for real root-cause review.

## Internal validation

The offline/mock suite includes original tests, new v0.2.14 replay tests, wrong-entity/wrong-context and wrong-dataset abstention, mislabeled equivalence vs source conflicts, approximate duration and extra-duty rejection, invalid task ID, and no-risk two-request preservation. A replay can test the host but **cannot guarantee live Qwen or Ministral will produce correct v0.2.14 structured outputs**. All old modes remain selectable for regression comparison.
