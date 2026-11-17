# v0.2.7.1: Paragraph protocol hotfix

## What was found

In v0.2.7, `app/llm.py` did not recognize the new `paragraph_extraction`,
`paragraph_extraction_fallback`, `paragraph_review`, `paragraph_arbitration`, or
`paragraph_alignment_support` schema names when constructing **retry feedback**.
If a model's first response was invalid, the adapter supplied feedback for an
unrelated *legacy* schema (`issues`, `resolved_missing`, `resolved_conflicts`).
Such feedback cannot repair a paragraph extraction result and may cause all
three retry attempts to fail. The first turn itself is unaffected.

The English-only `en_p1` *example* was also embedded in both paragraph
extraction system prompts, including when asked to extract Chinese. Both
prompts now refer only to source-supplied paragraph IDs, and the user payload
provides `allowed_paragraph_ids` populated from the actual source.

## Changes

- Stage-specific retry instructions for all five paragraph wire schemas.
- Retry for paragraph extraction echoes the actual allowed paragraph IDs,
  not an English-only demonstration ID.
- Removed the English-only output example in extraction/fallback prompts.
- No changes to the existing matching / review / arbitration decision rules.
- No new model-specific prompt content; model selections remain configuration.
- Added mocked HTTP tests reproducing wrong Chinese IDs and invalid first
  responses. These tests **do not** demonstrate improvements on real models.
- Existing 16-case paragraph validation, 120-case v2 validation and the
  previous 11-case regression set are still included under `examples/`.

## Installation

Create a new folder rather than overwriting a working project. Copy your
existing `.env` into this one and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Try the 16-case paragraph validation first:

```bash
python -m examples.evaluate \
  --cases examples/validation_v027_paragraph_16.json \
  --mode dual \
  --out eval_v0271_paragraph_16.json
```

Do **not** assume this patch resolves all v0.2.7 errors. The earlier run had
six `extract_chinese` errors and one `keyword_matching` error; the root cause
of each particular failure requires the case `error` details in
`eval_v027_paragraph_16.json`. The one matching error is not directly fixed
by this patch.

Extract the original failure diagnostics on your Mac with:

```bash
python - <<'PY'
import json
with open('eval_v027_paragraph_16.json', encoding='utf-8') as f:
    report = json.load(f)
for case in report['cases']:
    if case['predicted'] == 'ERROR':
        print(case['id'], case.get('error_stage'), case.get('error'))
PY
```

If LM Studio's model has thinking enabled, disable it in a way supported by
your selected backend before re-testing speed. This patch intentionally does
not change generation parameters or assume all models/backends support the
same thinking-control field.
