#!/usr/bin/env bash
# Run v0.2.7.5 model checks on a powered Mac with LM Studio server running.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0275_regression_11.json

python -m examples.evaluate \
  --cases examples/validation_v0272_constraints_12.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0275_constraints_12.json

python - <<'PY'
import json
for p in ['eval_v0275_regression_11.json', 'eval_v0275_constraints_12.json']:
    with open(p, encoding='utf-8') as f:
        result = json.load(f)
    summary = result['summary']
    print('\n===', p, '===')
    print('Correct:', summary['correct'], '/', summary['total'],
          '| ERROR:', summary['errors'],
          '| False PASS:', summary['false_pass'],
          '| False FAIL:', summary['false_fail'])
    print('Mismatches:', ', '.join(summary['mismatched_ids']) or '(none)')
    print('Execution errors:', ', '.join(summary['error_ids']) or '(none)')
    print('REVIEW_REQUIRED:', summary['review_required'], '|',
          ', '.join(summary['review_required_ids']) or '(none)')
PY
