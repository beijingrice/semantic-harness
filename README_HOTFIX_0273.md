# v0.2.7.3: LM Studio Structured JSON + Reasoning Compatibility

## Reproduced server failure

Some Qwen3.5 LM Studio OpenAI-compatible backends return HTTP 200 with `finish_reason=stop`, but `content=""` and a complete JSON object in `reasoning_content` when `response_format.type=json_schema` is supplied. See LM Studio bug tracker [#1773](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1773).

## Behavioral fix

- If and only if JSON Schema mode is on and `content` is empty, recover `reasoning_content` **only when it is a complete, standalone JSON object** (`json.loads`), then perform the *normal Pydantic schema and paragraph source-ID checks*. Arbitrary text and free-form reasoning cannot be promoted into a final answer. No model-name check.
- If there is no safe answer to recover, stop immediately with actionable error instead of burning two retries on repeatedly empty output.
- `examples.evaluate --stop-after-errors 1` prints the exact stage and error, saves a partial evaluation report, and exits nonzero; `scripts/run_local_regressions.sh` uses it to avoid running a second dataset after failure.
- The actual LLM API request (temperature / json_schema / token limit) is otherwise unchanged.

## Known limits

This hotfix addresses a **transport/protocol response routing** issue; it does not change semantic judgments. A reasoning response that is not a standalone valid JSON object is intentionally rejected. The same source grounding and independent review checks apply to all responses.

## Installation and verification

Install dependencies in your venv: `python -m pip install -e '.[dev]'`
Run offline tests: `python -m pytest -q`
Run the powered local regression: `bash scripts/run_local_regressions.sh`

If still broken, test disabling JSON Schema via `LM_JSON_SCHEMA=false` in `.env`, or disable Thinking in LM Studio's model chat template. Avoid changing multiple parameters at once when comparing results.
