# v0.2.7 Validation Notes

See `README.md` for installation, running commands, and known limitations.

- Offline regression: legacy quote mode tests are explicitly pinned to `LM_EVIDENCE_MODE=quote`; new paragraph tests exercise the default contract.
- New 16-case Paragraph Grounding set is in `examples/validation_v027_paragraph_16.json`.
- Original 120-case V2, its 20-case pilot and 11-case regression are also included.
- Run `python -m pytest -q` first. Live Qwen/independent-verifier benchmarks require your local API server.

If you need the historical v0.2.6 wire protocol for A/B, set `LM_EVIDENCE_MODE=quote`.
Default v0.2.7 uses immutable original paragraph IDs, not model-generated text quotes.
