# Evaluation dataset

- **File:** `core.jsonl` (one JSON object per line, validated by `EvalCase`).
- **Version:** `core-v1` (35 cases).
- **Splits:** `dev` (~70%) for tuning diagnostics; `holdout` for unbiased checks. Do not tune retrieval hyperparameters against holdout.
- **Chunk IDs:** deterministic UUIDs from `uuid.uuid5(NAMESPACE_DNS, "{fixture}-chunk-0")` matching [`evaluation/fixtures/offline_registry.py`](../fixtures/offline_registry.py).
- **No secrets:** fixtures use synthetic text only.

## Adding a case

1. Add or extend a fixture document in the offline registry (or seed DB for live eval).
2. Label `expected_chunk_ids`, optional `expected_key_facts`, `should_abstain`, `required_retrieval_sources`.
3. Set `category` and `split`.
4. Run `python -m evaluation --dataset evaluation/datasets/core.jsonl --limit 5 --offline`.
