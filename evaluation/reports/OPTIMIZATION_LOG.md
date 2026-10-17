# Optimization log (Subtask 5)

Measurements before any production optimization (offline dev split, 35 cases):

| Metric | Baseline (offline, 35 cases, dev+holdout) |
|--------|-------------------------------------------|
| recall_at_k (mean) | 0.729 |
| precision_at_k (mean) | 0.257 |
| mrr (mean) | 0.443 |
| key_fact_recall (mean) | 0.452 |
| faithfulness_offline (mean) | 1.0 |
| abstention_accuracy (mean) | 0.714 |

Source: `python -m evaluation --dataset evaluation/datasets/core.jsonl --offline` (2026-10-17).

## Decisions

| Candidate | Baseline evidence | Action | After metrics |
|-----------|-------------------|--------|---------------|
| Skip planner LLM on trivial queries | Not measured on live path in CI | **Deferred** | Revisit after live baseline |
| Skip memory retrieval when empty | Memory node already cheap offline | **Not implemented** | — |
| DB indexes beyond 0003 | No EXPLAIN baseline captured | **Deferred** | Run EXPLAIN on test DB first |
| Prompt size reduction | No duplicate-context regression shown | **Not implemented** | Measure token counts with `--live-llm` |

No optimization was merged without a before/after quality check. Re-run evaluation after any future change.
