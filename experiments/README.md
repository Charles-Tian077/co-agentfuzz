# Experiment index

This directory separates experiment **protocols and execution templates** from
the runtime implementation under `src/` and historical observations outside the
repository. Protocol JSON files are documentation, not an alternative runtime
or a substitute for a run's actual manifest.

| Cohort | Implementation | Inputs and boundary |
|---|---|---|
| Repair-space replication | `coagent_repro.replicate_records`, `.strong_records`, `.v32_records` | 18 training / 60 held-out fictional records, seeds 101/137/173, four conditions |
| Record smoke | Same adapters, `--mode smoke` | 18 training / 6 evaluation records (not a full cohort) |
| Matched communication-text cost probe | `benign_record_feedback.token_probe` | Explicit verified prior run, fixed matched role inputs (not total hardware cost) |
| Earlier record transport diagnostic | `benign_llm_orca.study` | Its own task offset, selector and reward protocol (separate cohort) |
| Arithmetic mechanism diagnostic | `benign_orca.benchmark` | Deterministic arithmetic fixtures, no LLM effectiveness claim |
| Historical comparison and auditing | `Experiment`, `co_agent_fuzz`, `PromptFuzz` | Explicit authorized historical data, exact original source/model/budget needed |

The generic cached-model Slurm template is in `slurm/`. It runs only on an
allocated compute node and neither downloads weights nor resubmits/retries jobs.
Use separate output folders for every seed and model. `experiments/outputs/` is
private and ignored. No measured result is committed here by default.

For an archived result, retain its original source snapshot and hashes, data
partition, model digest, token/call ledger, rewards, receipts, frozen evaluation
and audit. Moving release code does not retrospectively change archived source
identities. Incomplete runs are missing evidence, not zero-valued outcomes.

For complete replications, the independent statistical unit is the fixed record.
Aggregate seeds within each record. Do not treat 180 seed observations as 180
independent tasks. Keep model/protocol/budget-specific comparisons separate.
See [the reproducibility guide](../docs/REPRODUCIBILITY.md).
