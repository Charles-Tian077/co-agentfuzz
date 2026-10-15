# Models and adapters

This directory holds **metadata only**. `registry.json` maps the seven existing
local model tags to packaged execution adapters. No model weights, tokens,
credentials or inference service are bundled. No weights are downloaded by the
release or test commands.

The controllers and local model clients live under `src/co_agent_fuzz/`; the
bounded fictional-record adapters live under `src/coagent_repro/`. The registry
is a documentation index, not a new runtime configuration or a claim that every
adapter produced a complete successful replication.

Use an authorized pre-existing cache and record the **actual** model digest,
quantization, chat template, service version, GPU residency, CPU offload, token
usage and resource allocation for each run. Tags can change and are not immutable
checkpoints. Except for the pinned V3.2 manifest, the release does not assert a
universal checkpoint or quantization for a tag.

The DeepSeek-V3.2 Q4_K_M manifest remains at
`src/coagent_repro/weights_manifest.json`. Its nine shard hashes and repository
revision describe weights, not redistribute them. The retained V3.2 adapter is
not proof of a complete three-seed efficacy result. CPU/GPU-offload choices and
different reasoning/output budgets must be disclosed and not pooled as a
matched cross-model cost comparison.

Place any private weights outside this repository. `models/weights/` and
`models/cache/` are additionally ignored. GPT-5/GPT-6, Llama 4 and unspecified
external API models are not supplied or claimed as reproducible by this release.
