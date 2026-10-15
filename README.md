# Co-AgentFuzz

Source code for collaborative feedback controllers, local model clients, and
frozen-model mechanism experiments. This repository contains no manuscript,
figures, private experimental outputs, raw benchmark datasets, or model weights.
It includes a deterministic export of the existing harmless fictional records.

The cleanup preserves the existing controller algorithms and reward rules.
Package entry points, deployment configuration, and result parsing were made
portable; these changes do not constitute a new algorithm or a new experiment.

## Repository layout

| Directory | Purpose |
|---|---|
| `src/co_agent_fuzz/` | Original controllers, mutation operators, filters, oracle, memory, Q-learning, and local model client |
| `src/benign_record_feedback/` | Stage-relative rewards, record repair curriculum, compact peer priors, audit, and token-cost probe |
| `src/benign_llm_orca/` | Fixed fictional records, bounded loopback transport, strict selector protocol, and passive replay |
| `src/benign_orca/` | Deterministic arithmetic mechanism diagnostic and its auditor |
| `src/coagent_repro/` | Three-seed replication entry points and existing Qwen3/QwQ/DeepSeek-V3.2 transport presets |
| `src/Experiment/` | Legacy comparison entry points, metrics, and split/evidence checks |
| `src/PromptFuzz/` | Inherited compatibility baseline; upstream license retained |
| `datasets/synthetic/` | Existing 18/60 fictional-record partitions, schema and SHA manifest |
| `models/` | Supported-model registry and adapter guide, without weights |
| `experiments/` | Cohort index, protocol metadata and generic Slurm template |
| `tests/` | Core, legacy, release, portability, and dataset consistency tests |
| `release_tools/` | Offline dataset exporter and pre-publication hygiene check |

Legacy package names are retained where renaming would invalidate existing imports
or evidence replay. Historical method-name aliases are read-only compatibility
metadata, not separate methods.

The repository folder is `co-agentfuzz`; the distribution name is
`co-agentfuzz` and the main Python import remains `co_agent_fuzz`.
Historical raw inputs and observed experiment artifacts are classified in the
**separate private sibling** `co-agentfuzz-private`, not under this GitHub root.
Neither that private directory nor recovery backups should be uploaded.

## Installation and offline tests

The tested interpreter is Python 3.11. Runtime dependency versions are declared
in both `pyproject.toml` and `requirements.txt`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python release_tools/check_release.py
python release_tools/export_synthetic_data.py --check
```

Dependency installation may require internet access or a local wheelhouse. The
unit-test session itself blocks network and model connections, including loopback
HTTP; its outputs are synthetic fixtures, not measured model results. Tests do not
download models or call paid APIs. Only the formal test directories are collected.

## Reproduce the fixed-record study on an allocated GPU

These entry points require a real Slurm compute-node step, an already installed
Ollama-compatible service, and an authorized offline model cache. They refuse
arbitrary endpoints and reuse of an output directory. Do not run inference on a
login node. The repository does not include model download or account-login scripts.

On the allocated compute node, start a service bound to `127.0.0.1`, disable cloud
access, and use the existing cache. Then run a **new** output directory:

```bash
srun --ntasks=1 python -m coagent_repro.replicate_records \
  --port 11434 --model qwen2.5:7b --mode smoke --seed 101 \
  --output /your/private/run/smoke
```

Replace `--mode smoke` with `--mode replication` only after the smoke has completed
and its audit is valid. Run each of `101`, `137`, and `173` in a separate process
and a distinct directory. Keep the four conditions matched within each model.

The generic scheduler template is `experiments/slurm/slurm_records.sbatch`. Supply your own
partition, account, allocated GPU resources, existing `OLLAMA_MODELS` directory,
model, and private output path. It never submits additional jobs or retries a
failed phase. See [the execution guide](experiments/slurm/README.md).

### Retained model-specific entry points

| Model family | Entry point | Preset boundary |
|---|---|---|
| Qwen2.5-7B, DeepSeek-R1-7B, GPT-OSS-20B, Llama3.1-8B | `coagent_repro.replicate_records` | Existing fixed-record transport and seed protocol |
| Qwen3-32B, QwQ-32B | `coagent_repro.strong_records` | Existing non-thinking Qwen3 and model-default QwQ presets |
| DeepSeek-V3.2 Q4_K_M | `coagent_repro.v32_records` | Existing audited GPU/CPU-offload configurations; not a successful full replication claim |

Unscored protocol checks are available through `coagent_repro.preflight` and
`coagent_repro.v32_preflight`. A preflight verifies schema, nonempty response,
selector representation, measured token counts, and GPU residency; it must not
select models by held-out accuracy.

The V3.2 manifest contains the authorized repository revision, quantization,
split-file sizes, and SHA-256 values, **not weights**. Runtime flags
`V32_GPU_COUNT`/`V32_GPU_LAYERS` must match the recorded hardware preset. CPU
offload and actual GPU allocations must be reported. Hardware/preset changes
are not algorithmic improvements.

## Protocol and evidence boundaries

The repair-space replication uses 18 training records and 60 fixed held-out
fictional records, seeds `101/137/173`, and Full, No learning, No communication,
and Neither. Smoke uses six evaluation records. LLM weights remain frozen;
only small action-value tables and numeric peer priors adapt. Rewards, message
receipts, next decisions, frozen evaluation, and actual usage are replayed from
saved records. Missing final token counts or protocol failures stop the run.

Each run writes `manifest.json`, `episodes.jsonl`, `summary.json`, and `audit.json`.
Failures preserve a partial ledger and `failure.json`; do not fill missing seeds
or replace missing outputs with invented zeroes. Output directories are private
and excluded from source releases.

Record-level inference must treat the 60 records as independent units and nest
the three seeds within each record. Mechanism probes, repair-space replications,
legacy search panels, and token-text probes are distinct cohorts. Do not pool
their results or describe fictional-record accuracy as attack success.

Read [the reproducibility guide](docs/REPRODUCIBILITY.md) for protocol/version
boundaries, audit requirements, and the matched token probe.

## Legacy comparison code and external inputs

The original `src/co_agent_fuzz/` and `src/PromptFuzz/` implementations are retained for
authorized evaluation and regression compatibility. Their historical dataset
paths are configuration defaults, **not bundled data**. Point the legacy CLI
and split utilities at appropriately licensed inputs in a private directory.
The fixed-record study above needs no external dataset. The exported JSONL files
are verifier fixtures; the runtime continues to use its unchanged built-in tasks.

The retained Co-AgentFuzz wrappers resolve legacy `Datasets/...` paths through
`COAGENT_DATA_ROOT` and `Results/...` paths through `COAGENT_OUTPUT_ROOT`. Set
these explicitly to authorized private directories; defaults are ignored
`datasets/private/` and `experiments/outputs/` directories. Installed-wheel users
can set `COAGENT_PROJECT_ROOT` or use explicit absolute paths. Variables in
`.env.example` are instructions only and are not automatically loaded. Inherited
PromptFuzz defaults are preserved; give that CLI explicit data/output paths.
Use `python -m Experiment.<module>` instead of historical relative script paths.

Old result files, personal deployment scripts, writing assets, and obsolete paid
fine-tuning workflows are not part of this release. Reproducing an archived
numerical claim requires that claim's exact source snapshot, model digest, data
partition, inference options, resource boundary, and recorded budget. A successful
unit-test run alone does not reproduce an LLM result.

## Publication checklist

- Upload only this source directory, not its parent or private backup directories.
- Run the release hygiene check and inspect the files staged for Git.
- Do not commit `.env`, API keys, SSH keys, weights, private data, or run outputs.
- Preserve upstream notices and confirm contributor authorization before release.
- Tag the tested source revision and record model/runtime digests for each experiment.

See [source provenance and licensing](THIRD_PARTY.md). No GitHub repository is
created and no content is uploaded by any cleanup or test command.
The [release checklist](docs/RELEASE_CHECKLIST.md) and
[data-availability notes](docs/DATA_AVAILABILITY.md) describe what is and is not
reproducible or publishable from this release.
