<div align="center">

# Co-AgentFuzz

**Three-stage agent collaboration with auditable feedback**

Generation · Mutation · Selection · Role-specific learning

[![Offline tests](https://github.com/Charles-Tian077/co-agentfuzz/actions/workflows/tests.yml/badge.svg)](https://github.com/Charles-Tian077/co-agentfuzz/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-3B7A57)](LICENSE.txt)

[Overview](#overview) · [Installation](#installation) · [Model deployment](#model-deployment) · [Experiments](#experiments) · [Results and auditing](#results-and-auditing) · [Repository map](#repository-map)

</div>

## Overview

Co-AgentFuzz connects seed generation, mutation, and candidate selection through
shared outcome feedback. The source includes the prompt-search controllers,
local model clients, role-specific action-value updates, peer communication,
and replay auditors.

**Start with the fixed-record study** to inspect the learning and communication
mechanism. It uses built-in fictional records and a deterministic verifier, with
frozen LLM weights. The original prompt-search implementation and inherited
PromptFuzz/GPTFuzzer components are retained separately for historical comparisons.

| Component | What it does |
|:--|:--|
| Generator | Produces a seed or an initial record |
| Mutator / editor | Transforms the seed or corrects the draft |
| Selector | Chooses candidates for evaluation |
| External evaluator | Verifies outcomes and supplies role-specific feedback |
| Learning and communication | Update action values and compact peer priors; record receipts and the state used by later decisions |

### What is available

| Included | Required separately |
|:--|:--|
| Controller source and local inference adapters | Installed inference service and authorized model weights |
| 18 training and 60 held-out fictional records, with schema and SHA-256 manifest | Historical benchmark inputs and original run ledgers |
| Seven model tags, protocol metadata, and a generic Slurm template | Your compute allocation and private output storage |
| Offline tests, usage accounting, and replay checks | Exact archived checkpoints and source snapshots for historical numerical claims |

The fictional-record study measures controlled generation, repair, and selection.
Its accuracy is **not jailbreak or prompt-injection attack success**. Model adapter
availability does not imply that every model completed a three-seed experiment.

## Installation

Use **Python 3.11**, the interpreter selected by the repository's CI workflow.
Dependencies are pinned in [pyproject.toml](pyproject.toml) and
[requirements.txt](requirements.txt).

```bash
git clone https://github.com/Charles-Tian077/co-agentfuzz.git
cd co-agentfuzz
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the package and check the source release:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python release_tools/export_synthetic_data.py --check
python release_tools/check_release.py
```

Installation requires access to the declared packages or an existing wheelhouse.
The tests themselves block network and model connections, including loopback
HTTP. They need no GPU, weights, API key, or paid API. Passing them checks the
implementation and fixtures; it does not reproduce a measured LLM result.

## Model deployment

### 1. Prepare the inference environment

The fixed-record entry points require an **allocated Slurm compute-node step**.
Offline checks can run locally; the documented inference path runs on the
allocated compute node, with GPU residency checked after every call.

Before submitting a run, prepare:

| Requirement | Configuration |
|:--|:--|
| Python environment | Activate the installed environment before `sbatch`; `python` must be available inside the job |
| Inference service | An installed Ollama-compatible service supporting `/api/version`, `/api/tags`, `/api/generate`, and `/api/ps` |
| Existing model cache | Set `OLLAMA_MODELS` to an authorized cache visible on the compute node; the exact model tag must already be registered |
| GPU and host memory | Request resources that fit the checkpoint, context, and offload preset; the template defaults are not universal requirements |
| Job-local endpoint | Set an unused `PORT` in **20000–49999**; inference connects only to `127.0.0.1` |
| Run storage | Set `RUN_ROOT` to a new private directory for each model, phase, and seed |

Weights are not bundled or downloaded by the execution template. Prepare the
cache through your authorized model-provisioning process before running these
commands. A model tag is not an immutable checkpoint: retain its actual digest,
quantization, chat template, and service version.

### 2. Select a supported model

| Exact cached tag | Execution module | Output limit: generator/editor · selector | Reasoning preset |
|:--|:--|:--:|:--|
| `qwen2.5:7b` | `coagent_repro.replicate_records` | 64 · 16 | Non-thinking |
| `deepseek-r1:7b` | `coagent_repro.replicate_records` | 64 · 16 | Non-thinking |
| `gpt-oss:20b` | `coagent_repro.replicate_records` | 64 · 16 | Non-thinking |
| `llama3.1:8b` | `coagent_repro.replicate_records` | 64 · 16 | Non-thinking |
| `qwen3:32b` | `coagent_repro.strong_records` | 64 · 16 | Non-thinking |
| `qwq:32b` | `coagent_repro.strong_records` | 512 · 512 | Model default |
| `deepseek-v3.2:q4km-local` | `coagent_repro.v32_records` | 64 · 16 | Non-thinking; Q4_K_M |

These are per-call output-token limits, not measured usage. The registry is at
[models/registry.json](models/registry.json). Keep the preset identical across
the four conditions within a model. Different model presets and hardware do not
define a matched cross-model cost comparison.

### 3. Launch a smoke run

The supplied template **starts and stops its own loopback service**, disables
cloud access, checks the selected port, and selects the module from `MODEL`.
Do not start a second service on that port. Run the following from the repository
root, with the Python environment active, after replacing the cache, output, and
partition placeholders:

```bash
export OLLAMA_MODELS=/absolute/path/to/authorized/offline-cache
export MODEL=qwen2.5:7b
export PORT=23141
export PHASE=smoke
export SEED=101
export RUN_ROOT=/absolute/path/to/private-runs/qwen25-7b/smoke-101

sbatch --export=ALL --chdir="$PWD" \
  --partition=YOUR_GPU_PARTITION --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=02:00:00 \
  experiments/slurm/slurm_records.sbatch
```

Add your scheduler's `--account` or other resource flags as needed. The resource
values above illustrate a small-model run; adjust them to the actual checkpoint
and allocation. **Always set `PORT` explicitly**: the template's default `11434`
is outside the fixed-record transport's accepted range.

Submission is not completion. Check the scheduler's terminal state, then inspect
`audit.json` and `summary.json` as described below. The template runs one phase
and one seed, without automatic resubmission or failed-phase retries. It does not
run a separate preflight or launch the three replication seeds for you.

<details>
<summary><strong>Advanced: preflight and DeepSeek-V3.2 deployment</strong></summary>

For Qwen3/QwQ, `coagent_repro.preflight` checks nonempty output, record schema,
selector representation, actual token counts, and GPU residency. V3.2 has
`coagent_repro.v32_preflight`. These are unscored checks on three training
records; they do not select models using held-out accuracy.

When managing a service yourself, run a preflight only inside an allocated
compute-node step against that job's existing loopback service, for example:

```bash
srun --ntasks=1 python -m coagent_repro.preflight \
  --model qwen3:32b --port 23141 \
  --output /absolute/path/to/private-runs/qwen3-32b/preflight
```

`coagent_repro.preflight` accepts only `qwen3:32b` and `qwq:32b`; it is not the
preflight entry point for the four base-model tags. For V3.2, use the V3.2 module
and `--model deepseek-v3.2:q4km-local` instead. Each output directory must be new.

V3.2 requires an already imported offline model matching
[weights_manifest.json](src/coagent_repro/weights_manifest.json):
`unsloth/DeepSeek-V3.2-GGUF`, revision
`a787696863bafd5c736955ef81cc869a0bf6178a`, Q4_K_M, nine shards totaling
405,398,024,320 bytes. The manifest contains sizes and hashes, not weights or an
import utility. Verify the shards and registered model before inference.

The adapter accepts the following recorded GPU-count / offloaded-layer pairs:

| Hardware preset | `V32_GPU_COUNT` | `V32_GPU_LAYERS` |
|:--|:--:|:--:|
| Two H200 GPUs | 2 | 36 |
| Four A100 80 GB GPUs | 4 | 40 |
| Four A100 40 GB GPUs | 4 | 20 |

Set `MODEL=deepseek-v3.2:q4km-local` and both V3.2 variables before submitting.
Request the corresponding actual GPUs, sufficient host RAM, and at least the
16 CPU threads used by this adapter. The variables describe the offload preset;
they do not allocate resources. Record the actual GPUs, VRAM residency, CPU
offload, and wall time. The generic template's one-GPU / 32 GB host-memory
example is not suitable for V3.2.

The retained V3.2 attempt did not complete the three-seed study. This release
provides its adapter and metadata, not a successful V3.2 efficacy result or a
guarantee that a recorded preset will finish within its time limits.

</details>

See the [compute-node execution guide](experiments/slurm/README.md) for the
template's environment variables, lifecycle, and failure behavior.

## Experiments

### Fixed-record replication

| Protocol item | Setting |
|:--|:--|
| Task families | Contact, inventory, meeting; balanced within each partition |
| Training partition | 18 fictional records per condition |
| Held-out partition | 60 fixed records per condition; smoke uses 6 |
| Controlled draft interventions | Clean, missing field, extra field, wrong value, wrong type, syntax |
| Random seeds | `101`, `137`, `173`, with separate controller states |
| Adaptation | Role action values and compact numeric peer priors; LLM weights stay frozen |
| Evaluation | Exploration disabled; action values and peer history frozen |
| Primary outcome | Exact correctness of the selected record |
| Audit evidence | Actual tokens/calls, role rewards, Q changes, feedback receipts, consumed states, and GPU residency |

Each seed automatically runs all four conditions:

| Condition | Action-value learning | Peer communication |
|:--|:--:|:--:|
| **Full** | On | On |
| No learning | Off | On |
| No communication | On | Off |
| Neither | Off | Off |

After a successful, audited smoke run, use the same deployment command with
`PHASE=replication` and submit each seed to a distinct output directory:

```bash
export PHASE=replication
export SEED=101
export RUN_ROOT=/absolute/path/to/private-runs/qwen25-7b/seed-101

sbatch --export=ALL --chdir="$PWD" \
  --partition=YOUR_GPU_PARTITION --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=02:00:00 \
  experiments/slurm/slurm_records.sbatch
```

Repeat with `SEED=137` / `seed-137` and `SEED=173` / `seed-173`. Select a different
unused port if runs share a node. Do not launch an extra job for each condition:
the adapter already includes the complete four-condition matrix in each seed.
Keep the model digest, data partition, requests, resources, and budgets matched.

The exported [training](datasets/synthetic/train.jsonl) and
[test](datasets/synthetic/test.jsonl) JSONL files mirror the built-in runtime
tasks. They are inspectable verifier fixtures, not arbitrary input files loaded
by the CLI. Their `expected` fields are answer keys for verification and must
not be supplied to the policies or appended to prompts. See
[dataset documentation](datasets/README.md) and
[protocol metadata](experiments/protocols/repair_space.json).

### Other experiment paths

| Study | Implementation / guide | Interpretation |
|:--|:--|:--|
| Matched communication-text probe | `benign_record_feedback.token_probe`; [usage and input requirements](docs/REPRODUCIBILITY.md#matched-communication-text-probe) | Compares appended versus omitted training-history text on matched role inputs; requires a verified DeepSeek-R1-7B input run with the same model digest |
| Arithmetic mechanism diagnostic | `benign_orca.benchmark`, `benign_orca.audit` | Deterministic mechanism check, without LLM efficacy claims |
| Earlier record diagnostic | `benign_llm_orca.study` | Separate task-offset, selector, and reward protocol |
| Historical search comparisons | `Experiment.run_co_agent_fuzz`, `Experiment.compare_paper_metrics`, `Experiment.run_validation_matrix`, `Experiment.run_ablation_matrix` | Retained Co-AgentFuzz and PromptFuzz compatibility paths; require explicitly supplied, authorized historical inputs |

The token-text probe is a small matched prompt-cost study, not end-to-end
hardware savings or an accuracy noninferiority test. It currently requires an
input audit with both `verified: true` and `real_llm: true`; the replication
auditor's output does not include `real_llm`, so a replication directory is not
automatically a valid probe input. Do not add audit fields just to bypass this
check. See the [version boundaries](docs/REPRODUCIBILITY.md).

Legacy tools use `COAGENT_DATA_ROOT` and `COAGENT_OUTPUT_ROOT` for private input
and output paths. Installed-package users can additionally set
`COAGENT_PROJECT_ROOT`, or supply absolute paths. The variables in
[.env.example](.env.example) must be exported explicitly; that file is not
automatically loaded. Inspect each CLI with `python -m Experiment.<module> --help`
before configuring a historical study. Historical inputs and result ledgers
are not included in this repository.

## Results and auditing

### Run artifacts

| File | Contents |
|:--|:--|
| `manifest.json` | Model identity, source hashes, partition digest, seed, conditions, hyperparameters, and runtime limits |
| `episodes.jsonl` | Per-record decisions and outputs, measured token usage, verifier scores, rewards, updates, messages, receipts, and state digests |
| `summary.json` | Completion status, condition-level outcomes, call/token totals, and final policy state |
| `audit.json` | Replay verification of rewards, next decisions, peer state, frozen evaluation, GPU residency, and artifact hashes |
| `failure.json`, when written | Error, no-retry status, and available partial-run accounting |
| `<RUN_ROOT>.service.log` | The template-managed inference service's log, alongside the run directory |

Inspect a completed run without issuing new model calls:

```bash
python -m json.tool /absolute/path/to/private-runs/qwen25-7b/seed-101/audit.json
python -m json.tool /absolute/path/to/private-runs/qwen25-7b/seed-101/summary.json
```

Require a successful scheduler exit, `summary.json` with `status: "complete"`,
and `audit.json` with `verified: true` before treating a seed as complete.
Within each trial, held-out exact-match accuracy is `correct / evaluations`.
`tokens` includes training and evaluation; `training_tokens` identifies the
training portion. Repair and disagreement counters cover both phases; use the
episode `phase` field when reporting evaluation-only diagnostics.

Preserve incomplete runs and their partial ledgers. An error before backend
initialization may leave only an output directory or service/scheduler log,
without `failure.json`. Missing final token frames are missing usage evidence,
not zero-cost calls. Do not fill absent seeds or overwrite failed runs.

### Statistical reporting

For a complete three-seed replication:

1. Match conditions by fixed record ID and seed; verify all required runs and
   protocol/model identities first.
2. Use the **60 held-out records as independent units** and aggregate seeds
   within each record before computing paired effects.
3. Resample records for bootstrap 95% confidence intervals. For stable success
   (correct on all three seeds), use exact paired McNemar tests and model-wise
   Holm correction for the prespecified contrast family.
4. Report actual call/token totals with their phase boundary, model/runtime
   identity, hardware, and CPU offload. Do not treat 180 seed observations as
   independent tasks or pool different protocols and budgets.

These are analysis requirements; the per-seed CLI writes evidence and its replay
audit, not a combined three-seed statistical report. The detailed methodology
and limitations are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

<details>
<summary><strong>Troubleshooting</strong></summary>

| Symptom | Check |
|:--|:--|
| `Job-local port required` | Export an unused `PORT` in 20000–49999 before `sbatch`; do not use the template default `11434` |
| `A scheduled compute-node step is required` | Use the supplied Slurm template or a real `srun` step in your allocation; do not fabricate scheduler variables |
| `Existing local model/digest required; no download` | Check the existing cache, exact registered model tag, and service permissions |
| Port already serves a process | Select a different unused job-local port; leave other services alone |
| Output already exists | Use a new phase/seed directory; retain the previous evidence |
| GPU residency missing | Inspect the allocation, checkpoint fit, service log, and offload preset |
| Timeout, missing final token counts, or circuit breaker | Preserve the partial run and recorded failure; do not silently retry or change precision/budgets |
| Token probe rejects the prior run | Verify the required audit schema and identical DeepSeek-R1-7B digest; do not edit evidence to satisfy the gate |

</details>

## Repository map

```text
co-agentfuzz/
├── src/
│   ├── co_agent_fuzz/          # Prompt-search controllers and local model client
│   ├── benign_record_feedback/ # Repair-space policies, rewards, and text-cost probe
│   ├── coagent_repro/          # Fixed-record replication and model-specific adapters
│   ├── benign_llm_orca/        # Earlier fictional-record transport and diagnostics
│   ├── benign_orca/            # Deterministic arithmetic diagnostic
│   ├── Experiment/            # Legacy comparisons, split tools, and evidence checks
│   └── PromptFuzz/            # Inherited PromptFuzz / GPTFuzzer compatibility code
├── datasets/synthetic/        # 18/60-record export and integrity manifest
├── models/                   # Supported-model metadata; no weights
├── experiments/
│   ├── protocols/             # Protocol metadata
│   └── slurm/                 # Compute-node template and execution guide
├── tests/                    # Offline core, legacy, release, and consistency tests
├── release_tools/            # Dataset export and source-release checks
└── docs/                     # Reproducibility, availability, and release guidance
```

The distribution and repository are named `co-agentfuzz`; the main Python import
is `co_agent_fuzz`. Legacy package names are retained for import compatibility
and evidence replay. They do not identify additional methods.

## Availability and licensing

The release contains source, model/protocol metadata, and synthetic fixtures.
It does not redistribute weights, historical raw benchmarks, private run
outputs, manuscripts, or writing assets. Reproducing a historical numerical
claim additionally requires its exact data selection, archived source,
checkpoint, inference settings, and resource budget.

The [MIT license](LICENSE.txt) and inherited notices are retained. PromptFuzz and
its GPTFuzzer components are attributed in [THIRD_PARTY.md](THIRD_PARTY.md);
their presence is not a claim of original authorship by Co-AgentFuzz.

[Reproducibility guide](docs/REPRODUCIBILITY.md) ·
[Data availability](docs/DATA_AVAILABILITY.md) ·
[Model notes](models/README.md) ·
[Experiment index](experiments/README.md) ·
[Release checklist](docs/RELEASE_CHECKLIST.md)
