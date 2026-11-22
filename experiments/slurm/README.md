# Compute-node execution

Use [slurm_records.sbatch](slurm_records.sbatch) to run one fixed-record phase
and seed on an allocated GPU compute node. Activate the installed Python
environment before submission. The template assumes an installed
Ollama-compatible service, `curl`, and an authorized, already populated offline
model cache. It does not install software, download or import weights,
authenticate an account, submit additional jobs, or retry failed phases.

## Configure the job

Run from the repository root. Replace the cache, output, partition, and resource
values with those for your environment:

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

Add your scheduler's account and any required GPU-type constraints. Resource
values are an example for a small model, not requirements for all checkpoints.

| Variable | Meaning |
|:--|:--|
| `OLLAMA_MODELS` | Existing authorized cache, visible and accessible on the compute node |
| `MODEL` | Exact registered tag from [the model registry](../../models/registry.json) |
| `PORT` | Unused port in **20000–49999**. Set explicitly because the template default `11434` is rejected by the transport |
| `PHASE` | `smoke` or `replication` |
| `SEED` | `101`, `137`, or `173` |
| `RUN_ROOT` | New private output directory. Existing paths are rejected |
| `V32_GPU_COUNT`, `V32_GPU_LAYERS` | An allowed GPU-count / offload-layer pair, required only for the V3.2 adapter |

The job inherits the active Python environment through the exported `PATH`.
The batch script must run from the repository root. `--chdir="$PWD"` makes that
explicit. Slurm must provide the job's GPU mask. Do not fabricate allocation
variables to bypass the compute-node checks.

## Service lifecycle

1. Validate the cache, output path, GPU mask, model allowlist, and executable paths.
2. Reject a port that already serves an inference process.
3. Start a private service on `127.0.0.1:$PORT`, with cloud access disabled and
   at most one loaded model and one parallel request.
4. Wait for service readiness, then use `srun` to execute the selected adapter.
5. Stop the service process owned by this job when the batch script exits.

Do not prestart a second service when using this template. Service output goes
to `<RUN_ROOT>.service.log`, beside the run directory. The template does not run
a separate preflight. See the root [deployment instructions](../../README.md#model-deployment)
for the Qwen3/QwQ and V3.2 preflight entry points, which require an already running
job-local service when invoked manually.

## Select the adapter and resources

The script selects `coagent_repro.replicate_records` for the four base-model tags,
`coagent_repro.strong_records` for Qwen3/QwQ, and `coagent_repro.v32_records` for
DeepSeek-V3.2. Each run includes all four learning/communication conditions.

For V3.2, the permitted pairs are `(2, 36)`, `(4, 40)`, and `(4, 20)` for
`V32_GPU_COUNT` and `V32_GPU_LAYERS`. They correspond to the recorded two-H200,
four-A100-80GB, and four-A100-40GB presets. Request the actual matching GPUs,
sufficient host memory, and at least 16 CPU threads. Environment variables
do not allocate resources or verify that a checkpoint fits. The one-GPU / 32 GB
example is not a V3.2 deployment recommendation. Disclose actual GPU residency
and CPU offload, and keep the hardware boundary fixed within a comparison.

## Advance from smoke to replication

After a successful scheduler exit, require a complete `summary.json` and an
`audit.json` with `verified: true`. Then submit separate replication runs with
seeds `101`, `137`, and `173`, changing `RUN_ROOT` for each seed. Choose distinct
ports for concurrent jobs sharing a node. The template neither launches this
sequence automatically nor chains success dependencies for you.

Keep partial outputs and service/scheduler logs if a run fails. Early setup
errors can occur before a structured `failure.json` exists. Do not reuse an
output path, fill missing seeds with zeroes, or silently retry a failed phase.
Package/module `--help` commands do not perform inference or contact a model.

See [results and auditing](../../README.md#results-and-auditing) for artifact
contents, phase-specific counters, and the record-level statistical unit.
