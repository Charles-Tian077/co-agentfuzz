# Compute-node execution

The template assumes Python dependencies and model weights have already been
installed in your authorized environment. It does not install software, download
weights, authenticate an account, submit extra jobs, or retry failed phases.

Set `OLLAMA_MODELS` to your existing offline cache, `MODEL` to a supported cached
model, `PHASE` to `smoke` or `replication`, `SEED` to 101/137/173, and `RUN_ROOT`
to a **new** private output directory. Supply the partition/account/GPU/memory/time
resources required by your own scheduler. The script uses only its allocated
compute node and its own loopback service process.

```bash
export OLLAMA_MODELS=/your/authorized/offline-cache
export MODEL=qwen2.5:7b PHASE=smoke SEED=101
export RUN_ROOT=/your/private/new-smoke
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 experiments/slurm/slurm_records.sbatch
```

Qwen3/QwQ select the retained 32B wrapper. V3.2 selects the retained Q4_K_M wrapper
and requires explicit `V32_GPU_COUNT`/`V32_GPU_LAYERS` plus appropriate allocated
host/GPU memory. The template's default resources are **not** a V3.2 deployment
recommendation; replace scheduler requests to fit the recorded preset. No model
is accepted because of an accuracy threshold.

Each job runs a single seed. After a successful smoke, submit separate new runs
for the three replication seeds. Do not reuse output folders or combine incomplete
seeds into a complete statistical cohort. Package/module `--help` commands perform
no inference and do not connect to a model endpoint.
