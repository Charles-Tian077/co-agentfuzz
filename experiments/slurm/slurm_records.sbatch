#!/usr/bin/env bash
#SBATCH --job-name=coagent-records
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Supply your own --partition, --account, --gres, --mem and --time to sbatch.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run through an allocated Slurm job, not a login node}"
: "${CUDA_VISIBLE_DEVICES:?An allocated GPU mask is required}"
: "${OLLAMA_MODELS:?Set the existing authorized offline model cache}"
: "${MODEL:?Set a supported cached model}"
: "${RUN_ROOT:?Set a new private output directory}"
[[ -d "$OLLAMA_MODELS" ]] || { echo 'Model cache missing' >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo 'Output already exists; refusing overwrite' >&2; exit 2; }
[[ "$CUDA_VISIBLE_DEVICES" != '-1' ]] || { echo 'No GPU allocated' >&2; exit 2; }

PHASE="${PHASE:-smoke}"
SEED="${SEED:-101}"
PORT="${PORT:-11434}"
case "$PHASE" in smoke|replication) ;; *) exit 2 ;; esac
case "$SEED" in 101|137|173) ;; *) exit 2 ;; esac
case "$MODEL" in
  qwen2.5:7b|deepseek-r1:7b|gpt-oss:20b|llama3.1:8b) module=coagent_repro.replicate_records ;;
  qwen3:32b|qwq:32b) module=coagent_repro.strong_records ;;
  deepseek-v3.2:q4km-local)
    : "${V32_GPU_COUNT:?Set the recorded V3.2 GPU count}"
    : "${V32_GPU_LAYERS:?Set the recorded V3.2 layer offload}"
    module=coagent_repro.v32_records ;;
  *) echo 'Unsupported model; no inference' >&2; exit 2 ;;
esac
command -v ollama >/dev/null
command -v python >/dev/null
command -v curl >/dev/null

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
# Off-node HTTP is not used; the Python record transport explicitly disables proxies.
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
if curl --silent --fail --max-time 2 "http://${OLLAMA_HOST}/api/version" >/dev/null; then
  echo 'Port already serves a process; choose a private free PORT' >&2; exit 2
fi
mkdir -p "$(dirname -- "$RUN_ROOT")"
ollama serve > "${RUN_ROOT}.service.log" 2>&1 &
service_pid=$!
trap 'kill "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in $(seq 1 30); do
  kill -0 "$service_pid" 2>/dev/null || { echo 'Service failed' >&2; exit 1; }
  if curl --silent --fail --max-time 2 "http://${OLLAMA_HOST}/api/version" >/dev/null; then
    ready=1; break
  fi
  sleep 1
done
[[ "$ready" == 1 ]] || { echo 'Service startup timeout; no retry' >&2; exit 1; }
srun --ntasks=1 python -m "$module" --port "$PORT" --model "$MODEL" \
  --mode "$PHASE" --seed "$SEED" --output "$RUN_ROOT"
