#!/bin/bash
# Launch the SIOP scoring server with independent vLLM workers.
#
# Required:
#   SIOP_MODEL=<SCORER_MODEL_PATH>
#
# Optional:
#   SIOP_NLI_MODEL=<NLI_MODEL_PATH>
#   SIOP_PORT=8390
#   SIOP_NUM_GPUS=8

set -euo pipefail

: "${SIOP_MODEL:?Set SIOP_MODEL to the scorer model path or Hugging Face id.}"

CONDA_ENV="${CONDA_ENV:-siop}"
PORT="${SIOP_PORT:-8390}"
NUM_GPUS="${SIOP_NUM_GPUS:-8}"
GPU_MEM="${SIOP_GPU_MEM:-0.15}"
MAX_MODEL_LEN="${SIOP_MAX_MODEL_LEN:-8192}"
LOG_DIR="${LOG_DIR:-logs}"

export CUDA_VISIBLE_DEVICES="${SIOP_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

mkdir -p "$LOG_DIR"

args=(
    --model "$SIOP_MODEL"
    --port "$PORT"
    --num-gpus "$NUM_GPUS"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_MODEL_LEN"
)

if [[ -n "${SIOP_NLI_MODEL:-}" ]]; then
    args+=(--nli-model "$SIOP_NLI_MODEL")
fi

nohup python -m verl.utils.siop.scoring_server "${args[@]}" \
    > "$LOG_DIR/siop_scorer.log" 2>&1 &

echo "[SIOP-Scorer] PID=$! log=$LOG_DIR/siop_scorer.log"
echo "[SIOP-Scorer] SIOP_SCORER_URL=http://localhost:$PORT"
