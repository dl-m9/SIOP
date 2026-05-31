#!/bin/bash
# Launch a local dense retrieval service used by the SIOP search tool.
#
# Required:
#   RETRIEVAL_SERVER_SCRIPT=<PATH_TO_RETRIEVAL_SERVER>
#   RETRIEVER_CORPUS_DIR=<DATA_DIR>/corpus
#
# The referenced retrieval server should accept the Search-R1-style arguments
# used below. Replace this wrapper if your retriever uses a different CLI.

set -euo pipefail

: "${RETRIEVAL_SERVER_SCRIPT:?Set RETRIEVAL_SERVER_SCRIPT to your retrieval server Python file.}"
: "${RETRIEVER_CORPUS_DIR:?Set RETRIEVER_CORPUS_DIR to the directory containing index and corpus files.}"

CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"
INDEX_FILE="${RETRIEVER_INDEX_FILE:-$RETRIEVER_CORPUS_DIR/e5_Flat.index}"
CORPUS_FILE="${RETRIEVER_CORPUS_FILE:-$RETRIEVER_CORPUS_DIR/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
LOG_DIR="${LOG_DIR:-logs}"

export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPUS:-0,1,2,3,4,5,6,7}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

mkdir -p "$LOG_DIR"

nohup python "$RETRIEVAL_SERVER_SCRIPT" \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --topk "$RETRIEVER_TOPK" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu > "$LOG_DIR/retrieval_server.log" 2>&1 &

echo "[Retriever] PID=$! log=$LOG_DIR/retrieval_server.log"
echo "[Retriever] Default endpoint: http://localhost:8000/retrieve"
