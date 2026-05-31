#!/bin/bash
# Main SIOP training entrypoint.
#
# Required:
#   MODEL_PATH=<MODEL_PATH>
#   TRAIN_FILE=<DATA_DIR>/train_multiturn.parquet
#   VAL_FILE=<DATA_DIR>/test_multiturn.parquet
#
# Optional:
#   CONDA_ENV=siop
#   OUTPUT_DIR=<OUTPUT_DIR>/siop
#   RETRIEVER_URL=http://localhost:8000/retrieve
#   SIOP_SCORER_URL=http://localhost:8390

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the policy model path or Hugging Face id.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file.}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet file.}"

CONDA_ENV="${CONDA_ENV:-siop}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/siop}"
LOG_DIR="${LOG_DIR:-logs}"
RETRIEVER_URL="${RETRIEVER_URL:-http://localhost:8000/retrieve}"
SIOP_SCORER_URL="${SIOP_SCORER_URL:-http://localhost:8390}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export PYTHONUNBUFFERED=1
export SIOP_SCORER_URL

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TOOL_CONFIG="$(mktemp)"
trap 'rm -f "$TOOL_CONFIG"' EXIT
cat >"$TOOL_CONFIG" <<EOF
tools:
  - class_name: verl.tools.local_search_tool.LocalSearchTool
    config:
      retrieval_url: "$RETRIEVER_URL"
      topk: 3
      timeout: 30
    tool_schema:
      type: function
      function:
        name: search
        description: Search a local retrieval service for relevant passages.
        parameters:
          type: object
          properties:
            query:
              type: string
              description: Search query.
          required:
            - query
EOF

python -u -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-128}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.005}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-4}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.5}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-5}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tools_in_prompt=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.generate_timeout_s="${GENERATE_TIMEOUT_S:-120}" \
    actor_rollout_ref.rollout.multi_turn.verbose_rollout_logging=false \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-2048}" \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.adv_estimator=siop \
    algorithm.gamma=1.0 \
    +algorithm.siop_enable_two_pass=true \
    +algorithm.siop_lambda="${SIOP_LAMBDA:-0.5}" \
    +algorithm.siop_eta="${SIOP_ETA:-1.0}" \
    +algorithm.siop_num_refs="${SIOP_NUM_REFS:-3}" \
    +algorithm.siop_nli_device="${SIOP_NLI_DEVICE:-cpu}" \
    trainer.logger="${TRAINER_LOGGER:-[console]}" \
    trainer.project_name="${PROJECT_NAME:-siop}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-siop_main}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-false}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.resume_mode="${RESUME_MODE:-disable}" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/siop_train.log"
