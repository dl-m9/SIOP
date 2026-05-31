# SIOP

Official code release for **Self-Induced Outcome Potential: Turn-Level Credit Assignment for Agents without Verifiers**.

- Paper: [arXiv:2605.04984](https://arxiv.org/abs/2605.04984)
- Repository: [github.com/dl-m9/SIOP](https://github.com/dl-m9/SIOP)
- Base framework: [verl](https://github.com/volcengine/verl)

SIOP is a verifier-free reinforcement learning method for long-horizon LLM agents. It uses the model's own rollout distribution to build semantic outcome states, then assigns turn-level credit to intermediate information-gathering turns. The method is designed for settings where final-answer supervision or task-specific verifiers are unavailable or unreliable.

## What This Repository Contains

This repository is a compact public implementation of the main SIOP training path. It includes:

- SIOP process reward construction.
- Semantic answer clustering with NLI.
- vLLM-based scoring service for cluster support estimation.
- Turn-level token reward placement.
- SIOP advantage estimation inside verl PPO.
- Multi-turn search-agent training and evaluation launchers.
- Public configuration files with placeholders only.

It does not include datasets, checkpoints, logs, W&B runs, private machine paths, ablation wrappers, or one-off experiment scripts.

## Method Overview

For each prompt, SIOP samples multiple rollouts from the current policy. Final answers are clustered into semantic outcome modes using bidirectional entailment. These clusters act as latent future states. SIOP then builds a reliability-aware target distribution over the clusters and scores whether each intermediate assistant turn increases support for reliable future outcomes.

The implementation follows a three-stage pipeline:

1. **Cluster pass**: extract rollout answers, group outputs by prompt, and cluster final answers into semantic modes.
2. **Scoring pass**: query a SIOP scorer to estimate support for reference answers at each assistant turn.
3. **Reward pass**: convert consecutive support changes into per-turn process rewards and place them on token boundaries.

The resulting rewards are consumed by the `siop` advantage estimator in verl.

## Repository Layout

```text
.
├── train_siop.sh                       # Main training entrypoint
├── eval_siop.sh                        # Validation-only entrypoint
├── launch_siop_scorer.sh               # vLLM + NLI scoring server launcher
├── launch_retriever.sh                 # Local retrieval service wrapper
├── config/
│   ├── local_search_tool_config.yaml    # Local retriever tool config
│   └── serper_tool_config.yaml          # Optional web search tool config
└── verl/
    ├── utils/siop/                     # Clustering, scoring client/server, process rewards
    ├── utils/reward_score/siop_reward.py
    ├── trainer/ppo/core_algos.py       # SIOP advantage estimator
    ├── trainer/ppo/ray_trainer.py      # SIOP reward pipeline integration
    └── tools/                          # Multi-turn search tool implementations
```

## Main Components

| Component | Path | Purpose |
| --- | --- | --- |
| Process reward pipeline | `verl/utils/siop/siop_process_reward.py` | Clusters rollouts, scores cluster references, computes turn rewards |
| NLI clustering | `verl/utils/siop/nli_clustering.py` | Groups semantically equivalent answers |
| Scoring server | `verl/utils/siop/scoring_server.py` | Serves `/cluster`, `/score`, `/nli_score`, and `/health` |
| Scoring client | `verl/utils/siop/scoring_client.py` | Batched HTTP client used during training |
| Token reward placement | `verl/utils/reward_score/siop_reward.py` | Places turn-level rewards at assistant turn boundaries |
| Advantage estimator | `verl/trainer/ppo/core_algos.py` | Registers `adv_estimator=siop` |
| Trainer integration | `verl/trainer/ppo/ray_trainer.py` | Calls the SIOP reward pipeline during PPO rollout processing |
| Local search tool | `verl/tools/local_search_tool.py` | Calls a local retrieval endpoint for multi-turn search |

## Requirements

The code targets Python 3.10+ and GPU training with Ray, PyTorch, Transformers, and vLLM. A typical setup needs:

- CUDA-capable GPUs for rollout and training.
- A policy model path or Hugging Face model id.
- A scorer model path or Hugging Face model id.
- An NLI model path or model id, such as a DeBERTa MNLI model.
- Multi-turn training and validation data in parquet format.
- A retrieval service if using search-augmented tasks.

Install the package in a conda environment:

```bash
conda create -n siop python=3.10 -y
conda activate siop
pip install -r requirements.txt
pip install -e ".[vllm]"
```

If your cluster uses a custom PyTorch, CUDA, vLLM, or FlashAttention build, install those first and then run `pip install -e .`.

## Expected Inputs

Before launching training, prepare these paths:

| Variable | Meaning |
| --- | --- |
| `MODEL_PATH` | Policy model path or Hugging Face model id |
| `SIOP_MODEL` | Scorer model path or Hugging Face model id |
| `SIOP_NLI_MODEL` | Optional NLI model path or model id |
| `TRAIN_FILE` | Training parquet file |
| `VAL_FILE` | Validation parquet file |
| `RETRIEVER_URL` | HTTP endpoint for local retrieval, default `http://localhost:8000/retrieve` |
| `OUTPUT_DIR` | Directory for checkpoints and training artifacts |

The launch scripts intentionally use placeholders and environment variables. Do not hard-code private data paths, model paths, API keys, or service URLs in committed files.

## Launch Order

SIOP training normally uses three processes:

1. A retrieval service.
2. A SIOP scoring server.
3. The verl PPO trainer.

### 1. Start a Retriever

If your task uses search, start a local retrieval service. The provided wrapper expects a Search-R1-style retrieval script.

```bash
RETRIEVAL_SERVER_SCRIPT=<PATH_TO_RETRIEVAL_SERVER> \
RETRIEVER_CORPUS_DIR=<DATA_DIR>/corpus \
RETRIEVER_CONDA_ENV=<RETRIEVER_ENV> \
bash launch_retriever.sh
```

Common optional variables:

```bash
RETRIEVER_INDEX_FILE=<DATA_DIR>/corpus/e5_Flat.index
RETRIEVER_CORPUS_FILE=<DATA_DIR>/corpus/wiki-18.jsonl
RETRIEVER_MODEL=intfloat/e5-base-v2
RETRIEVER_TOPK=3
RETRIEVER_GPUS=0,1,2,3
```

If your retriever has a different CLI, replace `launch_retriever.sh` or run your service directly. The trainer only needs a compatible HTTP endpoint.

### 2. Start the SIOP Scorer

The scorer runs independent vLLM and NLI workers through Ray and exposes scoring and clustering endpoints.

```bash
CONDA_ENV=siop \
SIOP_MODEL=<SCORER_MODEL_PATH> \
SIOP_NLI_MODEL=<NLI_MODEL_PATH> \
SIOP_PORT=8390 \
SIOP_NUM_GPUS=8 \
bash launch_siop_scorer.sh
```

After launch, check the server:

```bash
curl http://localhost:8390/health
```

Training uses this endpoint through:

```bash
export SIOP_SCORER_URL=http://localhost:8390
```

### 3. Run Main SIOP Training

```bash
CONDA_ENV=siop \
MODEL_PATH=<MODEL_PATH> \
TRAIN_FILE=<DATA_DIR>/train_multiturn.parquet \
VAL_FILE=<DATA_DIR>/test_multiturn.parquet \
OUTPUT_DIR=<OUTPUT_DIR>/siop \
RETRIEVER_URL=http://localhost:8000/retrieve \
SIOP_SCORER_URL=http://localhost:8390 \
bash train_siop.sh
```

The public launcher uses the main SIOP configuration:

```text
algorithm.adv_estimator=siop
algorithm.siop_enable_two_pass=true
algorithm.siop_lambda=0.5
algorithm.siop_eta=1.0
algorithm.siop_num_refs=3
```

Useful overrides:

```bash
ROLLOUT_N=4
MAX_ASSISTANT_TURNS=5
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
N_GPUS_PER_NODE=8
TOTAL_EPOCHS=1
SAVE_FREQ=10
TEST_FREQ=50
```

## Evaluation

Run validation for a saved checkpoint:

```bash
CONDA_ENV=siop \
MODEL_PATH=<MODEL_PATH> \
VAL_FILE=<DATA_DIR>/test_multiturn.parquet \
OUTPUT_DIR=<OUTPUT_DIR>/siop_eval \
RETRIEVER_URL=http://localhost:8000/retrieve \
bash eval_siop.sh <CHECKPOINT_PATH>
```

The evaluation launcher sets `trainer.val_only=true` and resumes from the provided checkpoint path.

## Search Tool Configuration

The default public training scripts generate a temporary local search tool config at runtime. Static examples are also provided:

- `config/local_search_tool_config.yaml`: local retrieval endpoint.
- `config/serper_tool_config.yaml`: optional Serper web search backend.

`serper_api_key` is intentionally empty in the committed config. Inject real keys through your deployment environment or a private config file outside Git.

## Output Directories

Generated files are ignored by Git. Common outputs include:

```text
logs/
outputs/
checkpoints/
wandb/
cache/
search_cache/
```

The public repository should not contain training logs, checkpoints, datasets, downloaded model weights, or API keys.

## Minimal Sanity Checks

Before running a long job, check script syntax:

```bash
bash -n train_siop.sh
bash -n eval_siop.sh
bash -n launch_siop_scorer.sh
bash -n launch_retriever.sh
```

Check core SIOP modules:

```bash
python -m py_compile \
  verl/utils/siop/nli_clustering.py \
  verl/utils/siop/scoring_client.py \
  verl/utils/siop/scoring_server.py \
  verl/utils/siop/siop_process_reward.py \
  verl/utils/reward_score/siop_reward.py
```

If you use a dedicated conda environment, run the environment's Python directly or activate it first.

## Troubleshooting

**The trainer cannot reach the scorer.**

Confirm that `SIOP_SCORER_URL` points to the running scorer and that `curl http://localhost:8390/health` succeeds.

**The retriever returns no passages.**

Check that `RETRIEVER_URL` matches the retriever endpoint and that the retrieval script is using the expected index and corpus files.

**vLLM runs out of memory.**

Lower `SIOP_GPU_MEM`, `ROLLOUT_GPU_MEM`, `ROLLOUT_TP_SIZE`, batch sizes, or max sequence lengths.

**NLI clustering is slow.**

Use the scoring server with GPU-backed NLI workers, reduce rollout group size, or use a smaller NLI model.

**Training starts but SIOP rewards are zero.**

Check scorer logs in `logs/siop_scorer.log`, verify that final answers are being extracted, and confirm that multi-turn assistant ranges are present in the rollout text.

## Citation

If this code is useful for your research, please cite:

```bibtex
@misc{hu2026siop,
  title = {Self-Induced Outcome Potential: Turn-Level Credit Assignment for Agents without Verifiers},
  author = {Hu, Senkang and Dai, Yong and Han, Xudong and Fang, Zhengru and Zhao, Yuzhi and Kwong, Sam Tak Wu and Fang, Yuguang},
  year = {2026},
  eprint = {2605.04984},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2605.04984}
}
```
