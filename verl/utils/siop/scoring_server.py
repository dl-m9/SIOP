"""SIOP scoring server — multi-GPU vLLM + DeBERTa service.

Each GPU runs an independent vLLM instance + DeBERTa NLI model via Ray workers.
Provides two endpoints:
  POST /cluster — NLI semantic clustering (parallel across workers)
  POST /score   — prompt log-prob scoring (parallel across workers)

Usage:
    python -m verl.utils.siop.scoring_server \
        --model <SCORER_MODEL_PATH> \
        --port 8390 --num-gpus 8
"""

import argparse
import asyncio
import os
import time
from typing import Optional

import ray
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ScoreItem(BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    ref_start: int
    ref_end: int


class ScoreRequest(BaseModel):
    items: list[ScoreItem]


class ScoreResponse(BaseModel):
    scores: list[float]
    elapsed_ms: float


class ClusterGroup(BaseModel):
    """One query's rollout answers to cluster."""
    group_id: str
    question: str
    answers: list[str]
    strict: bool = True


class ClusterRequest(BaseModel):
    groups: list[ClusterGroup]


class ClusterGroupResult(BaseModel):
    group_id: str
    semantic_ids: list[int]
    cluster_info: dict  # {cluster_id: {count, frequency, reference, members}}


class ClusterResponse(BaseModel):
    results: list[ClusterGroupResult]
    elapsed_ms: float


class NliPair(BaseModel):
    premise: str
    hypothesis: str


class NliRequest(BaseModel):
    pairs: list[NliPair]


class NliResponse(BaseModel):
    scores: list[float]
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Ray Worker — one per GPU, loads vLLM + DeBERTa
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class ScoringWorker:
    """Single-GPU worker with vLLM (scoring) and DeBERTa (NLI clustering)."""

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.3,
                 max_model_len: int = 8192,
                 nli_model_name: str = "microsoft/deberta-v2-xlarge-mnli"):
        import torch
        from vllm import LLM

        # Determine device from Ray's CUDA_VISIBLE_DEVICES
        self.device = "cuda:0"  # Ray sets CUDA_VISIBLE_DEVICES to single GPU
        self.gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '?')

        # Load vLLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
        )
        print(f"[ScoringWorker GPU {self.gpu_id}] vLLM loaded", flush=True)

        # Load DeBERTa NLI model on the same GPU
        from verl.utils.siop.nli_clustering import EntailmentDeberta
        self.nli_model = EntailmentDeberta(nli_model_name, device=self.device)
        print(f"[ScoringWorker GPU {self.gpu_id}] DeBERTa loaded", flush=True)

    def score_batch(self, items_data: list[dict]) -> list[float]:
        """Score a batch of items. Takes dicts for Ray serialization."""
        if not items_data:
            return []

        import torch
        from vllm import SamplingParams

        full_prompts = [d["prompt_ids"] + d["response_ids"] for d in items_data]

        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=1,
        )

        # vLLM 0.11 uses TokensPrompt format
        prompts = [{"prompt_token_ids": p} for p in full_prompts]
        outputs = self.llm.generate(
            prompts,
            sampling_params=params,
            use_tqdm=False,
        )

        scores = []
        for i, output in enumerate(outputs):
            d = items_data[i]
            abs_start = len(d["prompt_ids"]) + d["ref_start"]
            abs_end = len(d["prompt_ids"]) + d["ref_end"]

            logprobs_list = output.prompt_logprobs
            if logprobs_list is None:
                scores.append(float("-inf"))
                continue

            ref_lps = []
            for pos in range(abs_start, min(abs_end, len(logprobs_list))):
                lp_dict = logprobs_list[pos]
                if lp_dict is None:
                    continue
                token_id = full_prompts[i][pos]
                if token_id in lp_dict:
                    ref_lps.append(lp_dict[token_id].logprob)

            scores.append(sum(ref_lps) / len(ref_lps) if ref_lps else float("-inf"))

        torch.cuda.empty_cache()
        return scores

    def cluster_group(self, group_id: str, question: str, answers: list[str],
                      strict: bool = True) -> dict:
        """Cluster one group of answers via NLI."""
        from verl.utils.siop.nli_clustering import cluster_answers, get_cluster_info

        if len(answers) < 2:
            semantic_ids = list(range(len(answers)))
            cluster_info = get_cluster_info(semantic_ids, answers)
        else:
            semantic_ids = cluster_answers(answers, question, self.nli_model, strict=strict)
            cluster_info = get_cluster_info(semantic_ids, answers)

        # Convert cluster_info keys to strings for JSON
        return {
            "group_id": group_id,
            "semantic_ids": semantic_ids,
            "cluster_info": {str(k): v for k, v in cluster_info.items()},
        }

    def nli_score_batch(self, pairs: list[dict], sub_batch: int = 16) -> list[float]:
        """Entailment probability for (premise, hypothesis) pairs.

        Chunks to small sub-batches to cap DeBERTa forward peak memory,
        then empties the torch cache so the pool returns to the driver
        (vLLM wake_up on the same GPU needs that memory back).
        """
        import torch
        import torch.nn.functional as F
        if not pairs:
            return []
        results: list[float] = []
        for start in range(0, len(pairs), sub_batch):
            chunk = pairs[start:start + sub_batch]
            premises = [p["premise"] for p in chunk]
            hypotheses = [p["hypothesis"] for p in chunk]
            inputs = self.nli_model.tokenizer(
                premises, hypotheses, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.nli_model.device)
            with torch.no_grad():
                logits = self.nli_model.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[:, 2]
            results.extend(probs.cpu().tolist())
            del inputs, logits, probs
        torch.cuda.empty_cache()
        return results

    def health_check(self) -> bool:
        return self.llm is not None and self.nli_model is not None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SIOP Scoring Server")
_workers: Optional[list] = None
_worker_lock: Optional[asyncio.Lock] = None
_worker_index: int = 0


@app.on_event("startup")
async def startup():
    global _workers, _worker_lock, _worker_index

    model_path = os.environ["SIOP_SCORER_MODEL"]
    num_gpus = int(os.environ.get("SIOP_SCORER_NUM_GPUS", "8"))
    gpu_mem = float(os.environ.get("SIOP_SCORER_GPU_MEM", "0.3"))
    max_model_len = int(os.environ.get("SIOP_SCORER_MAX_MODEL_LEN", "8192"))
    nli_model = os.environ.get("SIOP_SCORER_NLI_MODEL", "microsoft/deberta-v2-xlarge-mnli")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"[SIOP-Scorer] Creating {num_gpus} workers (gpu_mem={gpu_mem})...", flush=True)

    _workers = []
    for i in range(num_gpus):
        w = ScoringWorker.remote(model_path, gpu_mem, max_model_len, nli_model)
        _workers.append(w)
        print(f"[SIOP-Scorer] Created worker {i+1}/{num_gpus}", flush=True)

    # Wait for all workers to be ready
    ray.get([w.health_check.remote() for w in _workers])
    print(f"[SIOP-Scorer] All {num_gpus} workers ready!", flush=True)

    _worker_lock = asyncio.Lock()
    _worker_index = 0


@app.get("/health")
async def health():
    if _workers is None:
        return {"status": "loading"}
    return {"status": "healthy", "num_workers": len(_workers)}


@app.post("/cluster", response_model=ClusterResponse)
async def cluster(request: ClusterRequest):
    """Cluster answers for multiple query groups in parallel across workers."""
    global _worker_index

    if not _workers:
        return ClusterResponse(results=[], elapsed_ms=0)

    groups = request.groups
    if not groups:
        return ClusterResponse(results=[], elapsed_ms=0)

    t0 = time.time()
    num_workers = len(_workers)

    # Distribute groups round-robin
    async with _worker_lock:
        start_idx = _worker_index
        _worker_index = (_worker_index + len(groups)) % num_workers

    worker_counts = [0] * num_workers
    refs = []
    ref_order = []  # track original index
    for i, g in enumerate(groups):
        w_idx = (start_idx + i) % num_workers
        ref = _workers[w_idx].cluster_group.remote(
            g.group_id, g.question, g.answers, g.strict
        )
        refs.append(ref)
        ref_order.append(i)
        worker_counts[w_idx] += 1

    total_answers = sum(len(g.answers) for g in groups)
    dist_str = ",".join(str(c) for c in worker_counts)
    print(f"[/cluster] {len(groups)} groups, {total_answers} answers -> workers [{dist_str}]", flush=True)

    # Gather results
    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(None, lambda: ray.get(refs))

    results = []
    for raw in raw_results:
        results.append(ClusterGroupResult(
            group_id=raw["group_id"],
            semantic_ids=raw["semantic_ids"],
            cluster_info=raw["cluster_info"],
        ))

    elapsed = (time.time() - t0) * 1000
    print(f"[/cluster] done in {elapsed:.0f}ms", flush=True)
    return ClusterResponse(results=results, elapsed_ms=elapsed)


@app.post("/nli_score", response_model=NliResponse)
async def nli_score(request: NliRequest):
    global _worker_index

    if not _workers or not request.pairs:
        return NliResponse(scores=[], elapsed_ms=0)

    t0 = time.time()
    pairs_data = [p.model_dump() for p in request.pairs]
    num_workers = len(_workers)
    chunks: list[list[dict]] = [[] for _ in range(num_workers)]
    chunk_indices: list[list[int]] = [[] for _ in range(num_workers)]

    async with _worker_lock:
        start_idx = _worker_index
        _worker_index = (_worker_index + len(pairs_data)) % num_workers

    for i, d in enumerate(pairs_data):
        w_idx = (start_idx + i) % num_workers
        chunks[w_idx].append(d)
        chunk_indices[w_idx].append(i)

    refs = []
    active_workers = []
    for w_idx in range(num_workers):
        if chunks[w_idx]:
            refs.append(_workers[w_idx].nli_score_batch.remote(chunks[w_idx]))
            active_workers.append(w_idx)

    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(None, lambda: ray.get(refs))

    all_scores = [0.0] * len(pairs_data)
    for result, w_idx in zip(raw_results, active_workers):
        for s, orig_idx in zip(result, chunk_indices[w_idx]):
            all_scores[orig_idx] = s

    elapsed = (time.time() - t0) * 1000
    print(f"[/nli_score] {len(pairs_data)} pairs in {elapsed:.0f}ms", flush=True)
    return NliResponse(scores=all_scores, elapsed_ms=elapsed)


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest):
    global _worker_index

    if not _workers:
        return ScoreResponse(scores=[], elapsed_ms=0)

    items = request.items
    if not items:
        return ScoreResponse(scores=[], elapsed_ms=0)

    t0 = time.time()

    # Convert pydantic to dicts for Ray serialization
    items_data = [item.model_dump() for item in items]

    # Distribute items round-robin across workers
    num_workers = len(_workers)
    chunks: list[list[dict]] = [[] for _ in range(num_workers)]
    chunk_indices: list[list[int]] = [[] for _ in range(num_workers)]

    async with _worker_lock:
        start_idx = _worker_index
        _worker_index = (_worker_index + len(items_data)) % num_workers

    for i, item_data in enumerate(items_data):
        w_idx = (start_idx + i) % num_workers
        chunks[w_idx].append(item_data)
        chunk_indices[w_idx].append(i)

    dist_str = ",".join(str(len(c)) for c in chunks)
    print(f"[/score] {len(items_data)} items -> workers [{dist_str}]", flush=True)

    # Submit to workers in parallel
    refs = []
    active_workers = []
    for w_idx in range(num_workers):
        if chunks[w_idx]:
            ref = _workers[w_idx].score_batch.remote(chunks[w_idx])
            refs.append(ref)
            active_workers.append(w_idx)

    # Gather results
    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(None, lambda: ray.get(refs))

    # Reassemble in original order
    all_scores = [0.0] * len(items_data)
    for result, w_idx in zip(raw_results, active_workers):
        for score_val, orig_idx in zip(result, chunk_indices[w_idx]):
            all_scores[orig_idx] = score_val

    elapsed = (time.time() - t0) * 1000
    print(f"[/score] done {len(items_data)} items in {elapsed:.0f}ms", flush=True)
    return ScoreResponse(scores=all_scores, elapsed_ms=elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8390)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--nli-model", default="microsoft/deberta-v2-xlarge-mnli")
    args = parser.parse_args()

    os.environ["SIOP_SCORER_MODEL"] = args.model
    os.environ["SIOP_SCORER_NUM_GPUS"] = str(args.num_gpus)
    os.environ["SIOP_SCORER_GPU_MEM"] = str(args.gpu_memory_utilization)
    os.environ["SIOP_SCORER_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["SIOP_SCORER_NLI_MODEL"] = args.nli_model

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
