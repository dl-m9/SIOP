"""NLI-only server — DeBERTa clustering + entailment scoring, no vLLM.

Use when only EMPO/TTRL/EMPO-style cluster-frequency rewards are needed
(no log-prob scoring). Light-weight: 1 GPU at ~2GB, or CPU fallback.

Endpoints:
  POST /cluster    — semantic clustering of rollout answers
  POST /nli_score  — pairwise entailment probability
  GET  /health     — readiness probe

Usage:
    python -m verl.utils.siop.nli_server --port 8390 \
        --num-workers 1 --device cuda:0
    # or CPU
    python -m verl.utils.siop.nli_server --port 8390 --device cpu
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
# Schema (mirrors scoring_server.py for client compatibility)
# ---------------------------------------------------------------------------

class ClusterGroup(BaseModel):
    group_id: str
    question: str
    answers: list[str]
    strict: bool = True


class ClusterRequest(BaseModel):
    groups: list[ClusterGroup]


class ClusterGroupResult(BaseModel):
    group_id: str
    semantic_ids: list[int]
    cluster_info: dict


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
# Worker — DeBERTa only
# ---------------------------------------------------------------------------

class _NliWorkerBase:
    def __init__(self, nli_model_name: str, device: str):
        from verl.utils.siop.nli_clustering import EntailmentDeberta
        self.device = device
        self.nli_model = EntailmentDeberta(nli_model_name, device=device)
        print(f"[NliWorker {device}] DeBERTa loaded", flush=True)

    def cluster_group(self, group_id: str, question: str, answers: list[str],
                      strict: bool = True) -> dict:
        from verl.utils.siop.nli_clustering import cluster_answers, get_cluster_info
        if len(answers) < 2:
            semantic_ids = list(range(len(answers)))
            cluster_info = get_cluster_info(semantic_ids, answers)
        else:
            semantic_ids = cluster_answers(answers, question, self.nli_model, strict=strict)
            cluster_info = get_cluster_info(semantic_ids, answers)
        return {
            "group_id": group_id,
            "semantic_ids": semantic_ids,
            "cluster_info": {str(k): v for k, v in cluster_info.items()},
        }

    def nli_score_batch(self, pairs: list[dict], sub_batch: int = 16) -> list[float]:
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
        if str(self.nli_model.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return results

    def health_check(self) -> bool:
        return self.nli_model is not None


@ray.remote(num_gpus=1)
class _GpuNliWorker(_NliWorkerBase):
    def __init__(self, nli_model_name: str):
        super().__init__(nli_model_name, device="cuda:0")


@ray.remote(num_cpus=4)
class _CpuNliWorker(_NliWorkerBase):
    def __init__(self, nli_model_name: str):
        super().__init__(nli_model_name, device="cpu")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SIOP NLI-only Server")
_workers: Optional[list] = None
_worker_lock: Optional[asyncio.Lock] = None
_worker_index: int = 0


@app.on_event("startup")
async def startup():
    global _workers, _worker_lock, _worker_index

    nli_model = os.environ.get("SIOP_NLI_MODEL", "microsoft/deberta-v2-xlarge-mnli")
    num_workers = int(os.environ.get("SIOP_NLI_NUM_WORKERS", "1"))
    device = os.environ.get("SIOP_NLI_DEVICE", "cuda")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"[NLI-Server] {num_workers} {device} workers, model={nli_model}", flush=True)

    _workers = []
    if device.startswith("cuda"):
        for i in range(num_workers):
            w = _GpuNliWorker.remote(nli_model)
            _workers.append(w)
    else:
        for i in range(num_workers):
            w = _CpuNliWorker.remote(nli_model)
            _workers.append(w)

    ray.get([w.health_check.remote() for w in _workers])
    print(f"[NLI-Server] all {num_workers} workers ready", flush=True)
    _worker_lock = asyncio.Lock()


@app.get("/health")
async def health():
    if _workers is None:
        return {"status": "loading"}
    return {"status": "healthy", "num_workers": len(_workers), "mode": "nli-only"}


@app.post("/cluster", response_model=ClusterResponse)
async def cluster(request: ClusterRequest):
    global _worker_index
    if not _workers or not request.groups:
        return ClusterResponse(results=[], elapsed_ms=0)

    t0 = time.time()
    groups = request.groups
    num_workers = len(_workers)

    async with _worker_lock:
        start_idx = _worker_index
        _worker_index = (_worker_index + len(groups)) % num_workers

    refs = []
    for i, g in enumerate(groups):
        w_idx = (start_idx + i) % num_workers
        ref = _workers[w_idx].cluster_group.remote(
            g.group_id, g.question, g.answers, g.strict
        )
        refs.append(ref)

    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(None, lambda: ray.get(refs))

    results = [
        ClusterGroupResult(
            group_id=raw["group_id"],
            semantic_ids=raw["semantic_ids"],
            cluster_info=raw["cluster_info"],
        )
        for raw in raw_results
    ]
    elapsed = (time.time() - t0) * 1000
    print(f"[/cluster] {len(groups)} groups in {elapsed:.0f}ms", flush=True)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8390)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--nli-model", type=str,
                        default="microsoft/deberta-v2-xlarge-mnli",
                        help="HF id or absolute path to local snapshot")
    args = parser.parse_args()

    os.environ["SIOP_NLI_MODEL"] = args.nli_model
    os.environ["SIOP_NLI_NUM_WORKERS"] = str(args.num_workers)
    os.environ["SIOP_NLI_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
