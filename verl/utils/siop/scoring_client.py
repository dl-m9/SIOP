"""HTTP client for the SIOP scoring server."""

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SiopScoringClient:

    def __init__(self, service_url=None, timeout=120.0, max_retries=3):
        self.service_url = service_url or os.environ.get("SIOP_SCORER_URL", "http://localhost:8390")
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(total=max_retries, backoff_factor=1.0, status_forcelist=[502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.service_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "healthy"
        except Exception as e:
            print(f"[SIOP-Client] Health check failed: {e}", flush=True)
            return False

    def cluster_batch(self, groups: list[dict], timeout: float = None) -> list[dict]:
        """Cluster answers for multiple query groups.

        Args:
            groups: list of {group_id, question, answers, strict}

        Returns:
            list of {group_id, semantic_ids, cluster_info}
        """
        if not groups:
            return []

        timeout = timeout or max(self.timeout, self.timeout * len(groups) / 32.0)
        try:
            r = self.session.post(
                f"{self.service_url}/cluster",
                json={"groups": groups},
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            print(f"[SIOP-Client] Clustering {len(groups)} groups: "
                  f"{data.get('elapsed_ms', 0):.0f}ms", flush=True)
            return data["results"]
        except Exception as exc:
            print(f"[SIOP-Client] Clustering failed: {exc}", flush=True)
            return []

    def nli_score_batch(self, pairs: list[dict], chunk_size: int = 512) -> list[float]:
        """Entailment probabilities for (premise, hypothesis) dict pairs."""
        if not pairs:
            return []
        all_scores = [0.0] * len(pairs)
        for start in range(0, len(pairs), chunk_size):
            chunk = pairs[start:start + chunk_size]
            timeout = max(self.timeout, self.timeout * len(chunk) / 256.0)
            try:
                r = self.session.post(
                    f"{self.service_url}/nli_score",
                    json={"pairs": chunk},
                    timeout=timeout,
                )
                r.raise_for_status()
                data = r.json()
                for i, s in enumerate(data["scores"]):
                    all_scores[start + i] = s
            except Exception as exc:
                print(f"[SIOP-Client] NLI chunk {start}-{start+len(chunk)} failed: {exc}", flush=True)
        return all_scores

    def score_batch(self, items: list[dict], chunk_size: int = 256) -> list[float]:
        """Score pseudo-inputs.  items: list of {prompt_ids, response_ids, ref_start, ref_end}."""
        if not items:
            return []

        all_scores = [float("-inf")] * len(items)

        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            timeout = max(self.timeout, self.timeout * len(chunk) / 128.0)
            try:
                r = self.session.post(
                    f"{self.service_url}/score",
                    json={"items": chunk},
                    timeout=timeout,
                )
                r.raise_for_status()
                data = r.json()
                for i, s in enumerate(data["scores"]):
                    all_scores[start + i] = s
                print(f"[SIOP-Client] Chunk {start}-{start+len(chunk)}: "
                      f"{data.get('elapsed_ms', 0):.0f}ms", flush=True)
            except Exception as exc:
                print(f"[SIOP-Client] Chunk {start}-{start+len(chunk)} failed: {exc}", flush=True)

        return all_scores


# Singleton
_CLIENT = None


def get_siop_scoring_client():
    """Get or create the singleton client.  Returns None if unavailable."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    url = os.environ.get("SIOP_SCORER_URL")
    if not url:
        return None

    client = SiopScoringClient(service_url=url)
    if client.health_check():
        print(f"[SIOP-Client] Connected to {url}", flush=True)
        _CLIENT = client
        return _CLIENT
    else:
        print(f"[SIOP-Client] Server at {url} not available", flush=True)
        return None
