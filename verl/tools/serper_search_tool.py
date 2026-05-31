"""
Serper API Search Tool for verl multi-turn agent loop.
Directly calls Serper API (Google search) without needing a retrieval service.
Supports local disk cache to avoid redundant API calls.
"""

import json
import hashlib
import http.client
import asyncio
import logging
import os
import re
import threading
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

# Process-level stats shared across all SerperSearchTool instances in the same worker
_stats_lock = threading.Lock()
_stats = {"api_calls": 0, "cache_hits": 0}
_stats_file = os.environ.get("SERPER_STATS_FILE", "logs/search_stats.log")
_stats_thread_started = False


def _stats_reporter():
    """Background thread: write stats to file every 60s."""
    while True:
        time.sleep(60)
        with _stats_lock:
            api, hits = _stats["api_calls"], _stats["cache_hits"]
        total = api + hits
        if total > 0:
            rate = f"{hits * 100 // total}%"
            try:
                Path(_stats_file).parent.mkdir(parents=True, exist_ok=True)
                with open(_stats_file, "a") as f:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{ts} api={api} cache={hits} total={total} hit_rate={rate}\n")
            except Exception:
                pass


def _ensure_stats_thread():
    global _stats_thread_started
    if not _stats_thread_started:
        _stats_thread_started = True
        t = threading.Thread(target=_stats_reporter, daemon=True)
        t.start()


# Strip from both ends only; repeated so "??foo??" -> "foo"
_OUTER_PUNCT_RE = re.compile(
    r"^[\s\-–—.,;:!?。，、；：！？…\"'\"«»「」『』（）()\[\]]+|[\s\-–—.,;:!?。，、；：！？…\"'\"«»「」『』（）()\[\]]+$"
)


def _normalize_query_for_cache_key(query: str) -> str:
    """Conservative normalization for cache keys only (API still uses raw query)."""
    s = unicodedata.normalize("NFKC", query)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    prev = None
    while prev != s:
        prev = s
        s = _OUTER_PUNCT_RE.sub("", s)
        s = s.strip()
    return s


class SerperSearchTool(BaseTool):
    """Search tool using Serper API (Google search) directly."""

    def __init__(self, config: dict, tool_schema=None):
        self.serper_api_key = config.get("serper_api_key", "")
        self.top_k = config.get("search_top_k", 3)
        self.region = config.get("search_region", "us")
        self.lang = config.get("search_lang", "en")
        self.max_retries = config.get("max_retries", 3)
        self.request_timeout = float(config.get("request_timeout", 10.0))

        # Local cache
        cache_dir = config.get("cache_dir", None) or os.environ.get("SERPER_CACHE_DIR", "./search_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        _ensure_stats_thread()
        super().__init__(config, tool_schema)

    def _cache_key(self, query: str) -> str:
        normalized = _normalize_query_for_cache_key(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _cache_get(self, query: str) -> Optional[list[dict]]:
        path = self.cache_dir / f"{self._cache_key(query)}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data["results"]
            except Exception:
                return None
        return None

    def _cache_put(self, query: str, results: list[dict]):
        path = self.cache_dir / f"{self._cache_key(query)}.json"
        try:
            path.write_text(json.dumps({"query": query, "results": results}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[SerperSearchTool] Failed to write cache: {e}")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query_list = parameters.get("query", parameters.get("query_list", []))
        if isinstance(query_list, str):
            query_list = [query_list]
        query_list = query_list[:3]

        all_results = []
        for query in query_list:
            results = await asyncio.to_thread(self._search, query)
            formatted = self._format_results(query, results)
            all_results.append(formatted)

        response_text = "\n\n".join(all_results)
        return ToolResponse(text=response_text), 0.0, {"num_queries": len(query_list)}

    def _search(self, query: str) -> list[dict]:
        if os.environ.get("SERPER_MOCK", "0") == "1":
            return [{"title": f"Mock result for: {query[:60]}", "snippet": f"Mock search result for: {query[:100]}"}]

        cached = self._cache_get(query)
        if cached is not None:
            with _stats_lock:
                _stats["cache_hits"] += 1
            return cached

        if not self.serper_api_key:
            return []
        last_error = None
        for attempt in range(self.max_retries):
            conn = None
            try:
                https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
                if https_proxy:
                    parsed = urllib.parse.urlparse(https_proxy)
                    conn = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=self.request_timeout)
                    conn.set_tunnel("google.serper.dev", 443)
                else:
                    conn = http.client.HTTPSConnection("google.serper.dev", timeout=self.request_timeout)
                payload = json.dumps({"q": query, "num": self.top_k, "gl": self.region, "hl": self.lang})
                headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                data = json.loads(res.read().decode("utf-8"))
                if data and "organic" in data:
                    results = data["organic"][:self.top_k]
                    self._cache_put(query, results)
                    with _stats_lock:
                        _stats["api_calls"] += 1
                    return results
                return []
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(1)
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
        return []

    def _format_results(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            lines.append(f"Doc {i} (Title: {title})\n{snippet}")
        return "\n".join(lines)
