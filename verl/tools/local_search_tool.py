"""
Local Retrieval Search Tool for verl multi-turn agent loop.
Calls a local retrieval server (pyserini/FAISS on Wikipedia) instead of external API.
No API key needed, no rate limits, deterministic results.
"""

import asyncio
import logging
import os
from typing import Any

import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)


class LocalSearchTool(BaseTool):
    """Search tool using a local retrieval server (e5/BM25 on Wikipedia)."""

    def __init__(self, config: dict, tool_schema=None):
        self.retrieval_url = config.get(
            "retrieval_url",
            os.environ.get("RETRIEVAL_URL", "http://localhost:8000/retrieve"),
        )
        self.top_k = config.get("search_top_k", 3)
        self.timeout = float(config.get("request_timeout", 10.0))
        self._session = requests.Session()
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query_list = parameters.get("query", parameters.get("query_list", []))
        if isinstance(query_list, str):
            query_list = [query_list]
        query_list = query_list[:3]

        results = await asyncio.to_thread(self._batch_search, query_list)
        all_formatted = []
        for query, docs in zip(query_list, results):
            all_formatted.append(self._format_results(query, docs))

        response_text = "\n\n".join(all_formatted)
        return ToolResponse(text=response_text), 0.0, {"num_queries": len(query_list)}

    def _batch_search(self, query_list: list[str]) -> list[list[dict]]:
        """Call local retrieval server."""
        try:
            resp = self._session.post(
                self.retrieval_url,
                json={"queries": query_list, "topk": self.top_k},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("result", [])

            # Normalize response format: each item may be
            #   {"document": {...}, "score": float}  (return_scores=true)
            #   or just a dict with "contents"/"title"/"text"
            parsed = []
            for group in raw_results:
                docs = []
                for item in group:
                    if isinstance(item, dict) and "document" in item:
                        doc = item["document"]
                    else:
                        doc = item
                    # Extract title and text from "contents" if needed
                    contents = doc.get("contents", "")
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    if not title and contents:
                        parts = contents.split("\n", 1)
                        title = parts[0].strip()
                        text = parts[1].strip() if len(parts) > 1 else ""
                    docs.append({"title": title, "snippet": text})
                parsed.append(docs)
            return parsed
        except Exception as e:
            logger.warning(f"[LocalSearchTool] Retrieval failed: {e}")
            return [[] for _ in query_list]

    def _format_results(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            lines.append(f"Doc {i} (Title: {title})\n{snippet}")
        return "\n".join(lines)
