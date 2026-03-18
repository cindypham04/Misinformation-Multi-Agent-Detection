from __future__ import annotations

from typing import Any, Dict, List

from langchain_tavily import TavilySearch

from ..config import AppConfig
from ..schemas import Evidence


def tavily_search(
    *,
    query: str,
    config: AppConfig,
) -> List[Evidence]:
    """Run a Tavily search and normalize results into `Evidence` records."""
    search = TavilySearch(
        tavily_api_key=config.tavily_api_key,
        max_results=config.tavily_max_results,
        topic=config.tavily_topic,
        include_domains=config.reliable_domains,
    )

    raw = search.invoke({"query": query})
    return _normalize_tavily_results(raw)


def _normalize_tavily_results(raw: Any) -> List[Evidence]:
    # Tavily output shape can vary; normalize to a stable internal schema.
    # `langchain_tavily` may return a list[dict] or a dict with `results`.
    results: Any
    if isinstance(raw, dict) and "results" in raw:
        results = raw.get("results", [])
    else:
        results = raw

    if not isinstance(results, list):
        return []

    normalized: List[Evidence] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "")
        url = str(item.get("url", "") or "")
        content = str(item.get("content", "") or item.get("snippet", "") or "")
        score_val = item.get("score", 0.0)
        try:
            score = float(score_val)
        except Exception:
            score = 0.0

        source = ""
        # Some responses include a "source" field; otherwise infer from URL.
        if "source" in item and item["source"]:
            source = str(item["source"])
        elif url:
            source = url.split("/")[2] if "://" in url and len(url.split("/")) > 2 else url

        if not url and not content and not title:
            continue

        normalized.append(
            Evidence(
                title=title,
                url=url,
                content=content,
                score=score,
                source=source,
            )
        )

    return normalized


def search_tool_call(input: Dict[str, Any], *, config: AppConfig) -> Dict[str, List[Evidence]]:
    """ToolNode-friendly callable. Input is expected to contain {'query': <str>}."""
    query = str(input.get("query", "") or "").strip()
    if not query:
        return {"results": []}
    return {"results": tavily_search(query=query, config=config)}

