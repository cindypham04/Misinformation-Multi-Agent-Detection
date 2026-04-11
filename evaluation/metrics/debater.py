from __future__ import annotations
from evaluation.parsers import parse_debate_log_metrics

def compute_debater_metrics(result: dict) -> dict:
    query_count = result.get("evidence_pool_query_count", 0)
    total_items = result.get("evidence_pool_total_items", 0)
    debate_log = result.get("debate_log", [])

    log_metrics = parse_debate_log_metrics(debate_log)

    return {
        "evidence_pool_query_count": query_count,
        "evidence_pool_total_items": total_items,
        "evidence_retrieval_rate": total_items / query_count if query_count > 0 else 0.0,
        **log_metrics,
    }