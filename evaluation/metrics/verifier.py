from __future__ import annotations 
from evaluation.parsers import parse_final_report

def compute_verifier_metrics(result: dict) -> dict:
    report = parse_final_report(result.get("final_report"))

    query_count = result.get("verifier_evidence_query_count", 0)
    total_items = result.get("verifier_evidence_total_items", 0)

    neg_score = report["negative_score"]
    aff_score = report["affirmative_score"]

    if neg_score is not None and aff_score is not None:
        score_margin = abs(neg_score - aff_score)
    else:
        score_margin = None

    return {
        "decision_source": report["decision_source"],
        "n_verifier_queries": query_count,
        "n_verifier_evidence_items": total_items,
        "verifier_evidence_discovery_rate": total_items / query_count if query_count > 0 else 0.0,
        "negative_score": neg_score,
        "affirmative_score": aff_score,
        "score_margin": score_margin,
        "verdict": result.get("final_verdict"),
    }
