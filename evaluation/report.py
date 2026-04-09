from __future__ import annotations
from evaluation.metrics.system import compute_system_metrics
from evaluation.metrics.debater import compute_debater_metrics
from evaluation.metrics.advisor import compute_advisor_metrics
from evaluation.metrics.verifier import compute_verifier_metrics



def generate_weakness_report(results: list) -> dict:
    system = compute_system_metrics(results)
    debater_metrics = [compute_debater_metrics(r) for r in results]
    advisor_metrics = [compute_advisor_metrics(r) for r in results]
    verifier_metrics = [compute_verifier_metrics(r) for r in results]

    def avg(metrics_list: list, key: str) -> float:
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    
    debater_weaknesses = {
        "avg_evidence_retrieval_rate": avg(debater_metrics, "evidence_retrieval_rate"),
        "avg_llm_argument_rate": avg(debater_metrics, "llm_argument_rate"),
        "avg_url_cite_rate": avg(debater_metrics, "url_cite_rate"),
        "low_evidence_claim_ids": [
            r["claim_id"] for r, m in zip(results, debater_metrics)
            if m["evidence_pool_total_items"] < 3
        ],
        "fallback_argument_examples": [
            r["claim"] for r, m in zip(results, debater_metrics)
            if m["fallback_argument_count"] > 0
        ][:5],
    }

    advisor_weaknesses = {
        "classification_source_dist": {
            "Ollama LLM": sum(1 for m in advisor_metrics if m["classification_source"] == "Ollama LLM"),
            "heuristic fallback": sum(1 for m in advisor_metrics if m["classification_source"] == "heuristic fallback"),
        },
        "avg_valid_rate": avg(advisor_metrics, "valid_rate"),
        "avg_redundancy_rate": avg(advisor_metrics, "redundancy_rate"),
        "avg_irrelevancy_rate": avg(advisor_metrics, "irrelevancy_rate"),
        "avg_advice_section_coverage": avg(advisor_metrics, "advice_section_coverage"),
        "high_irrelevancy_claim_ids": [
            r["claim_id"] for r, m in zip(results, advisor_metrics)
            if m["irrelevancy_rate"] > 0.5
        ],
    }

    verifier_weaknesses = {
        "decision_source_dist": {
            "Ollama LLM": sum(1 for m in verifier_metrics if m["decision_source"] == "Ollama LLM"),
            "heuristic fallback": sum(1 for m in verifier_metrics if m["decision_source"] == "heuristic fallback"),
        },
        "avg_verifier_evidence_discovery_rate": avg(verifier_metrics, "verifier_evidence_discovery_rate"),
        "zero_margin_count": sum(1 for m in verifier_metrics if m["score_margin"] == 0),
        "low_evidence_claim_ids": [
            r["claim_id"] for r, m in zip(results, verifier_metrics)
            if m["n_verifier_evidence_items"] == 0
        ],
    }

    cm = system["binary_confusion_matrix"]
    error_analysis = {
        "false_positives": [
            {"claim_id": r["claim_id"], "claim": r["claim"], "verdict": r.get("final_verdict"), "label": r.get("ground_truth_label")}
            for r in results
            if r.get("final_verdict") == "supported" and r.get("ground_truth_label", 1.0) < 0.5
        ][:20],
        "false_negatives": [
            {"claim_id": r["claim_id"], "claim": r["claim"], "verdict": r.get("final_verdict"), "label": r.get("ground_truth_label")}
            for r in results
            if r.get("final_verdict") == "refuted" and r.get("ground_truth_label", 0.0) >= 0.5
        ][:20],
        "insufficient_on_true": [
            {"claim_id": r["claim_id"], "claim": r["claim"]}
            for r in results
            if r.get("final_verdict") == "insufficient" and r.get("ground_truth_label", 0.0) >= 0.5
        ][:20],
        "insufficient_on_false": [
            {"claim_id": r["claim_id"], "claim": r["claim"]}
            for r in results
            if r.get("final_verdict") == "insufficient" and r.get("ground_truth_label", 1.0) < 0.5
        ][:20],
    }

    return {
        "summary": {
            "total_claims": system["total_claims"],
            "completed_claims": system["completed_claims"],
            "binary_accuracy": system["binary_accuracy"],
            "inclusive_accuracy": system["inclusive_accuracy"],
            "binary_f1": system["binary_f1"],
            "insufficient_rate": system["insufficient_rate"],
            "verdict_counts": system["verdict_counts"],
            "confusion_matrix": cm,
        },
        "debater_weaknesses": debater_weaknesses,
        "advisor_weaknesses": advisor_weaknesses,
        "verifier_weaknesses": verifier_weaknesses,
        "error_analysis": error_analysis,
        "per_claim_metrics": [
            {
                "claim_id": r["claim_id"],
                "claim": r["claim"],
                "label": r.get("ground_truth_label"),
                "verdict": r.get("final_verdict"),
                "debater": debater_metrics[i],
                "advisor": advisor_metrics[i],
                "verifier": verifier_metrics[i],
            }
            for i, r in enumerate(results)
        ],
    }