from __future__ import annotations
from typing import Optional

def _map_verdict(verdict: Optional[str]) -> Optional[int]:
    if verdict == "supported":
        return 1
    if verdict == "refuted":
        return 0
    return None

def _map_label(label: float) -> int:
    return 1 if label >= 0.5 else 0

def compute_system_metrics(results: list) -> dict:
    total = len(results)
    timed_out = sum(1 for r in results if r.get("timed_out", False))
    errored = sum(1 for r in results if r.get("error") is not None)
    completed = total - timed_out - errored

    verdict_counts = {"supported": 0, "refuted": 0, "insufficient": 0, "none": 0}
    for r in results:
        v = r.get("final_verdict")
        if v in verdict_counts:
            verdict_counts[v] += 1
        else:
            verdict_counts["none"] += 1

    # Binary accuracy (excludes insufficient/erros)
    TP = TN = FP = FN = 0
    insufficient_on_true = 0
    insufficient_on_false = 0

    for r in results:
        predicted = _map_verdict(r.get("final_verdict"))
        actual = _map_label(r.get("ground_truth_label", 0.0))

        if predicted is None:
            if actual == 1:
                insufficient_on_true += 1
            elif actual == 0:
                insufficient_on_false += 1
            continue

        if predicted == 1 and actual == 1:
            TP += 1
        elif predicted == 1 and actual == 0:
            FP += 1
        elif predicted == 0 and actual == 0:
            TN += 1
        elif predicted == 0 and actual == 1:
            FN += 1
    
    binary_n = TP + TN + FP + FN
    binary_accuracy = (TP + TN) / binary_n if binary_n > 0 else 0.0
    binary_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    binary_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if (binary_precision + binary_recall) > 0 else 0.0
    )
    inclusive_accuracy = (TP + TN) / completed if completed > 0 else 0.0

    return {
        "total_claims": total,
        "completed_claims": completed,
        "timed_out_claims": timed_out,
        "errored_claims": errored,
        "verdict_counts": verdict_counts,
        "insufficient_rate": verdict_counts["insufficient"] / completed if completed > 0 else 0.0,
        "binary_n": binary_n,
        "binary_accuracy": binary_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "binary_confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
        "inclusive_accuracy": inclusive_accuracy,
        "insufficient_on_true": insufficient_on_true,
        "insufficient_on_false": insufficient_on_false,
    }