from __future__ import annotations
from evaluation.parsers import parse_advisor_analysis

def compute_advisor_metrics(result:dict) -> dict:
    analysis = parse_advisor_analysis(result.get("advisor_analysis"))

    category_counts = analysis["category_counts"]
    total_turns = analysis["n_debate_turns"]

    # Compute per-category rates (avoid division by zero)
    category_rates = {
        cat: count / total_turns if total_turns > 0 else 0.0
        for cat, count in category_counts.items()
    }

    # Check how many of the 5 advice sections have real content
    advice_text = result.get("advisor_advie") or ""

    advice_sections = [
        "Highest-priority valid points:",
        "Remaining gaps to resolve:",
        "Assertions that need stronger scrutiny:",
        "Low-value or noisy points to discount:",
        "Verifier focus:",
    ]

    sections_with_content = sum(
        1 for section in advice_sections
        if section in advice_text
    )

    advice_section_coverage = sections_with_content / len(advice_sections)

    return {
        "classification_source": analysis["classification_source"],
        "n_debate_turns_anlayzed": total_turns,
        "category_counts": category_counts,
        "category_rates": category_rates,
        "valid_rate": category_rates["valid"],
        "redundancy_rate": category_rates["redundant"],
        "irrelevancy_rate": category_rates["irrelevant"],
        "advice_section_coverage": advice_section_coverage,
    }
