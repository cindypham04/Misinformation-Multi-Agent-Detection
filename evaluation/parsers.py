"""
Parsers for agent free-text output fields.
All parsers are pure functions and never raise — on failure they return a safe empty dict.
"""

from __future__ import annotations
import re
from typing import Optional

_ANALYSIS_SECTION_MAP = {
    "Valid points:":       "valid",
    "Unresolved points:":  "unresolved",
    "Unsupported points:": "unsupported",
    "Logical leaps:":      "logical_leap",
    "Redundant points:":   "redundant",
    "Irrelevant points:":  "irrelevant",
}

def _empty_advisor_analysis() -> dict:
    return {
        "n_debate_turns": 0,
        "n_negative_turns": 0,
        "n_affirmative_turns": 0,
        "n_evidence_snippets": 0,
        "classification_source": "unknown",
        "category_counts": {
            "valid": 0, "unresolved": 0, "unsupported": 0,
            "logical_leap": 0, "redundant": 0, "irrelevant": 0,
        },
        "category_items": {
            "valid": [], "unresolved": [], "unsupported": [],
            "logical_leap": [], "redundant": [], "irrelevant": [],
        },
    }

def _parse_advisor_item_line(line: str) -> Optional[dict]:
    m = re.match(r"\[(\w+)\]\s*(.*)", line)
    if not m:
        return None
    role = m.group(1)
    rest = m.group(2).strip()
    reason_match = re.search(r"\(([^)]+)\)\s*$", rest)
    if reason_match:
        reason = reason_match.group(1)
        content = rest[: reason_match.start()].strip()
    else:
        reason = ""
        content = rest
    return {"role": role, "content": content, "reason": reason}

def parse_advisor_analysis(text: Optional[str]) -> dict:
    result = _empty_advisor_analysis()
    if not text:
        return result

    try:
        # --- scalar header fields ---
        scalar_patterns = {
            "n_debate_turns":      r"-\s*Debate turns:\s*(\d+)",
            "n_negative_turns":    r"-\s*Negative turns:\s*(\d+)",
            "n_affirmative_turns": r"-\s*Affirmative turns:\s*(\d+)",
            "n_evidence_snippets": r"-\s*Evidence snippets in pool:\s*(\d+)",
        }
        for key, pattern in scalar_patterns.items():
            m = re.search(pattern, text)
            if m:
                result[key] = int(m.group(1))

        # --- classification source ---
        if "Ollama LLM" in text:
            result["classification_source"] = "Ollama LLM"
        elif "heuristic fallback" in text:
            result["classification_source"] = "heuristic fallback"

        # --- section item parsing ---
        current_category: Optional[str] = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped in _ANALYSIS_SECTION_MAP:
                current_category = _ANALYSIS_SECTION_MAP[stripped]
                continue
            if current_category and stripped.startswith("- ["):
                item = _parse_advisor_item_line(stripped[2:])
                if item:
                    result["category_items"][current_category].append(item)
                    result["category_counts"][current_category] += 1

    except Exception:
        pass

    return result

def _empty_report() -> dict:
    return {
        "decision_source": "unknown",
        "n_shared_evidence": None,
        "n_verifier_evidence": None,
        "negative_score": None,
        "affirmative_score": None,
        "reported_verdict": None,
        "rationale": None,
    }

def parse_final_report(text: Optional[str]) -> dict:
    result = _empty_report()
    if not text:
        return result

    try:
        # --- decision source ---
        if "Decision source: Ollama LLM" in text:
            result["decision_source"] = "Ollama LLM"
        elif "Decision source: heuristic fallback" in text:
            result["decision_source"] = "heuristic fallback"

        # --- numeric fields (heuristic fallback only) ---
        int_fields = {
            "n_shared_evidence":   r"-\s*Shared evidence snippets:\s*(\d+)",
            "n_verifier_evidence": r"-\s*New verifier evidence snippets:\s*(\d+)",
            "negative_score":      r"-\s*Negative score:\s*(\d+)",
            "affirmative_score":   r"-\s*Affirmative score:\s*(\d+)",
        }
        for key, pattern in int_fields.items():
            m = re.search(pattern, text)
            if m:
                result[key] = int(m.group(1))

        m = re.search(r"-\s*Verdict:\s*(\w+)", text)
        if m:
            result["reported_verdict"] = m.group(1).strip()

        m = re.search(r"-\s*Rationale:\s*(.+)", text)
        if m:
            result["rationale"] = m.group(1).strip()

    except Exception:
        pass

    return result

def parse_debate_log_metrics(debate_log: list) -> dict:
    total = len(debate_log)
    if total == 0:
        return {
            "total_turns": 0,
            "negative_turns": 0,
            "affirmative_turns": 0,
            "llm_argument_count": 0,
            "fallback_argument_count": 0,
            "llm_argument_rate": 0.0,
            "avg_argument_length_chars": 0.0,
            "url_cited_argument_count": 0,
            "url_cite_rate": 0.0,
        }

    negative_turns = 0
    affirmative_turns = 0
    fallback_count = 0
    url_cited = 0
    total_chars = 0

    for entry in debate_log:
        text = entry.strip()

        if text.startswith("[negative]"):
            negative_turns += 1
        elif text.startswith("[affirmative]"):
            affirmative_turns += 1

        body = re.sub(r"^\[\w+\]\s*", "", text)

        if body.startswith("I argue FOR") or body.startswith("I argue AGAINST"):
            fallback_count += 1

        if "https://" in text:
            url_cited += 1

        total_chars += len(text)

    llm_count = total - fallback_count
    return {
        "total_turns": total,
        "negative_turns": negative_turns,
        "affirmative_turns": affirmative_turns,
        "llm_argument_count": llm_count,
        "fallback_argument_count": fallback_count,
        "llm_argument_rate": llm_count / total,
        "avg_argument_length_chars": total_chars / total,
        "url_cited_argument_count": url_cited,
        "url_cite_rate": url_cited / total,
    }
