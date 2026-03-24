from __future__ import annotations

import json
import os
import re
from typing import Dict, List
from urllib import error, request

from langgraph.graph import StateGraph

from ..config import AppConfig
from ..schemas import Evidence, ParentState, VerifierState, VerdictLabel
from ..tools.search import tavily_search

# ==============================
# Helper functions
# ==============================

def _extract_advice_section(advice: str, heading: str) -> list[str]:
    lines = advice.splitlines()
    captured: list[str] = []
    in_section = False

    for line in lines:
        stripped = line.strip()
        normalized = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if normalized == heading:
            in_section = True
            continue
        if not in_section:
            continue
        if normalized.endswith(":"):
            break
        if normalized:
            captured.append(normalized)

    return [item for item in captured if item and not item.lower().startswith("no clearly") and not item.lower().startswith("no major")]


def _clean_advice_line(text: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text


def _build_advice_queries(claim: str, advice: str) -> list[str]:
    sections = [
        "Highest-priority valid points:",
        "Remaining gaps to resolve:",
        "Assertions that need stronger scrutiny:",
    ]
    queries: list[str] = []

    for heading in sections:
        for item in _extract_advice_section(advice, heading):
            cleaned = _clean_advice_line(item)
            if not cleaned:
                continue
            queries.append(f"{claim} {cleaned[:120]}")

    return queries


# ==============================
# Query generation
# ==============================

def generate_queries(state: VerifierState) -> VerifierState:
    claim = state["claim"]
    advice = state.get("advisor_advice") or ""

    queries = [
        f"{claim} fact check",
        f"{claim} evidence systematic review",
    ]
    queries.extend(_build_advice_queries(claim, advice))

    state["generated_queries"] = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:5]
    return state


def retrieve_evidence(state: VerifierState, *, config: AppConfig) -> VerifierState:
    retrieved: Dict[str, List[Evidence]] = state.get("retrieved_evidence", {})
    for q in state.get("generated_queries", []):
        if q in retrieved:
            continue
        retrieved[q] = tavily_search(query=q, config=config)
    state["retrieved_evidence"] = retrieved
    return state


# ==============================
# Final evaluation
# ==============================

def _summarize_evidence(evidence_map: Dict[str, List[Evidence]], *, max_items: int = 6) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for query, evidences in evidence_map.items():
        for evidence in evidences:
            summary.append(
                {
                    "query": query,
                    "title": str(evidence.get("title", "") or ""),
                    "url": str(evidence.get("url", "") or ""),
                    "content": str(evidence.get("content", "") or "")[:240],
                    "source": str(evidence.get("source", "") or ""),
                }
            )
            if len(summary) >= max_items:
                return summary
    return summary


def _call_ollama_verifier(state: VerifierState) -> tuple[VerdictLabel, str] | None:
    model = os.getenv("OLLAMA_MODEL", "qwen:7b").strip() or "qwen:7b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    prompt = (
        "You are the final verification agent in a misinformation detection system.\n"
        "Review the claim, debate log, advisor advice, and available evidence.\n"
        "Return a verdict of supported, refuted, or insufficient.\n"
        "Supported means the claim is more strongly backed by the available evidence.\n"
        "Refuted means the claim is more strongly contradicted by the available evidence.\n"
        "Insufficient means the evidence is too limited, mixed, or ambiguous to decide.\n"
        'Return strict JSON with the shape {"verdict":"supported|refuted|insufficient","report":"short explanation"}.\n'
        f"Claim: {state['claim']}\n"
        f"Debate log: {json.dumps(state.get('debate_log', []), ensure_ascii=True)}\n"
        f"Advisor advice: {state.get('advisor_advice') or ''}\n"
        f"Shared evidence summary: {json.dumps(_summarize_evidence(state.get('evidence_pool', {})), ensure_ascii=True)}\n"
        f"Verifier evidence summary: {json.dumps(_summarize_evidence(state.get('retrieved_evidence', {})), ensure_ascii=True)}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=45) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    response_text = str(raw_payload.get("response", "") or "").strip()
    if not response_text:
        return None

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return None

    verdict = str(parsed.get("verdict", "") or "").strip().lower()
    if verdict not in {"supported", "refuted", "insufficient"}:
        return None

    report = str(parsed.get("report", "") or "").strip()
    if not report:
        return None

    return verdict, report


def _fallback_verdict(state: VerifierState) -> tuple[VerdictLabel, str]:
    claim = state["claim"]
    advice = state.get("advisor_advice") or ""
    verifier_evidence = state.get("retrieved_evidence", {})
    shared_evidence = state.get("evidence_pool", {})
    n_new = sum(len(v) for v in verifier_evidence.values())
    n_shared = sum(len(v) for v in shared_evidence.values())

    negative_cues = ("refute", "contradict", "false", "unsupported", "logical leap", "stronger scrutiny")
    affirmative_cues = ("valid points", "evidence-backed", "supported", "corroborates")

    negative_score = sum(advice.lower().count(cue) for cue in negative_cues)
    affirmative_score = sum(advice.lower().count(cue) for cue in affirmative_cues)

    if n_new == 0 and n_shared == 0:
        verdict: VerdictLabel = "insufficient"
        reason = "No evidence was available to support or refute the claim."
    elif negative_score > affirmative_score + 1:
        verdict = "refuted"
        reason = "The advisor guidance emphasized unresolved or weak arguments more than supported ones."
    elif affirmative_score > negative_score and (n_new + n_shared) > 0:
        verdict = "supported"
        reason = "The available evidence and advisor guidance leaned more toward supported arguments."
    else:
        verdict = "insufficient"
        reason = "The available evidence remained mixed or too limited for a confident decision."

    report = (
        "Verifier report.\n"
        f"- Claim: {claim}\n"
        f"- Shared evidence snippets: {n_shared}\n"
        f"- New verifier evidence snippets: {n_new}\n"
        f"- Verdict: {verdict}\n"
        f"- Rationale: {reason}\n"
        "- Decision source: heuristic fallback\n"
    )
    return verdict, report


def final_evaluation(state: VerifierState) -> VerifierState:
    llm_result = _call_ollama_verifier(state)
    if llm_result:
        verdict, llm_report = llm_result
        report = (
            "Verifier report.\n"
            f"- Claim: {state['claim']}\n"
            f"- Verdict: {verdict}\n"
            f"- Rationale: {llm_report}\n"
            "- Decision source: Ollama LLM\n"
        )
    else:
        verdict, report = _fallback_verdict(state)

    state["verdict"] = verdict
    state["report"] = report
    return state


# ==============================
# Build graph
# ==============================

def build_verifier_subgraph(*, config: AppConfig):
    builder: StateGraph = StateGraph(VerifierState)

    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_evidence", lambda s: retrieve_evidence(s, config=config))
    builder.add_node("final_evaluation", final_evaluation)

    builder.add_edge("generate_queries", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "final_evaluation")

    builder.set_entry_point("generate_queries")
    builder.set_finish_point("final_evaluation")
    compiled = builder.compile()

    def run_on_parent(parent: ParentState) -> ParentState:
        verifier_state: VerifierState = VerifierState(
            claim=parent["claim"],
            debate_log=list(parent.get("debate_log", [])),
            evidence_pool=dict(parent.get("evidence_pool", {})),
            advisor_advice=parent.get("advisor_advice"),
            generated_queries=[],
            retrieved_evidence={},
            verdict=None,
            report=None,
        )

        out: VerifierState = compiled.invoke(verifier_state)

        parent["verifier_evidence"] = dict(out.get("retrieved_evidence", {}))
        parent["final_verdict"] = out.get("verdict")
        parent["final_report"] = out.get("report")
        return parent

    return run_on_parent
