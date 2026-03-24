from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, TypedDict
from urllib import error, request

from langgraph.graph import StateGraph

from ..config import AppConfig
from ..schemas import DebaterRole, Evidence, ParentState
from ..tools.search import tavily_search


class BilateralDebateState(TypedDict):
    """
    Private subgraph state for one bilateral debate invocation.

    This is intentionally separate from `ParentState`, but it carries the shared artifacts so negative
    and affirmative steps can see each other's updates inside a single subgraph run.
    """

    claim: str
    guidance: str

    # Shared artifacts for both internal agents.
    evidence_pool: Dict[str, List[Evidence]]  # query -> evidence list
    debate_log: List[str]
    latest_negative_argument: Optional[str]
    latest_affirmative_argument: Optional[str]

    # Transient fields for the currently executing role turn.
    generated_queries: List[str]
    retrieved_evidence: Dict[str, List[Evidence]]  # evidence retrieved during this role turn

def _opponent_argument_for_role(state: BilateralDebateState, *, role: DebaterRole) -> Optional[str]:
    return state["latest_affirmative_argument"] if role == "negative" else state["latest_negative_argument"]


_QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "about",
    "into",
    "over",
    "under",
    "between",
    "after",
    "before",
    "during",
    "than",
    "then",
    "so",
    "if",
    "fact",
    "check",
    "news",
    "evidence",
}


def _normalize_query_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize_query_text(text: str) -> set[str]:
    return {
        token
        for token in _normalize_query_text(text).split(" ")
        if token and len(token) > 2 and token not in _QUERY_STOPWORDS
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _find_similar_existing_query(
    candidate_query: str,
    existing_queries: List[str],
    *,
    jaccard_threshold: float = 0.72,
    sequence_threshold: float = 0.86,
    combined_threshold: float = 0.78,
) -> Optional[str]:
    candidate_normalized = _normalize_query_text(candidate_query)
    if not candidate_normalized:
        return None

    candidate_tokens = _tokenize_query_text(candidate_query)
    best_match: Optional[str] = None
    best_score = 0.0

    for existing in existing_queries:
        existing_normalized = _normalize_query_text(existing)
        if not existing_normalized:
            continue

        token_score = _jaccard_similarity(candidate_tokens, _tokenize_query_text(existing))
        sequence_score = _sequence_similarity(candidate_normalized, existing_normalized)
        combined_score = 0.6 * token_score + 0.4 * sequence_score

        if (
            token_score >= jaccard_threshold
            or sequence_score >= sequence_threshold
            or combined_score >= combined_threshold
        ) and combined_score > best_score:
            best_score = combined_score
            best_match = existing

    return best_match


def _fallback_queries(*, claim: str, opponent_argument: Optional[str]) -> List[str]:
    queries: List[str] = []
    if opponent_argument:
        queries.append(f"{claim} fact check {opponent_argument[:100]}")
    queries.extend([f"{claim} fact check", f"{claim} evidence", f"{claim} Reuters", f"{claim} AP News"])
    return _dedupe_preserve_order([q.strip() for q in queries if q.strip()])


def _build_query_planner_prompt(
    *,
    role: DebaterRole,
    claim: str,
    guidance: str,
    opponent_argument: Optional[str],
    debate_log_tail: List[str],
    existing_queries: List[str],
) -> str:
    role_goal = (
        "Challenge and stress-test the claim using the strongest contradictory evidence."
        if role == "negative"
        else "Support and defend the claim using the strongest corroborating evidence."
    )
    return (
        "You are a query planner for a structured misinformation debate.\n"
        "Generate concise, high-value web search queries.\n"
        "Focus on factual verification, source quality, and unresolved points.\n"
        "Avoid duplicate intent with already-searched queries unless necessary.\n"
        "Return strict JSON only with this shape:\n"
        '{"queries":["q1","q2","q3"]}\n'
        f"Role: {role}\n"
        f"Role objective: {role_goal}\n"
        f"Claim: {claim}\n"
        f"Guidance: {guidance}\n"
        f"Latest opponent argument: {opponent_argument or 'None'}\n"
        f"Recent debate log (most recent last): {json.dumps(debate_log_tail, ensure_ascii=True)}\n"
        f"Existing evidence queries already searched: {json.dumps(existing_queries, ensure_ascii=True)}\n"
        "Rules:\n"
        "- Produce 3 to 5 queries.\n"
        "- Keep each query short and directly searchable.\n"
        "- Include at least one source-targeted query when useful (Reuters/AP/BBC/CDC/WHO/etc).\n"
        "- Output valid JSON only. No markdown. No extra keys.\n"
    )


def _call_ollama_query_planner(prompt: str) -> Optional[List[str]]:
    model = os.getenv("OLLAMA_MODEL", "qwen:7b").strip() or "qwen:7b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

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
        with request.urlopen(http_request, timeout=30) as response:
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

    queries = parsed.get("queries")
    if not isinstance(queries, list):
        return None

    out: List[str] = []
    for candidate in queries:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if cleaned:
            out.append(cleaned)
    out = _dedupe_preserve_order(out)
    return out if out else None


def _search_with_retry(
    *,
    query: str,
    config: AppConfig,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 0.2,
) -> List[Evidence]:
    # Retry a small fixed number of times; if all attempts fail, return empty evidence and continue.
    attempts = max(1, max_attempts)
    delay_seconds = max(0.0, initial_backoff_seconds)

    for attempt in range(1, attempts + 1):
        try:
            return tavily_search(query=query, config=config)
        except Exception:
            if attempt >= attempts:
                return []
            if delay_seconds > 0:
                time.sleep(delay_seconds)
                delay_seconds *= 2

    return []


def _generate_queries_for_role(state: BilateralDebateState, *, role: DebaterRole) -> BilateralDebateState:
    claim = state["claim"]
    opponent = _opponent_argument_for_role(state, role=role)
    guidance = state.get("guidance", "")
    debate_log_tail = state.get("debate_log", [])[-8:]
    existing_queries = list(state.get("evidence_pool", {}).keys())

    prompt = _build_query_planner_prompt(
        role=role,
        claim=claim,
        guidance=guidance,
        opponent_argument=opponent,
        debate_log_tail=debate_log_tail,
        existing_queries=existing_queries,
    )
    llm_queries = _call_ollama_query_planner(prompt)
    fallback_queries = _fallback_queries(claim=claim, opponent_argument=opponent)
    raw_candidates = llm_queries if llm_queries else fallback_queries

    canonicalized: List[str] = []
    for query in raw_candidates:
        cleaned = query.strip()
        if not cleaned:
            continue
        similar = _find_similar_existing_query(cleaned, existing_queries)
        canonicalized.append(similar if similar else cleaned)

    final_queries = _dedupe_preserve_order(canonicalized)

    # Keep small and deterministic while guaranteeing useful fallback coverage.
    if len(final_queries) < 3:
        for query in fallback_queries:
            similar = _find_similar_existing_query(query, existing_queries)
            final_queries.append(similar if similar else query)
        final_queries = _dedupe_preserve_order(final_queries)

    state["generated_queries"] = final_queries[:5]
    return state


def _retrieve_evidence_for_role(
    state: BilateralDebateState,
    *,
    role: DebaterRole,
    config: AppConfig,
) -> BilateralDebateState:
    # Retrieval updates the shared evidence_pool inside the subgraph, so negative and affirmative steps
    # share one view of what was already retrieved.
    evidence_pool = state.get("evidence_pool", {})
    retrieved: Dict[str, List[Evidence]] = {}

    for q in state.get("generated_queries", []):
        if q in evidence_pool:
            retrieved[q] = evidence_pool[q]
            continue
        fetched = _search_with_retry(query=q, config=config)
        if fetched:
            evidence_pool[q] = fetched
        retrieved[q] = fetched

    state["evidence_pool"] = evidence_pool
    state["retrieved_evidence"] = retrieved
    return state


def _write_argument_for_role(state: BilateralDebateState, *, role: DebaterRole) -> BilateralDebateState:
    claim = state["claim"]
    evidence = state.get("retrieved_evidence", {}) or {}
    n_sources = sum(len(v) for v in evidence.values())
    stance = "AGAINST" if role == "negative" else "FOR"

    new_argument = (
        f"[{role}] Stub argument {stance} the claim: '{claim}'. "
        f"Retrieved {n_sources} evidence snippets this turn. "
        "LLM reasoning not wired yet."
    )

    state["debate_log"] = state.get("debate_log", []) + [new_argument]
    if role == "negative":
        state["latest_negative_argument"] = new_argument
    else:
        state["latest_affirmative_argument"] = new_argument

    return state


def build_debater_subgraph(*, config: AppConfig):
    """
    Returns a compiled debater subgraph that runs:
      1) negative: generate_queries -> retrieve_evidence -> write_argument
      2) affirmative: generate_queries -> retrieve_evidence -> write_argument

    Both steps share one `BilateralDebateState` instance, so they see each other's updates.
    """

    builder: StateGraph = StateGraph(BilateralDebateState)

    def negative_generate_queries(state: BilateralDebateState) -> BilateralDebateState:
        return _generate_queries_for_role(state, role="negative")

    def negative_retrieve_evidence(state: BilateralDebateState) -> BilateralDebateState:
        return _retrieve_evidence_for_role(state, role="negative", config=config)

    def negative_write_argument(state: BilateralDebateState) -> BilateralDebateState:
        return _write_argument_for_role(state, role="negative")

    def affirmative_generate_queries(state: BilateralDebateState) -> BilateralDebateState:
        return _generate_queries_for_role(state, role="affirmative")

    def affirmative_retrieve_evidence(state: BilateralDebateState) -> BilateralDebateState:
        return _retrieve_evidence_for_role(state, role="affirmative", config=config)

    def affirmative_write_argument(state: BilateralDebateState) -> BilateralDebateState:
        return _write_argument_for_role(state, role="affirmative")

    builder.add_node("negative_generate_queries", negative_generate_queries)
    builder.add_node("negative_retrieve_evidence", negative_retrieve_evidence)
    builder.add_node("negative_write_argument", negative_write_argument)
    builder.add_node("affirmative_generate_queries", affirmative_generate_queries)
    builder.add_node("affirmative_retrieve_evidence", affirmative_retrieve_evidence)
    builder.add_node("affirmative_write_argument", affirmative_write_argument)

    builder.add_edge("negative_generate_queries", "negative_retrieve_evidence")
    builder.add_edge("negative_retrieve_evidence", "negative_write_argument")
    builder.add_edge("negative_write_argument", "affirmative_generate_queries")
    builder.add_edge("affirmative_generate_queries", "affirmative_retrieve_evidence")
    builder.add_edge("affirmative_retrieve_evidence", "affirmative_write_argument")

    builder.set_entry_point("negative_generate_queries")
    builder.set_finish_point("affirmative_write_argument")

    compiled = builder.compile()

    def run_on_parent(parent: ParentState) -> ParentState:
        # Project shared parent fields into the subgraph; negative/affirmative updates happen in-place
        # within the bilateral subgraph state.
        bilateral_state: BilateralDebateState = BilateralDebateState(
            claim=parent["claim"],
            guidance=parent["guidance"],
            evidence_pool=dict(parent.get("evidence_pool", {})),
            debate_log=list(parent.get("debate_log", [])),
            latest_negative_argument=parent.get("latest_negative_argument"),
            latest_affirmative_argument=parent.get("latest_affirmative_argument"),
            generated_queries=[],
            retrieved_evidence={},
        )

        out: BilateralDebateState = compiled.invoke(bilateral_state)

        # Project bilateral updates back into ParentState once per invocation.
        parent["debate_log"] = list(out.get("debate_log", []))
        parent["latest_negative_argument"] = out.get("latest_negative_argument")
        parent["latest_affirmative_argument"] = out.get("latest_affirmative_argument")
        parent["evidence_pool"] = dict(out.get("evidence_pool", {}))
        return parent

    return run_on_parent

