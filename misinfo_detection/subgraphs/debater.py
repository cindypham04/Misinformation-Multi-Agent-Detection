from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

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


def _generate_queries_for_role(state: BilateralDebateState, *, role: DebaterRole) -> BilateralDebateState:
    claim = state["claim"]
    opponent = _opponent_argument_for_role(state, role=role)

    queries: List[str] = []
    if opponent:
        queries.append(f"{claim} fact check {opponent[:80]}")

    queries.extend(
        [
            f"{claim} evidence",
            f"{claim} Reuters",
            f"{claim} AP News",
        ]
    )

    # Keep small and deterministic for the skeleton.
    state["generated_queries"] = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:5]
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
            continue
        evidence_pool[q] = tavily_search(query=q, config=config)
        retrieved[q] = evidence_pool[q]

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

