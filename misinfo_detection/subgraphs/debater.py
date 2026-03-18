from __future__ import annotations

from typing import Dict, List

from langgraph.graph import StateGraph

from ..config import AppConfig
from ..schemas import DebaterRole, DebaterState, Evidence, ParentState
from ..tools.search import tavily_search


def generate_queries(state: DebaterState) -> DebaterState:
    claim = state["claim"]
    opponent = state.get("latest_opponent_argument")

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


def write_argument(state: DebaterState) -> DebaterState:
    role = state["role"]
    claim = state["claim"]
    evidence = state.get("retrieved_evidence", {})
    n_sources = sum(len(v) for v in evidence.values())
    stance = "AGAINST" if role == "negative" else "FOR"

    state["new_argument"] = (
        f"[{role}] Stub argument {stance} the claim: '{claim}'. "
        f"Retrieved {n_sources} evidence snippets this turn. "
        "LLM reasoning not wired yet."
    )
    return state


def build_debater_subgraph(*, role: DebaterRole, config: AppConfig):
    """Returns a compiled debater subgraph for the given role."""
    builder: StateGraph = StateGraph(DebaterState)

    def retrieve_evidence(state: DebaterState) -> DebaterState:
        retrieved: Dict[str, List[Evidence]] = state.get("retrieved_evidence", {})
        for q in state.get("generated_queries", []):
            if q in retrieved:
                continue
            retrieved[q] = tavily_search(query=q, config=config)
        state["retrieved_evidence"] = retrieved
        return state

    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("write_argument", write_argument)

    builder.add_edge("generate_queries", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "write_argument")

    builder.set_entry_point("generate_queries")
    builder.set_finish_point("write_argument")
    compiled = builder.compile()

    def run_on_parent(parent: ParentState) -> ParentState:
        debater_state: DebaterState = DebaterState(
            claim=parent["claim"],
            guidance=parent["guidance"],
            debate_log=list(parent["debate_log"]),
            role=role,
            latest_opponent_argument=parent["latest_affirmative_argument"]
            if role == "negative"
            else parent["latest_negative_argument"],
            generated_queries=[],
            retrieved_evidence={},
            new_argument=None,
        )

        out: DebaterState = compiled.invoke(debater_state)

        # Project changes back into ParentState
        new_argument = out.get("new_argument")
        if new_argument:
            parent["debate_log"] = parent.get("debate_log", []) + [new_argument]
            if role == "negative":
                parent["latest_negative_argument"] = new_argument
            else:
                parent["latest_affirmative_argument"] = new_argument

        # Merge evidence into shared pool
        pool = parent.get("evidence_pool", {})
        for q, evidences in out.get("retrieved_evidence", {}).items():
            pool[q] = pool.get(q, []) + list(evidences)
        parent["evidence_pool"] = pool
        return parent

    return run_on_parent

