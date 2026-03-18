from __future__ import annotations

from typing import Dict, List

from langgraph.graph import StateGraph

from ..config import AppConfig
from ..schemas import Evidence, ParentState, VerifierState, VerdictLabel
from ..tools.search import tavily_search


def generate_queries(state: VerifierState) -> VerifierState:
    claim = state["claim"]
    advice = state.get("advisor_advice") or ""

    queries = [
        f"{claim} fact check",
        f"{claim} evidence systematic review",
    ]
    if advice:
        queries.append(f"{claim} {advice[:80]}")

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


def final_evaluation(state: VerifierState) -> VerifierState:
    # Stubbed evaluation: real implementation would weigh evidence + debate.
    n_new = sum(len(v) for v in state.get("retrieved_evidence", {}).values())
    claim = state["claim"]

    verdict: VerdictLabel = "insufficient"
    report = (
        "Stub verifier report.\n"
        f"- Claim: {claim}\n"
        f"- New verifier evidence snippets: {n_new}\n"
        "- Verdict: insufficient (LLM reasoning not wired yet)\n"
    )
    state["verdict"] = verdict
    state["report"] = report
    return state


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

