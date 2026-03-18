from __future__ import annotations

from langgraph.graph import StateGraph

from ..schemas import AdvisorState, ParentState


def advisor_analyze(state: AdvisorState) -> AdvisorState:
    claim = state["claim"]
    n_turns = len(state.get("debate_log", []))
    n_evidence = sum(len(v) for v in state.get("evidence_pool", {}).values())

    state["analysis"] = (
        "Stub advisor analysis.\n"
        f"- Claim: {claim}\n"
        f"- Debate turns: {n_turns}\n"
        f"- Evidence snippets in pool: {n_evidence}\n"
        "LLM reasoning not wired yet."
    )
    return state


def advisor_advice(state: AdvisorState) -> AdvisorState:
    analysis = state.get("analysis") or ""
    state["advice"] = (
        "Stub advisor advice: focus the verifier on the strongest evidence, "
        "and resolve any remaining ambiguities.\n\n"
        + analysis
    )
    return state


def build_advisor_subgraph():
    builder: StateGraph = StateGraph(AdvisorState)
    builder.add_node("advisor_analyze", advisor_analyze)
    builder.add_node("advisor_advice", advisor_advice)
    builder.add_edge("advisor_analyze", "advisor_advice")
    builder.set_entry_point("advisor_analyze")
    builder.set_finish_point("advisor_advice")
    compiled = builder.compile()

    def run_on_parent(parent: ParentState) -> ParentState:
        advisor_state: AdvisorState = AdvisorState(
            claim=parent["claim"],
            debate_log=list(parent.get("debate_log", [])),
            evidence_pool=dict(parent.get("evidence_pool", {})),
            analysis=None,
            advice=None,
        )

        out: AdvisorState = compiled.invoke(advisor_state)
        parent["advisor_analysis"] = out.get("analysis")
        parent["advisor_advice"] = out.get("advice")
        return parent

    return run_on_parent

