from __future__ import annotations

from langgraph.graph import StateGraph

from ..config import AppConfig, load_config
from ..nodes.guidance import build_guidance
from ..schemas import ParentState
from ..subgraphs.advisor import build_advisor_subgraph
from ..subgraphs.debater import build_debater_subgraph
from ..subgraphs.verifier import build_verifier_subgraph


def _increment_round(state: ParentState) -> ParentState:
    state["current_round"] = int(state.get("current_round", 0)) + 1
    return state


def _continue_debate(state: ParentState) -> str:
    if int(state.get("current_round", 0)) >= int(state.get("max_rounds", 0)):
        return "advisor"
    return "debate"


def build_parent_graph(*, config: AppConfig | None = None):
    config = config or load_config()

    run_debate = build_debater_subgraph(config=config)
    run_advisor = build_advisor_subgraph()
    run_verifier = build_verifier_subgraph(config=config)

    builder: StateGraph = StateGraph(ParentState)

    builder.add_node("build_guidance", build_guidance)
    builder.add_node("debate", run_debate)
    builder.add_node("increment_round", _increment_round)
    builder.add_node("advisor", run_advisor)
    builder.add_node("verifier", run_verifier)

    builder.add_edge("build_guidance", "debate")
    builder.add_edge("debate", "increment_round")

    # Loop debate until max rounds, then transition to advisor/verifier.
    builder.add_conditional_edges("increment_round", _continue_debate)

    builder.add_edge("advisor", "verifier")

    builder.set_entry_point("build_guidance")
    builder.set_finish_point("verifier")

    return builder.compile()

