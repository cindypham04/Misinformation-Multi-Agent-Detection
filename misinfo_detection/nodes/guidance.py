from __future__ import annotations

from ..schemas import ParentState


def build_guidance(state: ParentState) -> ParentState:
    claim = state["claim"].strip()
    guidance = (
        "You are participating in a structured debate about the following claim.\n"
        "Your job is to provide concise, evidence-grounded reasoning.\n"
        "Cite evidence by URL when possible.\n"
        f"\nCLAIM: {claim}\n"
    )
    state["guidance"] = guidance
    return state

