from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Running `python tests/this_file.py` (or from `tests/`) does not put the repo root on sys.path;
# pytest from the repo root does. Ensure imports work in both cases.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from misinfo_detection.config import AppConfig
from misinfo_detection.graph.parent import build_parent_graph
from misinfo_detection.schemas import Evidence, ParentState
from misinfo_detection.subgraphs.debater import build_debater_subgraph


def _make_parent(claim: str, *, max_rounds: int) -> ParentState:
    return ParentState(
        claim=claim,
        guidance="",
        current_round=0,
        max_rounds=max_rounds,
        evidence_pool={},
        debate_log=[],
        latest_negative_argument=None,
        latest_affirmative_argument=None,
        advisor_analysis=None,
        advisor_advice=None,
        verifier_evidence={},
        final_verdict=None,
        final_report=None,
    )


def _fake_tavily_search(*, query: str, config: AppConfig) -> List[Evidence]:
    # Single deterministic evidence item per query.
    slug = query.replace(" ", "_").replace("/", "_")
    return [
        Evidence(
            title=f"Title for {query}",
            url=f"https://example.com/{slug}",
            content=f"Snippet for {query}",
            score=1.0,
            source="mock",
        )
    ]


def test_debater_subgraph_bilateral_updates_shared_state(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )

    monkeypatch.setattr(debater_module, "tavily_search", _fake_tavily_search)

    run_debate = build_debater_subgraph(config=config)
    claim = "The sky is green"
    parent = _make_parent(claim, max_rounds=1)

    run_debate(parent)

    assert len(parent["debate_log"]) == 2
    assert parent["latest_negative_argument"] is not None
    assert parent["latest_affirmative_argument"] is not None

    assert f"{claim} evidence" in parent["evidence_pool"]
    assert f"{claim} Reuters" in parent["evidence_pool"]
    assert f"{claim} AP News" in parent["evidence_pool"]
    assert any(k.startswith(f"{claim} fact check") for k in parent["evidence_pool"].keys())

    # Negative retrieves 3 queries; affirmative adds only the new fact-check query because the base
    # evidence queries are already present in the shared evidence_pool.
    assert len(parent["evidence_pool"]) == 4
    assert all(len(v) == 1 for v in parent["evidence_pool"].values())


def test_parent_graph_runs_to_verifier_with_bilateral_debate(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module
    import misinfo_detection.subgraphs.verifier as verifier_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )

    monkeypatch.setattr(debater_module, "tavily_search", _fake_tavily_search)
    monkeypatch.setattr(verifier_module, "tavily_search", _fake_tavily_search)

    graph = build_parent_graph(config=config)
    claim = "Vaccines cause autism"
    initial = _make_parent(claim, max_rounds=1)

    out: ParentState = graph.invoke(initial)

    assert out["final_verdict"] == "insufficient"
    assert len(out["debate_log"]) == 2
    assert out["latest_negative_argument"] is not None
    assert out["latest_affirmative_argument"] is not None

    # Verifier evidence is stored in `verifier_evidence` (separate from `evidence_pool`).
    assert out["verifier_evidence"]

    assert f"{claim} evidence" in out["evidence_pool"]

