from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Running `python tests/this_file.py` (or from `tests/`) does not put the repo root on sys.path;
# pytest from the repo root does. Ensure imports work in both cases.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from misinfo_detection.config import AppConfig
from misinfo_detection.schemas import Evidence
from misinfo_detection.subgraphs.debater import (
    BilateralDebateState,
    _call_ollama_query_planner,
    _fallback_queries,
    _find_similar_existing_query,
    _generate_queries_for_role,
    _retrieve_evidence_for_role,
    _search_with_retry,
)


def _state(
    *,
    claim: str = "Vaccines cause autism",
    guidance: str = "Use evidence-grounded reasoning.",
    debate_log: List[str] | None = None,
    evidence_pool: dict[str, list[Evidence]] | None = None,
    latest_negative_argument: str | None = None,
    latest_affirmative_argument: str | None = None,
) -> BilateralDebateState:
    return BilateralDebateState(
        claim=claim,
        guidance=guidance,
        evidence_pool=evidence_pool or {},
        debate_log=debate_log or [],
        latest_negative_argument=latest_negative_argument,
        latest_affirmative_argument=latest_affirmative_argument,
        generated_queries=[],
        retrieved_evidence={},
    )


def _fake_tavily_search(*, query: str, config: AppConfig) -> List[Evidence]:
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


def test_call_ollama_query_planner_parses_valid_json(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {"response": json.dumps({"queries": ["q1", "q2", "q2", ""]})}
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(_request, timeout=30):
        return _FakeResponse()

    monkeypatch.setattr(debater_module.request, "urlopen", _fake_urlopen)

    out = _call_ollama_query_planner("prompt")
    assert out == ["q1", "q2"]


def test_call_ollama_query_planner_returns_none_on_invalid_response(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {"response": "not-json"}
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(_request, timeout=30):
        return _FakeResponse()

    monkeypatch.setattr(debater_module.request, "urlopen", _fake_urlopen)

    assert _call_ollama_query_planner("prompt") is None


def test_fallback_queries_include_claim_and_sources():
    claim = "The sky is green"
    opponent = "[affirmative] It is proven by study X"
    queries = _fallback_queries(claim=claim, opponent_argument=opponent)

    assert len(queries) >= 4
    assert f"{claim} fact check" in queries
    assert f"{claim} evidence" in queries
    assert f"{claim} Reuters" in queries
    assert f"{claim} AP News" in queries
    assert any(q.startswith(f"{claim} fact check ") for q in queries)


def test_find_similar_existing_query_matches_semantic_duplicate():
    existing = [
        "Vaccines cause autism fact check",
        "Vaccines cause autism Reuters",
    ]
    candidate = "Vaccines and autism fact-check"

    match = _find_similar_existing_query(candidate, existing)
    assert match == "Vaccines cause autism fact check"


def test_generate_queries_for_role_uses_llm_output_and_reuses_existing(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    monkeypatch.setattr(
        debater_module,
        "_call_ollama_query_planner",
        lambda prompt: [
            "Vaccines and autism fact-check",
            "Vaccines cause autism Reuters",
            "vaccines cause autism CDC statement",
        ],
    )

    state = _state(
        claim="Vaccines cause autism",
        debate_log=["[negative] early point"],
        latest_negative_argument="[negative] prior argument",
        evidence_pool={
            "Vaccines cause autism fact check": [Evidence(title="t", url="u", content="c", score=1.0, source="s")],
            "Vaccines cause autism Reuters": [Evidence(title="t2", url="u2", content="c2", score=0.9, source="s2")],
        },
    )

    out = _generate_queries_for_role(state, role="affirmative")

    assert "Vaccines cause autism fact check" in out["generated_queries"]
    assert "Vaccines cause autism Reuters" in out["generated_queries"]
    assert "vaccines cause autism CDC statement" in out["generated_queries"]
    assert len(out["generated_queries"]) <= 5


def test_generate_queries_for_role_falls_back_when_llm_unavailable(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    monkeypatch.setattr(debater_module, "_call_ollama_query_planner", lambda prompt: None)

    state = _state(
        claim="The sky is green",
        latest_affirmative_argument="[affirmative] It appears green at sunset",
    )

    out = _generate_queries_for_role(state, role="negative")

    assert f"{state['claim']} evidence" in out["generated_queries"]
    assert f"{state['claim']} Reuters" in out["generated_queries"]
    assert f"{state['claim']} AP News" in out["generated_queries"]
    assert any(q.startswith(f"{state['claim']} fact check") for q in out["generated_queries"])


def test_retrieve_evidence_reuses_cached_query_results(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )

    calls: List[str] = []

    def _spy_tavily_search(*, query: str, config: AppConfig) -> List[Evidence]:
        calls.append(query)
        return _fake_tavily_search(query=query, config=config)

    monkeypatch.setattr(debater_module, "tavily_search", _spy_tavily_search)

    cached_query = "Vaccines cause autism fact check"
    cached_evidence = [Evidence(title="cached", url="u", content="c", score=1.0, source="cache")]
    state = _state(
        claim="Vaccines cause autism",
        evidence_pool={cached_query: cached_evidence},
    )
    state["generated_queries"] = [cached_query, "Vaccines cause autism Reuters"]

    out = _retrieve_evidence_for_role(state, role="negative", config=config)

    # Cached query should be reused (not re-fetched).
    assert calls == ["Vaccines cause autism Reuters"]
    assert out["retrieved_evidence"][cached_query] == cached_evidence
    assert "Vaccines cause autism Reuters" in out["retrieved_evidence"]


def test_search_with_retry_eventually_succeeds(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )
    calls = {"count": 0}

    def _flaky_search(*, query: str, config: AppConfig) -> List[Evidence]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary failure")
        return _fake_tavily_search(query=query, config=config)

    monkeypatch.setattr(debater_module, "tavily_search", _flaky_search)

    out = _search_with_retry(query="retry query", config=config, max_attempts=3, initial_backoff_seconds=0.0)
    assert calls["count"] == 3
    assert len(out) == 1
    assert out[0]["title"] == "Title for retry query"


def test_search_with_retry_returns_empty_after_exhausting_retries(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )
    calls = {"count": 0}

    def _always_fail(*, query: str, config: AppConfig) -> List[Evidence]:
        calls["count"] += 1
        raise RuntimeError("network failure")

    monkeypatch.setattr(debater_module, "tavily_search", _always_fail)

    out = _search_with_retry(query="will fail", config=config, max_attempts=3, initial_backoff_seconds=0.0)
    assert calls["count"] == 3
    assert out == []


def test_retrieve_evidence_mixed_batch_continues_after_failure(monkeypatch):
    import misinfo_detection.subgraphs.debater as debater_module

    config = AppConfig(
        tavily_api_key="dummy",
        reliable_domains=[],
        max_rounds=1,
        tavily_max_results=1,
        tavily_topic="general",
    )

    def _fake_search_with_retry(*, query: str, config: AppConfig, max_attempts: int = 3, initial_backoff_seconds: float = 0.2) -> List[Evidence]:
        if "bad query" in query:
            return []
        return _fake_tavily_search(query=query, config=config)

    monkeypatch.setattr(debater_module, "_search_with_retry", _fake_search_with_retry)

    state = _state(claim="Vaccines cause autism")
    state["generated_queries"] = [
        "good query one",
        "bad query two",
        "good query three",
    ]

    out = _retrieve_evidence_for_role(state, role="negative", config=config)

    assert "good query one" in out["retrieved_evidence"]
    assert "good query three" in out["retrieved_evidence"]
    assert out["retrieved_evidence"]["bad query two"] == []
    assert "bad query two" not in out["evidence_pool"]

