from pathlib import Path
import sys
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

langchain_tavily = types.ModuleType("langchain_tavily")


class _DummyTavilySearch:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return []


langchain_tavily.TavilySearch = _DummyTavilySearch
sys.modules.setdefault("langchain_tavily", langchain_tavily)

from misinfo_detection.subgraphs.verifier import final_evaluation, generate_queries


def test_verifier_generate_queries_uses_advisor_guidance():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": [],
        "evidence_pool": {},
        "advisor_advice": (
            "Advisor advice for verifier.\n"
            "- Claim: Vaccines reduce severe COVID-19 outcomes\n"
            "- Relevance classification source: heuristic fallback\n"
            "- Highest-priority valid points:\n"
            "- [affirmative] CDC hospitalization data shows reduced severe outcomes.\n"
            "- Remaining gaps to resolve:\n"
            "- [negative] Age-stratified outcome data is still missing.\n"
            "- Assertions that need stronger scrutiny:\n"
            "- [negative] The claim proves all risks disappear.\n"
            "- Low-value or noisy points to discount:\n"
            "- [negative] The moderator is unfair.\n"
            "- Verifier focus: Prioritize validating the strongest evidence-backed claims before broadening the search."
        ),
        "generated_queries": [],
        "retrieved_evidence": {},
        "verdict": None,
        "report": None,
    }

    out = generate_queries(state)
    queries = out["generated_queries"]

    assert any("fact check" in query for query in queries)
    assert any("CDC hospitalization data shows reduced severe outcomes" in query for query in queries)
    assert any("Age-stratified outcome data is still missing" in query for query in queries)


def test_verifier_final_evaluation_uses_ollama_result_when_available():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": ["[affirmative] CDC says hospitalization risk is lower after vaccination."],
        "evidence_pool": {},
        "advisor_advice": "Advisor advice for verifier.",
        "generated_queries": [],
        "retrieved_evidence": {},
        "verdict": None,
        "report": None,
    }

    with patch(
        "misinfo_detection.subgraphs.verifier._call_ollama_verifier",
        return_value=("supported", "Most reliable evidence in the record supports the claim."),
    ):
        out = final_evaluation(state)

    assert out["verdict"] == "supported"
    assert "Decision source: Ollama LLM" in (out["report"] or "")
    assert "supports the claim" in (out["report"] or "")


def test_verifier_final_evaluation_falls_back_to_insufficient_without_evidence():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": [],
        "evidence_pool": {},
        "advisor_advice": "Advisor advice for verifier.",
        "generated_queries": [],
        "retrieved_evidence": {},
        "verdict": None,
        "report": None,
    }

    with patch(
        "misinfo_detection.subgraphs.verifier._call_ollama_verifier",
        return_value=None,
    ):
        out = final_evaluation(state)

    assert out["verdict"] == "insufficient"
    assert "Decision source: heuristic fallback" in (out["report"] or "")
