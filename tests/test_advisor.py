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

from misinfo_detection.subgraphs.advisor import advisor_advice, advisor_analyze


def test_sanity():
    assert True


def test_advisor_analysis_classifies_turns():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": [
            "[affirmative] Vaccines reduce severe COVID-19 outcomes according to CDC hospitalization data.",
            "[negative] Bananas are yellow and this debate is biased.",
            "[negative] Vaccines reduce severe COVID-19 outcomes is overstated without age-stratified data.",
        ],
        "evidence_pool": {
            "vaccines severe covid cdc": [
                {
                    "title": "CDC reports lower hospitalization risk after vaccination",
                    "url": "https://www.cdc.gov/example",
                    "content": "COVID-19 vaccines reduce hospitalization and severe outcomes.",
                    "score": 0.9,
                    "source": "cdc.gov",
                }
            ]
        },
        "analysis": None,
        "advice": None,
    }

    out = advisor_analyze(state)
    analysis = out["analysis"] or ""

    assert "Valid points:" in analysis
    assert "Irrelevant points:" in analysis
    assert "Unresolved points:" in analysis
    assert "[affirmative]" in analysis
    assert "Bananas are yellow" in analysis
    assert "age-stratified data" in analysis


def test_advisor_analysis_uses_ollama_relevance_when_available():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": [
            "[affirmative] Vaccines reduce severe COVID-19 outcomes according to CDC hospitalization data.",
            "[negative] This moderator is unfair and everyone knows it.",
        ],
        "evidence_pool": {
            "vaccines severe covid cdc": [
                {
                    "title": "CDC reports lower hospitalization risk after vaccination",
                    "url": "https://www.cdc.gov/example",
                    "content": "COVID-19 vaccines reduce hospitalization and severe outcomes.",
                    "score": 0.9,
                    "source": "cdc.gov",
                }
            ]
        },
        "analysis": None,
        "advice": None,
    }

    with patch(
        "misinfo_detection.subgraphs.advisor._classify_turns_with_ollama",
        return_value={
            0: {"relevance": "relevant", "reason": "Addresses the claim directly."},
            1: {"relevance": "irrelevant", "reason": "Complains about moderation instead of the claim."},
        },
    ):
        out = advisor_analyze(state)

    analysis = out["analysis"] or ""
    assert "Relevance classification source: Ollama LLM." in analysis
    assert "LLM relevance filter: Complains about moderation instead of the claim." in analysis
    assert "LLM assessment: Addresses the claim directly." in analysis


def test_advisor_advice_summarizes_valid_points_gaps_and_noise():
    state = {
        "claim": "Vaccines reduce severe COVID-19 outcomes",
        "debate_log": [
            "[affirmative] Vaccines reduce severe COVID-19 outcomes according to CDC hospitalization data.",
            "[negative] This moderator is unfair and everyone knows it.",
            "[negative] The claim may be overstated without age-stratified outcome data.",
        ],
        "evidence_pool": {
            "vaccines severe covid cdc": [
                {
                    "title": "CDC reports lower hospitalization risk after vaccination",
                    "url": "https://www.cdc.gov/example",
                    "content": "COVID-19 vaccines reduce hospitalization and severe outcomes.",
                    "score": 0.9,
                    "source": "cdc.gov",
                }
            ]
        },
        "analysis": None,
        "advice": None,
    }

    with patch(
        "misinfo_detection.subgraphs.advisor._classify_turns_with_ollama",
        return_value={
            0: {"relevance": "relevant", "reason": "Directly addresses the claim with supporting evidence."},
            1: {"relevance": "irrelevant", "reason": "Discusses moderation instead of the claim."},
            2: {"relevance": "relevant", "reason": "Raises a legitimate scope limitation that still needs support."},
        },
    ):
        out = advisor_advice(state)

    advice = out["advice"] or ""
    assert "Advisor advice for verifier." in advice
    assert "Highest-priority valid points:" in advice
    assert "Remaining gaps to resolve:" in advice
    assert "Assertions that need stronger scrutiny:" in advice
    assert "Low-value or noisy points to discount:" in advice
    assert "CDC hospitalization data" in advice
    assert "age-stratified outcome data" in advice
    assert "moderator is unfair" in advice


def test_advisor_analysis_tracks_quality_categories():
    state = {
        "claim": "Electric vehicles reduce urban air pollution",
        "debate_log": [
            "[affirmative] Electric vehicles reduce urban air pollution because city tailpipe emissions fall according to transport studies.",
            "[negative] Electric vehicles reduce urban air pollution therefore they completely eliminate pollution everywhere.",
            "[negative] Electric vehicles reduce urban air pollution because city tailpipe emissions fall according to transport studies.",
            "[affirmative] Electric vehicles reduce urban air pollution and that settles it.",
        ],
        "evidence_pool": {
            "electric vehicles urban air pollution studies": [
                {
                    "title": "Transport studies link EV adoption to lower tailpipe emissions",
                    "url": "https://example.org/ev-study",
                    "content": "Urban tailpipe emissions fall when internal combustion vehicles are replaced by EVs.",
                    "score": 0.8,
                    "source": "example.org",
                }
            ]
        },
        "analysis": None,
        "advice": None,
    }

    out = advisor_analyze(state)
    analysis = out["analysis"] or ""

    assert "Logical leaps:" in analysis
    assert "Redundant points:" in analysis
    assert "Unsupported points:" in analysis
    assert "completely eliminate pollution everywhere" in analysis
    assert "that settles it" in analysis
