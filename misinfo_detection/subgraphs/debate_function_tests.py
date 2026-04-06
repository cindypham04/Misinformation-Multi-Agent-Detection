from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from misinfo_detection.subgraphs.debater import _build_argument_prompt
from misinfo_detection.subgraphs.debater import _call_ollama_argument_writer
from misinfo_detection.subgraphs.debater import _fallback_argument_text
from misinfo_detection.subgraphs.debater import _generate_queries_for_role
from misinfo_detection.subgraphs.debater import _opponent_argument_for_role
from misinfo_detection.subgraphs.debater import _retrieve_evidence_for_role
from misinfo_detection.subgraphs.debater import _summarize_retrieved_evidence
from misinfo_detection.subgraphs.debater import _write_argument_for_role
from misinfo_detection.subgraphs.debater import BilateralDebateState
from dotenv import load_dotenv
from misinfo_detection.config import load_config

'''
This file idividually tests our debate agent functions. Replace the function call
in main with what currently needs to be tested.
'''


def _write_trace_step(path: Path, *, title: str, payload: object) -> None:
    section = (
        f"\n{title}\n"
        f"{'-' * len(title)}\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=True, default=str)}\n"
    )
    path.open("a", encoding="utf-8").write(section)


def test_generate_queries_for_role():
    state = BilateralDebateState(
        claim="Vaccines cause autism",
        guidance="Use evidence-grounded reasoning.",
        debate_log=[],
        latest_negative_argument=None,
        latest_affirmative_argument=None,
        evidence_pool={},
        generated_queries=[],
        retrieved_evidence={},
    )
    out = _generate_queries_for_role(state, role="negative")
    print(out)
    return state

def test_retrieve_evidence_for_role():
    state = BilateralDebateState(
        claim="Vaccines cause autism",
        guidance="Use evidence-grounded reasoning.",
        debate_log=[],
        latest_negative_argument=None,
        latest_affirmative_argument=None,
        evidence_pool={},
    )
    state["generated_queries"] = ["Vaccines cause autism fact check", "Vaccines cause autism evidence", "Vaccines cause autism Reuters", "Vaccines cause autism AP News"]
    out = _retrieve_evidence_for_role(state, role="negative", config=load_config())
    print(out)


def test_write_argument_for_role_trace():
    trace_path = Path("output.txt")
    header = (
        "TRACE FOR _write_argument_for_role\n"
        "================================\n"
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )
    trace_path.write_text(header, encoding="utf-8")

    state = BilateralDebateState(
        claim="Vaccines cause autism",
        guidance="Use evidence-grounded reasoning and cite reliable sources when possible.",
        debate_log=[
            "[affirmative] Prior stance in favor of the claim without enough support."
        ],
        latest_negative_argument=None,
        latest_affirmative_argument=(
            "[affirmative] Prior stance in favor of the claim without enough support."
        ),
        evidence_pool={},
        generated_queries=[],
        retrieved_evidence={
            "Vaccines cause autism fact check": [
                {
                    "title": "Fact check: no evidence vaccines cause autism",
                    "url": "https://www.reuters.com/fact-check-example",
                    "content": "Large studies have not found evidence that vaccines cause autism.",
                    "score": 0.95,
                    "source": "reuters.com",
                }
            ],
            "Vaccines cause autism CDC": [
                {
                    "title": "Autism and vaccines",
                    "url": "https://www.cdc.gov/vaccine-safety/about/autism.html",
                    "content": "CDC states there is no link between vaccines and autism.",
                    "score": 0.92,
                    "source": "cdc.gov",
                }
            ],
        },
    )

    _write_trace_step(trace_path, title="STEP 1 - Initial State", payload=state)

    opponent_argument = _opponent_argument_for_role(state, role="negative")
    _write_trace_step(
        trace_path,
        title="STEP 2 - Opponent Argument",
        payload={"role": "negative", "opponent_argument": opponent_argument},
    )

    evidence_summary = _summarize_retrieved_evidence(state.get("retrieved_evidence", {}))
    _write_trace_step(
        trace_path,
        title="STEP 3 - Evidence Summary",
        payload={"summary_count": len(evidence_summary), "summary": evidence_summary},
    )

    prompt = _build_argument_prompt(
        role="negative",
        claim=state["claim"],
        guidance=state["guidance"],
        opponent_argument=opponent_argument,
        debate_log_tail=state.get("debate_log", [])[-8:],
        evidence_summary=evidence_summary,
    )
    _write_trace_step(
        trace_path,
        title="STEP 4 - Argument Prompt",
        payload={"prompt_length": len(prompt), "prompt": prompt},
    )

    llm_argument = _call_ollama_argument_writer(prompt)
    _write_trace_step(
        trace_path,
        title="STEP 5 - LLM Argument Result",
        payload={"llm_argument": llm_argument},
    )

    fallback_argument = _fallback_argument_text(
        role="negative",
        claim=state["claim"],
        opponent_argument=opponent_argument,
        evidence_summary=evidence_summary,
    )
    _write_trace_step(
        trace_path,
        title="STEP 6 - Fallback Argument Result",
        payload={"fallback_argument": fallback_argument},
    )

    out = _write_argument_for_role(state, role="negative")
    _write_trace_step(trace_path, title="STEP 7 - Final Output State", payload=out)

    print(f"Saved _write_argument_for_role trace to: {trace_path}")


if __name__ == "__main__":
    load_dotenv()
    # Query generation works
    #test_generate_queries_for_role()

    # Evidence retrieval works
    #test_retrieve_evidence_for_role()

    # Debug Argument Generation
    test_write_argument_for_role_trace()