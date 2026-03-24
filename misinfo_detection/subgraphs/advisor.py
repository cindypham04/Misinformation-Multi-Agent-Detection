from __future__ import annotations

import json
import os
import re
from urllib import error, request
from typing import Any

from langgraph.graph import StateGraph

from ..schemas import AdvisorState, ParentState

# ==============================
# Helper functions
# ==============================

def parse_debate_log(debate_log: list[str]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []

    for raw_turn in debate_log:
        text = raw_turn.strip()
        role = "unknown"
        content = text

        if text.startswith("[negative]"):
            role = "negative"
            content = text[len("[negative]") :].strip()
        elif text.startswith("[affirmative]"):
            role = "affirmative"
            content = text[len("[affirmative]") :].strip()

        # Remove the current skeleton's boilerplate so downstream analysis
        # works on the actual argument text instead of formatter noise.
        for prefix in ("Stub argument", "AGAINST the claim:", "FOR the claim:"):
            if content.startswith(prefix):
                content = content[len(prefix) :].strip()

        turns.append(
            {
                "role": role,
                "raw": raw_turn,
                "content": content.strip(" '\""),
            }
        )

    return turns


def group_debate_by_role(debate_log: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "negative": [],
        "affirmative": [],
        "unknown": [],
    }

    for turn in debate_log:
        role = str(turn.get("role", "unknown"))
        if role not in grouped:
            role = "unknown"
        grouped[role].append(turn)

    return grouped


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _build_evidence_lexicon(evidence_pool: dict[str, list[dict[str, Any]]]) -> set[str]:
    tokens: set[str] = set()
    for query, evidences in evidence_pool.items():
        tokens.update(_tokenize(query))
        for evidence in evidences:
            tokens.update(_tokenize(str(evidence.get("title", ""))))
            tokens.update(_tokenize(str(evidence.get("content", ""))))
            tokens.update(_tokenize(str(evidence.get("source", ""))))
    return tokens


def _classify_turns_with_ollama(
    *,
    claim: str,
    turns: list[dict[str, Any]],
) -> dict[int, dict[str, str]]:
    model = os.getenv("OLLAMA_MODEL", "qwen:7b").strip() or "qwen:7b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    prompt = (
        "You are a debate advisor at expert level.\n"
        "For each debate turn, classify both relevance and reasoning quality.\n"
        "Relevance must be either relevant or irrelevant.\n"
        "Quality must be one of valid, unresolved, unsupported, logical_leap, redundant, or irrelevant.\n"
        "Mark a turn as relevant only if it materially addresses the truth, falsity, "
        "scope, evidence, assumptions, or framing of the claim.\n"
        "Use quality=valid when a relevant turn is materially useful and grounded.\n"
        "Use quality=unresolved when a relevant turn raises a real issue but still needs evidence.\n"
        "Use quality=unsupported when the turn makes a claim confidently without support.\n"
        "Use quality=logical_leap when it jumps from premise to conclusion too aggressively.\n"
        "Use quality=redundant when it mostly repeats prior points without adding substance.\n"
        "Use quality=irrelevant when it is off-topic, rhetorical noise, or does not engage the claim.\n"
        "Return strict JSON with the shape "
        '{"items":[{"index":0,"relevance":"relevant|irrelevant","quality":"valid|unresolved|unsupported|logical_leap|redundant|irrelevant","reason":"short reason"}]}.\n'
        f"Claim: {claim}\n"
        f"Turns: {json.dumps(turns, ensure_ascii=True)}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=30) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    response_text = str(raw_payload.get("response", "") or "").strip()
    if not response_text:
        return {}

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {}

    items = parsed.get("items", [])
    if not isinstance(items, list):
        return {}

    classifications: dict[int, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int):
            continue
        relevance = str(item.get("relevance", "") or "").strip().lower()
        if relevance not in {"relevant", "irrelevant"}:
            continue
        quality = str(item.get("quality", "") or "").strip().lower()
        if quality not in {"valid", "unresolved", "unsupported", "logical_leap", "redundant", "irrelevant"}:
            quality = "irrelevant" if relevance == "irrelevant" else "unresolved"
        classifications[index] = {
            "relevance": relevance,
            "quality": quality,
            "reason": str(item.get("reason", "") or "").strip() or "no reason provided",
        }
    return classifications


def _find_redundant_turn_indexes(turns: list[dict[str, Any]]) -> set[int]:
    seen_signatures: dict[tuple[str, ...], int] = {}
    redundant_indexes: set[int] = set()

    for index, turn in enumerate(turns):
        tokens = sorted(_tokenize(str(turn.get("content", "") or "")))
        signature = tuple(tokens[:20])
        if len(signature) < 5:
            continue
        if signature in seen_signatures:
            redundant_indexes.add(index)
        else:
            seen_signatures[signature] = index

    return redundant_indexes


def _infer_quality_label(
    *,
    content: str,
    claim_overlap: int,
    evidence_overlap: int,
    has_url: bool,
    redundant: bool,
) -> tuple[str, str]:
    lower_content = content.lower()
    hedge_markers = ("may ", "might ", "unclear", "uncertain", "needs", "without", "lack of", "question")
    leap_markers = ("therefore", "proves", "clearly means", "must mean", "obviously", "undeniable")

    if not content.strip():
        return "irrelevant", "empty or unparsable turn"
    if claim_overlap == 0:
        return "irrelevant", "does not engage with the claim"
    if redundant:
        return "redundant", "repeats an earlier argument without adding substantive new information"
    if evidence_overlap > 0 or has_url:
        return "valid", "relevant to the claim and grounded in retrieved evidence"
    if any(marker in lower_content for marker in leap_markers):
        return "logical_leap", "relevant to the claim but jumps to a conclusion without enough support"
    if any(marker in lower_content for marker in hedge_markers):
        return "unresolved", "raises a relevant issue but still needs evidentiary support"
    return "unsupported", "relevant to the claim but presented without clear support"


def _analyze_turn(
    *,
    index: int,
    turn: dict[str, Any],
    claim_tokens: set[str],
    evidence_tokens: set[str],
    llm_classification: dict[str, str] | None,
    redundant: bool,
) -> dict[str, Any]:
    content = str(turn.get("content", "") or "").strip()
    content_tokens = _tokenize(content)
    claim_overlap = len(content_tokens & claim_tokens)
    evidence_overlap = len(content_tokens & evidence_tokens)
    has_url = "http://" in content or "https://" in content

    if llm_classification:
        relevance = llm_classification.get("relevance", "")
        llm_reason = llm_classification.get("reason", "LLM did not provide a reason")
        llm_quality = llm_classification.get("quality", "")
        if relevance == "irrelevant":
            label = "irrelevant"
            reason = f"LLM relevance filter: {llm_reason}"
        else:
            if llm_quality == "valid" and not (evidence_overlap > 0 or has_url):
                label = "unresolved"
                reason = f"LLM marked the turn valid, but retrieved evidence does not clearly support it yet. {llm_reason}"
            else:
                label = llm_quality or "unresolved"
                reason = f"LLM assessment: {llm_reason}"
    else:
        label, reason = _infer_quality_label(
            content=content,
            claim_overlap=claim_overlap,
            evidence_overlap=evidence_overlap,
            has_url=has_url,
            redundant=redundant,
        )

    return {
        "index": index,
        "role": turn.get("role", "unknown"),
        "content": content,
        "label": label,
        "reason": reason,
    }


def _format_bucket(items: list[dict[str, Any]], *, empty_text: str) -> str:
    if not items:
        return f"- {empty_text}"

    lines: list[str] = []
    for item in items:
        role = str(item.get("role", "unknown"))
        content = str(item.get("content", "") or "")
        reason = str(item.get("reason", "") or "")
        lines.append(f"- [{role}] {content} ({reason})")
    return "\n".join(lines)


def _compute_analysis_data(
    *,
    claim: str,
    debate_log: list[str],
    evidence_pool: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    parsed_turns = parse_debate_log(debate_log)
    grouped_turns = group_debate_by_role(parsed_turns)
    claim_tokens = _tokenize(claim)
    evidence_tokens = _build_evidence_lexicon(evidence_pool)
    llm_classifications = _classify_turns_with_ollama(claim=claim, turns=parsed_turns)
    redundant_indexes = _find_redundant_turn_indexes(parsed_turns)

    analyzed_turns = [
        _analyze_turn(
            index=index,
            turn=turn,
            claim_tokens=claim_tokens,
            evidence_tokens=evidence_tokens,
            llm_classification=llm_classifications.get(index),
            redundant=index in redundant_indexes,
        )
        for index, turn in enumerate(parsed_turns)
    ]

    return {
        "parsed_turns": parsed_turns,
        "grouped_turns": grouped_turns,
        "analyzed_turns": analyzed_turns,
        "valid_points": [turn for turn in analyzed_turns if turn["label"] == "valid"],
        "irrelevant_points": [turn for turn in analyzed_turns if turn["label"] == "irrelevant"],
        "unresolved_points": [turn for turn in analyzed_turns if turn["label"] == "unresolved"],
        "unsupported_points": [turn for turn in analyzed_turns if turn["label"] == "unsupported"],
        "logical_leap_points": [turn for turn in analyzed_turns if turn["label"] == "logical_leap"],
        "redundant_points": [turn for turn in analyzed_turns if turn["label"] == "redundant"],
        "n_evidence": sum(len(v) for v in evidence_pool.values()),
        "classification_source": "Ollama LLM" if llm_classifications else "heuristic fallback",
    }


def _summarize_points(items: list[dict[str, Any]], *, empty_text: str, limit: int = 2) -> str:
    if not items:
        return f"- {empty_text}"

    lines: list[str] = []
    for item in items[:limit]:
        role = str(item.get("role", "unknown"))
        content = str(item.get("content", "") or "")
        reason = str(item.get("reason", "") or "")
        lines.append(f"- [{role}] {content} ({reason})")
    return "\n".join(lines)

# ==============================
# Advisor Analysis
# ==============================

def advisor_analyze(state: AdvisorState) -> AdvisorState:
    claim = state["claim"]
    evidence_pool = state.get("evidence_pool", {})
    analysis_data = _compute_analysis_data(
        claim=claim,
        debate_log=state.get("debate_log", []),
        evidence_pool=evidence_pool,
    )
    parsed_turns = analysis_data["parsed_turns"]
    grouped_turns = analysis_data["grouped_turns"]
    negative_turns = [turn["content"] for turn in grouped_turns["negative"]]
    affirmative_turns = [turn["content"] for turn in grouped_turns["affirmative"]]
    unknown_turns = [turn["content"] for turn in grouped_turns["unknown"]]
    n_turns = len(parsed_turns)
    valid_points = analysis_data["valid_points"]
    irrelevant_points = analysis_data["irrelevant_points"]
    unresolved_points = analysis_data["unresolved_points"]
    unsupported_points = analysis_data["unsupported_points"]
    logical_leap_points = analysis_data["logical_leap_points"]
    redundant_points = analysis_data["redundant_points"]

    state["analysis"] = (
        "Advisor analysis.\n"
        f"- Claim: {claim}\n"
        f"- Debate turns: {n_turns}\n"
        f"- Negative turns: {len(negative_turns)}\n"
        f"- Affirmative turns: {len(affirmative_turns)}\n"
        f"- Unknown turns: {len(unknown_turns)}\n"
        f"- Evidence snippets in pool: {analysis_data['n_evidence']}\n"
        f"- Negative arguments: {negative_turns or ['None']}\n"
        f"- Affirmative arguments: {affirmative_turns or ['None']}\n"
        f"- Unclassified arguments: {unknown_turns or ['None']}\n"
        "\nValid points:\n"
        f"{_format_bucket(valid_points, empty_text='No clearly evidence-grounded points identified.')}\n"
        "\nUnresolved points:\n"
        f"{_format_bucket(unresolved_points, empty_text='No unresolved points identified.')}\n"
        "\nUnsupported points:\n"
        f"{_format_bucket(unsupported_points, empty_text='No clearly unsupported points identified.')}\n"
        "\nLogical leaps:\n"
        f"{_format_bucket(logical_leap_points, empty_text='No clear logical leaps identified.')}\n"
        "\nRedundant points:\n"
        f"{_format_bucket(redundant_points, empty_text='No clearly redundant points identified.')}\n"
        "\nIrrelevant points:\n"
        f"{_format_bucket(irrelevant_points, empty_text='No clearly irrelevant points identified.')}\n"
        "\nRelevance classification source: "
        f"{analysis_data['classification_source']}."
    )
    return state

# ==============================
# Advisor Advice
# ==============================

def advisor_advice(state: AdvisorState) -> AdvisorState:
    claim = state["claim"]
    analysis_data = _compute_analysis_data(
        claim=claim,
        debate_log=state.get("debate_log", []),
        evidence_pool=state.get("evidence_pool", {}),
    )
    valid_points = analysis_data["valid_points"]
    irrelevant_points = analysis_data["irrelevant_points"]
    unresolved_points = analysis_data["unresolved_points"]
    unsupported_points = analysis_data["unsupported_points"]
    logical_leap_points = analysis_data["logical_leap_points"]
    redundant_points = analysis_data["redundant_points"]

    if valid_points:
        verifier_focus = (
            "Prioritize validating the strongest evidence-backed claims before broadening the search."
        )
    elif unresolved_points or unsupported_points or logical_leap_points:
        verifier_focus = (
            "Prioritize resolving claim-relevant gaps and stress-testing assertions that appear unsupported or overstated."
        )
    else:
        verifier_focus = (
            "Prioritize establishing a reliable evidence base because the debate did not produce useful, claim-relevant signal."
        )

    state["advice"] = (
        "Advisor advice for verifier.\n"
        f"- Claim: {claim}\n"
        f"- Relevance classification source: {analysis_data['classification_source']}\n"
        "- Highest-priority valid points:\n"
        f"{_summarize_points(valid_points, empty_text='No clearly valid evidence-backed points yet.')}\n"
        "- Remaining gaps to resolve:\n"
        f"{_summarize_points(unresolved_points, empty_text='No major unresolved gaps identified.')}\n"
        "- Assertions that need stronger scrutiny:\n"
        f"{_summarize_points(unsupported_points + logical_leap_points, empty_text='No clearly unsupported or overstated points identified.')}\n"
        "- Low-value or noisy points to discount:\n"
        f"{_summarize_points(redundant_points + irrelevant_points, empty_text='No clearly irrelevant or redundant points identified.')}\n"
        f"- Verifier focus: {verifier_focus}"
    )
    return state

# ==============================
# Build Graph
# ==============================

def build_advisor_subgraph():
    builder: StateGraph = StateGraph(AdvisorState)
    builder.add_node("advisor_analyze", advisor_analyze)
    builder.add_node("advisor_advice", advisor_advice)
    builder.add_edge("advisor_analyze", "advisor_advice")
    builder.set_entry_point("advisor_analyze")
    builder.set_finish_point("advisor_advice")
    compiled = builder.compile()

    def run_on_parent(parent: ParentState) -> ParentState:
        # One-shot: advisor doesn't mutate evidence, only adds analysis/advice fields.
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
