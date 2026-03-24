from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


class Evidence(TypedDict):
    title: str
    url: str
    content: str
    score: float
    source: str


VerdictLabel = Literal["supported", "refuted", "insufficient"]


class ParentState(TypedDict):
    # Shared state owned by the parent graph; subgraphs project outputs back into this.
    # Input + configuration
    claim: str
    guidance: str
    current_round: int
    max_rounds: int

    # Shared cross-agent artifacts
    evidence_pool: Dict[str, List[Evidence]]  # query -> evidence list (shared across agents)
    debate_log: List[str]
    latest_negative_argument: Optional[str]
    latest_affirmative_argument: Optional[str]

    # Advisor outputs
    advisor_analysis: Optional[str]
    advisor_advice: Optional[str]

    # Verifier outputs
    verifier_evidence: Dict[str, List[Evidence]]  # verifier-only evidence keyed by query
    final_verdict: Optional[VerdictLabel]
    final_report: Optional[str]

DebaterRole = Literal["negative", "affirmative"]


class DebaterState(TypedDict):
    # Private per-debater state used inside the debater subgraph.
    claim: str
    guidance: str
    debate_log: List[str]
    role: DebaterRole
    latest_opponent_argument: Optional[str]

    generated_queries: List[str]
    retrieved_evidence: Dict[str, List[Evidence]]  # local, keyed by query
    new_argument: Optional[str]


class AdvisorState(TypedDict):
    # Advisor runs once after debate to highlight gaps/weaknesses.
    claim: str
    debate_log: List[str]
    evidence_pool: Dict[str, List[Evidence]]

    analysis: Optional[str]
    advice: Optional[str]


class VerifierState(TypedDict):
    # Verifier runs once at the end and may retrieve additional evidence.
    claim: str
    debate_log: List[str]
    evidence_pool: Dict[str, List[Evidence]]
    advisor_advice: Optional[str]

    generated_queries: List[str]
    retrieved_evidence: Dict[str, List[Evidence]]  # local, keyed by query
    verdict: Optional[VerdictLabel]
    report: Optional[str]
