from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


class Evidence(TypedDict):
    title: str
    url: str
    content: str
    score: float
    source: str


VerdictLabel = Literal["true", "false", "insufficient"]


class ParentState(TypedDict):
    # Input + configuration
    claim: str
    guidance: str
    current_round: int
    max_rounds: int

    # Shared cross-agent artifacts
    evidence_pool: Dict[str, List[Evidence]]  # keyed by query
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
    claim: str
    guidance: str
    debate_log: List[str]
    role: DebaterRole
    latest_opponent_argument: Optional[str]

    generated_queries: List[str]
    retrieved_evidence: Dict[str, List[Evidence]]  # local, keyed by query
    new_argument: Optional[str]


class AdvisorState(TypedDict):
    claim: str
    debate_log: List[str]
    evidence_pool: Dict[str, List[Evidence]]

    analysis: Optional[str]
    advice: Optional[str]


class VerifierState(TypedDict):
    claim: str
    debate_log: List[str]
    evidence_pool: Dict[str, List[Evidence]]
    advisor_advice: Optional[str]

    generated_queries: List[str]
    retrieved_evidence: Dict[str, List[Evidence]]  # local, keyed by query
    verdict: Optional[VerdictLabel]
    report: Optional[str]

