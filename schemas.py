from typing import TypedDict, List

class Evidence(TypedDict):
    title: str
    url: str
    content: str
    score: float # how similar a source is to the claim

class DebateState(TypedDict):
    claim: str
    round: int
    pro_agent_argument: List[str]
    cons_agent_argument: List[str]
    pro_evidence: List[Evidence]
    cons_evidence: List[Evidence]
