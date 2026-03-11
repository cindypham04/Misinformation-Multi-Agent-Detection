import os
from typing import List

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph

from schemas import DebateState, Evidence
from dotenv import load_dotenv

# ------------------------
# 1. API Calls
# ------------------------

# API of TAVILY for searching articles
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("Missing TAVILY_API_KEY in .env")

# Since we already have ollama locally, we might call LLM without API

# ------------------------
# 2. Building Nodes
# ------------------------

# Affirmative agent
def pro_agent(state: DebateState) -> DebateState:
    claim = state["claim"] # Getting claim from the current state

    # 1. Search for evidence from the claim 
    results = external_search(claim)

    # 2. Call LLM to construct generation from the evidence 

    # 3. Update state with new evidence and claim 

    # 4. Return new state
    


# Negative agent
def cons_agent(state: DebateState) -> DebateState:
    claim = state["claim"] # Getting claim from the current state

    # 1. Search for evidence from the claim 
    results = external_search(claim)

    # 2. Call LLM to construct generation from the evidence 

    # 3. Update state with new evidence and claim 

    # 4. Return new state


# ------------
# 3. Tools
# ------------

# External search tool
def external_search(claim: str) -> List[Evidence]:
    # List of reliable sources 
    reliable_publishers = [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "politifact.com",
        "snopes.com",
        "cdc.gov",
        "who.int"]
    
    # Conduct searching
    search = TavilySearch(
        tavily_api_key=tavily_api_key,
        max_results=5,
        topic="general",
        include_domains=reliable_publishers,
    )

    # Return results
    results = search.invoke({"query": claim})

    return results

print(external_search("Cigarettes cause lung cancer"))

# ------------------------
# 4. Build Debate Loop
# ------------------------

# Pass the type of the state to the graph builder
builder = StateGraph(DebateState) 

# Add nodes to graph
builder.add_node("pro", pro_agent)
builder.add_node("cons", cons_agent)

# Add edges to graph
builder.add_edge("START", "pro")
builder.add_edge("pro", "cons")
builder.add_edge("cons", "pro")
builder.add_edge("pro", "cons")
builder.add_edge("cons", "END")

graph = builder.compile()

# Initial state
initial_state = DebateState(
    claim="Covid vaccines cause infertility",
    round=0,
    pro_agent_argument=[],
    cons_agent_argument=[],
    pro_evidence=[],
    cons_evidence=[]
)

# Run graph
graph.invoke(initial_state)
