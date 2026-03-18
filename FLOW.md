## Misinformation Multi-Agent Detection – Codebase Flow

This document describes how a claim moves through the current LangGraph-based multi-agent workflow and what each component does today (with stubbed reasoning).

---

## High-level flow

1. **Entry point**
   - You run the system via:
     - `python main.py --claim "..."` (root entrypoint), which calls
     - `misinfo_detection.cli.main()` → `run_claim(claim: str)`.

2. **Configuration and graph build**
   - `misinfo_detection.cli.run_claim`:
     - Loads configuration from `misinfo_detection.config.load_config()`:
       - Reads `TAVILY_API_KEY` (required).
       - Sets defaults: `reliable_domains`, `max_rounds`, Tavily params.
     - Builds the parent graph via `misinfo_detection.graph.parent.build_parent_graph(config)`.
     - Constructs an initial `ParentState` (from `misinfo_detection.schemas`).

3. **Graph execution**
   - The compiled parent graph is invoked with the initial `ParentState`.
   - Execution proceeds through:
     1. `build_guidance` node
     2. Debate loop:
        - Negative debater subgraph
        - Affirmative debater subgraph
        - Round increment + conditional loop
     3. Advisor subgraph (one-shot)
     4. Verifier subgraph (one-shot)
   - Final outputs are written into `ParentState.final_verdict` and `ParentState.final_report`, which the CLI prints.

---

## State model

### ParentState (shared)

Defined in `misinfo_detection/schemas.py`:

- **Inputs/configuration**
  - `claim: str`
  - `guidance: str`
  - `current_round: int`
  - `max_rounds: int`
- **Shared artifacts**
  - `evidence_pool: dict[str, list[Evidence]]`
  - `debate_log: list[str]`
  - `latest_negative_argument: str | None`
  - `latest_affirmative_argument: str | None`
- **Advisor outputs**
  - `advisor_analysis: str | None`
  - `advisor_advice: str | None`
- **Verifier outputs**
  - `verifier_evidence: dict[str, list[Evidence]]`
  - `final_verdict: Literal["true","false","insufficient"] | None`
  - `final_report: str | None`

Subgraphs receive the current `ParentState`, work inside a **private state** (e.g. `DebaterState`), and then project updates back into `ParentState` (e.g. appending to `debate_log`, merging into `evidence_pool`, or setting `final_verdict`).

---

## Tools and external calls

### Tavily search wrapper

File: `misinfo_detection/tools/search.py`

- `tavily_search(query: str, config: AppConfig) -> list[Evidence]`
  - Configures `langchain_tavily.TavilySearch` with:
    - `tavily_api_key` from env
    - `max_results`, `topic`, `include_domains` from `AppConfig`
  - Calls `search.invoke({"query": query})`
  - Normalizes the response into a list of `Evidence` TypedDicts.

This function is used by:

- Debater subgraphs (to gather supporting/attacking evidence).
- Verifier subgraph (to gather clarifying evidence late in the flow).

---

## Parent graph orchestration

File: `misinfo_detection/graph/parent.py`

1. **Node: `build_guidance`**
   - Implementation: `misinfo_detection.nodes.guidance.build_guidance`.
   - Reads `claim` and writes a text prompt into `ParentState.guidance` to guide all agents.

2. **Debate loop**
   - Uses two compiled debater subgraphs from `misinfo_detection.subgraphs.debater`:
     - `build_debater_subgraph(role="negative", config)`
     - `build_debater_subgraph(role="affirmative", config)`
   - Node sequence:
     - `negative` → `affirmative` → `increment_round`.
   - Conditional edge from `increment_round`:
     - If `current_round < max_rounds` → go back to `negative` for another round.
     - Else → move to `advisor`.

3. **Advisor**
   - Node: `advisor` from `misinfo_detection.subgraphs.advisor.build_advisor_subgraph()`.
   - Runs **once**, after the debate rounds complete.

4. **Verifier**
   - Node: `verifier` from `misinfo_detection.subgraphs.verifier.build_verifier_subgraph(config)`.
   - Runs **once**, after the advisor.
   - This is the graph’s finish point.

---

## Debater subgraphs

File: `misinfo_detection/subgraphs/debater.py`

### Private state: DebaterState

Contains:

- `claim`, `guidance`, `debate_log`
- `role: "negative" | "affirmative"`
- `latest_opponent_argument`
- `generated_queries: list[str]`
- `retrieved_evidence: dict[str, list[Evidence]]`
- `new_argument: str | None`

### Node sequence inside the subgraph

1. `generate_queries(DebaterState) -> DebaterState`
   - Builds a small set of search queries using:
     - The `claim`
     - The opponent’s last argument (if available)
   - Writes them into `generated_queries`.

2. `retrieve_evidence(DebaterState) -> DebaterState`
   - For each `generated_queries` entry, calls `tavily_search`.
   - Stores results in `retrieved_evidence`.

3. `write_argument(DebaterState) -> DebaterState`
   - **Current behavior (stub):**
     - Constructs a short text string summarizing:
       - The role (negative/affirmative)
       - The claim
       - How many evidence snippets were retrieved
     - Does **not** call an LLM yet.
   - Writes this into `new_argument`.

### Projection back into ParentState

The compiled debater graph is wrapped by `build_debater_subgraph` into a callable:

- It:
  - Initializes `DebaterState` from the current `ParentState`.
  - Invokes the debater subgraph.
  - Projects changes back to `ParentState`:
    - Appends `new_argument` to `debate_log`.
    - Updates `latest_negative_argument` or `latest_affirmative_argument`.
    - Merges `retrieved_evidence` into the shared `evidence_pool`.

---

## Advisor subgraph

File: `misinfo_detection/subgraphs/advisor.py`

### Private state: AdvisorState

Contains:

- `claim`
- `debate_log`
- `evidence_pool`
- `analysis`, `advice`

### Node sequence

1. `advisor_analyze(AdvisorState) -> AdvisorState`
   - **Current behavior (stub):**
     - Counts debate turns and evidence snippets.
     - Writes a short, descriptive “analysis” string.

2. `advisor_advice(AdvisorState) -> AdvisorState`
   - **Current behavior (stub):**
     - Wraps the analysis into a generic advisory message for the verifier.

### Projection back into ParentState

- After the advisor subgraph finishes, `build_advisor_subgraph`:
  - Writes `analysis` to `ParentState.advisor_analysis`.
  - Writes `advice` to `ParentState.advisor_advice`.

---

## Verifier subgraph

File: `misinfo_detection/subgraphs/verifier.py`

### Private state: VerifierState

Contains:

- `claim`, `debate_log`, `evidence_pool`, `advisor_advice`
- `generated_queries`
- `retrieved_evidence` (verifier-only)
- `verdict`, `report`

### Node sequence

1. `generate_queries(VerifierState) -> VerifierState`
   - Builds a small set of clarifying queries using:
     - The `claim`
     - (Optionally) the advisor advice text.

2. `retrieve_evidence(VerifierState, config) -> VerifierState`
   - Calls `tavily_search` for each query.
   - Populates verifier-local `retrieved_evidence`.

3. `final_evaluation(VerifierState) -> VerifierState`
   - **Current behavior (stub):**
     - Counts new verifier evidence snippets.
     - Sets:
       - `verdict = "insufficient"`
       - `report` describing the claim and evidence count.

### Projection back into ParentState

- After the verifier subgraph finishes, `build_verifier_subgraph`:
  - Copies `retrieved_evidence` into `ParentState.verifier_evidence`.
  - Writes `verdict` into `ParentState.final_verdict`.
  - Writes `report` into `ParentState.final_report`.

---

## CLI and user-facing behavior

File: `misinfo_detection/cli.py`

- `run_claim(claim: str) -> ParentState`
  - Loads config.
  - Builds the parent graph.
  - Creates an initial `ParentState` and invokes the graph.
  - Returns the final `ParentState`.

- `main(argv: list[str] | None) -> int`
  - Parses `--claim "..."`.
  - Calls `run_claim`.
  - Prints:
    - `final_verdict` (e.g. `insufficient`).
    - `final_report` (stub text for now).

Root `main.py` simply delegates to this CLI, so:

```bash
python main.py --claim "Cigarettes cause lung cancer"
```

runs the full multi-agent workflow and prints the current (stubbed) decision. As you replace the stubbed nodes with real LLM-backed reasoning, this flow and structure remain the same. 

