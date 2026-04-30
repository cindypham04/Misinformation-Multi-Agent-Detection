## Misinformation Multi-Agent Detection – Codebase Flow

This document describes how a claim moves through the LangGraph-based multi-agent workflow and what each component does.

---

## High-level flow

1. **Entry point**
   - You run the system via:
     - `python main.py --claim "..."` (root entrypoint), which calls
     - `misinfo_detection.cli.main()` → `run_claim(claim: str)`.
   - Alternatively, the Streamlit web UI (`app.py`) calls `run_claim` directly.

2. **Configuration and graph build**
   - `misinfo_detection.cli.run_claim`:
     - Loads configuration from `misinfo_detection.config.load_config()`:
       - Reads `TAVILY_API_KEY` (required).
       - Sets defaults: `reliable_domains`, `max_rounds`, Tavily params.
       - Reads optional Ollama settings: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`.
     - Builds the parent graph via `misinfo_detection.graph.parent.build_parent_graph(config=config)`.
     - Constructs an initial `ParentState` (from `misinfo_detection.schemas`).

3. **Graph execution**
   - The compiled parent graph is invoked with the initial `ParentState`.
   - Execution proceeds through:
     1. `build_guidance` node
     2. Debate loop:
        - `debate` node (bilateral subgraph — negative then affirmative in one call)
        - `increment_round` node + conditional loop back to `debate`
     3. `advisor` node (one-shot)
     4. `verifier` node (one-shot)
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
  - `final_verdict: Literal["supported", "refuted", "insufficient"] | None`
  - `final_report: str | None`

Subgraphs receive the current `ParentState`, work inside a **private state**, and then project updates back into `ParentState` (e.g. appending to `debate_log`, merging into `evidence_pool`, or setting `final_verdict`).

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

- The bilateral debater subgraph (to gather supporting/attacking evidence for both roles).
- The verifier subgraph (to gather clarifying evidence late in the flow).

---

## Parent graph orchestration

File: `misinfo_detection/graph/parent.py`

1. **Node: `build_guidance`**
   - Implementation: `misinfo_detection.nodes.guidance.build_guidance`.
   - Reads `claim` and writes a text prompt into `ParentState.guidance` to guide all agents.

2. **Debate loop**
   - Uses a single bilateral debater callable from `misinfo_detection.subgraphs.debater`:
     - `build_debater_subgraph(config=config)`
   - The returned callable handles both the negative and affirmative turns internally in one subgraph run.
   - Node sequence:
     - `debate` → `increment_round`.
   - Conditional edge from `increment_round`:
     - If `current_round < max_rounds` → go back to `debate` for another round.
     - Else → move to `advisor`.

3. **Advisor**
   - Node: `advisor` from `misinfo_detection.subgraphs.advisor.build_advisor_subgraph()`.
   - Runs **once**, after the debate rounds complete.

4. **Verifier**
   - Node: `verifier` from `misinfo_detection.subgraphs.verifier.build_verifier_subgraph(config=config)`.
   - Runs **once**, after the advisor.
   - This is the graph's finish point.

---

## Debater subgraph

File: `misinfo_detection/subgraphs/debater.py`

### Private state: BilateralDebateState

A single shared state used for both the negative and affirmative turns inside one subgraph run. Contains:

- `claim`, `guidance`
- `evidence_pool: dict[str, list[Evidence]]` — shared across both roles during this round
- `debate_log: list[str]`
- `latest_negative_argument: str | None`
- `latest_affirmative_argument: str | None`
- `generated_queries: list[str]` — transient, for the currently executing role turn
- `retrieved_evidence: dict[str, list[Evidence]]` — transient, for the currently executing role turn

### Node sequence inside the subgraph

Both roles follow the same three-step pipeline, executed sequentially:

1. `negative_generate_queries` → `negative_retrieve_evidence` → `negative_write_argument`
2. `affirmative_generate_queries` → `affirmative_retrieve_evidence` → `affirmative_write_argument`

**generate_queries** (per role):
   - Calls `_call_ollama_query_planner` with a structured prompt containing the role objective, claim, guidance, opponent's last argument, recent debate log tail, and already-searched queries.
   - If Ollama is unavailable or returns an invalid response, falls back to `_fallback_queries` (deterministic queries from the claim and opponent argument).
   - Applies similarity-based deduplication: candidates are compared against `evidence_pool` keys using Jaccard token similarity and sequence similarity; near-duplicates are replaced with the canonical existing query.
   - At most 5 queries are kept.

**retrieve_evidence** (per role):
   - For each query in `generated_queries`:
     - Reuses cached results from `evidence_pool` if the query was already searched.
     - Otherwise calls `_search_with_retry` (up to 3 attempts with exponential backoff) → `tavily_search`.
   - Updates the shared `evidence_pool` and the transient `retrieved_evidence`.

**write_argument** (per role):
   - Calls `_call_ollama_argument_writer` with a structured prompt containing the role instruction, claim, guidance, compact opponent argument, recent debate log tail, and a summarized evidence list.
   - If Ollama is unavailable or returns an invalid response, falls back to `_fallback_argument_text`, which selects aligned or conflicting evidence heuristically and constructs a stance-preserving fallback argument.
   - Prepends `[negative]` or `[affirmative]` to the argument and appends it to `debate_log`.
   - Updates `latest_negative_argument` or `latest_affirmative_argument`.

### Projection back into ParentState

After the bilateral subgraph finishes, `build_debater_subgraph` projects:
- `debate_log` (both new arguments appended)
- `latest_negative_argument`, `latest_affirmative_argument`
- `evidence_pool` (merged with any new queries retrieved this round)

---

## Advisor subgraph

File: `misinfo_detection/subgraphs/advisor.py`

### Private state: AdvisorState

Contains:

- `claim`
- `debate_log`
- `evidence_pool`
- `analysis: str | None`
- `advice: str | None`
- `analysis_data: dict | None` — cached from `advisor_analyze` to avoid a double LLM call

### Node sequence

1. `advisor_analyze(AdvisorState) -> AdvisorState`
   - Parses `debate_log` into structured turns, grouping by role.
   - Calls `_classify_turns_with_ollama` to get per-turn relevance and quality labels (valid, unresolved, unsupported, logical_leap, redundant, irrelevant) from the Ollama LLM. Falls back to heuristic classification if Ollama is unavailable.
   - Detects redundant turns via token-signature comparison.
   - Builds an `evidence_pool` lexicon for overlap scoring.
   - Writes a structured `analysis` string covering turn counts, role breakdowns, and labeled point buckets.
   - Caches the computed analysis into `analysis_data` to share with the next node.

2. `advisor_advice(AdvisorState) -> AdvisorState`
   - Reads `analysis_data` (cached from step 1, recomputed only if missing).
   - Selects the highest-priority valid points, remaining gaps, assertions needing scrutiny, and low-value/noisy points.
   - Determines a `verifier_focus` directive based on which buckets are non-empty.
   - Writes a structured `advice` string for the verifier.

### Projection back into ParentState

- `analysis` → `ParentState.advisor_analysis`
- `advice` → `ParentState.advisor_advice`

---

## Verifier subgraph

File: `misinfo_detection/subgraphs/verifier.py`

### Private state: VerifierState

Contains:

- `claim`, `debate_log`, `evidence_pool`, `advisor_advice`
- `generated_queries: list[str]`
- `retrieved_evidence: dict[str, list[Evidence]]` — verifier-only
- `verdict: VerdictLabel | None`
- `report: str | None`

### Node sequence

1. `generate_queries(VerifierState) -> VerifierState`
   - Starts with two base queries: `"{claim} fact check"` and `"{claim} evidence systematic review"`.
   - Extracts additional queries from the advisor advice sections (valid points, unresolved gaps, scrutiny assertions) via `_build_advice_queries`.
   - Deduplicates and keeps at most 5 queries.

2. `retrieve_evidence(VerifierState, config) -> VerifierState`
   - Calls `tavily_search` for each query, skipping already-retrieved ones.
   - Populates verifier-local `retrieved_evidence`.

3. `final_evaluation(VerifierState) -> VerifierState`
   - Calls `_call_ollama_verifier` with a structured prompt containing the claim, truncated debate log, advisor advice, shared evidence summary, and verifier evidence summary.
   - Ollama returns `{"verdict": "supported|refuted|insufficient", "report": "..."}`.
   - If Ollama is unavailable or returns an invalid response, falls back to `_fallback_verdict`:
     - Scores the negative and affirmative cases by counting role mentions in each advisor advice section with weighted rules.
     - Returns `supported`, `refuted`, or `insufficient` based on the score differential and evidence availability.
   - Sets `verdict` and `report`.

### Projection back into ParentState

- `retrieved_evidence` → `ParentState.verifier_evidence`
- `verdict` → `ParentState.final_verdict`
- `report` → `ParentState.final_report`

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
    - `final_verdict` (e.g. `supported`, `refuted`, or `insufficient`).
    - `final_report` (structured text from the verifier).

Root `main.py` simply delegates to this CLI, so:

```bash
python main.py --claim "Cigarettes cause lung cancer"
```

runs the full multi-agent workflow and prints the final verdict and report.

### Streamlit UI

File: `app.py`

- Provides a web interface for interactive claim evaluation.
- Calls `run_claim` and displays `final_verdict` and `final_report` in the browser.
- Run with:

```bash
streamlit run app.py
```
