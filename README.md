# Misinformation Multi-Agent Detection

This project is a small prototype for evaluating misinformation claims with a multi-agent debate workflow built with LangGraph and Tavily search.

## What It Does

- Defines a shared debate state in `schemas.py`
- Uses Tavily to retrieve evidence from a small set of reliable publishers
- Builds a simple LangGraph loop with a `pro` agent and a `cons` agent
- Provides both CLI and web UI interfaces

## Architecture Documentation

For a deeper understanding of how the codebase is structured and how data flows through the system, see:

- **[FLOW.md](FLOW.md)** — Describes the overall system flow from entry point through the parent graph, including the state model, Tavily integration, and the sequence of guidance → debate → advisor → verifier stages.
- **[DEBATER_QUERY_PLANNER.md](DEBATER_QUERY_PLANNER.md)** — Details the debater subgraph's query planning subsystem, including Ollama LLM integration, query normalization, similarity-based deduplication, fallback logic, and retry policy.

## Setup

### 1. Create and activate a virtual environment

**Linux/macOS:**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langgraph langchain-tavily python-dotenv streamlit
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_api_key_here

# Optional: Configure Ollama (defaults shown)
OLLAMA_MODEL=qwen:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Optional: Configure debate rounds
MAX_ROUNDS=3
```

### 4. Start Ollama (required for LLM inference)

The system uses Ollama for local LLM inference. Make sure Ollama is installed and running:

```bash
# Start Ollama server (if not already running)
ollama serve

# Pull the required model (in a separate terminal)
ollama pull qwen:7b
```

Extra Notes:

- A claim may take up to 5 min to run depending on the model chosen, larger models ( at least 7b or 8b parameters) is preferred.

## Run

### Option 1: Web UI (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

This will:

- Open your browser automatically at [http://localhost:8501](http://localhost:8501)
- Provide a text input for claims
- Display the verdict, report, advisor analysis, evidence, and debate trace

Extra notes: 

- When launching the streamlit app it will prompt for an email in the terminal which can be skipped, but it can prevent the browser from popping up if unanswered.

### Option 2: Command Line

Run a single claim via CLI:

```bash
python main.py --claim "Your claim to analyze here"
```

## Project Structure

```
├── app.py                    # Streamlit web UI
├── main.py                   # CLI entry point
├── misinfo_detection/
│   ├── cli.py               # Core run_claim() function
│   ├── config.py            # Configuration loader
│   ├── schemas.py           # Data types (ParentState, Evidence, etc.)
│   ├── graph/
│   │   └── parent.py        # Main LangGraph orchestration
│   ├── nodes/
│   │   └── guidance.py      # Initial guidance builder
│   └── subgraphs/
│       ├── debater.py       # Bilateral debate subgraph
│       ├── advisor.py       # Post-debate advisor
│       └── verifier.py      # Final verdict generator
├── evaluation/              # Batch evaluation tools
├── requirements.txt
└── README.md
```

## Status

The early prototype is complete. The search tool, graph structure, multi-agent debate loop, advisor, verifier, evaluation pipeline, and Streamlit UI are all in place.


## AI-Generated Code Declaration

Parts of this codebase were generated or substantially shaped with AI assistance (Cursor AI agent / Claude). The following files and components are AI-generated or AI-assisted:

### AI-Generated


| File                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `misinfo_detection/subgraphs/debater.py` | Ollama-backed LLM query planner (`_call_ollama_query_planner`), query normalization and similarity-based deduplication (`_normalize_query`, `_tokenize`, `_query_similarity`, `_is_similar_to_existing`), fallback query generation (`_fallback_queries`), Ollama argument writer (`_call_ollama_argument_writer`), Ollama reachability check with caching (`_is_ollama_reachable`), structured error types (`OllamaErrorInfo`, `QueryPlannerCallResult`, `ArgumentWriterCallResult`), and error logging (`_log_ollama_error`) |
| `DEBATER_QUERY_PLANNER.md`               | Architecture documentation for the debater query planning subsystem, including Ollama integration, fallback logic, similarity reuse policy, and retry policy                                                                                                                                                                                                                                                                                                                                                                   |
| `FLOW.md`                                | Architecture narrative describing the overall system flow, state model, Tavily integration, and parent graph sequence                                                                                                                                                                                                                                                                                                                                                                                                          |
| `tests/test_guided_debate.py`            | Unit and integration tests for the debater subgraph, including query planner tests, similarity scoring tests, and argument generation tests                                                                                                                                                                                                                                                                                                                                                                                    |
| `app.py`                                 | Streamlit UI for interactive claim evaluation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |


### AI-Assisted


| File                                      | Description                                                                                                                       |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `misinfo_detection/cli.py`                | CLI entry point wiring `argparse`, config loading, and graph execution                                                            |
| `misinfo_detection/config.py`             | `AppConfig` dataclass and `load_config()` from environment variables                                                              |
| `misinfo_detection/schemas.py`            | TypedDicts for `Evidence`, `ParentState`, `DebaterState`, `AdvisorState`, `VerifierState`, and verdict literals                   |
| `misinfo_detection/graph/parent.py`       | Parent LangGraph definition connecting guidance, debate, advisor, and verifier subgraphs                                          |
| `misinfo_detection/nodes/guidance.py`     | Guidance node that prepares claim context for all agents                                                                          |
| `misinfo_detection/tools/search.py`       | Tavily search wrapper normalizing results to `Evidence`                                                                           |
| `misinfo_detection/subgraphs/advisor.py`  | Advisor subgraph parsing the debate log, classifying turns via Ollama, and producing structured advice for the verifier           |
| `misinfo_detection/subgraphs/verifier.py` | Verifier subgraph generating queries, performing Tavily retrieval, and producing a final verdict via Ollama or heuristic fallback |
| `evaluation/runner.py`                    | Batch evaluation driver with checkpointing, timeouts, and report generation                                                       |
| `evaluation/report.py`                    | Aggregation of per-agent metrics into a weakness-oriented report                                                                  |
| `evaluation/parsers.py`                   | Pure parsers for advisor and verifier free-text output sections                                                                   |
| `evaluation/metrics/advisor.py`           | Advisor-specific evaluation metrics                                                                                               |
| `evaluation/metrics/debater.py`           | Debater-specific evaluation metrics                                                                                               |
| `evaluation/metrics/system.py`            | System-level evaluation metrics                                                                                                   |
| `evaluation/metrics/verifier.py`          | Verifier-specific evaluation metrics                                                                                              |
| `tests/test_advisor.py`                   | Unit tests for the advisor subgraph                                                                                               |
| `tests/test_verifier.py`                  | Unit tests for the verifier subgraph                                                                                              |


