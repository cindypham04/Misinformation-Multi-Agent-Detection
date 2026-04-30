# Misinformation Multi-Agent Detection

This project is a small prototype for evaluating misinformation claims with a multi-agent debate workflow built with LangGraph and Tavily search.

## What It Does

- Defines a shared debate state in `schemas.py`
- Uses Tavily to retrieve evidence from a small set of reliable publishers
- Builds a simple LangGraph loop with a `pro` agent and a `cons` agent

## Setup

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install langgraph langchain-tavily python-dotenv
```

Add your Tavily API key to `.env`:

```env
TAVILY_API_KEY=your_api_key_here
```

## Run

```bash
python main.py
```

## Status

The early prototype is complete. The search tool, graph structure, multi-agent debate loop, advisor, verifier, evaluation pipeline, and Streamlit UI are all in place.

## AI-Generated Code Declaration

Parts of this codebase were generated or substantially shaped with AI assistance (Cursor AI agent / Claude). The following files and components are AI-generated or AI-assisted:

### AI-Generated

| File | Description |
|------|-------------|
| `misinfo_detection/subgraphs/debater.py` | Ollama-backed LLM query planner (`_call_ollama_query_planner`), query normalization and similarity-based deduplication (`_normalize_query`, `_tokenize`, `_query_similarity`, `_is_similar_to_existing`), fallback query generation (`_fallback_queries`), Ollama argument writer (`_call_ollama_argument_writer`), Ollama reachability check with caching (`_is_ollama_reachable`), structured error types (`OllamaErrorInfo`, `QueryPlannerCallResult`, `ArgumentWriterCallResult`), and error logging (`_log_ollama_error`) |
| `DEBATER_QUERY_PLANNER.md` | Architecture documentation for the debater query planning subsystem, including Ollama integration, fallback logic, similarity reuse policy, and retry policy |
| `FLOW.md` | Architecture narrative describing the overall system flow, state model, Tavily integration, and parent graph sequence |
| `tests/test_guided_debate.py` | Unit and integration tests for the debater subgraph, including query planner tests, similarity scoring tests, and argument generation tests |

### AI-Assisted 

| File | Description |
|------|-------------|
| `misinfo_detection/cli.py` | CLI entry point wiring `argparse`, config loading, and graph execution |
| `misinfo_detection/config.py` | `AppConfig` dataclass and `load_config()` from environment variables |
| `misinfo_detection/schemas.py` | TypedDicts for `Evidence`, `ParentState`, `DebaterState`, `AdvisorState`, `VerifierState`, and verdict literals |
| `misinfo_detection/graph/parent.py` | Parent LangGraph definition connecting guidance, debate, advisor, and verifier subgraphs |
| `misinfo_detection/nodes/guidance.py` | Guidance node that prepares claim context for all agents |
| `misinfo_detection/tools/search.py` | Tavily search wrapper normalizing results to `Evidence` |
| `misinfo_detection/subgraphs/advisor.py` | Advisor subgraph parsing the debate log, classifying turns via Ollama, and producing structured advice for the verifier |
| `misinfo_detection/subgraphs/verifier.py` | Verifier subgraph generating queries, performing Tavily retrieval, and producing a final verdict via Ollama or heuristic fallback |
| `evaluation/runner.py` | Batch evaluation driver with checkpointing, timeouts, and report generation |
| `evaluation/report.py` | Aggregation of per-agent metrics into a weakness-oriented report |
| `evaluation/parsers.py` | Pure parsers for advisor and verifier free-text output sections |
| `evaluation/metrics/advisor.py` | Advisor-specific evaluation metrics |
| `evaluation/metrics/debater.py` | Debater-specific evaluation metrics |
| `evaluation/metrics/system.py` | System-level evaluation metrics |
| `evaluation/metrics/verifier.py` | Verifier-specific evaluation metrics |
| `tests/test_advisor.py` | Unit tests for the advisor subgraph |
| `tests/test_verifier.py` | Unit tests for the verifier subgraph |
| `app.py` | Streamlit UI for interactive claim evaluation |
