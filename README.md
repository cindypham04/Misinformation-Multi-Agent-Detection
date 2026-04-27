# Misinformation Multi-Agent Detection

This project is a small prototype for evaluating misinformation claims with a multi-agent debate workflow built with LangGraph and Tavily search.

## What It Does

- Defines a shared debate state in `schemas.py`
- Uses Tavily to retrieve evidence from a small set of reliable publishers
- Builds a simple LangGraph loop with a `pro` agent and a `cons` agent
- Provides both CLI and web UI interfaces

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

## Run

### Option 1: Web UI (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

This will:
- Open your browser automatically at http://localhost:8501
- Provide a text input for claims
- Display the verdict, report, advisor analysis, evidence, and debate trace

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

The project is currently an early prototype. The search tool and graph structure are in place, but the agent reasoning and debate updates are still incomplete.
