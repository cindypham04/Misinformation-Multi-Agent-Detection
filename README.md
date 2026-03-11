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

The project is currently an early prototype. The search tool and graph structure are in place, but the agent reasoning and debate updates are still incomplete.
