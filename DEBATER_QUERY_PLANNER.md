# Debater Query Planning Enhancements

## Overview

The debater subgraph now supports LLM-assisted query planning with deterministic fallbacks and
similarity-based query reuse. This improves retrieval quality and avoids repeated external searches
for near-duplicate query intent.

## Added Functionality

- Added `_find_similar_existing_query(...)` to detect near-duplicate query intent against
  existing keys in `evidence_pool`.
- Added normalization, tokenization, and scoring helpers:
  - `_normalize_query_text(...)`
  - `_tokenize_query_text(...)`
  - `_jaccard_similarity(...)`
  - `_sequence_similarity(...)`
  - `_dedupe_preserve_order(...)`
- Added LLM query-planning helpers:
  - `_build_query_planner_prompt(...)`
  - `_call_ollama_query_planner(...)`
- Added deterministic fallback query generation via `_fallback_queries(...)`, which produces:
  - `"{claim} fact check {opponent_argument[:100]}"` (if an opponent argument exists)
  - `"{claim} fact check"`
  - `"{claim} evidence"`
  - `"{claim} Reuters"`
  - `"{claim} AP News"`

## Bilateral State Sharing

Both the negative and affirmative turns run inside a single `BilateralDebateState` instance per round. Because they share one `evidence_pool`, the existing query keys visible to the affirmative role's query planner already include any queries the negative role retrieved earlier in the same round. This means the affirmative's deduplication and canonicalization logic works against a fully up-to-date pool, reducing redundant retrieval within a round.

## Ollama Configuration

Both `_call_ollama_query_planner` and `_call_ollama_argument_writer` read the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen:7b` | The Ollama model to use for all LLM calls |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Base URL of the Ollama HTTP API |

## `_generate_queries_for_role(...)` Behavior

For each debater role turn:

1. Builds context from:
   - current claim
   - parent guidance prompt
   - latest opponent argument
   - last 8 entries of the debate log
   - existing `evidence_pool` query keys (includes queries from both roles so far)
2. Attempts to get 3–5 strict-JSON queries from Ollama via `_call_ollama_query_planner`.
3. Falls back to `_fallback_queries` if LLM output is unavailable or invalid.
4. Canonicalizes generated queries by mapping similar query text to an already-existing query key
   from `evidence_pool`.
5. Deduplicates and caps final queries to 5.
6. Ensures a minimum of 3 queries by supplementing with fallback queries when needed.

## Similarity Reuse Logic

`_find_similar_existing_query(...)` compares candidate queries to existing query keys using:

- token overlap (Jaccard similarity)
- normalized string similarity (SequenceMatcher ratio)
- weighted combined score

Thresholds are currently:

- Jaccard >= 0.72, or
- sequence similarity >= 0.86, or
- combined score >= 0.78

If a match is found, the candidate is replaced with the canonical existing query key so retrieval
can reuse cached evidence.

## Retrieval Cache Reuse Update

In `_retrieve_evidence_for_role(...)`, if a generated query already exists in `evidence_pool`,
the function now copies it into `retrieved_evidence` for the current turn (instead of skipping it
entirely). This preserves turn-level evidence accounting without re-calling external search.

## Retrieval Retry Policy

New searches use `_search_with_retry(...)`:

- retries Tavily calls up to 3 attempts (`max_attempts=3`)
- uses exponential backoff starting at 0.2 seconds (`initial_backoff_seconds=0.2`), doubling on each retry
- returns an empty list after all retries fail, allowing the turn to continue

This keeps debate execution resilient when one query fails while still collecting evidence for
other queries in the same turn.

## Expected Impact

- Better role-aware and context-aware query generation.
- Reduced duplicate retrieval across rounds and across both debaters.
- Stronger alignment with guided debate design using claim, opponent argument, and debate history.
