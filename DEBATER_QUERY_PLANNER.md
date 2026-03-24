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
- Added deterministic fallback query generation via `_fallback_queries(...)`.

## `_generate_queries_for_role(...)` Behavior

For each debater role turn:

1. Builds context from:
   - current claim
   - parent guidance prompt
   - latest opponent argument
   - recent debate history
   - existing evidence query keys
2. Attempts to get 3-5 strict-JSON queries from Ollama.
3. Falls back to deterministic base queries if LLM output is unavailable or invalid.
4. Canonicalizes generated queries by mapping similar query text to an already-existing query key
   from `evidence_pool`.
5. Deduplicates and caps final queries to 5.
6. Ensures minimum query coverage by supplementing with fallback queries when needed.

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

## Expected Impact

- Better role-aware and context-aware query generation.
- Reduced duplicate retrieval across rounds and across both debaters.
- Stronger alignment with guided debate design using claim, opponent argument, and debate history.
