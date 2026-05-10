# Approach Document — SHL Assessment Recommender

## 1. Problem Decomposition

The core challenge is converting a vague, natural-language hiring intent into a grounded shortlist of SHL assessments through dialogue — without hallucinating products or URLs. Three sub-problems drive the design:

1. **Grounding**: the LLM must only recommend products that exist in the catalogue.
2. **Conversation control**: the agent must know when to ask, when to recommend, and when to stop.
3. **Reliability**: the output schema must be machine-parseable on every turn.

---

## 2. Stack and Justification

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **LLM** | Groq + `llama-3.1-70b-versatile` | Fastest free-tier inference (~250 tok/s); 70B parameters give strong instruction-following and JSON adherence; JSON mode eliminates format failures |
| **Agent framework** | LangChain (langchain-groq, langchain-core) | Thin wrapper for message formatting and async invocation; avoids reinventing chat history management |
| **Vector store** | FAISS `IndexFlatIP` | No server needed; installs as a single pip package; exact cosine search is fine for a 60-item catalogue (no approximation error) |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | 22 MB; 5 ms per query; strong semantic quality for product retrieval; no API dependency |
| **API** | FastAPI + Uvicorn | Async-first; Pydantic validation enforces schema on both request and response; auto-generated docs at /docs |
| **Deployment** | Render (Docker) / Fly.io | Free tier; Docker support; health-check aware; cold-start within the 2-min evaluator window |

**What I tried and dropped:**
- **OpenRouter free models**: lower JSON compliance, higher latency.
- **ChromaDB**: overkill for 60 items; FAISS is simpler and faster.
- **LangChain agents with tools**: added latency and complexity; a single-shot prompt with retrieved context is faster and more predictable within the 30-second timeout.

---

## 3. Retrieval Setup

Each catalogue item is embedded as a concatenation of its name, test type, category keys, description, use cases, level, and domain. This rich text means a query like "contact centre agents English US" retrieves both the spoken-language screen and the simulation — not just keyword-matched items.

At request time, the last four user messages are concatenated into a search query. The top 20 results are injected into the system prompt as a formatted catalogue block. Only items in that block can appear in recommendations (enforced by a URL prefix check on the LLM output).

The FAISS index is built once (either at Docker build time or via `scripts/build_index.py`) and loaded from disk on startup, keeping cold-start latency low.

---

## 4. Prompt Design

The system prompt has three parts:

1. **Rules** (10 numbered): clarify before recommending, max 10 items, URL must come from context, refuse off-topic, `end_of_conversation` semantics, etc. Numbered rules increase LLM compliance.
2. **Catalogue context**: the 20 retrieved items formatted as structured bullet points. Injected fresh per request so the agent always has the most relevant subset.
3. **Output schema**: explicit JSON template with field-level comments. Combined with Groq's JSON mode this gives near-100% parseable output.

**Conversation control rules in the prompt:**
- `recommendations` must be `null` while still clarifying.
- `end_of_conversation` is `true` only when the user has explicitly confirmed the final shortlist.
- The agent should ask at most one clarifying question per turn.

---

## 5. Output Parsing

Even with JSON mode, the parser applies three fallback strategies in order:
1. Direct `json.loads` on the raw response.
2. Strip markdown fences, then parse.
3. Regex-extract the first `{…}` block, then parse.

Recommendations undergo URL validation: any URL not starting with `https://www.shl.com/` is silently dropped. This prevents hallucinated URLs from reaching the evaluator.

---

## 6. Evaluation Approach

**Hard evals** (schema + scope) are tested by `scripts/test_api.py`:
- Vague first message → no recommendations.
- Off-topic / injection → no recommendations.
- Every returned URL starts with `https://www.shl.com/`.
- Schema keys present on every response.

**Recall@10** is measured by `scripts/eval_traces.py` against all ten C1–C10 traces using hardcoded expected shortlists derived from the trace endings. A simplified one-shot conversation (persona summary) is sent to the agent; if recommendations are returned they are compared against expected names (case-insensitive).

**Improvements measured:**
- Switching from a keyword system prompt to RAG-injected context raised Recall@10 from ~0.55 to ~0.82 on the 10 public traces.
- Adding `use_cases` and `level` fields to the catalogue embeddings improved retrieval of domain-specific items (e.g. DSI for healthcare admin, SVAR for contact centre).
- Groq JSON mode eliminated ~15% of turns that previously returned non-parseable text.

---

## 7. AI Tools Used

Claude (Anthropic) was used for: drafting the system prompt rules, writing the catalogue item descriptions, and reviewing edge cases in the conversation traces. All code was written by the author and is defensible in a technical interview.
