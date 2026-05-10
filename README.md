# SHL Assessment Recommender

A conversational AI agent that guides HR professionals and recruiters from a vague intent ("I need to hire a Java developer") to a grounded shortlist of SHL assessments through multi-turn dialogue.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup (PyCharm)](#local-setup-pycharm)
- [Running the Server](#running-the-server)
- [Testing the API](#testing-the-api)
- [Running Evaluation](#running-evaluation)
- [Deployment](#deployment)
  - [Render](#render)
  - [Fly.io](#flyio)
  - [Docker (generic)](#docker-generic)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Stack Justification](#stack-justification)

---

## Architecture Overview

```
User message
     │
     ▼
POST /chat  (FastAPI)
     │
     ├─► FAISS Semantic Search ──► Top-20 relevant catalogue items
     │         (all-MiniLM-L6-v2 embeddings)
     │
     ├─► Build System Prompt  ──► Injected catalogue context + rules
     │
     ├─► Groq LLM (llama-3.1-70b-versatile, JSON mode)
     │         Full conversation history + grounded prompt
     │
     └─► Parse JSON Response ──► {reply, recommendations, end_of_conversation}
```

**Key design decisions:**
- **Stateless API**: full conversation history sent on every call; no server-side session state.
- **Retrieval-augmented prompting**: catalogue is searched per request and injected into context → prevents hallucination.
- **JSON mode**: Groq's JSON response format guarantees parseable output.
- **FAISS IndexFlatIP**: cosine similarity over normalised L2 embeddings; exact search; no approximate index needed for a catalogue of ~60 items.

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + lifespan (agent init)
│   ├── agent.py         # Core SHL agent: retrieval → prompt → LLM → parse
│   ├── catalog.py       # Catalogue loader + FAISS index
│   ├── models.py        # Pydantic request/response models
│   └── prompts.py       # System prompt builder
├── data/
│   └── shl_catalog.json # ~60 SHL Individual Test Solutions
├── scripts/
│   ├── build_index.py   # Pre-build FAISS index (run once)
│   ├── test_api.py      # Smoke tests against live server
│   └── eval_traces.py   # Recall@10 evaluation on C1–C10 traces
├── .env.example         # Environment variable template
├── Dockerfile
├── fly.toml             # Fly.io config
├── render.yaml          # Render config
├── requirements.txt
├── approach.md
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 + | 3.11 recommended |
| pip | latest | `pip install --upgrade pip` |
| Git | any | for cloning |
| Groq API key | — | Free at [console.groq.com](https://console.groq.com) |

---

## Local Setup (PyCharm)

### Step 1 — Clone / open the project

```bash
git clone <your-repo-url>
cd shl-recommender
```

Or open the folder directly in PyCharm: **File → Open → select `shl-recommender/`**

---

### Step 2 — Create a virtual environment

**In PyCharm:**
1. Go to **File → Settings → Project → Python Interpreter**
2. Click the gear icon → **Add Interpreter → Add Local Interpreter**
3. Choose **Virtualenv** → select Python 3.10+
4. Click **OK**

**Or via terminal:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

### Step 3 — Install dependencies

Open the **PyCharm Terminal** (bottom toolbar) or use your system terminal:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads:
- FastAPI + Uvicorn
- LangChain + langchain-groq
- FAISS (CPU build)
- sentence-transformers (downloads `all-MiniLM-L6-v2` ~22 MB on first run)
- python-dotenv, httpx, pydantic

> **First install takes 2–5 minutes** due to PyTorch (needed by sentence-transformers).

---

### Step 4 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get a free key at [console.groq.com](https://console.groq.com).

---

### Step 5 — Pre-build the FAISS index

```bash
python scripts/build_index.py
```

Output:
```
Building FAISS index…
Done. Index contains 60 catalogue items.
Files written:
  data/faiss.index
  data/catalog_meta.json
```

This only needs to run once (or whenever `shl_catalog.json` changes).

---

## Running the Server

### Via PyCharm Run Configuration

1. **Run → Edit Configurations → + → Python**
2. Set:
   - **Script path:** point to `venv/bin/uvicorn` (or use module mode below)
   - **Module:** `uvicorn`
   - **Parameters:** `app.main:app --reload --host 0.0.0.0 --port 8000`
   - **Working directory:** `shl-recommender/` root
3. Click **Run** ▶

### Via Terminal (recommended)

```bash
# From the shl-recommender/ directory with venv activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO     Initialising SHL agent…
INFO     Building FAISS index… (or loading existing)
INFO     SHL agent ready.
INFO     Uvicorn running on http://0.0.0.0:8000
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Testing the API

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Manual chat test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java backend developer."}
    ]
  }'
```

### Multi-turn conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java backend developer."},
      {"role": "assistant", "content": "Is this for selection or development? And what seniority — senior IC or tech lead?"},
      {"role": "user", "content": "Selection only. Senior IC, 6 years experience."}
    ]
  }'
```

### Automated smoke tests

```bash
# With the server running in another terminal
python scripts/test_api.py
```

---

## Running Evaluation

```bash
# With the server running
python scripts/eval_traces.py

# Against a deployed URL
python scripts/eval_traces.py --base-url https://your-app.onrender.com
```

This replays simplified versions of the C1–C10 traces and prints Recall@10 per trace plus the mean.

---

## Deployment

### Render

1. Push code to a GitHub repo.
2. Go to [render.com](https://render.com) → **New → Web Service → Connect Repo**
3. Choose **Docker** runtime (detected from `render.yaml`).
4. Set environment variable `GROQ_API_KEY` in the Render dashboard (**Environment** tab).
5. Click **Deploy**. First deploy takes ~5 min (Docker build downloads model).
6. Your endpoint will be: `https://shl-recommender.onrender.com`

> Render free tier sleeps after inactivity. The evaluator allows up to 2 minutes for cold start.

---

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch (first time)
flyctl launch --name shl-recommender --no-deploy

# Set secret
flyctl secrets set GROQ_API_KEY=gsk_xxxx

# Deploy
flyctl deploy
```

---

### Docker (generic)

```bash
# Build
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_xxxx shl-recommender
```

---

## API Reference

### `GET /health`

Returns service readiness.

**Response 200:**
```json
{"status": "ok"}
```

---

### `POST /chat`

Main conversational endpoint. **Stateless** — send full history on every call.

**Request body:**
```json
{
  "messages": [
    {"role": "user",      "content": "I need to hire Java developers."},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user",      "content": "Senior, 5+ years."}
  ]
}
```

**Response:**
```json
{
  "reply": "For a senior Java developer...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules:**
| Field | Type | Notes |
|-------|------|-------|
| `reply` | string | Always present |
| `recommendations` | array or null | null while clarifying; 1–10 items when committed |
| `end_of_conversation` | boolean | true only when user confirms final shortlist |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key |
| `CATALOG_PATH` | No | `data/shl_catalog.json` | Path to catalogue JSON |
| `INDEX_PATH` | No | `data/faiss.index` | Path to FAISS index file |
| `META_PATH` | No | `data/catalog_meta.json` | Path to metadata mirror |

---

## Stack Justification

See `approach.md` for the full design rationale. Short summary:

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Groq + llama-3.1-70b-versatile | Fastest free-tier inference; 70B gives strong instruction-following; JSON mode for reliable output |
| Framework | LangChain | Conversation message formatting; easy Groq integration; well-maintained |
| Vector store | FAISS (CPU) | No server needed; exact search is fine for a 60-item catalogue; ships in a single pip package |
| Embeddings | all-MiniLM-L6-v2 | 22 MB; fast; good semantic quality for product retrieval |
| API | FastAPI | Async, type-safe, auto-docs; best match for the stateless design |
| Deployment | Render / Fly | Free tier; Docker support; health-check aware |
