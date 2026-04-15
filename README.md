# CortexOps(OrionAI) — Production Multi-Agent Inference Engine

> **CortexOps(OrionAI)** is a scalable, production-grade agent orchestration system that routes user queries through a structured four-stage AI pipeline with memory, retry logic, structured JSON outputs, latency tracking, and a live monitoring dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      User / HTTP Client                          │
└──────────────────────────────┬───────────────────────────────────┘
                               │  POST /run  { query, session_id }
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Layer  (:8000)                         │
│  • Request validation (Pydantic v2)                              │
│  • Async request handling                                         │
│  • Global exception handler                                       │
│  • Request timing middleware                                       │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Orchestrator Engine                            │
│  • Cache check (skip pipeline if duplicate query)                │
│  • Drives the 4-agent pipeline                                   │
│  • Manages Executor→Critic retry loop (max 3)                    │
│  • Tracks per-agent and total latency                            │
│  • Writes structured step records to memory                      │
│  • Records metrics for the live dashboard                        │
└──┬──────────────┬──────────────┬──────────────┬─────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌──────┐     ┌─────────┐   ┌─────────┐   ┌─────────┐
│Plann-│     │Retriever│   │Executor │   │ Critic  │
│  er  │────▶│  Agent  │──▶│  Agent  │──▶│  Agent  │
│Agent │     │         │   │(concurr-│   │(scores, │
│      │     │2-tier:  │   │ent step │   │ retries)│
│Steps │     │corpus + │   │ exec +  │   │approves │
│plan  │     │LLM synth│   │synthesis│   │or rejects│
└──────┘     └─────────┘   └─────────┘   └────┬────┘
                                              │  rejected?
                                              ▼
                                    ┌──────────────────┐
                                    │  Retry Loop      │
                                    │  (feedback →     │
                                    │   next attempt)  │
                                    └──────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                            │
│  ┌─────────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Memory Store        │  │   TTL Cache  │  │  Metrics Store │  │
│  │  (Redis / in-memory) │  │  (in-process)│  │  (rolling 200) │  │
│  └─────────────────────┘  └──────────────┘  └────────────────┘  │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐   │
│  │  LLM Client          │  │  Structured JSON Logger           │   │
│  │  (OpenAI / Mock)     │  │  (stdout + rotating file)        │   │
│  └─────────────────────┘  └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Live Dashboard  GET /dashboard             │
│  Charts: Latency · Retries · Confidence     │
│  Agent latency bars · Request history table │
└─────────────────────────────────────────────┘
```

---

## Agents

| Agent | Role | Output Schema |
|---|---|---|
| **Planner** | Decomposes the query into 2–6 ordered, typed steps | `{ steps: [{id, description, type}] }` |
| **Retriever** | Fetches context (keyword corpus first, LLM synthesis fallback) | `{ context, sources: [{title, relevance}] }` |
| **Executor** | Runs each step concurrently via LLM, then synthesises a final answer | `{ results: [{step_id, output}], combined }` |
| **Critic** | Scores the answer on 4 dimensions (relevance·accuracy·completeness·clarity), approves or rejects | `{ approved, confidence, feedback, scores }` |

### Retry Logic

The **Critic Agent** gates the Executor's output. If `confidence < threshold` (default 0.70):
1. The Critic's `feedback` is passed back to the Executor as context.
2. The Executor retries (same plan, same context, improved prompt).
3. This repeats up to `MAX_RETRIES` times (default 3).
4. After exhausting retries, the last answer is always returned.

---

## Project Structure

```
CortexOps/
├── main.py                    ← FastAPI app factory + entry point
├── requirements.txt
├── .env.example
│
├── api/
│   ├── models.py              ← Pydantic request/response models
│   ├── routes.py              ← POST /run, GET /health, /session/* endpoints
│   └── metrics_routes.py      ← GET /metrics/summary, /history, /dashboard
│
├── agents/
│   ├── base_agent.py          ← Abstract base + AgentResult envelope
│   ├── planner.py             ← PlannerAgent
│   ├── retriever.py           ← RetrieverAgent (2-tier)
│   ├── executor.py            ← ExecutorAgent (concurrent steps + synthesis)
│   └── critic.py              ← CriticAgent (4-dimension scoring)
│
├── orchestrator/
│   └── engine.py              ← OrchestratorEngine (pipeline + retry loop)
│
├── memory/
│   └── store.py               ← MemoryStore (Redis / in-memory fallback)
│
├── utils/
│   ├── inference.py           ← LLMClient (OpenAI async + mock mode)
│   ├── logger.py              ← StructuredLogger (JSON, rotating file)
│   ├── cache.py               ← SimpleCache (TTL, thread-safe)
│   └── metrics.py             ← MetricsStore (rolling 200-request window)
│
├── config/
│   └── settings.py            ← Pydantic BaseSettings (config-driven)
│
├── dashboard/
│   └── index.html             ← Live Latency & Retry Dashboard (SPA)
│
└── logs/
    └── orion.log              ← JSON-structured rotating log file
```

---

## Setup

### 1. Clone & install

```bash
cd CortexOps
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

> **No API key?** Leave `OPENAI_API_KEY=sk-placeholder` — the system runs in **Mock Mode** using deterministic responses so the full pipeline executes without any real LLM calls.

### 3. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the dashboard

```
http://localhost:8000/dashboard
```

---

## API Reference

### POST /run — Run the inference pipeline

**Request**
```json
{
  "query": "Explain how transformer models work",
  "session_id": "user-session-001"
}
```

**Response**
```json
{
  "request_id": "a1b2c3d4-...",
  "session_id": "user-session-001",
  "query": "Explain how transformer models work",
  "steps": [
    {"id": 1, "description": "Understand the architecture of transformers", "type": "analysis"},
    {"id": 2, "description": "Retrieve context on self-attention mechanisms", "type": "retrieval"},
    {"id": 3, "description": "Synthesise a comprehensive explanation", "type": "synthesis"}
  ],
  "final_answer": "Transformer models use self-attention to weigh the importance of each token...",
  "retries": 1,
  "latency": 2.847,
  "critic_confidence": 0.88,
  "critic_feedback": "Clear and well-structured. Covers key concepts.",
  "agent_latencies_ms": {
    "planner": 312.4,
    "retriever": 48.1,
    "executor_attempt_0": 1402.7,
    "critic_attempt_0": 289.8,
    "executor_attempt_1": 689.4,
    "critic_attempt_1": 104.6
  },
  "from_cache": false,
  "error": null
}
```

### GET /health
```json
{ "status": "ok", "version": "1.0.0", "memory_backend": "redis", "cache_enabled": true }
```

### GET /dashboard
Opens the live **Latency & Retry Dashboard** in the browser.

### GET /metrics/summary
Returns aggregate KPIs: avg/P95 latency, retry rate, critic confidence, error rate, per-agent averages.

### GET /metrics/history?limit=50
Returns the rolling history of the last N requests for timeline charts.

### GET /session/{session_id}
Retrieve all memory stored for a session (steps, outputs, status).

---

## Dashboard Features

| Widget | Description |
|---|---|
| **KPI Cards** | Total requests, avg/P95 latency, avg retries, critic confidence, error rate |
| **Latency Timeline** | Line chart of end-to-end seconds per request (last 50) |
| **Retry Distribution** | Bar chart: how many requests needed 0/1/2/3 retries |
| **Confidence Timeline** | Critic score per request with approval threshold line |
| **Agent Latency Bars** | Horizontal bars showing avg ms per agent |
| **Request History** | Table with query, latency, retries, confidence, cache status, request ID |

Auto-refreshes every **5 seconds**. Refresh button available for manual sync.

---

## Advanced Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `sk-placeholder` | OpenAI key (mock mode if placeholder) |
| `MODEL_NAME` | `gpt-4o-mini` | Any OpenAI chat model |
| `MAX_RETRIES` | `3` | Critic-triggered retry budget |
| `CRITIC_CONFIDENCE_THRESHOLD` | `0.70` | Min score to approve without retry |
| `REDIS_URL` | `redis://localhost:6379/0` | Falls back to in-memory if unavailable |
| `CACHE_TTL_SECONDS` | `300` | Deduplication window (seconds) |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## Swapping the LLM

To replace OpenAI with a local model (vLLM, Ollama, etc.), edit only `utils/inference.py`:

```python
# Replace the OpenAI call in LLMClient.chat_complete() with:
response = await your_local_client.generate(prompt=user_prompt)
```

Everything else — agents, orchestrator, memory, dashboard — remains unchanged.

---

## Why This System Matters

Modern AI applications need more than a single LLM call. OrionAI demonstrates:

- **Reliability** — The Critic/retry loop catches low-quality outputs before they reach the user.
- **Observability** — Every step is logged as structured JSON and visualized on the live dashboard.
- **Modularity** — Each agent is independent; swap any one without touching the others.
- **Production patterns** — Async throughout, config-driven, graceful Redis fallback, request deduplication.
- **Extensibility** — Drop in a real vector DB for the Retriever, or swap the LLM backend with one file change.
