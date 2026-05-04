# Valura AI — Team Lead Assignment

## Defence Video
`https://youtu.be/KCKKD0Xi8gg?si=5hzEt3xl8tJSa2hz`

#Commit SHA: cf6b9146c11fb16569705fbace96f985a3ad4d4f

---

## Setup

```bash
git clone https://github.com/Raka-Jhonsan/Valura
cd Valura
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in OPENAI_API_KEY
uvicorn src.main:app --reload
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CLASSIFIER_MODEL` | No | `gpt-4o-mini` | Classifier model (use `gpt-4.1` for eval) |
| `AGENT_MODEL` | No | `gpt-4o-mini` | Agent prose generation model |
| `MEMORY_DB_PATH` | No | `valura_memory.db` | SQLite session memory path |
| `MAX_CONTEXT_TURNS` | No | `6` | Prior turns injected into classifier |
| `PIPELINE_TIMEOUT_SECONDS` | No | `8` | SSE pipeline timeout |

---

## Running Tests

```bash
pytest tests/ -v
```

No `OPENAI_API_KEY` needed — all LLM calls are mocked. CI runs clean.

---

## Architecture

```
POST /query
     │
     ▼
Safety Guard      ← pure Python regex, no LLM, < 10ms
     │ blocked → SSE error event, stop
     ▼
Session Memory    ← load last 6 turns from SQLite
     ▼
Classifier        ← one LLM call → intent + entities + target_agent
     ▼
Router            ← dispatches to correct agent
     │
     ├── portfolio_health  → fully implemented
     └── all others        → structured stub (never crashes)
     ▼
SSE Stream        ← metadata → report → text chunks → done
```

**Key files:**
```
src/
├── safety_guard.py       ← regex patterns, no LLM
├── classifier.py         ← single LLM call, follow-up resolution
├── memory.py             ← SQLite session store
├── router.py             ← agent registry
├── main.py               ← FastAPI, /query endpoint
├── models.py             ← all Pydantic schemas
└── agents/
    ├── base_agent.py     ← abstract base, defines run() contract
    ├── portfolio_health.py  ← MONITOR + PROTECT, fully built
    └── stub_agent.py     ← structured not-implemented for all others
```

---

## Design Decisions

**Safety guard is the only blocking authority.**
The classifier also returns a `safety_verdict` but it is informational only — it never re-blocks. The regex guard is deterministic, under 10ms, and never depends on OpenAI availability.

**Educational queries pass through the guard.**
Queries with framing markers (`what is`, `explain`, `how does`) pass even if harmful keywords are present. `how do I` is excluded — it requests instructions, not explanation.

**LLM handles prose only, not maths.**
In the portfolio health agent, concentration, performance, and benchmark metrics are computed locally in pure Python. The LLM is called once at the end to write plain-language observations. LLMs are unreliable at arithmetic — compute locally, narrate with LLM.

**SQLite over in-memory.**
`uvicorn --reload` wipes in-memory state on every restart, breaking multi-turn follow-up testing. SQLite persists to disk, costs zero infrastructure.

**Empty portfolio → BUILD guidance.**
`user_004` has no holdings. Rather than returning zeros, the portfolio health agent switches mission — from MONITOR to BUILD — and streams practical first-step guidance.

---

## Test Thresholds

| Test | Threshold |
|---|---|
| Safety guard recall (harmful queries) | ≥ 95% |
| Safety guard pass-through (educational) | ≥ 90% |
| Classifier routing accuracy | ≥ 85% |
| Empty portfolio (`user_004`) no crash | must pass |

---

## Cost & Latency

Measured locally with `gpt-4o-mini`, 20 sequential requests:

| Metric | Target | Measured |
|---|---|---|
| p95 first-token latency | < 2s | ~0.8s |
| p95 end-to-end | < 6s | ~3.2s |
| Cost per query (gpt-4.1 pricing) | < $0.05 | ~$0.008 |

---

## What I'd Build Next

1. **Market research agent** — stub already routing correctly, one new file needed
2. **Embedding pre-classifier** — skip the LLM call for obvious-intent queries, cut cost and latency for the common case
