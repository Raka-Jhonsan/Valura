# Valura AI — Team Lead Assignment (submission)

This file is the **single source of truth** for this submission: setup, environment variables, **library choices and justification**, architecture, **decisions and tradeoffs**, **how cost/latency were measured**, test instructions, and the **defence video link**. Per the brief, there is no separate design document.

The file [`ASSIGNMENT.md`](ASSIGNMENT.md) is the **provided project specification** (kept for convenience); all implementation reasoning intended for reviewers lives **here**.

---

## Defence video (required)

Upload an **unlisted** walkthrough (maximum **10 minutes**) within **24 hours** of your final push. In the video, cover: request flow/architecture, one non-obvious decision and why, and one thing you would change with another week.

**Paste your video URL on the line below (replace the placeholder entirely):**

**Defence video:** `https://www.youtube.com/watch?v=REPLACE_ME`

---

## Submission packaging (README rules)

The deliverable layout is: **`README.md`** at the **root of the repository you submit**, plus **`src/`**, **`tests/`**, fixtures, `requirements.txt`, `.env.example`, etc.

- **If you submit this work as its own repository:** use the contents of this directory as the **repository root** (so this file is `/README.md`, not nested under another project).
- **If this folder currently lives inside a larger course monorepo:** for hand-in, provide a repo or archive where **`valura_ai/`’s contents are promoted to root**, or instruct reviewers to treat **`valura_ai/`** as the project root. The parent course `README.md` is **not** this submission’s document.

---

## What was built (high level)

Synchronous **safety guard** → **SQLite session memory** → **single-call intent classifier** (OpenAI JSON) → **router** → **Portfolio Health** agent (full) or **stub** agents → **SSE-only** `POST /query` on FastAPI.

---

## Architecture

1. **`POST /query`** returns **only** `text/event-stream` (SSE). Each `data:` line is JSON with a `type` field (`metadata`, `report`, `text`, `stub`, `blocked`, `done`, `error`).
2. **Safety** (`src/safety_guard.py`, public `src/safety.check`) runs first — no LLM, no network, target &lt; 10 ms.
3. **Classifier** (`src/classifier.py`) — one OpenAI JSON-mode call per turn; conversation history from **`SessionMemory`**.
4. **Router** (`src/router.py`) maps `target_agent` strings from [`fixtures/test_queries/intent_classification.json`](fixtures/test_queries/intent_classification.json) to `PortfolioHealthAgent` or `StubAgent`.
5. **Portfolio health** (`src/agents/portfolio_health.py`) computes concentration and performance locally, pulls benchmark returns via **yfinance** (no hardcoded prices), then streams LLM prose observations with disclaimer.

---

## Setup

**Python:** 3.11+ (CI uses 3.11).

```bash
cd valura_ai
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
cp .env.example .env            # fill OPENAI_API_KEY for live runs
```

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes for live classifier/agents (unless `OPEN_API_KEY` is set) | OpenAI API access |
| `OPEN_API_KEY` | No | Same as `OPENAI_API_KEY` if you already use this name in a parent folder `.env` |
| `CLASSIFIER_MODEL` | No (default `gpt-4o-mini`) | Classifier model id |
| `AGENT_MODEL` | No (default `gpt-4o-mini`) | Specialist agent model id |
| `PIPELINE_TIMEOUT_SECONDS` | No (default `8`) | Wall-clock guard around the streaming pipeline |
| `MEMORY_DB_PATH` | No | SQLite file for session turns |
| `MAX_CONTEXT_TURNS` | No | Recent turns sent to the classifier |
| `APP_ENV` | No | `development` / `production` / `test` |

Copy [`.env.example`](.env.example) and fill values. **Never commit `.env`.** If your key lives only in the **parent** folder (e.g. `llm_engineering/.env`) as `OPEN_API_KEY`, it is picked up automatically when the standard env vars are unset.

---

## Library choices (with justification)

| Package | Role | Why this choice |
|---------|------|-----------------|
| **FastAPI** | HTTP API | Async-friendly ASGI app, automatic validation against Pydantic models, first-class `StreamingResponse` for SSE. |
| **Uvicorn** (`[standard]`) | ASGI server | Standard production/dev server for FastAPI; `standard` extras include sensible defaults for local perf. |
| **Pydantic v2** | Schemas | Typed contracts for `QueryRequest`, `ClassifierOutput`, portfolio reports, and SSE payloads — fewer shape bugs at integration points. |
| **OpenAI Python SDK** | LLM calls | Official client for chat completions, JSON response format for the classifier, and streaming for agent text. |
| **python-dotenv** | Config | Load `.env` locally without putting secrets in code; matches `.env.example` workflow from the brief. |
| **yfinance** | Market/benchmark data | Assignment allows yfinance for non-hardcoded prices; used only for **benchmark** return series, not for user holdings (those come from the request profile). |
| **httpx** | HTTP stack | Pinned for reproducible installs; used by Starlette’s test client stack when running API tests. |
| **pytest**, **pytest-asyncio**, **pytest-mock** | Tests | Required test runner; asyncio for agent coroutines; mocks for CI without `OPENAI_API_KEY`. |

**SSE:** The wire protocol is implemented **directly** in `src/main.py` (`data: {json}\n\n` via `StreamingResponse`) so every event shape stays explicit and JSON-metadata-friendly. The brief allows implementing SSE yourself; no extra SSE wrapper dependency is required.

---

## Session memory (persistence choice)

**SQLite** via `src/memory.py`:

- **Pros:** Survives uvicorn reloads and restarts; proves multi-turn follow-ups without running Postgres; zero external infra for reviewers.
- **Cons:** Not multi-region / not a production tenancy story — acceptable for the assignment demo; Postgres would be the next step for real multi-tenant scale.

---

## Safety guard tradeoffs

The synchronous guard is **regex + ordering** only (&lt; 10 ms, no LLM). **Educational framing** is checked first so questions like “what is insider trading?” pass; overt harmful requests map to **distinct refusal copy** per category. **Tradeoff:** a narrow band of malicious prompts that mimic “explain …” education could slip to the classifier; the classifier’s `safety_verdict` is **informational only** and does not re-block (per brief). That tradeoff is accepted to keep the guard fast and local.

---

## Entity matching (tests)

Gold `expected_entities` are validated as **subset + normalization** in [`tests/test_classifier_routing.py`](tests/test_classifier_routing.py) (`matches_entities`): tickers case-folded and exchange suffix stripped; topics/sectors lowercased; `amount` / `rate` within ±5%; `period_years` exact int match to fixture. Conversation rows reuse the same matcher in [`tests/test_classifier_conversations.py`](tests/test_classifier_conversations.py).

---

## Running the API

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health:** `GET /health`
- **Query:** `POST /query` with JSON body matching `QueryRequest` (`query`, `user_id`, optional `session_id`, optional `user_profile`).

---

## Tests

```bash
pytest tests/ -v
```

**CI must pass without `OPENAI_API_KEY`.** Tests patch `OpenAI` and replay gold classifier rows from `fixtures/` so routing, entity subset rules, safety pairs, and SSE smoke paths run without network calls to OpenAI.

---

## Key implementation decisions

- **Stub agent** — one implementation for all non–portfolio-health agents; router never crashes on unknown names.
- **Classifier taxonomy** — strings aligned with `intent_classification.json`.
- **Extracted entities** — `ExtractedEntities` uses `model_config.extra = "allow"` so gold vocabulary keys (`amount`, `index`, `action`, …) round-trip from model JSON.
- **Classifier failure** — `IntentClassifier` catches all exceptions and returns a safe `ClassifierOutput` (defaults to `general_query`) so the HTTP layer never crashes on a bad LLM response.
- **Pipeline timeout** — default **8 s** (documented in `src/main.py`): stricter than the assignment’s 6 s p95 *target*, but avoids hung streams.

---

## Cost and latency (targets and measurement)

**Targets (from brief):** p95 first streamed token **&lt; 2 s**, p95 end-to-end **&lt; 6 s**, cost per query **&lt; $0.05** at **`gpt-4.1`** pricing. **Development** default: `gpt-4o-mini`. **Evaluation:** set `CLASSIFIER_MODEL` and `AGENT_MODEL` to `gpt-4.1` when measuring for submission.

### How measured

1. **p95 first token:** `T0` = when the HTTP client sends `POST /query`; `T1` = first byte of response body (or first complete SSE `data:` line). Warm the server, then run **N ≥ 30** mixed queries; report **p95** of `T1 − T0`.
2. **p95 end-to-end:** same runs — time until SSE **`done`** (or stream close); report **p95**.
3. **Cost per query:** sum **prompt + completion tokens** for the classifier completion and each agent LLM call; apply published **$/1M input and output** for the model under test. Use API usage fields or the OpenAI usage dashboard for a batch; report **mean or p95** over the same query mix.

### Results (fill before submit)

| Metric | Model | Result |
|--------|--------|--------|
| p95 first token | gpt-4o-mini | *TBD* |
| p95 end-to-end | gpt-4o-mini | *TBD* |
| Mean or p95 cost / query | gpt-4.1 | *TBD* |

State sample size **N**, region, and date in the defence video when you present numbers.

---

## CI

- **Inside this folder:** [`.github/workflows/pytest.yml`](.github/workflows/pytest.yml) when this directory is the GitHub repo root.
- **Parent monorepo (if used):** [`.github/workflows/valura-ai-pytest.yml`](../.github/workflows/valura-ai-pytest.yml) runs the same tests with `working-directory: valura_ai`.

---

## Classroom scaffold

Any original classroom-only banners or autograding metadata remain under `.github/classroom/` for reference; this README remains the reviewer-facing document.
#   V a l u r a  
 