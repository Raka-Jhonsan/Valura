"""
Valura AI Microservice — FastAPI entry point.

One endpoint: POST /query
Pipeline: Safety Guard -> Session Memory -> Classifier -> Router -> Agent -> SSE stream

SSE event types emitted:
  {"type": "metadata",   "data": {...}}   — classifier output, first event
  {"type": "report",     "data": {...}}   — structured portfolio health report
  {"type": "text",       "data": "..."}   — streamed prose chunks
  {"type": "stub",       "data": {...}}   — stub agent response
  {"type": "done",       "data": ""}      — stream complete
  {"type": "error",      "data": "..."}   — structured error, never a raw traceback

Timeout: 8 seconds
  p95 target is 6s. 8s gives headroom without hanging forever.
  First token must arrive within 2s — enforced by the classifier being one fast call.
"""
import os
import uuid
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from src.models import QueryRequest, PipelineMetadata, init_logging
from src.safety_guard import SafetyGuard
from src.classifier import IntentClassifier
from src.memory import SessionMemory, init_db
from src.router import AgentRouter

load_dotenv(override=True)
init_logging()

# Colors (same pattern as deal_agent_framework.py)
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "8"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    log("Valura AI Microservice started")
    yield


app = FastAPI(
    title="Valura AI Microservice",
    description="AI co-investor — build, monitor, grow, and protect your portfolio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singletons (lazy-init inside lifespan) ───────────────────────────────────
safety_guard = SafetyGuard()
classifier = IntentClassifier()
router = AgentRouter()


def log(message: str):
    text = BG_BLUE + WHITE + "[Pipeline] " + message + RESET
    logging.info(text)


def _sse_event(event_type: str, data) -> str:
    """Format a single SSE event as a string."""
    payload = json.dumps({"type": event_type, "data": data})
    return f"data: {payload}\n\n"


def _sse_error(message: str) -> str:
    return _sse_event("error", message)


def _sse_done() -> str:
    return _sse_event("done", "")


def _sse_from_agent_ndjson_chunks(chunk: str) -> list[str]:
    """
    Agents emit newline-delimited JSON objects. SSE requires each logical event as
    `data: …\\n\\n`. Split on lines so streamed tokens still produce valid frames.
    """
    events: list[str] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(f"data: {line}\n\n")
    return events


@app.get("/health")
async def health():
    return {"status": "ok", "service": "valura-ai"}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Main pipeline endpoint. Returns an SSE stream.
    All responses are streamed — there is no JSON fallback path.
    """
    session_id = request.session_id or str(uuid.uuid4())
    log(f"Incoming query | session={session_id} | user={request.user_id}")

    async def _query_pipeline() -> AsyncGenerator[str, None]:
        # ── 1. Safety Guard ───────────────────────────────────────────────
        is_safe, category, block_message = safety_guard.check(request.query)
        if not is_safe:
            log(f"Query blocked by safety guard — category: {category}")
            yield _sse_event(
                "blocked",
                {"category": category, "message": block_message},
            )
            yield _sse_done()
            return

        # ── 2. Load session memory ─────────────────────────────────────────
        memory = SessionMemory(session_id=session_id, user_id=request.user_id)
        history = memory.get_recent_turns()

        # ── 3. Classify intent ─────────────────────────────────────────────
        classifier_output = classifier.classify(query=request.query, history=history)

        # ── 4. Emit metadata event ─────────────────────────────────────────
        metadata = PipelineMetadata(
            session_id=session_id,
            intent=classifier_output.intent,
            target_agent=classifier_output.target_agent,
            entities=classifier_output.entities,
            safety_verdict=classifier_output.safety_verdict,
            safety_reason=classifier_output.safety_reason,
        )
        yield _sse_event("metadata", metadata.model_dump())

        # ── 5. Route & stream agent (NDJSON chunks → SSE `data:` frames) ───
        agent = router.route(classifier_output.target_agent)
        log(f"Dispatching to {agent.NAME}")

        async for chunk in agent.run(
            classifier_output=classifier_output,
            user_profile=request.user_profile,
        ):
            for frame in _sse_from_agent_ndjson_chunks(chunk):
                yield frame

        # ── 6. Persist turns ───────────────────────────────────────────────
        resolved = classifier_output.resolved_query or request.query
        memory.save_turn("user", resolved)
        memory.save_turn("assistant", f"[{classifier_output.intent}]")

        yield _sse_done()

    async def pipeline_stream() -> AsyncGenerator[str, None]:
        try:
            async with asyncio.timeout(PIPELINE_TIMEOUT):
                async for event in _query_pipeline():
                    yield event
        except TimeoutError:
            log(f"Pipeline timeout after {PIPELINE_TIMEOUT}s for session={session_id}")
            yield _sse_error(
                f"Request timed out after {PIPELINE_TIMEOUT} seconds. Please try again."
            )
            yield _sse_done()
        except Exception as e:
            log(f"Unhandled pipeline error: {e}")
            yield _sse_error("An unexpected error occurred. Please try again.")
            yield _sse_done()

    return StreamingResponse(
        pipeline_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Valura AI</title>
    </head>
    <body>
        <h2>Valura AI Query</h2>

        <input id="query" placeholder="Enter your query" style="width:300px;">
        <button onclick="sendQuery()">Submit</button>

        <pre id="output"></pre>

        <script>
        async function sendQuery() {
            const query = document.getElementById("query").value;

            const response = await fetch("/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    query: query,
                    user_id: "user_001"
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let output = document.getElementById("output");
            output.textContent = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                output.textContent += chunk;
            }
        }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)