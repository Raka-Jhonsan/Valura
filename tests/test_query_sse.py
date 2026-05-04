"""POST /query SSE contract: blocked path and stub path (no live OpenAI)."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models import ClassifierOutput, ExtractedEntities


def _parse_sse_events(raw: bytes) -> list[dict]:
    """Parse `data: {json}\\n\\n` frames into dicts (best-effort)."""
    events: list[dict] = []
    text = raw.decode("utf-8", errors="replace")
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append({"_raw": payload})
    return events


@pytest.fixture
def sse_client():
    with TestClient(app) as client:
        yield client


def test_query_sse_blocked_emits_blocked_then_done(sse_client: TestClient):
    with patch("src.main.safety_guard") as mock_guard:
        mock_guard.check.return_value = (
            False,
            "insider_trading",
            "I'm not able to assist with trading on material non-public information.",
        )
        with sse_client.stream(
            "POST",
            "/query",
            json={"query": "trade on a tip before earnings", "user_id": "u1"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            buf = b"".join(resp.iter_bytes())

    events = _parse_sse_events(buf)
    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert "blocked" in types
    assert "done" in types
    blocked = next(e for e in events if e.get("type") == "blocked")
    assert blocked["data"]["category"] == "insider_trading"
    assert "material" in blocked["data"]["message"].lower() or "insider" in blocked["data"]["message"].lower()


def test_query_sse_stub_emits_metadata_stub_done(sse_client: TestClient):
    co = ClassifierOutput(
        intent="stock research",
        target_agent="market_research",
        entities=ExtractedEntities(tickers=["AAPL"]),
        resolved_query="What is happening with AAPL?",
    )
    with patch("src.main.safety_guard") as mock_guard:
        mock_guard.check.return_value = (True, None, "")
        with patch("src.main.classifier") as mock_classifier:
            mock_classifier.classify.return_value = co
            with sse_client.stream(
                "POST",
                "/query",
                json={"query": "What is happening with AAPL?", "user_id": "u1"},
            ) as resp:
                assert resp.status_code == 200
                buf = b"".join(resp.iter_bytes())

    events = _parse_sse_events(buf)
    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert "metadata" in types
    assert "stub" in types
    assert "done" in types
    meta = next(e for e in events if e.get("type") == "metadata")
    assert meta["data"]["target_agent"] == "market_research"
    assert meta["data"]["intent"] == "stock research"
