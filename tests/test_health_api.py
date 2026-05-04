"""Lightweight HTTP smoke tests (no OpenAI)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"
    assert "valura" in body.get("service", "").lower()
