"""OPEN_API_KEY fallback (no network)."""
from __future__ import annotations


def test_get_openai_api_key_prefers_openai_then_open_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_API_KEY", "sk-from-open-api")
    from src.openai_key import get_openai_api_key

    assert get_openai_api_key() == "sk-from-open-api"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-priority")
    assert get_openai_api_key() == "sk-priority"
