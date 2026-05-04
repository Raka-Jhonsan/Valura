"""Resolve the OpenAI API key from the environment or nearby `.env` files."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def _read_key_from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = dotenv_values(path)
    for name in ("OPENAI_API_KEY", "OPEN_API_KEY"):
        raw = data.get(name)
        if raw:
            s = str(raw).strip()
            if s:
                return s
    return None


def get_openai_api_key() -> str | None:
    """
    Prefer `OPENAI_API_KEY`, then `OPEN_API_KEY` (common alternate name).

    If neither is set in `os.environ`, look for `.env` in `valura_ai/` and in the
    parent directory (e.g. monorepo root) so a single `llm_engineering/.env` works
    when running uvicorn from `valura_ai/`.
    """
    for name in ("OPENAI_API_KEY", "OPEN_API_KEY"):
        raw = os.environ.get(name)
        if raw:
            s = raw.strip()
            if s:
                return s

    valura_root = Path(__file__).resolve().parent.parent
    for directory in (valura_root, valura_root.parent):
        key = _read_key_from_file(directory / ".env")
        if key:
            return key
    return None
