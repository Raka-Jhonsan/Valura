"""
Public safety surface for the pipeline and tests.

`check(query)` wraps `SafetyGuard` in a stable `SafetyVerdict` object so gold-set
tests (and callers) never depend on tuple unpacking.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.safety_guard import SafetyGuard


class SafetyVerdict(BaseModel):
    """Result of synchronous safety screening before any LLM call."""

    blocked: bool
    category: Optional[str] = None
    message: str = Field(default="", description="Block explanation; empty when allowed.")


_guard = SafetyGuard()


def check(query: str) -> SafetyVerdict:
    """
    Run the safety guard. Returns `blocked=False` when the pipeline may proceed.
    """
    is_safe, category, msg = _guard.check(query)
    if is_safe:
        return SafetyVerdict(blocked=False, message="")
    return SafetyVerdict(blocked=True, category=category, message=msg or "")
