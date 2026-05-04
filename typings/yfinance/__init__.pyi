"""Partial stubs for yfinance (upstream package is untyped)."""

from typing import Any

def download(
    tickers: str | list[str],
    *,
    period: str | None = None,
    group_by: str | None = None,
    progress: bool = True,
    auto_adjust: bool | None = None,
    **kwargs: Any,
) -> Any: ...

class Ticker:
    def __init__(self, ticker: str, session: Any = None) -> None: ...
