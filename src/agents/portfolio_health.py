"""
Portfolio Health Check Agent — the first fully-implemented specialist.

Covers the MONITOR and PROTECT halves of the Valura mission:
  - Concentration risk
  - Performance vs benchmark
  - Plain-language observations for novice investors
  - Empty portfolio → BUILD guidance (never crashes)
  - Regulatory disclaimer on every response

Data flow:
  1. Receives UserProfile (portfolio already attached — agent never fetches it)
  2. Computes concentration + performance metrics locally (no LLM needed for maths)
  3. Fetches benchmark return via yfinance (non-blocking, falls back gracefully)
  4. Calls LLM once to write plain-language observations
  5. Streams the structured report back chunk by chunk
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import yfinance as yf  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from src.models import (
    ClassifierOutput,
    UserProfile,
    Portfolio,
    PortfolioHealthReport,
    ConcentrationRisk,
    Performance,
    BenchmarkComparison,
    Observation,
)
from src.agents.base_agent import BaseAgent

load_dotenv(override=True)

# Colors
BG_GREEN = "\033[42m"
WHITE = "\033[37m"
RESET = "\033[0m"

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# Benchmark ticker map by user country
BENCHMARK_MAP = {
    "US": ("^GSPC", "S&P 500"),
    "GB": ("^FTSE", "FTSE 100"),
    "DE": ("^GDAXI", "DAX"),
    "FR": ("^FCHI", "CAC 40"),
    "JP": ("^N225", "Nikkei 225"),
    "IN": ("^BSESN", "BSE Sensex"),
    "DEFAULT": ("^GSPC", "S&P 500"),
}

DISCLAIMER = (
    "This analysis is for informational purposes only and does not constitute investment advice. "
    "Past performance is not indicative of future results. "
    "Please consult a qualified financial advisor before making investment decisions."
)


def _fetch_benchmark_return(ticker: str) -> Optional[float]:
    """
    Fetch 1-year return for a benchmark index via yfinance.
    Returns None on any failure — caller handles gracefully.
    """
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return None
        close = data["Close"]
        if hasattr(close, "columns"):  # MultiIndex / multi-ticker frame
            close = close.iloc[:, 0]
        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])
        if start_price == 0:
            return None
        return round(((end_price - start_price) / start_price) * 100, 2)
    except Exception:
        return None


def _compute_concentration(portfolio: Portfolio) -> ConcentrationRisk:
    """Compute concentration metrics from portfolio holdings."""
    total = portfolio.total_market_value
    if total == 0:
        return ConcentrationRisk(top_position_pct=0.0, top_3_positions_pct=0.0, flag="low")

    sorted_holdings = sorted(portfolio.holdings, key=lambda h: h.market_value, reverse=True)
    top1_pct = round((sorted_holdings[0].market_value / total) * 100, 1) if sorted_holdings else 0.0
    top3_pct = (
        round(sum(h.market_value for h in sorted_holdings[:3]) / total * 100, 1)
        if len(sorted_holdings) >= 3
        else top1_pct
    )

    if top1_pct >= 50 or top3_pct >= 75:
        flag = "high"
    elif top1_pct >= 30 or top3_pct >= 55:
        flag = "medium"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=top1_pct,
        top_3_positions_pct=top3_pct,
        flag=flag,
    )


def _compute_performance(portfolio: Portfolio) -> Performance:
    """Compute total return and annualised return from portfolio holdings."""
    total_cost = sum(h.total_cost for h in portfolio.holdings)
    total_value = sum(h.market_value for h in portfolio.holdings)

    if total_cost == 0:
        return Performance(total_return_pct=0.0)

    total_return_pct = round(((total_value - total_cost) / total_cost) * 100, 2)

    annualised = None
    if portfolio.inception_date:
        try:
            inception = datetime.fromisoformat(portfolio.inception_date)
            years = (datetime.now(timezone.utc) - inception.replace(tzinfo=timezone.utc)).days / 365.25
            if years > 0.1:
                annualised = round(((1 + total_return_pct / 100) ** (1 / years) - 1) * 100, 2)
        except Exception:
            pass

    sorted_by_return = sorted(portfolio.holdings, key=lambda h: h.return_pct, reverse=True)
    best = sorted_by_return[0].ticker if sorted_by_return else None
    worst = sorted_by_return[-1].ticker if sorted_by_return else None

    return Performance(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualised,
        best_performer=best,
        worst_performer=worst,
    )


class PortfolioHealthAgent(BaseAgent):
    """
    Analyses a user's portfolio and streams a structured health report.
    Handles empty portfolios with BUILD guidance instead of crashing.
    """

    NAME = "Portfolio Health Agent"
    DESCRIPTION = "Portfolio health check, diversification, performance vs benchmark"
    LOG_COLOR = BG_GREEN
    LOG_TEXT = WHITE
    LOG_RESET = RESET

    def __init__(self):
        super().__init__()

    async def run(
        self,
        classifier_output: ClassifierOutput,
        user_profile: Optional[UserProfile] = None,
    ) -> AsyncGenerator[str, None]:
        self.log(f"Starting portfolio health check (intent: {classifier_output.intent})")
        self.init_client_as_needed()

        portfolio = user_profile.portfolio if user_profile else Portfolio()
        country = user_profile.country if user_profile else "US"

        if portfolio.is_empty:
            self.log("Empty portfolio — switching to BUILD guidance")
            async for chunk in self._build_guidance_stream(user_profile):
                yield chunk
            return

        self.log("Computing concentration and performance metrics")
        concentration = _compute_concentration(portfolio)
        performance = _compute_performance(portfolio)

        benchmark_ticker, benchmark_name = BENCHMARK_MAP.get(country, BENCHMARK_MAP["DEFAULT"])
        self.log(f"Fetching benchmark: {benchmark_name} ({benchmark_ticker})")
        benchmark_return = _fetch_benchmark_return(benchmark_ticker)

        benchmark_comparison = None
        if benchmark_return is not None:
            alpha = round(performance.total_return_pct - benchmark_return, 2)
            benchmark_comparison = BenchmarkComparison(
                benchmark=benchmark_name,
                portfolio_return_pct=performance.total_return_pct,
                benchmark_return_pct=benchmark_return,
                alpha_pct=alpha,
            )

        report = PortfolioHealthReport(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark_comparison,
            observations=[],
            disclaimer=DISCLAIMER,
        )

        yield json.dumps({"type": "report", "data": report.model_dump()}) + "\n"

        self.log("Generating plain-language observations")
        async for chunk in self._observations_stream(report, user_profile):
            yield chunk

        self.log("Portfolio health check complete")

    async def _observations_stream(
        self,
        report: PortfolioHealthReport,
        user_profile: Optional[UserProfile],
    ) -> AsyncGenerator[str, None]:
        """Call LLM to generate novice-friendly observations. Stream the response."""
        risk_profile = user_profile.risk_profile if user_profile else "moderate"
        holdings_summary = []
        if user_profile and not user_profile.portfolio.is_empty:
            total = user_profile.portfolio.total_market_value
            top = sorted(user_profile.portfolio.holdings, key=lambda x: x.market_value, reverse=True)[:5]
            for h in top:
                pct = round(h.market_value / total * 100, 1) if total > 0 else 0
                holdings_summary.append(f"{h.ticker}: {pct}% (return: {round(h.return_pct, 1)}%)")

        prompt = f"""You are the Valura AI co-investor. Analyse this portfolio health report and write
2-4 plain-language observations for a {risk_profile} investor. Focus on what matters most.
Use simple language — no jargon without explanation. Surface the 1-2 most important things.

Portfolio summary:
- Concentration: top position = {report.concentration_risk.top_position_pct}%, top 3 = {report.concentration_risk.top_3_positions_pct}%, flag = {report.concentration_risk.flag}
- Total return: {report.performance.total_return_pct}%
- Annualised return: {report.performance.annualized_return_pct}
- Best performer: {report.performance.best_performer}
- Worst performer: {report.performance.worst_performer}
- Top holdings: {', '.join(holdings_summary) if holdings_summary else 'N/A'}
- Benchmark: {report.benchmark_comparison.benchmark if report.benchmark_comparison else 'N/A'}, alpha: {report.benchmark_comparison.alpha_pct if report.benchmark_comparison else 'N/A'}%

Write observations as flowing prose. Each observation should start on a new line with either
[WARNING] or [INFO] prefix. Do not use bullet points or markdown headers.
End with exactly this disclaimer on its own line:
{DISCLAIMER}"""

        try:
            stream = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
                stream=True,
            )
            yield '{"type": "observations_start"}\n'
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield json.dumps({"type": "text", "data": delta}) + "\n"
            yield '{"type": "observations_end"}\n'
        except Exception as e:
            self.log(f"LLM observations call failed: {e}")
            yield json.dumps({"type": "text", "data": f"\n{DISCLAIMER}"}) + "\n"

    async def _build_guidance_stream(
        self,
        user_profile: Optional[UserProfile],
    ) -> AsyncGenerator[str, None]:
        """Generate BUILD guidance for users with no portfolio."""
        risk_profile = user_profile.risk_profile if user_profile else "moderate"
        goals = user_profile.investment_goals if user_profile else []
        currency = user_profile.preferred_currency if user_profile else "USD"

        prompt = f"""You are the Valura AI co-investor helping a new investor get started.
This user has no portfolio yet. Their risk profile is: {risk_profile}.
Investment goals: {', '.join(goals) if goals else 'not specified'}.
Preferred currency: {currency}.

Write 3-4 paragraphs of warm, practical guidance to help them take their first steps:
1. Acknowledge where they are (starting from zero is completely normal)
2. Suggest 2-3 concrete starting points appropriate for their risk profile
3. Mention one thing to avoid as a beginner
4. Encourage them to start small and stay consistent

Plain language only. No jargon. No bullet points. Be encouraging but realistic.
End with exactly this disclaimer on its own line:
{DISCLAIMER}"""

        try:
            stream = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
                stream=True,
            )
            empty_report = PortfolioHealthReport(
                concentration_risk=ConcentrationRisk(
                    top_position_pct=0.0, top_3_positions_pct=0.0, flag="low"
                ),
                performance=Performance(total_return_pct=0.0),
                observations=[
                    Observation(severity="info", text="No portfolio found. Here is how to get started.")
                ],
                disclaimer=DISCLAIMER,
            )
            yield json.dumps({"type": "report", "data": empty_report.model_dump()}) + "\n"
            yield '{"type": "build_guidance_start"}\n'
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield json.dumps({"type": "text", "data": delta}) + "\n"
            yield '{"type": "build_guidance_end"}\n'
        except Exception as e:
            self.log(f"BUILD guidance LLM call failed: {e}")
            yield json.dumps(
                {
                    "type": "text",
                    "data": (
                        "Welcome to Valura. You don't have a portfolio yet — that's a great place to start. "
                        "Consider beginning with a diversified index fund aligned to your risk profile. "
                        f"\n{DISCLAIMER}"
                    ),
                }
            ) + "\n"
