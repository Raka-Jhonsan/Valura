"""Portfolio Health agent — structured report, concentration, disclaimer (mocked LLM)."""
from __future__ import annotations

import json

import pytest

from src.agents.portfolio_health import PortfolioHealthAgent
from src.models import ClassifierOutput, ExtractedEntities


def _portfolio_classifier_output() -> ClassifierOutput:
    return ClassifierOutput(
        intent="portfolio health check",
        target_agent="portfolio_health",
        entities=ExtractedEntities(),
        resolved_query="How is my portfolio doing?",
    )


async def _first_report_payload(agent: PortfolioHealthAgent, user) -> dict:
    """Return the first `type=report` JSON payload from the agent stream."""
    co = _portfolio_classifier_output()
    async for raw in agent.run(co, user_profile=user):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "report":
                return obj["data"]
    raise AssertionError("no report event yielded")


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(mock_openai, user_empty):
    """Empty portfolio must yield a report with disclaimer (BUILD path)."""
    _mock_client, _ = mock_openai
    _ = _mock_client  # patches applied by fixture
    agent = PortfolioHealthAgent()
    report = await _first_report_payload(agent, user_empty)
    assert report is not None
    assert "disclaimer" in report
    assert "investment advice" in report["disclaimer"].lower()


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(mock_openai, user_concentrated):
    """Highly concentrated portfolio must surface elevated concentration flag."""
    _mock_client, _ = mock_openai
    _ = _mock_client
    agent = PortfolioHealthAgent()
    report = await _first_report_payload(agent, user_concentrated)
    flag = report["concentration_risk"]["flag"]
    assert flag in {"high", "medium"}


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(mock_openai, user_active_trader):
    _mock_client, _ = mock_openai
    _ = _mock_client
    agent = PortfolioHealthAgent()
    report = await _first_report_payload(agent, user_active_trader)
    assert report["disclaimer"]
    d = report["disclaimer"].lower()
    assert "investment advice" in d and "informational" in d
