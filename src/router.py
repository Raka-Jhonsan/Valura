"""
Router — maps target_agent string from the classifier to the correct agent instance.

Design:
  - One registry dict. Adding a new agent is one line.
  - Unknown agent names fall through to StubAgent — never crashes.
  - Non-portfolio agents share StubAgent (cached by class name).
"""
import logging
from typing import Dict
from src.agents.base_agent import BaseAgent
from src.agents.portfolio_health import PortfolioHealthAgent
from src.agents.stub_agent import StubAgent

# Colors (same pattern as deal_agent_framework.py)
BG_CYAN = "\033[46m"
WHITE = "\033[37m"
RESET = "\033[0m"

_STUB = StubAgent

# Keys must match `expected_agent` strings in fixtures/test_queries/intent_classification.json
_REGISTRY: Dict[str, type] = {
    "portfolio_health": PortfolioHealthAgent,
    "market_research": _STUB,
    "investment_strategy": _STUB,
    "financial_planning": _STUB,
    "financial_calculator": _STUB,
    "risk_assessment": _STUB,
    "product_recommendation": _STUB,
    "predictive_analysis": _STUB,
    "customer_support": _STUB,
    "general_query": _STUB,
    # Legacy / alias names (safe if classifier ever emits them)
    "support": _STUB,
}


class AgentRouter:
    """
    Resolves a target_agent name to a live agent instance.
    Agents are lazy-instantiated and cached — one instance per agent type.
    """

    def __init__(self):
        self._instances: Dict[str, BaseAgent] = {}
        self.log("Agent Router initialised")

    def log(self, message: str):
        text = BG_CYAN + WHITE + "[Router] " + message + RESET
        logging.info(text)

    def get_agent(self, target_agent: str) -> BaseAgent:
        """
        Return the agent instance for the given target_agent name.
        Falls back to StubAgent if the name is not in the registry.
        """
        agent_class = _REGISTRY.get(target_agent, StubAgent)
        cache_key = agent_class.__name__

        if cache_key not in self._instances:
            self.log(f"Instantiating {agent_class.__name__} for '{target_agent}'")
            self._instances[cache_key] = agent_class()

        return self._instances[cache_key]

    def route(self, target_agent: str) -> BaseAgent:
        """Alias for get_agent — matches the verb used in main.py."""
        agent = self.get_agent(target_agent)
        self.log(f"Routing '{target_agent}' -> {agent.NAME}")
        return agent
