"""
Intent Classifier — one LLM call that drives the entire pipeline.

Responsibilities:
  1. Resolve follow-up queries ("what about Apple?") into standalone sentences
     using the last N turns of conversation context
  2. Classify intent and extract entities in one shot
  3. Return a structured ClassifierOutput (Pydantic)
  4. Never crash — LLM failure falls back to a safe default

One call. One model. One structured output.
"""
import os
import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from src.models import ClassifierOutput, ExtractedEntities
from src.openai_key import get_openai_api_key

load_dotenv(override=True)

# Colors (same pattern as deal_agent_framework.py)
BG_MAGENTA = "\033[45m"
WHITE = "\033[37m"
RESET = "\033[0m"

MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")

# Full agent taxonomy — must match `expected_agent` in fixtures/test_queries/intent_classification.json
AGENT_TAXONOMY = [
    "portfolio_health",
    "market_research",
    "investment_strategy",
    "financial_planning",
    "financial_calculator",
    "risk_assessment",
    "product_recommendation",
    "predictive_analysis",
    "customer_support",
    "general_query",
]

SYSTEM_PROMPT = f"""You are the intent classifier for Valura, a wealth management AI platform.

Your job is to analyse a user's query — given conversation history for context — and return a
single JSON object. No preamble. No markdown fences. JSON only.

AGENT TAXONOMY (use exactly these strings for target_agent):
{json.dumps(AGENT_TAXONOMY, indent=2)}

Agent responsibilities:
- portfolio_health: portfolio overview, health check, diversification, how am I doing, performance, is my portfolio OK
- market_research: stock research, company analysis, sector or index moves, market news, price of X, ticker-only queries
- investment_strategy: should I buy/sell/rebalance, allocation guidance, hedging, timing
- financial_planning: retirement savings goals, college/house/FIRE planning, long-horizon savings questions
- financial_calculator: deterministic math — DCA/FV, mortgage payments, tax estimates, FX conversion
- risk_assessment: VaR, beta, drawdown, stress tests, exposure to currencies or scenarios
- product_recommendation: recommend specific funds/ETFs/products for a profile
- predictive_analysis: forecasts, where will X be in N months/years
- customer_support: login, bank linking, transaction history, app how-to, billing
- general_query: greetings, thanks, definitions, educational comparisons, gibberish that should not crash

FOLLOW-UP RESOLUTION:
If the query references something from a prior turn ("what about that", "and Apple?", "same for bonds"),
rewrite it as a complete standalone sentence in resolved_query. If no resolution is needed, copy the
original query into resolved_query.

ENTITY EXTRACTION rules (use arrays below; you may also add scalar keys from the taxonomy):
- tickers: uppercase, e.g. ["AAPL", "MSFT"]. Include exchange suffix if user mentions it.
- amounts: numeric values only, e.g. [10000.0, 500.0]
- time_periods: e.g. ["1 year", "5 years", "since inception", "YTD"]
- sectors: e.g. ["technology", "healthcare"]
- topics: e.g. ["dividends", "rebalancing", "inflation"]
- currencies: e.g. ["USD", "GBP", "EUR"]
- Optional scalars when clearly stated: amount, rate, period_years, frequency, horizon, time_period,
  index, action, goal, currency (ISO 4217 string), or other short keys from the assignment vocabulary.

SAFETY VERDICT:
- "safe": normal investment question
- "review": borderline — may involve speculative or aggressive strategies
- "unsafe": clear harmful intent (manipulation, fraud, insider trading)
Note: this is INFORMATIONAL only. The safety guard upstream has already blocked truly unsafe queries.

Return ONLY this JSON shape — no other text:
{{
  "intent": "string describing the user's intent in 3-6 words",
  "target_agent": "one of the agent taxonomy strings above",
  "entities": {{
    "tickers": [],
    "amounts": [],
    "time_periods": [],
    "sectors": [],
    "topics": [],
    "currencies": []
  }},
  "safety_verdict": "safe | review | unsafe",
  "safety_reason": "null or brief reason if review/unsafe",
  "confidence": 0.0-1.0,
  "resolved_query": "the full standalone query"
}}"""

# Fallback when the LLM call fails entirely
_FALLBACK = ClassifierOutput(
    intent="unknown",
    target_agent="general_query",
    entities=ExtractedEntities(),
    safety_verdict="safe",
    confidence=0.0,
    resolved_query="",
)


class IntentClassifier:
    """
    Single-call LLM classifier with follow-up resolution and graceful fallback.
    Lazy-initialises the OpenAI client (same pattern as PlanningAgent).
    """

    def __init__(self):
        self.client = None

    def init_client_as_needed(self):
        if not self.client:
            self.log("Initialising OpenAI client")
            self.client = OpenAI(api_key=get_openai_api_key())
            self.log("OpenAI client ready")

    def log(self, message: str):
        text = BG_MAGENTA + WHITE + "[Classifier] " + message + RESET
        logging.info(text)

    def classify(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ClassifierOutput:
        """
        Classify a query, resolving follow-ups against history.

        Args:
            query:   The raw user query from the current turn.
            history: Prior turns [{"role": ..., "content": ...}] oldest-first.

        Returns:
            ClassifierOutput — never raises.
        """
        self.init_client_as_needed()
        self.log(f"Classifying: {query[:80]}{'...' if len(query) > 80 else ''}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject conversation history so the model can resolve follow-ups
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            self.log(f"Raw classifier response: {raw[:120]}{'...' if len(raw) > 120 else ''}")

            data = json.loads(raw)
            output = ClassifierOutput(
                intent=data.get("intent", "unknown"),
                target_agent=data.get("target_agent", "support"),
                entities=ExtractedEntities(**data.get("entities", {})),
                safety_verdict=data.get("safety_verdict", "safe"),
                safety_reason=data.get("safety_reason"),
                confidence=float(data.get("confidence", 1.0)),
                resolved_query=data.get("resolved_query", query),
            )
            self.log(f"Classified as: {output.target_agent} (intent: {output.intent})")
            return output

        except Exception as e:
            self.log(f"Classifier failed — using fallback. Error: {e}")
            fallback = _FALLBACK.model_copy()
            fallback.resolved_query = query
            return fallback


if __name__ == "__main__":
    classifier = IntentClassifier()
    result = classifier.classify("How is my portfolio doing?")
    print(json.dumps(result.model_dump(), indent=2))
