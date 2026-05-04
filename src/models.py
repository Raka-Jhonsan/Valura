"""
Shared Pydantic models — the data contracts between every component.
All inter-agent schemas live here. Nothing else imports from each other's internals.
"""
from __future__ import annotations
import sys
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

load_dotenv(override=True)


# ─── Logging ──────────────────────────────────────────────────────────────────

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Valura] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


# ─── User / Portfolio ─────────────────────────────────────────────────────────

class Holding(BaseModel):
    ticker: str
    name: str
    quantity: float
    purchase_price: float
    current_price: float
    currency: str = "USD"
    sector: Optional[str] = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def total_cost(self) -> float:
        return self.quantity * self.purchase_price

    @property
    def unrealised_pnl(self) -> float:
        return self.market_value - self.total_cost

    @property
    def return_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return (self.unrealised_pnl / self.total_cost) * 100


class Portfolio(BaseModel):
    holdings: List[Holding] = Field(default_factory=list)
    cash_balance: float = 0.0
    currency: str = "USD"
    inception_date: Optional[str] = None  # ISO date string e.g. "2023-01-15"

    @property
    def total_market_value(self) -> float:
        return sum(h.market_value for h in self.holdings) + self.cash_balance

    @property
    def is_empty(self) -> bool:
        return len(self.holdings) == 0


class UserProfile(BaseModel):
    user_id: str
    name: str
    kyc_status: Literal["verified", "pending", "rejected"] = "pending"
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    investment_goals: List[str] = Field(default_factory=list)
    portfolio: Portfolio = Field(default_factory=Portfolio)
    preferred_currency: str = "USD"
    country: str = "US"


# ─── Classifier ───────────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    """
    Core list fields plus optional scalars from fixtures/README vocabulary.
    Extra keys are allowed so gold JSON (e.g. index, action, goal) round-trip from the LLM.
    """

    model_config = ConfigDict(extra="allow")

    tickers: List[str] = Field(default_factory=list)
    amounts: List[float] = Field(default_factory=list)
    time_periods: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    currencies: List[str] = Field(default_factory=list)


class ClassifierOutput(BaseModel):
    intent: str
    target_agent: str
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: Literal["safe", "review", "unsafe"] = "safe"
    safety_reason: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    resolved_query: str = ""  # follow-up resolution — full standalone query


# ─── Portfolio Health Agent output ────────────────────────────────────────────

class Observation(BaseModel):
    severity: Literal["info", "warning", "critical"]
    text: str


class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low", "medium", "high"]


class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: Optional[float] = None
    best_performer: Optional[str] = None
    worst_performer: Optional[str] = None


class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class PortfolioHealthReport(BaseModel):
    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: List[Observation] = Field(default_factory=list)
    disclaimer: str = (
        "This analysis is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results. "
        "Please consult a qualified financial advisor before making investment decisions."
    )


# ─── Stub Agent output ────────────────────────────────────────────────────────

class StubAgentResponse(BaseModel):
    classified_intent: str
    extracted_entities: ExtractedEntities
    target_agent: str
    message: str = "This agent is not implemented in the current build."


# ─── HTTP layer ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None


class PipelineMetadata(BaseModel):
    session_id: str
    intent: str
    target_agent: str
    entities: ExtractedEntities
    safety_verdict: str
    safety_reason: Optional[str] = None
