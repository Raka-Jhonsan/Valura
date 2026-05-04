"""
Valura AI — assignment spine: synchronous safety, single-call classifier, router,
one full agent (portfolio health), stubs for the rest. Import the FastAPI ``app``
with ``from src.main import app`` when mounting Uvicorn.

Subpackages: ``src.agents`` (specialists + stubs).
"""
from __future__ import annotations

__version__ = "1.0.0"

from src.classifier import IntentClassifier
from src.memory import SessionMemory, init_db
from src.models import (
    ClassifierOutput,
    ExtractedEntities,
    PipelineMetadata,
    Portfolio,
    PortfolioHealthReport,
    QueryRequest,
    UserProfile,
)
from src.router import AgentRouter
from src.safety import SafetyVerdict, check
from src.safety_guard import SafetyGuard

__all__ = [
    "__version__",
    "IntentClassifier",
    "SessionMemory",
    "AgentRouter",
    "SafetyGuard",
    "SafetyVerdict",
    "check",
    "init_db",
    "ClassifierOutput",
    "ExtractedEntities",
    "UserProfile",
    "Portfolio",
    "PortfolioHealthReport",
    "QueryRequest",
    "PipelineMetadata",
]
