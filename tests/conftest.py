"""
Shared pytest fixtures for the Valura AI assignment.

Key design: patch `OpenAI` where code constructs the client so CI runs without
OPENAI_API_KEY. The classifier path returns canned JSON; agent paths use a tiny
streaming mock for observation / BUILD guidance chunks. This exercises parsing,
routing, and agent logic — not production model behaviour.

Gold routing and safety cases still load from `fixtures/`.

Use `mock_openai` when you need to reconfigure completions; tests that request
only `mock_llm` pull in the same underlying mock automatically.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.classifier import IntentClassifier
from src.models import Holding, Portfolio, UserProfile


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_DEFAULT_STREAM_TAIL = (
    "This analysis is for informational purposes only and does not constitute investment advice. "
    "Past performance is not indicative of future results. "
    "Please consult a qualified financial advisor before making investment decisions."
)


def _make_mock_completion(content: str):
    """Build a mock OpenAI Chat Completions API response."""
    choice = MagicMock()
    choice.message.content = content
    choice.delta.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_stream_chunks(text: str):
    """Yield chunk objects resembling `stream=True` completion chunks."""
    step = max(16, len(text) // 8 or 1)
    for i in range(0, len(text), step):
        fragment = text[i : i + step]
        choice = MagicMock()
        choice.delta.content = fragment
        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


def classifier_response_for(
    target_agent: str,
    intent: str,
    *,
    tickers=None,
    topics=None,
) -> str:
    """Stable JSON matching `ClassifierOutput` / `ExtractedEntities` parsing."""
    return json.dumps(
        {
            "intent": intent,
            "target_agent": target_agent,
            "entities": {
                "tickers": tickers or [],
                "amounts": [],
                "time_periods": [],
                "sectors": [],
                "topics": topics or [],
                "currencies": [],
            },
            "safety_verdict": "safe",
            "safety_reason": None,
            "confidence": 0.95,
            "resolved_query": "mocked resolved query",
        }
    )


def merge_entities_for_mock(expected: dict | None) -> dict:
    """Merge gold `expected_entities` into the classifier JSON shape."""
    out: dict = {
        "tickers": [],
        "amounts": [],
        "time_periods": [],
        "sectors": [],
        "topics": [],
        "currencies": [],
    }
    for key, val in (expected or {}).items():
        if key in out and isinstance(val, list):
            out[key] = val
        else:
            out[key] = val
    return out


def normalize_target_agent(agent: str) -> str:
    """Align fixture vocabulary with production agent names."""
    return {"portfolio_query": "portfolio_health"}.get(agent, agent)


def _agent_entities_resolved_query(case: dict) -> tuple[str, dict, str]:
    """Normalise gold / conversation rows to (target_agent, entities, resolved_query)."""
    if "expected_agent" in case:
        return (
            case["expected_agent"],
            case.get("expected_entities") or {},
            case["query"],
        )
    exp = case["expected"]
    return (
        exp["agent"],
        exp.get("entities") or {},
        case["current_user_turn"],
    )


def make_classifier_side_effect(
    case_by_user_message: dict[str, dict],
    stream_text: str | None = None,
):
    """
    `case_by_user_message` maps the **last user** message in the chat payload to a gold row.
    """
    text = stream_text or f"[INFO] Mock observation for tests.\n\n{_DEFAULT_STREAM_TAIL}"

    def _side_effect(*_args, **kwargs):
        if kwargs.get("stream"):
            return _mock_stream_chunks(text)
        messages = kwargs.get("messages") or []
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m["content"]
                break
        row = case_by_user_message.get(user_msg)
        if row is None:
            agent, entities, rq = "general_query", {}, user_msg
        else:
            agent, entities, rq = _agent_entities_resolved_query(row)
            agent = normalize_target_agent(agent)
        body = {
            "intent": "classified",
            "target_agent": agent,
            "entities": merge_entities_for_mock(entities),
            "safety_verdict": "safe",
            "safety_reason": None,
            "confidence": 0.95,
            "resolved_query": rq,
        }
        return _make_mock_completion(json.dumps(body))

    return _side_effect


# ---------------------------------------------------------------------------
# Fixture loaders (gold sets, users JSON)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. load_user('usr_001')."""

    def _load(user_id: str) -> dict:
        for path in (FIXTURES_DIR / "users").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user["user_id"] == user_id:
                return user
        raise FileNotFoundError(f"No fixture for user {user_id}")

    return _load


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    with open(
        FIXTURES_DIR / "test_queries" / "intent_classification.json",
        encoding="utf-8",
    ) as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    with open(
        FIXTURES_DIR / "test_queries" / "safety_pairs.json",
        encoding="utf-8",
    ) as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: conversation_test_cases('follow_up_session')."""

    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]

    return _load


# ---------------------------------------------------------------------------
# OpenAI client mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai():
    """
    Patch `OpenAI` so classifier and agents never hit the network.

    Yields `(mock_client, set_classifier_payload)` where `set_classifier_payload`
    accepts the raw JSON **string** (use `classifier_response_for(...)`) returned
    on non-streaming `chat.completions.create` calls.

    Streaming create() calls yield canned observation text ending with disclaimer.
    """
    mock_client = MagicMock()
    state = {
        "classifier_content": classifier_response_for(
            "portfolio_health", "portfolio health check"
        ),
        "stream_text": f"[INFO] Mock observation for tests.\n\n{_DEFAULT_STREAM_TAIL}",
    }

    def set_classifier_payload(content: str) -> None:
        state["classifier_content"] = content

    def create_side_effect(*_args, **kwargs):
        if kwargs.get("stream"):
            return _mock_stream_chunks(state["stream_text"])
        return _make_mock_completion(state["classifier_content"])

    mock_client.chat.completions.create.side_effect = create_side_effect

    # Classifier binds `OpenAI` at import time; agents import it lazily inside `init_client_as_needed`.
    with patch("src.classifier.OpenAI", return_value=mock_client):
        with patch("openai.OpenAI", return_value=mock_client):
            yield mock_client, set_classifier_payload


@pytest.fixture
def mock_llm(mock_openai):
    """Same mock client as `mock_openai` — for stubs that inject `llm=mock_llm`."""
    client, _setter = mock_openai
    return client


@pytest.fixture
def intent_classifier_gold(mock_openai, gold_classifier_queries):
    """
    `IntentClassifier` with OpenAI patched so each user message replays the gold
    `expected_agent` / `expected_entities` for that exact query string.
    """
    mock_client, _ = mock_openai
    by_query = {c["query"]: c for c in gold_classifier_queries}
    mock_client.chat.completions.create.side_effect = make_classifier_side_effect(by_query)
    return IntentClassifier()


# ---------------------------------------------------------------------------
# Sample UserProfile fixtures (prefer `load_user` + adapters for JSON fidelity)
# ---------------------------------------------------------------------------


@pytest.fixture
def user_empty():
    """usr_004 — no positions; aligns with skeleton portfolio-health tests."""
    return UserProfile(
        user_id="usr_004",
        name="Empty Portfolio User",
        kyc_status="verified",
        risk_profile="moderate",
        investment_goals=["long-term growth"],
        portfolio=Portfolio(holdings=[], cash_balance=0.0),
        country="US",
    )


@pytest.fixture
def user_concentrated():
    """usr_003 — top-heavy NVDA for concentration-risk assertions."""
    return UserProfile(
        user_id="usr_003",
        name="Concentrated Holder",
        kyc_status="verified",
        risk_profile="aggressive",
        investment_goals=["capital appreciation"],
        portfolio=Portfolio(
            holdings=[
                Holding(
                    ticker="NVDA",
                    name="NVIDIA Corp",
                    quantity=100,
                    purchase_price=400.0,
                    current_price=850.0,
                    sector="technology",
                ),
                Holding(
                    ticker="AAPL",
                    name="Apple Inc",
                    quantity=20,
                    purchase_price=150.0,
                    current_price=190.0,
                    sector="technology",
                ),
                Holding(
                    ticker="MSFT",
                    name="Microsoft Corp",
                    quantity=10,
                    purchase_price=300.0,
                    current_price=380.0,
                    sector="technology",
                ),
            ],
            cash_balance=5000.0,
            inception_date="2023-01-01",
        ),
        country="US",
    )


@pytest.fixture
def user_active_trader():
    return UserProfile(
        user_id="usr_001",
        name="Active Trader",
        kyc_status="verified",
        risk_profile="aggressive",
        investment_goals=["short-term gains", "active trading"],
        portfolio=Portfolio(
            holdings=[
                Holding(
                    ticker="TSLA",
                    name="Tesla Inc",
                    quantity=50,
                    purchase_price=200.0,
                    current_price=250.0,
                    sector="automotive",
                ),
                Holding(
                    ticker="AMZN",
                    name="Amazon.com Inc",
                    quantity=15,
                    purchase_price=130.0,
                    current_price=175.0,
                    sector="technology",
                ),
                Holding(
                    ticker="META",
                    name="Meta Platforms",
                    quantity=25,
                    purchase_price=280.0,
                    current_price=400.0,
                    sector="technology",
                ),
                Holding(
                    ticker="GOOGL",
                    name="Alphabet Inc",
                    quantity=10,
                    purchase_price=140.0,
                    current_price=165.0,
                    sector="technology",
                ),
            ],
            cash_balance=10000.0,
            inception_date="2022-06-01",
        ),
        country="US",
    )
