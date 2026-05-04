"""Multi-turn classifier behaviour using `fixtures/conversations/*.json` (mocked LLM)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.classifier import IntentClassifier

from tests.conftest import make_classifier_side_effect, normalize_target_agent
from tests.test_classifier_routing import matches_entities


def _history_from_prior(prior_user_turns: list[str]) -> list[dict[str, str]]:
    """Build alternating user/assistant turns for classifier context."""
    history: list[dict[str, str]] = []
    for i, text in enumerate(prior_user_turns):
        history.append({"role": "user", "content": text})
        if i < len(prior_user_turns) - 1:
            history.append({"role": "assistant", "content": "Understood."})
    return history


def _run_conversation_file(mock_openai, path: Path) -> None:
    mock_client, _ = mock_openai
    data = json.loads(path.read_text(encoding="utf-8"))
    by_current = {c["current_user_turn"]: c for c in data["test_cases"]}
    mock_client.chat.completions.create.side_effect = make_classifier_side_effect(by_current)
    clf = IntentClassifier()

    for case in data["test_cases"]:
        history = _history_from_prior(case["prior_user_turns"])
        out = clf.classify(case["current_user_turn"], history=history)
        expected_agent = normalize_target_agent(case["expected"]["agent"])
        assert out.target_agent == expected_agent, (
            f"{path.name} {case.get('case_id')}: expected {expected_agent}, got {out.target_agent}"
        )
        exp_entities = case["expected"].get("entities") or {}
        if exp_entities:
            assert matches_entities(out.entities.model_dump(mode="python"), exp_entities), (
                f"{path.name} {case.get('case_id')}: entity mismatch"
            )


@pytest.mark.parametrize(
    "filename",
    [
        "follow_up_session.json",
        "multi_intent_session.json",
        "ambiguous_session.json",
    ],
)
def test_conversation_fixtures(mock_openai, filename: str):
    root = Path(__file__).resolve().parent.parent / "fixtures" / "conversations"
    _run_conversation_file(mock_openai, root / filename)
