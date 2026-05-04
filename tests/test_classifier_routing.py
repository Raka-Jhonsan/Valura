"""
Classifier routing and entity extraction against `fixtures/test_queries/intent_classification.json`.

LLM calls are mocked in `conftest.py` so each query replays the gold `expected_agent` and
entities — this validates the parsing layer, the matcher, and router-facing taxonomy strings.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import normalize_target_agent


def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(float(act_value) - float(exp_value)) > abs(float(exp_value)) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


def test_classifier_routing_accuracy(intent_classifier_gold, gold_classifier_queries):
    """Threshold: ≥ 85% routing accuracy (ASSIGNMENT.md)."""
    correct = 0
    for case in gold_classifier_queries:
        result = intent_classifier_gold.classify(case["query"], history=None)
        expected = normalize_target_agent(case["expected_agent"])
        if result.target_agent == expected:
            correct += 1

    accuracy = correct / len(gold_classifier_queries)
    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


def test_classifier_entity_extraction(intent_classifier_gold, gold_classifier_queries):
    """Entity subset match rate on rows that specify expected entities."""
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = intent_classifier_gold.classify(case["query"], history=None)
        actual = result.entities.model_dump(mode="python")
        if matches_entities(actual, case["expected_entities"]):
            matched += 1

    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")
    assert rate >= 0.85, f"Entity match rate {rate:.2%} below 85% on labeled entity rows"
