from dataclasses import dataclass

import pytest

from application.router.router import Router


@dataclass
class FakeClassificationResult:
    label: str
    confidence: float


class FakePredictor:

    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def predict(self, text):
        return FakeClassificationResult(
            label=self.label,
            confidence=self.confidence,
        )


def test_routes_chat():
    predictor = FakePredictor(
        label="chat",
        confidence=0.95,
    )

    router = Router(predictor)

    result = router.route("thanks")

    assert result.label == "chat"
    assert result.confidence == 0.95
    assert result.accepted is True


def test_routes_rag():
    predictor = FakePredictor(
        label="rag",
        confidence=0.91,
    )

    router = Router(predictor)

    result = router.route(
        "what is python?"
    )

    assert result.label == "rag"
    assert result.confidence == 0.91
    assert result.accepted is True


def test_low_confidence_is_rejected():
    predictor = FakePredictor(
        label="rag",
        confidence=0.54,
    )

    router = Router(
        predictor,
        confidence_threshold=0.80,
    )

    result = router.route(
        "something ambiguous"
    )

    assert result.label == "rag"
    assert result.confidence == 0.54
    assert result.accepted is False


def test_custom_threshold():
    predictor = FakePredictor(
        label="rag",
        confidence=0.75,
    )

    router = Router(
        predictor,
        confidence_threshold=0.70,
    )

    result = router.route("explain python")

    assert result.accepted is True


def test_invalid_threshold():
    predictor = FakePredictor(
        label="chat",
        confidence=0.95,
    )

    with pytest.raises(ValueError):
        Router(
            predictor,
            confidence_threshold=1.5,
        )