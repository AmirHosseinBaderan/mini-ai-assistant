from dataclasses import dataclass

from intent_classifier.predictor import (
    ClassificationResult,
)


@dataclass(frozen=True)
class RouteResult:
    label: str
    confidence: float
    accepted: bool


class Router:
    def __init__(
        self,
        predictor,
        confidence_threshold: float = 0.80,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

    def route(self, text: str) -> RouteResult:
        result: ClassificationResult = (
            self.predictor.predict(text)
        )

        accepted = (
            result.confidence
            >= self.confidence_threshold
        )

        return RouteResult(
            label=result.label,
            confidence=result.confidence,
            accepted=accepted,
        )