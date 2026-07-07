"""
Every classifier in the ensemble implements this interface. This is what makes
the ensemble pluggable: adding a new HF model to the gateway means writing one
small class, not touching the orchestration code.
"""
from abc import ABC, abstractmethod

from app.schemas import ClassifierResult


class BaseClassifier(ABC):
    #: Unique name used in logs, dashboards, and eval reports.
    name: str = "base"

    @abstractmethod
    def _predict(self, text: str) -> tuple[str, float]:
        """Return (label, risk_score in [0,1]). Implemented per-classifier.

        risk_score is normalized so 1.0 always means "highest risk" regardless
        of what the underlying model's label scheme looks like -- this is what
        lets the ensemble compare/aggregate scores across models that were
        never designed to be compared to each other.
        """
        raise NotImplementedError

    def predict(self, text: str) -> ClassifierResult:
        import time

        start = time.perf_counter()
        try:
            label, score = self._predict(text)
            error = None
        except Exception as exc:  # noqa: BLE001 - we want to surface any model failure
            label, score, error = "error", 0.0, str(exc)
        latency_ms = (time.perf_counter() - start) * 1000
        return ClassifierResult(
            name=self.name,
            label=label,
            score=score,
            latency_ms=latency_ms,
            error=error,
        )
