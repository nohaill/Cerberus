from app.decision import decide
from app.schemas import ClassifierResult, Verdict


def _result(name="prompt_injection", label="INJECTION", score=0.0, error=None) -> ClassifierResult:
    return ClassifierResult(name=name, label=label, score=score, latency_ms=1.0, error=error)


def test_high_risk_blocks():
    results = [_result(score=0.95)]
    verdict, rationale = decide(results)
    assert verdict == Verdict.BLOCK
    assert "prompt_injection" in rationale


def test_low_risk_allows():
    results = [_result(label="SAFE", score=0.05)]
    verdict, _ = decide(results)
    assert verdict == Verdict.ALLOW


def test_mid_risk_escalates():
    results = [_result(score=0.6)]
    verdict, rationale = decide(results)
    assert verdict == Verdict.ESCALATE
    assert "ambiguous" in rationale


def test_pii_forces_redact_regardless_of_other_scores():
    results = [_result(score=0.05), _result(name="pii", label="EMAIL", score=1.0)]
    verdict, rationale = decide(results)
    assert verdict == Verdict.REDACT
    assert "PII" in rationale


def test_all_classifiers_failed_fails_closed_to_escalate():
    # PII classifier is excluded from the "all scored" check by design (it
    # always returns score 0.0 or 1.0, never errors). Only non-PII classifiers
    # count for fail-closed behavior.
    results = [_result(score=0.0, error="model unavailable")]
    verdict, rationale = decide(results)
    assert verdict == Verdict.ESCALATE
    assert "failed" in rationale.lower()


def test_highest_score_wins_when_multiple_classifiers_disagree():
    results = [_result(name="emotion", score=0.2), _result(name="prompt_injection", score=0.9)]
    verdict, rationale = decide(results)
    assert verdict == Verdict.BLOCK
    assert "prompt_injection" in rationale
