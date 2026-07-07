"""
Structured audit logging.

Every decision gets one JSON log line -- this is the audit trail a real
deployment would ship to its log aggregator (Datadog, CloudWatch, ELK,
whatever). Deliberately does NOT log the raw message text: only a truncated
SHA-256 hash, so the audit log itself can't become a PII/security liability
just by existing. Logging a hash still lets you correlate "this exact input
was seen N times" without storing the input.

This lives outside app/gateway.py on purpose -- Gateway.check() stays a pure
request-in/response-out function with no logging side effects, which is what
keeps tests/test_gateway_integration.py able to assert on its output without
mocking a logger. The HTTP layer (app/main.py) is responsible for calling
log_decision() after it gets a response back.
"""
import hashlib
import json
import logging
import sys
import time

from app.schemas import CheckRequest, CheckResponse

logger = logging.getLogger("cerberus.audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))  # message IS the JSON
    logger.addHandler(handler)
    logger.propagate = False


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_decision(request: CheckRequest, response: CheckResponse) -> None:
    record = {
        "event": "cerberus.decision",
        "timestamp": time.time(),
        "request_id": response.request_id,
        "verdict": response.verdict.value,
        "max_risk_score": response.max_risk_score,
        "total_latency_ms": response.total_latency_ms,
        "text_length": len(request.text),
        "text_hash": _text_hash(request.text),
        "direction": request.direction,
        "classifier_scores": {r.name: r.score for r in response.classifier_results},
        "classifier_errors": {r.name: r.error for r in response.classifier_results if r.error},
        "policy_matched_doc_id": response.policy_citation.matched_doc_id if response.policy_citation else None,
        "policy_similarity": response.policy_citation.similarity if response.policy_citation else None,
        "redacted": response.redacted_text is not None,
    }
    logger.info(json.dumps(record))
