"""
Prometheus metrics.

Exposed at GET /metrics in Prometheus text format -- the standard scrape
endpoint, so this drops straight into any existing Prometheus + Grafana
setup with no custom exporter needed. See docs/OBSERVABILITY.md for an
example Grafana panel/query set.
"""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas import CheckResponse

REQUESTS_TOTAL = Counter(
    "cerberus_requests_total", "Total gateway requests processed, by final verdict.", ["verdict"]
)

CLASSIFIER_ERRORS_TOTAL = Counter(
    "cerberus_classifier_errors_total", "Classifier failures/timeouts, by classifier name.", ["classifier"]
)

POLICY_CITATIONS_TOTAL = Counter(
    "cerberus_policy_citations_total",
    "Requests resolved via the agent pipeline's retrieve+decide path, by matched policy doc.",
    ["matched_doc_id"],
)

REDACTIONS_TOTAL = Counter("cerberus_redactions_total", "Requests that triggered PII redaction.")

REQUEST_LATENCY_MS = Histogram(
    "cerberus_request_latency_ms",
    "End-to-end gateway request latency in milliseconds.",
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

CLASSIFIER_MAX_RISK_SCORE = Histogram(
    "cerberus_max_risk_score", "Highest ensemble risk score per request.", buckets=tuple(i / 10 for i in range(11))
)


def record_response(response: CheckResponse) -> None:
    REQUESTS_TOTAL.labels(verdict=response.verdict.value).inc()
    REQUEST_LATENCY_MS.observe(response.total_latency_ms)
    CLASSIFIER_MAX_RISK_SCORE.observe(response.max_risk_score)

    for r in response.classifier_results:
        if r.error:
            CLASSIFIER_ERRORS_TOTAL.labels(classifier=r.name).inc()

    if response.policy_citation:
        POLICY_CITATIONS_TOTAL.labels(matched_doc_id=response.policy_citation.matched_doc_id or "none").inc()

    if response.redacted_text is not None:
        REDACTIONS_TOTAL.inc()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
