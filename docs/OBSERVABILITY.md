# Observability

Three layers, in increasing order of "real production infrastructure":

## 1. Zero-infrastructure: the built-in dashboard

```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/dashboard
```

Polls `GET /v1/stats` every 3s. Process-local, in-memory, resets on restart --
this exists for local dev and demos/screenshots, not as a production
monitoring solution. Useful for: "run the gateway, immediately see something,"
no Prometheus/Grafana setup required.

## 2. Structured audit logs

Every decision emits one JSON line to stdout (`app/observability/logging_config.py`):

```json
{"event": "cerberus.decision", "timestamp": 1782989411.64, "request_id": null,
 "verdict": "redact", "max_risk_score": 1.0, "total_latency_ms": 1.73,
 "text_length": 30, "text_hash": "d1c96e6831336021", "direction": "inbound",
 "classifier_scores": {"prompt_injection": 0.05, "emotion": 0.1, "ai_text_detector": 0.1, "pii": 1.0},
 "classifier_errors": {}, "policy_matched_doc_id": null, "policy_similarity": null,
 "redacted": true}
```

Deliberately logs `text_hash` (truncated SHA-256), not the raw message --
the audit log can confirm "this exact input was seen N times" for incident
investigation without itself becoming a store of user PII or attack payloads.
In production, ship stdout to whatever log aggregator you already run
(CloudWatch, Datadog, ELK) -- no code change needed, this is just structured
stdout logging.

## 3. Prometheus metrics

```bash
curl http://127.0.0.1:8000/metrics
```

Standard Prometheus text format, drops into any existing scrape config:

```yaml
scrape_configs:
  - job_name: cerberus
    static_configs:
      - targets: ["localhost:8000"]
```

Metrics exposed (`app/observability/metrics.py`):

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `cerberus_requests_total` | Counter | `verdict` | Traffic volume by outcome |
| `cerberus_request_latency_ms` | Histogram | - | p50/p95/p99 latency, alertable |
| `cerberus_max_risk_score` | Histogram | - | Distribution of ensemble risk scores -- shifts here can indicate a changing traffic pattern (e.g. an ongoing attack) before verdict counts alone would show it |
| `cerberus_classifier_errors_total` | Counter | `classifier` | Which classifier is failing, if any -- feeds an alert on the fail-closed-to-escalate path getting hit unexpectedly often |
| `cerberus_policy_citations_total` | Counter | `matched_doc_id` | Which policy docs are actually being matched in production -- a policy doc that's never cited is worth reviewing (too narrow?), one dominating is worth reviewing too (too broad?) |
| `cerberus_redactions_total` | Counter | - | PII redaction volume |

Example Grafana panels worth building from these:
- **Verdict mix over time** (stacked area, `rate(cerberus_requests_total[5m])` by `verdict`) -- a sudden shift toward `block`/`escalate` is often the first signal of an actual attack campaign against the product.
- **p99 latency** (from the histogram) -- alert if it climbs, especially once real LLM calls are in the decision path (Phase 4's `decide_llm` node is the latency-variable part of the pipeline).
- **Classifier error rate** -- alert on this directly; a silently-failing classifier degrades detection without ever showing up in the verdict mix.

## 4. Tracing the agent pipeline (LangSmith)

`app/agents/graph.py` builds a standard LangGraph `StateGraph`, which
LangSmith can trace automatically -- no code change needed, only environment
variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_PROJECT=cerberus
```

With these set, every graph invocation (which nodes ran, in what order, with
what state at each step, and timing per node) shows up in the LangSmith UI.
This is especially useful for debugging *why* the `decide_llm` node reached a
particular verdict once real LLM calls are in play -- you get the full
retrieved-docs-plus-prompt-plus-response for each ambiguous case, not just
the final JSON. Not exercised live in this repo's dev environment (no
LangSmith API key available) -- same caveat as `AnthropicLLMClient` in
Phase 4.
