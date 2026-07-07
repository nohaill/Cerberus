"""
In-memory recent-request stats.

Prometheus (app/observability/metrics.py) is the real production path -- it
needs a Prometheus server + Grafana to actually look at. This module exists
so the repo has a *zero-infrastructure* dashboard: run the gateway, hit
GET /dashboard, see live numbers immediately. It's not a replacement for
Prometheus (it's process-local, resets on restart, no history) -- it's a
"batteries included" view for local dev, demos, and screenshots.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.schemas import CheckResponse

_MAX_RECENT = 500


@dataclass
class _RecentRecord:
    timestamp: float
    verdict: str
    max_risk_score: float
    total_latency_ms: float
    matched_doc_id: str | None
    redacted: bool


class StatsAggregator:
    def __init__(self, max_recent: int = _MAX_RECENT):
        self._lock = threading.Lock()
        self._recent: deque[_RecentRecord] = deque(maxlen=max_recent)
        self._total_count = 0
        self._verdict_counts: dict[str, int] = {}

    def record(self, response: CheckResponse) -> None:
        with self._lock:
            self._total_count += 1
            v = response.verdict.value
            self._verdict_counts[v] = self._verdict_counts.get(v, 0) + 1
            self._recent.append(
                _RecentRecord(
                    timestamp=time.time(),
                    verdict=v,
                    max_risk_score=response.max_risk_score,
                    total_latency_ms=response.total_latency_ms,
                    matched_doc_id=response.policy_citation.matched_doc_id if response.policy_citation else None,
                    redacted=response.redacted_text is not None,
                )
            )

    def snapshot(self) -> dict:
        with self._lock:
            recent = list(self._recent)
            total = self._total_count
            verdict_counts = dict(self._verdict_counts)

        latencies = sorted(r.total_latency_ms for r in recent)
        n = len(latencies)

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return latencies[idx]

        doc_counts: dict[str, int] = {}
        for r in recent:
            if r.matched_doc_id:
                doc_counts[r.matched_doc_id] = doc_counts.get(r.matched_doc_id, 0) + 1

        return {
            "total_requests_lifetime": total,
            "recent_window_size": n,
            "verdict_counts_lifetime": verdict_counts,
            "verdict_counts_recent": _count_by(recent, "verdict"),
            "latency_p50_ms": pct(0.50),
            "latency_p95_ms": pct(0.95),
            "latency_p99_ms": pct(0.99),
            "redaction_count_recent": sum(1 for r in recent if r.redacted),
            "policy_doc_hits_recent": doc_counts,
            "recent_requests": [
                {
                    "timestamp": r.timestamp,
                    "verdict": r.verdict,
                    "max_risk_score": r.max_risk_score,
                    "total_latency_ms": r.total_latency_ms,
                    "matched_doc_id": r.matched_doc_id,
                }
                for r in recent[-50:]
            ],
        }


def _count_by(records: list[_RecentRecord], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        key = getattr(r, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


# Process-local singleton -- fine for a single gateway instance / local dev.
# A multi-instance deployment would read dashboard numbers from Prometheus
# instead (see app/observability/metrics.py), not this.
stats = StatsAggregator()
