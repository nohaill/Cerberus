"""
API contracts for Cerberus. Kept separate from business logic so the schema
can be versioned independently (this is v1) as the decision pipeline grows.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ESCALATE = "escalate"  # ambiguous case, routed to policy/agent layer


class ClassifierResult(BaseModel):
    """Raw output of a single classifier in the ensemble."""

    name: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    error: Optional[str] = None


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    # Arbitrary caller-supplied id so callers can correlate a decision with
    # their own request/trace id in their logs.
    request_id: Optional[str] = None
    # Direction matters: injection detectors mostly care about inbound user
    # text, PII detectors matter on both directions.
    direction: str = Field(default="inbound", pattern="^(inbound|outbound)$")


class PolicyCitation(BaseModel):
    """Present only when an ambiguous case was resolved by the RAG policy
    layer -- gives the audit trail a concrete pointer to *why* a verdict was
    reached beyond just classifier scores."""

    matched_doc_id: Optional[str]
    matched_category: Optional[str]
    similarity: Optional[float]


class CheckResponse(BaseModel):
    request_id: Optional[str]
    verdict: Verdict
    # Highest-risk score across the ensemble, surfaced for easy alerting/dashboards.
    max_risk_score: float
    classifier_results: list[ClassifierResult]
    rationale: str
    redacted_text: Optional[str] = None
    policy_citation: Optional[PolicyCitation] = None
    total_latency_ms: float
