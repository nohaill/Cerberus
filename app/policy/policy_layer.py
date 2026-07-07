"""
The policy resolution layer -- Phase 3's rule-based approach.

As of Phase 4, this class is no longer on the gateway's production path;
app/agents/graph.py's AgentPipeline (retrieve + decide_llm nodes) supersedes
it, using the same vector store and corpus but replacing the mechanical
top-1-match mapping below with an LLM reasoning step. PolicyLayer is kept
and still tested (tests/test_policy_layer.py) as the deterministic baseline
this phase improved on, and because MockLLMClient's offline behavior
intentionally mirrors it exactly (see app/agents/llm_client.py docstring).

Only invoked when the ensemble's threshold-based `decide()` returns ESCALATE
-- clear allows and clear blocks never touch this code path, which keeps the
(slower, retrieval-based) policy layer off the hot path for the vast majority
of requests.

Implementation is rule-based: retrieve the top-matching policy document, and
if its similarity clears a confidence bar, map its `action` field directly to
a verdict.
"""
from dataclasses import dataclass

from app.config import get_settings
from app.policy.documents import POLICY_DOCUMENTS
from app.policy.vector_store import BaseVectorStore, InMemoryVectorStore, PgVectorStore, RetrievalMatch

_ACTION_TO_VERDICT = {
    "block": "block",
    "redact": "redact",
    "allow": "allow",
    "escalate": "escalate",
}


def _build_default_vector_store() -> BaseVectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        if not settings.database_url:
            raise RuntimeError(
                "CERBERUS_VECTOR_STORE_BACKEND=pgvector requires CERBERUS_DATABASE_URL to be set."
            )
        # NOTE: not exercised against a live Postgres instance in this repo's
        # dev environment -- see docs/POLICY_LAYER.md before relying on this
        # path in production. The corpus also needs to be ingested separately
        # (docs/POLICY_LAYER.md step 3) before querying it here will return
        # anything.
        return PgVectorStore(dsn=settings.database_url)
    return InMemoryVectorStore(POLICY_DOCUMENTS)


@dataclass
class PolicyResolution:
    verdict: str
    rationale: str
    matched_doc_id: str | None
    matched_category: str | None
    similarity: float | None


class PolicyLayer:
    def __init__(self, vector_store: BaseVectorStore | None = None):
        self.vector_store = vector_store or _build_default_vector_store()

    def resolve(self, text: str) -> PolicyResolution:
        settings = get_settings()
        matches: list[RetrievalMatch] = self.vector_store.query(text, k=3)

        if not matches or matches[0].similarity < settings.policy_similarity_threshold:
            # No confident policy match -- stay escalated for human review
            # rather than guessing. This is a deliberate fail-safe: a weak
            # retrieval match is worse than an honest "we don't know."
            best = matches[0] if matches else None
            return PolicyResolution(
                verdict="escalate",
                rationale=(
                    "No policy document matched with sufficient confidence "
                    f"(best similarity={best.similarity:.2f})" if best else "No policy documents available"
                ) + "; keeping this escalated for human review.",
                matched_doc_id=best.document.id if best else None,
                matched_category=best.document.category if best else None,
                similarity=best.similarity if best else None,
            )

        top = matches[0]
        verdict = _ACTION_TO_VERDICT[top.document.action]
        rationale = (
            f"Policy layer matched '{top.document.category}' (doc={top.document.id}, "
            f"similarity={top.similarity:.2f}) -> {verdict}."
        )
        return PolicyResolution(
            verdict=verdict,
            rationale=rationale,
            matched_doc_id=top.document.id,
            matched_category=top.document.category,
            similarity=top.similarity,
        )
