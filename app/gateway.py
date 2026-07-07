"""
Gateway orchestration.

As of Phase 4, the entire triage -> policy retrieval -> decision flow is one
LangGraph pipeline (app/agents/graph.py). This module is now a thin wrapper:
run the classifier ensemble, hand the results to the graph, translate its
result into the HTTP response shape. The "skip the expensive path for clear
cases" behavior lives in the graph's own conditional routing, not here.
"""
import time

from app.agents.graph import AgentPipeline
from app.classifiers.ensemble import ClassifierEnsemble
from app.schemas import CheckRequest, CheckResponse, PolicyCitation, Verdict


class Gateway:
    def __init__(self, ensemble: ClassifierEnsemble | None = None, agent_pipeline: AgentPipeline | None = None):
        self.ensemble = ensemble or ClassifierEnsemble()
        self.agent_pipeline = agent_pipeline if agent_pipeline is not None else AgentPipeline()

    async def check(self, request: CheckRequest) -> CheckResponse:
        import asyncio

        start = time.perf_counter()
        results = await self.ensemble.run(request.text)

        # AgentPipeline.run is sync (LangGraph's .invoke API, and the real
        # LLM client's blocking HTTP call) -- runs in the default executor so
        # it doesn't block the event loop for concurrent requests. For clear
        # allow/block/redact cases this returns almost immediately since the
        # graph's conditional routing skips straight to END.
        loop = asyncio.get_event_loop()
        agent_result = await loop.run_in_executor(None, self.agent_pipeline.run, request.text, results)

        verdict = Verdict(agent_result.verdict)
        rationale = agent_result.rationale
        policy_citation = None
        if agent_result.matched_doc_id:
            policy_citation = PolicyCitation(
                matched_doc_id=agent_result.matched_doc_id,
                matched_category=agent_result.matched_category,
                similarity=agent_result.similarity,
            )

        redacted_text = None
        if verdict == Verdict.REDACT:
            pii_clf = self.ensemble.get_pii_classifier()
            if pii_clf:
                redacted_text = pii_clf.redact(request.text)

        max_risk = max((r.score for r in results if r.error is None), default=0.0)
        total_latency_ms = (time.perf_counter() - start) * 1000

        return CheckResponse(
            request_id=request.request_id,
            verdict=verdict,
            max_risk_score=max_risk,
            classifier_results=results,
            rationale=rationale,
            redacted_text=redacted_text,
            policy_citation=policy_citation,
            total_latency_ms=total_latency_ms,
        )
