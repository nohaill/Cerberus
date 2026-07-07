"""
The full decision pipeline as a LangGraph StateGraph.

    START -> triage -> [conditional] -> END                    (fast path)
                     -> retrieve -> decide_llm -> END           (ambiguous path)

`triage` is the same cheap, deterministic threshold check from Phase 1/2
(app/decision.py) -- it lives inside the graph now rather than being called
separately by the gateway, so the entire request-handling decision logic is
one graph invocation, and the "skip the expensive path for clear cases"
behavior is the graph's own conditional routing rather than an if-statement
in gateway.py. `retrieve` and `decide_llm` are only reached for ambiguous
(ESCALATE) triage results, which is what keeps the LLM call off the hot path
for the large majority of requests.
"""
from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.agents.llm_client import BaseLLMClient, build_llm_client
from app.agents.state import GraphState, RetrievedDoc
from app.decision import decide
from app.policy.documents import POLICY_DOCUMENTS
from app.policy.vector_store import BaseVectorStore, InMemoryVectorStore, PgVectorStore
from app.config import get_settings
from app.schemas import ClassifierResult


@dataclass
class AgentResult:
    verdict: str
    rationale: str
    matched_doc_id: Optional[str]
    matched_category: Optional[str]
    similarity: Optional[float]


def _summarize_classifiers(results: list[ClassifierResult]) -> str:
    lines = []
    for r in results:
        status = f"error={r.error}" if r.error else f"label={r.label} score={r.score:.2f}"
        lines.append(f"- {r.name}: {status}")
    return "\n".join(lines)


def _build_default_vector_store() -> BaseVectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        if not settings.database_url:
            raise RuntimeError(
                "CERBERUS_VECTOR_STORE_BACKEND=pgvector requires CERBERUS_DATABASE_URL to be set."
            )
        return PgVectorStore(dsn=settings.database_url)
    return InMemoryVectorStore(POLICY_DOCUMENTS)


class AgentPipeline:
    def __init__(self, vector_store: BaseVectorStore | None = None, llm_client: BaseLLMClient | None = None):
        self.vector_store = vector_store or _build_default_vector_store()
        self.llm_client = llm_client or build_llm_client()
        self._graph = self._build_graph()

    # --- nodes ---

    def _triage_node(self, state: GraphState) -> dict:
        verdict, rationale = decide(state["classifier_results"])
        return {"base_verdict": verdict.value, "base_rationale": rationale}

    def _route_after_triage(self, state: GraphState) -> str:
        return "retrieve" if state["base_verdict"] == "escalate" else "end"

    def _retrieve_node(self, state: GraphState) -> dict:
        matches = self.vector_store.query(state["text"], k=3)
        retrieved: list[RetrievedDoc] = [
            RetrievedDoc(
                doc_id=m.document.id,
                category=m.document.category,
                action=m.document.action,
                similarity=m.similarity,
                text=m.document.text,
            )
            for m in matches
        ]
        return {"retrieved_docs": retrieved}

    def _decision_node(self, state: GraphState) -> dict:
        decision = self.llm_client.decide(state["text"], state["classifier_summary"], state["retrieved_docs"])
        matched = next((d for d in state["retrieved_docs"] if d["doc_id"] == decision.matched_doc_id), None)
        return {
            "final_verdict": decision.verdict,
            "final_rationale": decision.rationale,
            "matched_doc_id": decision.matched_doc_id,
            "matched_category": matched["category"] if matched else None,
            "similarity": matched["similarity"] if matched else None,
        }

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("triage", self._triage_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("decide_llm", self._decision_node)

        graph.add_edge(START, "triage")
        graph.add_conditional_edges("triage", self._route_after_triage, {"retrieve": "retrieve", "end": END})
        graph.add_edge("retrieve", "decide_llm")
        graph.add_edge("decide_llm", END)

        return graph.compile()

    # --- public API ---

    def run(self, text: str, classifier_results: list[ClassifierResult]) -> AgentResult:
        initial_state: GraphState = {
            "text": text,
            "classifier_summary": _summarize_classifiers(classifier_results),
            "classifier_results": classifier_results,
        }
        final_state = self._graph.invoke(initial_state)

        if final_state.get("base_verdict") != "escalate":
            return AgentResult(
                verdict=final_state["base_verdict"],
                rationale=final_state["base_rationale"],
                matched_doc_id=None,
                matched_category=None,
                similarity=None,
            )

        return AgentResult(
            verdict=final_state["final_verdict"],
            rationale=f"{final_state['base_rationale']} Agent decision: {final_state['final_rationale']}",
            matched_doc_id=final_state.get("matched_doc_id"),
            matched_category=final_state.get("matched_category"),
            similarity=final_state.get("similarity"),
        )
