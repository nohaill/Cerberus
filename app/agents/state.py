"""
Shared state passed between LangGraph nodes.

Deliberately a plain TypedDict (not a pydantic model) -- LangGraph merges
partial dict updates returned by each node into this state, which is the
standard pattern for its StateGraph API.
"""
from typing import Optional, TypedDict

from app.schemas import ClassifierResult


class RetrievedDoc(TypedDict):
    doc_id: str
    category: str
    action: str
    similarity: float
    text: str


class GraphState(TypedDict, total=False):
    # --- input ---
    text: str
    classifier_summary: str  # human-readable summary the LLM node reads
    classifier_results: list[ClassifierResult]  # used by the triage node

    # --- triage node output ---
    base_verdict: str
    base_rationale: str

    # --- retrieve node output ---
    retrieved_docs: list[RetrievedDoc]

    # --- decision node output ---
    final_verdict: str
    final_rationale: str
    matched_doc_id: Optional[str]
    matched_category: Optional[str]
    similarity: Optional[float]
