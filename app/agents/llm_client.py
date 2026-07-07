"""
LLM client for the decision node.

Same pattern as every other real/mock pair in this repo (classifiers,
embeddings, vector store): a real backend that needs a live credential, and a
deterministic mock that keeps the graph fully testable offline.

Important honesty note, unlike the other mocks in this repo: MockLLMClient
is NOT a reduced-fidelity stand-in for the real thing's behavior -- it has no
semantic reasoning capability at all, it just mechanically picks the top
retrieved doc above a confidence bar (the same rule Phase 3's PolicyLayer
used). It exists to validate the graph's plumbing (state passing, conditional
routing, response shape) offline, not to demonstrate improved detection.
The actual capability upgrade Phase 4 claims -- catching semantic paraphrases
that lexical retrieval misses -- only exists when AnthropicLLMClient is used
with a real API key. This repo's dev environment had no API key available,
so that capability is implemented and ready but not live-verified here; see
README for how to verify it once you have a key.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.agents.prompts import DECISION_SYSTEM_PROMPT, build_decision_prompt
from app.agents.state import RetrievedDoc
from app.config import get_settings

_VALID_VERDICTS = {"allow", "block", "redact", "escalate"}


@dataclass
class LLMDecision:
    verdict: str
    rationale: str
    matched_doc_id: str | None


class BaseLLMClient(ABC):
    @abstractmethod
    def decide(self, text: str, classifier_summary: str, retrieved_docs: list[RetrievedDoc]) -> LLMDecision:
        raise NotImplementedError


class AnthropicLLMClient(BaseLLMClient):
    """Real backend. Requires ANTHROPIC_API_KEY in the environment and the
    `anthropic` package. Not exercised against a live API in this repo's dev
    environment -- see README for how to validate it."""

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.model = model or settings.agent_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    def decide(self, text: str, classifier_summary: str, retrieved_docs: list[RetrievedDoc]) -> LLMDecision:
        client = self._get_client()
        prompt = build_decision_prompt(text, classifier_summary, retrieved_docs)

        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            system=DECISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_llm_response(raw_text, retrieved_docs)


class MockLLMClient(BaseLLMClient):
    """Offline stand-in -- see module docstring. Mirrors Phase 3's rule-based
    policy mapping exactly, so it validates the graph without claiming a
    detection improvement it can't actually deliver offline."""

    def decide(self, text: str, classifier_summary: str, retrieved_docs: list[RetrievedDoc]) -> LLMDecision:
        settings = get_settings()
        if not retrieved_docs or retrieved_docs[0]["similarity"] < settings.policy_similarity_threshold:
            best = retrieved_docs[0] if retrieved_docs else None
            return LLMDecision(
                verdict="escalate",
                rationale=(
                    f"[mock] No policy document matched confidently "
                    f"(best similarity={best['similarity']:.2f})" if best else "[mock] No policy documents retrieved"
                ) + "; escalating for human review.",
                matched_doc_id=best["doc_id"] if best else None,
            )
        top = retrieved_docs[0]
        return LLMDecision(
            verdict=top["action"],
            rationale=f"[mock] Top retrieved doc {top['doc_id']} (similarity={top['similarity']:.2f}) -> {top['action']}.",
            matched_doc_id=top["doc_id"],
        )


def _parse_llm_response(raw_text: str, retrieved_docs: list[RetrievedDoc]) -> LLMDecision:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
        verdict = data.get("verdict", "escalate")
        if verdict not in _VALID_VERDICTS:
            verdict = "escalate"
        matched_id = data.get("matched_doc_id")
        valid_ids = {d["doc_id"] for d in retrieved_docs}
        if matched_id not in valid_ids:
            matched_id = None  # don't trust a hallucinated doc id
        rationale = data.get("rationale", "(no rationale provided)")
        return LLMDecision(verdict=verdict, rationale=rationale, matched_doc_id=matched_id)
    except (json.JSONDecodeError, AttributeError):
        # Malformed LLM output -- fail closed to escalate, don't crash the request.
        return LLMDecision(
            verdict="escalate",
            rationale=f"LLM response could not be parsed as JSON; escalating for human review. Raw: {raw_text[:200]}",
            matched_doc_id=None,
        )


def build_llm_client() -> BaseLLMClient:
    settings = get_settings()
    if settings.mock_mode or not os.environ.get("ANTHROPIC_API_KEY"):
        return MockLLMClient()
    return AnthropicLLMClient()
