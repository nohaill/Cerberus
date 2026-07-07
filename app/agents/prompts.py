"""
Prompt for the decision node's LLM call.

The prompt deliberately gives the model the retrieved policy chunks *as
context to reason with*, not as a lookup table to match against. This is the
actual point of Phase 4: Phase 3's rule-based policy layer could only act on
a confident top-1 lexical match; this node can reason about intent even when
retrieval surfaces only weak/partial matches -- which is exactly the
semantic-paraphrase gap documented in Phase 3 (see app/policy/documents.py
docstring and README).
"""
from app.agents.state import RetrievedDoc

DECISION_SYSTEM_PROMPT = """You are the decision-making component of an LLM security gateway. \
You will be shown a user message, automated classifier scores for it, and the most relevant \
internal policy documents retrieved for it. Your job is to decide the final handling verdict.

You must respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:
{"verdict": "<allow|block|redact|escalate>", "matched_doc_id": "<policy doc id or null>", "rationale": "<one or two sentences>"}

Guidance:
- Reason about the user's actual intent, not just surface keyword overlap with the policy text.
- If a retrieved policy document's action clearly applies given the message's real intent, use it, \
even if the wording doesn't literally match (paraphrases, indirect phrasing, and obfuscation attempts \
should be judged by what they're trying to accomplish, not their literal words).
- If none of the retrieved documents genuinely apply and you are not confident, return "escalate" -- \
an honest "I'm not sure, a human should look at this" is always safer than a confident wrong guess.
- Never invent a matched_doc_id that wasn't in the retrieved documents list."""


def build_decision_prompt(text: str, classifier_summary: str, retrieved_docs: list[RetrievedDoc]) -> str:
    docs_block = "\n\n".join(
        f"[{d['doc_id']}] category={d['category']} action={d['action']} similarity={d['similarity']:.2f}\n{d['text']}"
        for d in retrieved_docs
    ) or "(no policy documents retrieved)"

    return f"""User message:
\"\"\"{text}\"\"\"

Classifier signals:
{classifier_summary}

Retrieved policy documents (top {len(retrieved_docs)}, ranked by lexical similarity -- \
similarity is a hint, not ground truth; judge relevance yourself):

{docs_block}

Decide the final verdict."""
