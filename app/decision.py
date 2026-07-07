"""
The threshold-based triage decision. Extracted into its own module (rather
than living in gateway.py) specifically so app/agents/graph.py can import it
without creating gateway.py <-> app/agents/graph.py circular imports --
gateway.py uses the agent pipeline, and the agent pipeline's triage node uses
this same function, so it needs a home neither of them owns.
"""
from app.config import get_settings
from app.schemas import ClassifierResult, Verdict


def decide(results: list[ClassifierResult]) -> tuple[Verdict, str]:
    """Pure function: classifier results in, verdict + human-readable rationale out.

    Kept pure and side-effect free on purpose so it's trivial to unit test.
    This is the fast, cheap "triage" step -- it never makes a network call,
    which is what lets clear allow/block/redact cases skip the (slower, LLM-
    backed) agent pipeline entirely. See app/agents/graph.py for what happens
    to the ESCALATE case this function produces.

    Priority order (top wins):
    1. PII detected -> REDACT (before injection check, because a credit card
       number can confuse the injection classifier into a false INJECTION at
       high confidence -- "4111 1111 1111 1111" looks like data exfiltration
       to a model trained on text attacks. Redacting is always correct for
       PII regardless of what other classifiers say.)
    2. All classifiers errored -> ESCALATE (fail closed)
    3. Max risk above block threshold -> BLOCK
    4. Max risk below allow threshold -> ALLOW
    5. Otherwise -> ESCALATE (ambiguous, routes to agent pipeline)
    """
    settings = get_settings()

    # 1. PII always wins -- redact before any other verdict.
    pii_result = next((r for r in results if r.name == "pii"), None)
    if pii_result and pii_result.score >= 1.0:
        return Verdict.REDACT, f"PII detected ({pii_result.label}); redacting before pass-through."

    scored = [r for r in results if r.error is None and r.name != "pii"]
    if not scored:
        return Verdict.ESCALATE, "All classifiers failed or timed out; escalating for human review."

    max_result = max(scored, key=lambda r: r.score)

    if max_result.score >= settings.block_threshold:
        return Verdict.BLOCK, f"{max_result.name} flagged risk={max_result.score:.2f} (label={max_result.label}), above block threshold."

    if max_result.score <= settings.allow_threshold:
        return Verdict.ALLOW, f"All classifiers below allow threshold (max={max_result.score:.2f} from {max_result.name})."

    return (
        Verdict.ESCALATE,
        f"{max_result.name} risk={max_result.score:.2f} is ambiguous "
        f"(between allow={settings.allow_threshold} and block={settings.block_threshold}); "
        "routing to the agent pipeline.",
    )
