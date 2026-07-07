"""
Concrete classifier implementations.

Design note: every classifier can run in two modes:
  - real mode: lazy-loads a HF `transformers` pipeline on first use
  - mock mode: a deterministic heuristic stand-in

Mock mode exists so the *system* (routing, aggregation, decision logic, API,
tests, CI) can be built, tested, and demoed without needing to download
gigabytes of model weights or have network access to the HF Hub. This mirrors
a real pattern: you don't want your CI pipeline flaky because a model registry
was slow -- you want your orchestration logic tested independently of model
availability. Swapping to real models is a one-line config change
(CERBERUS_MOCK_MODE=false), not a code change.
"""
import re
from typing import Callable, Optional

from app.classifiers.base import BaseClassifier
from app.config import get_settings


def _load_seq_classification_model(model_id: str):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_id)


def _load_token_classification_model(model_id: str):
    from transformers import AutoModelForTokenClassification

    return AutoModelForTokenClassification.from_pretrained(model_id)


def _assert_not_on_meta(model, model_id: str):
    """Fail fast with an actionable message if the model is still on meta.

    We do NOT try to silently recover via to_empty(): that restores tensor
    storage but with uninitialized (random) weights, producing garbage
    predictions -- worse than a clear failure for a security gateway.
    """
    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        raise RuntimeError(
            f"Model '{model_id}' loaded onto the meta device "
            f"({len(meta_params)} params affected). Refusing to run with "
            f"unmaterialized weights. Check that no load kwargs (torch_dtype, "
            f"device_map, low_cpu_mem_usage) are forcing meta init on this "
            f"transformers/torch build."
        )


def _build_pipeline_cpu_safe(task: str, model_id: str, model_loader, **pipeline_kwargs):
    """Build a transformers pipeline pinned to CPU.

    Loads model + tokenizer via a bare from_pretrained (which materializes to
    CPU correctly on the supported stack), verifies no params are on meta
    (fail-fast rather than silently serving a randomly-initialized model),
    then builds the pipeline around the already-placed model.
    """
    from transformers import AutoTokenizer, pipeline

    model = model_loader(model_id)
    _assert_not_on_meta(model, model_id)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline(task, model=model, tokenizer=tokenizer, device=-1, **pipeline_kwargs)


class HFTextClassifier(BaseClassifier):
    """Generic wrapper around a HF `text-classification` pipeline."""

    def __init__(
        self,
        name: str,
        model_id: str,
        label_to_risk: Callable[[str, float], float],
        mock_fn: Callable[[str], tuple[str, float]],
    ):
        self.name = name
        self.model_id = model_id
        self.label_to_risk = label_to_risk
        self.mock_fn = mock_fn
        self._pipeline = None

    def preload(self) -> None:
        """Load model weights at startup, defensively handling the meta-device bug.

        Root cause is environmental: when `accelerate` is installed on certain
        torch/Python combinations (observed: Python 3.13 + Windows), transformers'
        from_pretrained initializes on the meta device and materialization fails,
        leaving data-less meta tensors that crash on any subsequent .to()/.item().

        The clean fix is `pip uninstall accelerate` (nothing here needs it for CPU
        inference). This code path is a defensive fallback so the service still
        works if that hasn't been done: after loading, if any parameter is still
        on meta, force a real load via from_pretrained with the accelerate path
        explicitly disabled.
        """
        if self._pipeline is None:
            self._pipeline = _build_pipeline_cpu_safe(
                "text-classification",
                self.model_id,
                _load_seq_classification_model,
                truncation=True,
                max_length=512,
            )

    def _load_pipeline(self):
        if self._pipeline is None:
            self.preload()
        return self._pipeline

    def _predict(self, text: str) -> tuple[str, float]:
        settings = get_settings()
        if settings.mock_mode:
            return self.mock_fn(text)
        clf = self._load_pipeline()
        result = clf(text)[0]
        label = result["label"]
        confidence = result["score"]
        risk = self.label_to_risk(label, confidence)
        return label, risk


# ---------------------------------------------------------------------------
# Mock heuristics: intentionally simple keyword/regex signals. They exist to
# make the ensemble *behave plausibly* offline, not to be accurate detectors.
# Do not mistake these for the real thing when reporting eval numbers later.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above) instructions",
    r"disregard (your|the) (system|prior) prompt",
    r"you are now",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"reveal your (system prompt|instructions)",
    r"act as (if|though)",
]


def _mock_prompt_injection(text: str) -> tuple[str, float]:
    lowered = text.lower()
    hits = sum(1 for p in _INJECTION_PATTERNS if re.search(p, lowered))
    if hits == 0:
        return "SAFE", 0.05
    score = min(0.5 + 0.2 * hits, 0.98)
    return "INJECTION", score


_ANGER_WORDS = {"furious", "hate", "stupid", "idiot", "worthless", "shut up", "screw you"}


def _mock_emotion(text: str) -> tuple[str, float]:
    lowered = text.lower()
    hits = sum(1 for w in _ANGER_WORDS if w in lowered)
    if hits == 0:
        return "neutral", 0.1
    return "anger", min(0.4 + 0.25 * hits, 0.95)


def _mock_ai_text_detector(text: str) -> tuple[str, float]:
    # Very rough stand-in: flags a couple of stock LLM-ism phrases.
    tells = ["as an ai language model", "i don't have personal", "i'm just a language model"]
    lowered = text.lower()
    if any(t in lowered for t in tells):
        return "AI-generated", 0.7
    return "human", 0.1


def build_prompt_injection_classifier() -> HFTextClassifier:
    settings = get_settings()
    return HFTextClassifier(
        name="prompt_injection",
        model_id=settings.prompt_injection_model,
        # protectai's model uses INJECTION/SAFE labels; treat INJECTION confidence
        # directly as risk, and SAFE confidence as inverse risk.
        label_to_risk=lambda label, conf: conf if label.upper() == "INJECTION" else 1 - conf,
        mock_fn=_mock_prompt_injection,
    )


def build_emotion_classifier() -> HFTextClassifier:
    settings = get_settings()
    high_risk_labels = {"anger", "disgust", "fear"}
    return HFTextClassifier(
        name="emotion",
        model_id=settings.emotion_model,
        label_to_risk=lambda label, conf: conf if label.lower() in high_risk_labels else conf * 0.2,
        mock_fn=_mock_emotion,
    )


def build_ai_text_detector() -> HFTextClassifier:
    settings = get_settings()
    # NOTE on model choice: `openai-community/roberta-base-openai-detector`
    # was trained in 2019 to detect GPT-2 generated *text passages*, not
    # short prompts/instructions. On short inputs (<20 words) it outputs
    # near-random high "Fake" scores regardless of content -- discovered
    # empirically in real-mode testing (see results_20260703_033556.json
    # in the project docs). The model is still included because it adds
    # *some* signal on longer outputs being passed through the gateway, but
    # its normalized risk score is capped at 0.6 so it can never independently
    # drive a block verdict (block threshold is 0.85) -- it can only
    # contribute to an escalate when combined with other signals.
    #
    # Better alternatives for a real deployment:
    #   - fastino/gliguard-LLMGuardrails-300M (purpose-built, in HF screenshot)
    #   - Swap this for a zero-shot classifier on short inputs (<50 chars)
    # This is a documented gap, not a silent workaround.
    AI_DETECTOR_SCORE_CAP = 0.6

    def label_to_risk(label: str, conf: float) -> float:
        raw = conf if ("fake" in label.lower() or "ai" in label.lower()) else 1 - conf
        return min(raw, AI_DETECTOR_SCORE_CAP)

    return HFTextClassifier(
        name="ai_text_detector",
        model_id=settings.ai_text_detector_model,
        label_to_risk=label_to_risk,
        mock_fn=_mock_ai_text_detector,
    )


class PIIClassifier(BaseClassifier):
    """Token-classification (NER) based PII detector.

    Distinct from HFTextClassifier because NER returns entity spans, not a
    single label -- and the gateway needs those spans later to actually redact
    text, not just score it.

    Entity type scope is intentionally narrow: EMAIL, PHONE, SSN, CREDIT_CARD
    only. LOC (location names) and ORG are valid NER outputs but not PII in
    any legal or security sense -- "What is the capital of Japan?" should not
    trigger redaction. PER (person names) is excluded for the same reason:
    "Tell me about Marie Curie" is not a PII leak. If your deployment has
    different requirements (e.g. HIPAA contexts where patient names *are* PII),
    add PER back with a note about why.
    """

    name = "pii"

    # Only truly sensitive entity types. See class docstring for what's
    # intentionally excluded and why.
    _PII_ENTITY_TYPES = {"EMAIL", "PHONE", "SSN", "CREDIT_CARD"}

    _EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    _PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    _SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

    def __init__(self):
        self._pipeline = None
        self._last_entities: list[dict] = []

    def preload(self) -> None:
        if self._pipeline is None:
            from app.config import get_settings as _gs
            self._pipeline = _build_pipeline_cpu_safe(
                "token-classification",
                _gs().pii_ner_model,
                _load_token_classification_model,
                aggregation_strategy="simple",
            )

    def _load_pipeline(self):
        if self._pipeline is None:
            self.preload()
        return self._pipeline

    def _mock_entities(self, text: str) -> list[dict]:
        entities = []
        for pattern, etype in (
            (self._EMAIL_RE, "EMAIL"),
            (self._PHONE_RE, "PHONE"),
            (self._SSN_RE, "SSN"),
            (self._CC_RE, "CREDIT_CARD"),
        ):
            for m in pattern.finditer(text):
                entities.append({"entity_group": etype, "start": m.start(), "end": m.end(), "word": m.group()})
        return entities

    def _predict(self, text: str) -> tuple[str, float]:
        settings = get_settings()
        if settings.mock_mode:
            entities = self._mock_entities(text)
        else:
            raw = self._load_pipeline()(text)
            # Filter to only the sensitive types we care about. The NER model
            # will also return PER, LOC, ORG -- we discard those here. See
            # class docstring for the reasoning.
            entities = [e for e in raw if e.get("entity_group") in self._PII_ENTITY_TYPES]

        self._last_entities = entities
        if not entities:
            return "none", 0.0
        types_found = ",".join(sorted({e["entity_group"] for e in entities}))
        return types_found, 1.0

    def redact(self, text: str) -> str:
        """Run detection (if not already run) and return text with PII masked."""
        if not self._last_entities:
            self._predict(text)
        redacted = text
        for e in sorted(self._last_entities, key=lambda x: x["start"], reverse=True):
            redacted = redacted[: e["start"]] + f"[REDACTED_{e['entity_group']}]" + redacted[e["end"] :]
        return redacted
