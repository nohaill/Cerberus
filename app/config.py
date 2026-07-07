"""
Central configuration for Cerberus.

Everything that might change between local dev, CI, and production lives here,
loaded from environment variables so no code changes are needed to swap models,
flip mock mode, or retune thresholds.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CERBERUS_")

    # --- Mode ---
    # MOCK_MODE=true runs deterministic heuristic stand-ins instead of downloading
    # real HF models. This keeps local dev, unit tests, and CI fast and offline-safe.
    # Flip to false (and have network access) to run the real ensemble.
    mock_mode: bool = True

    # --- Model selection (HF Hub repo ids) ---
    prompt_injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    ai_text_detector_model: str = "openai-community/roberta-base-openai-detector"
    pii_ner_model: str = "dslim/bert-base-NER"

    # --- Decision thresholds ---
    # Score above this -> hard block, no agent/RAG step needed.
    block_threshold: float = 0.85
    # Score below this -> hard allow.
    allow_threshold: float = 0.35
    # Anything in between is "ambiguous" and gets routed to the policy/RAG step
    # once that's wired up (Week 5-6 of the build plan).

    # --- Service ---
    max_concurrent_classifiers: int = 8
    # 30s covers CPU inference on first request after model warm-up.
    # For GPU or ONNX-optimized models this can be reduced significantly.
    request_timeout_seconds: float = 30.0

    # --- Policy / RAG layer ---
    policy_enabled: bool = True
    # HashingEmbeddingProvider dimensionality (mock mode only).
    embedding_dim: int = 512
    # Real embedding model used when mock_mode=false.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Minimum cosine similarity for the top policy match to be trusted.
    # Below this, an ambiguous case stays ESCALATE rather than guessing.
    # Tuned empirically against the mock hashing embeddings -- if you swap in
    # real sentence-transformer embeddings, re-tune this (real embeddings
    # produce a different similarity distribution than bag-of-words hashing).
    policy_similarity_threshold: float = 0.15
    # "memory" (default, in-process) or "pgvector" (production; requires
    # CERBERUS_DATABASE_URL to be set).
    vector_store_backend: str = "memory"
    database_url: str | None = None

    # --- Agent pipeline (Phase 4) ---
    # Model used by the LLM decision node. Only relevant when mock_mode=false
    # and ANTHROPIC_API_KEY is set -- otherwise MockLLMClient is used instead.
    agent_model: str = "claude-sonnet-4-6"


@lru_cache
def get_settings() -> Settings:
    return Settings()
