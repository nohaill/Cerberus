"""
Embedding providers for the policy retrieval step.

Same pattern as app/classifiers/hf_classifier.py: an abstract interface with
a real backend (sentence-transformers, lazy-loaded, needs network) and a mock
backend that runs fully offline so the retrieval pipeline can be built and
tested without downloading model weights.

The mock backend is a feature-hashing bag-of-words vectorizer (hash each
token into one of N buckets, count, L2-normalize) -- this is a real, if
crude, retrieval technique (see sklearn's HashingVectorizer), not a fake
stand-in. It approximates lexical/word-overlap similarity, which is enough to
validate the retrieval -> policy-mapping -> decision pipeline end to end. It
will NOT catch true semantic paraphrases with no shared vocabulary -- that's
exactly the gap real sentence embeddings close, and is worth calling out
explicitly rather than overselling the mock's quality.
"""
import hashlib
import re
from abc import ABC, abstractmethod

from app.config import get_settings

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str):
        """Return a 1D numpy array embedding for `text`."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]):
        return [self.embed(t) for t in texts]


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Offline, dependency-free bag-of-words hashing vectorizer.

    Supports optional IDF weighting (`fit_idf`). Without it, raw term
    frequency alone tends to let common words shared across many corpus
    documents (e.g. "instructions" appearing in half the policy docs here)
    dominate the similarity score and drown out the words that actually
    distinguish one policy from another. This was found empirically: an
    early version of this class without IDF weighting caused several
    prompt-injection cases to mismatch to an unrelated "creative writing is
    fine" policy doc purely because both texts contained the word
    "instructions". IDF weighting is the standard fix.
    """

    def __init__(self, dim: int | None = None, idf: "np.ndarray | None" = None):
        settings = get_settings()
        self.dim = dim or settings.embedding_dim
        self.idf = idf  # set via fit_idf() once a corpus is known

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _raw_counts(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in self._tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        return vec

    def fit_idf(self, corpus_texts: list[str]) -> None:
        """Compute per-bucket IDF weights from a corpus and store them on
        this provider so subsequent embed() calls apply them."""
        n_docs = len(corpus_texts)
        doc_counts = np.zeros(self.dim, dtype=np.float64)
        for text in corpus_texts:
            raw = self._raw_counts(text)
            doc_counts += (raw > 0).astype(np.float64)
        # smoothed IDF, standard formula: log((N+1)/(df+1)) + 1
        self.idf = np.log((n_docs + 1) / (doc_counts + 1)) + 1.0

    def embed(self, text: str):
        vec = self._raw_counts(text)
        if self.idf is not None:
            vec = vec * self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Real semantic embeddings via sentence-transformers. Requires network
    access to the HF Hub on first use and the `sentence-transformers` package.
    """

    def __init__(self, model_id: str | None = None):
        settings = get_settings()
        self.model_id = model_id or settings.embedding_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def embed(self, text: str):
        model = self._load()
        return model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]):
        model = self._load()
        return list(model.encode(texts, normalize_embeddings=True))


def build_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    if settings.mock_mode:
        return HashingEmbeddingProvider()
    return SentenceTransformerEmbeddingProvider()
