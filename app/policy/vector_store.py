"""
Vector store abstraction for policy retrieval.

InMemoryVectorStore is what runs in dev, tests, and CI: it embeds the (small,
static) policy corpus once at startup and does cosine similarity search in
numpy. That's a legitimate choice at this scale -- a few dozen policy
documents easily fits in memory, and the interface below is what makes it a
one-class swap to PgVectorStore once the corpus grows past what you want to
re-embed on every process start, or once you need it shared across multiple
gateway instances.

PgVectorStore is written for that production case: policy docs live in
Postgres with a pgvector column, embedded once via a separate ingestion step,
queried with a standard ANN similarity query. It's included and documented
here but not exercised in this environment (no live Postgres instance) --
see docs/POLICY_LAYER.md for the schema and how to actually stand it up.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.policy.documents import PolicyDocument
from app.policy.embeddings import BaseEmbeddingProvider, HashingEmbeddingProvider, build_embedding_provider

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass
class RetrievalMatch:
    document: PolicyDocument
    similarity: float


class BaseVectorStore(ABC):
    @abstractmethod
    def query(self, text: str, k: int = 3) -> list[RetrievalMatch]:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    def __init__(
        self,
        documents: list[PolicyDocument],
        embedding_provider: BaseEmbeddingProvider | None = None,
    ):
        self.documents = documents
        self.embedding_provider = embedding_provider or build_embedding_provider()

        if isinstance(self.embedding_provider, HashingEmbeddingProvider):
            # Fit IDF from this corpus so common cross-document words (e.g.
            # "instructions") don't dominate over the words that actually
            # distinguish one policy from another. See embeddings.py docstring.
            self.embedding_provider.fit_idf([doc.text for doc in documents])

        # Embed the whole corpus once at construction time -- fine for a
        # corpus of this size; see module docstring for when to graduate
        # to PgVectorStore instead.
        self._doc_embeddings = np.stack(
            [self.embedding_provider.embed(doc.text) for doc in self.documents]
        )

    def query(self, text: str, k: int = 3) -> list[RetrievalMatch]:
        query_vec = self.embedding_provider.embed(text)
        # doc embeddings are already L2-normalized, so this dot product IS
        # cosine similarity.
        sims = self._doc_embeddings @ query_vec
        top_k_idx = np.argsort(-sims)[:k]
        return [
            RetrievalMatch(document=self.documents[i], similarity=float(sims[i]))
            for i in top_k_idx
        ]


class PgVectorStore(BaseVectorStore):
    """Production backend: Postgres + pgvector extension.

    NOT exercised in this environment (no live Postgres instance available
    here) -- included for completeness and to show the intended production
    path. See docs/POLICY_LAYER.md for the schema, ingestion script, and how
    to run it locally via `docker compose up` (the `policy-db` service in
    docker-compose.yml).
    """

    def __init__(self, dsn: str, embedding_provider: BaseEmbeddingProvider | None = None, table: str = "policy_documents"):
        self.dsn = dsn
        self.table = table
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            import psycopg2  # requires: pip install psycopg2-binary

            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def query(self, text: str, k: int = 3) -> list[RetrievalMatch]:
        query_vec = self.embedding_provider.embed(text)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, category, action, text, 1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (list(query_vec), list(query_vec), k),
            )
            rows = cur.fetchall()
        return [
            RetrievalMatch(
                document=PolicyDocument(id=r[0], category=r[1], action=r[2], text=r[3]),
                similarity=float(r[4]),
            )
            for r in rows
        ]
