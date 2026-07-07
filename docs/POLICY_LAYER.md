# Running the policy layer against real pgvector

The default `InMemoryVectorStore` (mock and real embedding modes both use it
by default) is fine for a corpus of a few dozen policy documents and a single
gateway process. This doc covers switching to `PgVectorStore` for a real
deployment: a shared, persistent corpus queried by multiple gateway
instances.

This has **not** been run against a live Postgres instance in the dev
environment this repo was built in (no local Postgres/pgvector available
there) -- the schema and code below are correct and ready to run, but treat
this as the next thing to validate against a real database before relying on
it in production.

## 1. Start Postgres with pgvector

```bash
docker compose up -d policy-db   # uncomment the policy-db service in docker-compose.yml first
```

## 2. Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE policy_documents (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    action TEXT NOT NULL,      -- one of: block, redact, escalate, allow
    text TEXT NOT NULL,
    embedding VECTOR(384)      -- 384 = all-MiniLM-L6-v2 dimensionality; match your embedding model
);

CREATE INDEX ON policy_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## 3. Ingestion script (run once, and again whenever the policy corpus changes)

```python
# scripts/ingest_policy_docs.py
import psycopg2
from app.policy.documents import POLICY_DOCUMENTS
from app.policy.embeddings import SentenceTransformerEmbeddingProvider

provider = SentenceTransformerEmbeddingProvider()
conn = psycopg2.connect("postgresql://cerberus:cerberus@localhost:5432/cerberus_policy")

with conn.cursor() as cur:
    for doc in POLICY_DOCUMENTS:
        embedding = provider.embed(doc.text)
        cur.execute(
            """
            INSERT INTO policy_documents (id, category, action, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                category = EXCLUDED.category,
                action = EXCLUDED.action,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """,
            (doc.id, doc.category, doc.action, doc.text, list(embedding)),
        )
conn.commit()
```

## 4. Switch the gateway to use it

```bash
export CERBERUS_MOCK_MODE=false
export CERBERUS_VECTOR_STORE_BACKEND=pgvector
export CERBERUS_DATABASE_URL=postgresql://cerberus:cerberus@localhost:5432/cerberus_policy
```

And in `app/policy/policy_layer.py`, `_build_default_vector_store()` already
branches on `CERBERUS_VECTOR_STORE_BACKEND` and constructs a `PgVectorStore`
when set to `pgvector` -- no code change needed. What *is* still needed
before this works end to end: running the ingestion script in step 3 against
your actual Postgres instance, and validating the round trip (this repo's dev
environment had no live Postgres available to test against).

## 5. Re-tune the similarity threshold

`CERBERUS_POLICY_SIMILARITY_THRESHOLD` was empirically tuned (0.15) against
the mock hashing+IDF embeddings in this repo. Real sentence-transformer
embeddings produce a meaningfully different similarity distribution (usually
higher baseline cosine similarity between semantically related sentences) --
re-run `python -m eval.run_eval` after switching and re-tune the threshold
against real numbers rather than reusing the mock-mode value.
