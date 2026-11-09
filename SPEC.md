# Spec: PostgreSQL storage backend (`JEV_MEM_STORAGE=postgres`)

Design source of truth: `docs/postgresql_storage.md` (revised). This spec
covers **phases 0-2** of that document. Phase 3 (traversal pushdown) is a
follow-up and out of scope.

## Objective

Move Jev-Mem's per-namespace state (vector index, graph, keyword index,
resolved config) from file persistence to PostgreSQL behind a new storage
backend, so multiple processes/containers can serve the same namespaces from
one shared authority.

Consumers: the core `JevMemSystem` API (unchanged) and the Dify sidecar
(8-namespace LRU registry, which gains table-backed namespace management).

The file layout remains the default and stays fully supported; nothing in
this change is required for a file-based deployment.

Acceptance (summary — see Success Criteria):

1. `JEV_MEM_STORAGE=file` (default): zero behavior change; full existing
   suite green.
2. `JEV_MEM_STORAGE=postgres` + valid DSN: ingest / query / consolidate all
   work end to end against a live PostgreSQL with pgvector.
3. `scripts/migrate_to_postgres.py` imports an existing file layout into
   Postgres losslessly (row-count verified).

## Tech Stack

- Python 3.13, synchronous code only (no async, no new web/ORM frameworks).
- **psycopg 3** (sync API) — raw SQL per the design document; no ORM layer.
- PostgreSQL 14+ with pgvector >= 0.7 (`halfvec` column type).
- Local/CI database image: `pgvector/pgvector:pg18`.
- Embedding dimension is deployment-fixed and **parameterized**:
  `JEV_MEM_PG_VECTOR_DIM` (int, default `2560`) sets the `halfvec(N)` column
  at `initialize()`; a dimension mismatch with the active embedding model
  fails fast at import (same style as `validate_embedding_backend`).
- New env vars (all optional, file mode needs none):
  `JEV_MEM_STORAGE`, `JEV_MEM_PG_DSN`, `JEV_MEM_PG_VECTOR_DIM`
  (plus `JEV_MEM_GRAPH_STORAGE=table|age`, table default; `age` is
  documented but not implemented in phases 0-2).

## Commands

```text
# Full test suite (file mode; PG-gated tests auto-skip without a DSN)
.venv/bin/pytest tests/ -q

# Local PG for integration work
docker run -d --name jev-mem-pg -e POSTGRES_PASSWORD=jev -p 5432:5432 pgvector/pgvector:pg18
JEV_MEM_PG_DSN=postgresql://jev:jev@localhost:5432/jev_mem .venv/bin/pytest tests/test_pg_storage.py -q

# Migration from a file layout (mock models, offline)
JEV_MEM_MOCK=1 JEV_MEM_STORAGE=postgres JEV_MEM_PG_DSN=postgresql://jev:jev@localhost:5432/jev_mem \
  .venv/bin/python scripts/migrate_to_postgres.py --source <cache_dir>/<namespace>

# CI (unchanged entry points; the ubuntu job gains a pgvector service container)
.venv/bin/python -m build
.venv/bin/python scripts/check_public_release.py
```

## Project Structure

New files (flat, matching the existing `memory/` module style):

```
memory/pg_common.py           -> DSN/env resolution, connection helper,
                                 halfvec dimension validation, shared DDL
memory/pg_vector_store.py     -> PgVectorStore (exact L2 scan, no ANN index)
memory/pg_table_graph_store.py -> PGTableGraphStorage + in-memory cache with
                                 graph_updated_at version check + TTL
scripts/migrate_to_postgres.py -> file layout -> Postgres, one transaction,
                                 row-count verification
tests/test_pg_storage.py      -> PG-gated integration tests (skip without DSN)
SPEC.md                       -> this file
```

Modified in place:

```
memory/vector_db.py     -> VectorDatabase renamed to FileVectorStore (Phase 0)
memory/graph_db.py      -> concrete class renamed to FileGraphStore (Phase 0)
memory/memory_builder.py -> write/delete hooks delegate through VectorStore /
                           GraphStore protocols; keyword posting written through
memory/jev_mem_config.py -> JEV_MEM_STORAGE / JEV_MEM_PG_DSN / JEV_MEM_PG_VECTOR_DIM
.github/workflows/ci.yml -> ubuntu job: pgvector/pgvector:pg18 service container
                           + JEV_MEM_PG_DSN; macos job unchanged (PG tests skip)
```

Public API (`JevMemSystem`, sidecar endpoints) is unchanged.

## Code Style

Follow the existing codebase: dataclasses + `Protocol` type hints,
snake_case, one class per concern, module-level constants for DDL,
context-managed connections, fail-fast validation at import for
misconfiguration (pattern: `validate_embedding_backend`).

Example — psycopg 3 usage shape:

```python
def upsert_vector(self, namespace_id: str, vector_id: str,
                  vector: np.ndarray, metadata: dict) -> None:
    payload = json.dumps(metadata, ensure_ascii=False)
    with self._connect() as conn, conn.transaction():
        conn.execute(
            "INSERT INTO memory_vectors (namespace_id, vector_id, embedding, payload) "
            "VALUES (%s, %s, %s::halfvec, %s::jsonb) "
            "ON CONFLICT (namespace_id, vector_id) DO UPDATE "
            "SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
            (namespace_id, vector_id, np.asarray(vector, dtype=np.float16).tobytes(), payload),
        )
```

Conventions:

- One `memory_namespaces` row per namespace, created lazily
  (`INSERT ... ON CONFLICT DO NOTHING`), never deleted except via the
  namespace-delete path.
- Every node/link mutation bumps `memory_namespaces.graph_updated_at` in the
  same transaction (cache-inversion contract from the design doc).
- JSONB payloads are canonical; scalar columns are denormalizations written
  from the same payload, never read back independently.

## Testing Strategy

Framework: pytest (existing). Three layers:

1. **Unit (no PG, always run):** protocol extraction (Phase 0 renames),
   env/config parsing incl. dimension validation, DDL string sanity,
   migration-script argument handling.
2. **Integration (PG-gated, `pytest.mark` skipif on unset `JEV_MEM_PG_DSN`):**
   - `PgVectorStore`: add/search/delete/size parity with `FileVectorStore` —
     identical top-k ids, L2 scores within 1e-3 relative (fp16 `halfvec`
     vs float32 FAISS).
   - `PGTableGraphStorage`: upsert/load round trip, cascade delete,
     cross-process invalidation (two connections, bump + version check).
   - Keyword store: posting rows equal the output of `_index_text_basic` for
     the same input (word + bigram rules, incl. length-1 words in bigrams).
   - End-to-end: `JevMemSystem` on a PG backend with `JEV_MEM_MOCK=1` —
     ingest, query (all 4 views exercised), consolidate.
   - Migration: import a file fixture, verify node/link/vector/keyword/config
     row counts, then run the e2e flow against the imported data.
3. **CI:** ubuntu job runs a `pgvector/pgvector:pg18` service container and
   sets `JEV_MEM_PG_DSN`, so layer 2 always runs there; the macos job keeps
   skipping the PG-gated tests (file-mode coverage only).

The 233 existing tests must stay green untouched (file mode is the default).

## Boundaries

- **Always:**
  - Run the full suite (file mode) before committing.
  - Keep `JEV_MEM_STORAGE=file` as the default and behaviorally untouched.
  - Write DDL/constants in `memory/pg_common.py` only; no DDL strings
    scattered across stores.
  - Keep the schema exactly as `docs/postgresql_storage.md` specifies
    (incl. `graph_updated_at` / `last_used_at` contract).
  - Type-hint all new public methods; fail fast on config errors.
- **Ask first:**
  - Adding `psycopg` to `requirements.txt` (it is a new dependency).
  - Any CI workflow change beyond the documented ubuntu service container.
  - Any schema divergence from the design document.
  - Changing the `JEV_MEM_PG_VECTOR_DIM` default (2560).
- **Never:**
  - Commit a DSN, password, or real `.env` content.
  - Require PostgreSQL when `JEV_MEM_STORAGE=file` (the default).
  - Delete or rename the existing file formats (`graph.json`, `index.faiss`,
    `keyword_index.json`) — they remain the file backend and the export
    format.
  - Disable or remove failing tests to get the suite green.
  - Add an ANN index (HNSW/IVFFlat) in phases 0-2 — exact scan only.

## Success Criteria

- [ ] `JEV_MEM_STORAGE=file` default: full existing suite green, no behavior
      diff (the rename commits are refactors only).
- [ ] Phase 0: `FileVectorStore` / `FileGraphStore` behind the
      `VectorStore` / `GraphStore` protocols; sidecar and core tests green.
- [ ] Phase 1: `PgVectorStore` passes the vector parity tests against
      `pgvector/pgvector:pg18`; `halfvec(JEV_MEM_PG_VECTOR_DIM)` created at
      initialize; dimension mismatch fails fast.
- [ ] Phase 2: graph + keyword + config in Postgres; namespace management
      (create/list/delete cascade, `last_used_at`) works; file layout can be
      retired for migrated namespaces; keyword posting dict parity holds.
- [ ] `scripts/migrate_to_postgres.py` round-trips a real file layout with
      row-count verification, in one transaction.
- [ ] CI ubuntu job: PG-gated tests execute (not skipped) and pass.
- [ ] `scripts/check_public_release.py` clean (no secrets in the diff).

## Open Questions

None — all resolved (see below).

Resolved decisions:

- Connection pooling: one in-process `psycopg.ConnectionPool` (min=1,
  max=5) created in `pg_common.py` and shared by all stores; PgBouncer is a
  documented deployment option for multiple sidecars only, not built.
- Keyword partial matching stays in Python over the in-memory posting dict
  for phases 0-2 (behavior-identical migration); the pg_trgm GIN variant is
  a documented future option only.
- `JEV_MEM_GRAPH_STORAGE=age` (PGGraphStorage): the knob is preserved and
  documented, but not implemented in this scope.
