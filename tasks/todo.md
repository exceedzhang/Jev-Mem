# Tasks: PostgreSQL storage backend (phases 0–2)

Plan: [plan.md](plan.md) · Spec: [SPEC.md](../SPEC.md)

## Phase 0: Foundation

### Task 1: config env vars + fail-fast validation

**Description:** Add `JEV_MEM_STORAGE` (file|postgres, default file),
`JEV_MEM_PG_DSN`, `JEV_MEM_PG_VECTOR_DIM` (default 2560), and
`JEV_MEM_GRAPH_STORAGE` (table|age, age reserved/unimplemented) to
`JevMemConfig`. With `storage=postgres` and unset DSN, fail fast at import,
matching the `validate_embedding_backend` style. Document the vars in
`.env.example` and the sidecar README env table.

**Acceptance criteria:**
- [x] All four vars load from env with the documented defaults
- [x] `storage=postgres` + no DSN → clear error naming the missing var
- [x] `storage=file` (default) behavior completely unchanged

**Verification:**
- [x] `.venv/bin/python -m pytest -q` (full suite green — 236 passed)
- [x] New/updated config unit tests pass

**Dependencies:** None
**Files:** `memory/jev_mem_config.py`, `tests/` (config tests), `.env.example`, `integrations/dify/service/README.md`
**Scope:** S

### Task 2: extract VectorStore / GraphStore protocols; rename file backends

**Description:** Extract the `VectorStore` / `GraphStore` protocols (surfaces
per the design doc: the real `VectorStore` method set, and the six-method
`GraphStore` sketch) so `vector_db.py` / `graph_db.py` and the new PG
implementations share them. Rename `VectorDBInterface`→`VectorStore`,
`FAISSVectorDB`→`FileVectorStore`, `GraphDBInterface`→`GraphStore`, and
`NetworkXGraphDB`→`FileGraphStore`, updating all references (core + Dify
sidecar + tests). Pure refactor, no behavior change.

**Acceptance criteria:**
- [x] Both protocols exist and are importable; file backends satisfy them
- [x] All old-name references updated (grep shows no stragglers)
- [x] Public behavior unchanged — full suite green

**Verification:**
- [x] `.venv/bin/python -m pytest -q` (236 passed)
- [x] `grep -rn "VectorDBInterface|FAISSVectorDB|GraphDBInterface|NetworkXGraphDB"` shows no hits

**Dependencies:** Task 1
**Files:** `memory/vector_db.py`, `memory/graph_db.py`, plus every importer (`memory/memory_builder.py`, `memory/query_engine.py`, `jev_mem/system.py`, `integrations/dify/service/app.py`, `tests/*`)
**Scope:** M

### Checkpoint: Foundation
- [x] Full suite green, no behavior diff (236 passed)
- [ ] Review with human before proceeding

## Phase 1: pgvector

### Task 3: `memory/pg_common.py`

**Description:** Shared Postgres plumbing: DSN resolution + validation,
module-level `psycopg.ConnectionPool(min=1, max=5)` with lazy connect, the
full DDL from the design doc (all tables/indexes/FKs, parameterized
`halfvec(JEV_MEM_PG_VECTOR_DIM)`), dimension validation (fail fast if the
table dimension disagrees with the resolved embedding dimension), and the
`halfvec` cast helper.

**Acceptance criteria:**
- [ ] `psycopg` added to `requirements.txt`
- [ ] DDL executes cleanly on a stock `pgvector/pgvector:pg18` instance
- [ ] Dimension mismatch raises a clear error before any data is written

**Verification:**
- [ ] Unit: DDL string sanity + config wiring (no PG)
- [ ] PG-gated smoke: connect + `initialize()` creates all 5 tables

**Dependencies:** Task 1
**Files:** `memory/pg_common.py` (new), `requirements.txt`, `tests/test_pg_storage.py` (new)
**Scope:** M

### Task 4: `memory/pg_vector_store.py` — `PgVectorStore`

**Description:** Implement the `VectorStore` protocol over Postgres: exact L2
search (`ORDER BY embedding <-> $1::halfvec`, `WHERE namespace_id=$2`, no ANN
index), add/update/delete with `ON CONFLICT`, `size()`, `clear()`. `load()` /
`save()` are no-ops (search goes straight to SQL). Wire backend selection
(`JEV_MEM_STORAGE=postgres`) at the factory site used by `JevMemSystem`.

**Acceptance criteria:**
- [ ] All protocol methods work against a live `pgvector/pgvector:pg18`
- [ ] `storage=file` never imports psycopg at runtime (lazy import)

**Verification:**
- [ ] PG-gated unit tests for add/search/delete/size
- [ ] `.venv/bin/python -m pytest -q` (file-mode suite still green)

**Dependencies:** Task 2, Task 3
**Files:** `memory/pg_vector_store.py` (new), `memory/vector_db.py` or `jev_mem/system.py` (factory), `tests/test_pg_storage.py`
**Scope:** M

### Task 5: vector parity tests

**Description:** PG-gated integration tests proving `PgVectorStore` matches
`FileVectorStore`: same top-k ids for the same queries, L2 distances within
1e-3 relative tolerance (fp16 vs fp32), and metadata round-trip.

**Acceptance criteria:**
- [ ] Parity test inserts the same vectors into both backends and compares top-k ids + distances
- [ ] Test skips (not fails) when `JEV_MEM_PG_DSN` is unset

**Verification:**
- [ ] `docker run -d --name jev-mem-pg -e POSTGRES_PASSWORD=postgres -p 15432:5432 pgvector/pgvector:pg18` then run with `JEV_MEM_PG_DSN=postgresql://postgres:postgres@localhost:15432/jev_mem_test`

**Dependencies:** Task 4
**Files:** `tests/test_pg_storage.py`
**Scope:** S

### Checkpoint: Phase 1
- [ ] Parity tests pass against `pgvector/pgvector:pg18`
- [ ] File mode fully unaffected; review with human before proceeding

## Phase 2: graph, keyword, namespace, migration

### Task 6: `memory/pg_table_graph_store.py` — `PGTableGraphStorage`

**Description:** Implement the `GraphStore` protocol: `initialize()` (DDL),
`load(namespace_id)` (node/link rows → in-memory NetworkX, with
`graph_updated_at` version check + TTL), `upsert_node` / `upsert_link`
(jsonb canonical; bump `graph_updated_at` in the same transaction),
`delete_namespace`, `flush`. Node jsonb deserializes via the
`node_type` dispatch (as `import_from_json` does today).

**Acceptance criteria:**
- [ ] Round-trip: upsert node/link → delete in-memory cache → load → identical object model
- [ ] Two connections: a mutation on A invalidates B's cache (version check)
- [ ] jsonb payload is the only thing read back into the object model

**Verification:**
- [ ] PG-gated tests (round-trip, invalidation, cascade delete)

**Dependencies:** Task 2, Task 3
**Files:** `memory/pg_table_graph_store.py` (new), `tests/test_pg_storage.py`
**Scope:** M

### Task 7: keyword posting + `memory_builder` write-path hooks

**Description:** Route `memory_builder.py` writes/updates/deletes through the
protocols: keyword posting rows (`INSERT ... ON CONFLICT DO NOTHING`) in the
same transaction as the node write (mirroring `_index_text_basic`); on text
update, delete old terms + insert new; on node delete, rely on the FK cascade
for links/keywords plus the paired vector delete. The keyword posting dict is
reloaded with the graph cache.

**Acceptance criteria:**
- [ ] Keyword posting parity: PG rows == `keyword_index.json` dict for the same writes (incl. length-1 words inside bigrams)
- [ ] Update/delete paths remove stale terms/rows
- [ ] File-mode write path unchanged (delegation only)

**Verification:**
- [ ] PG-gated keyword parity test; full suite (file mode) green

**Dependencies:** Task 4, Task 6
**Files:** `memory/memory_builder.py`, `memory/pg_table_graph_store.py` (keyword table helpers), `tests/test_pg_storage.py`
**Scope:** M

### Task 8: namespace management

**Description:** Make `memory_namespaces` the registry: lazy create
(`INSERT ... ON CONFLICT DO NOTHING`), existence check, `list()` (new
capability), `DELETE` with cascade, `last_used_at` bumped on load. Update the
sidecar `NamespaceRegistry` to consult the table (not only its in-process
dict) in postgres mode, with `last_used_at`-based eviction/recovery.

**Acceptance criteria:**
- [ ] create/list/delete work per namespace; delete cascades to all child rows
- [ ] Sidecar on postgres mode recovers the namespace set across restarts
- [ ] File mode: sidecar behavior unchanged

**Verification:**
- [ ] PG-gated tests (registry + cascade); sidecar tests green

**Dependencies:** Task 6
**Files:** `memory/pg_table_graph_store.py` (or `pg_common.py`), `integrations/dify/service/app.py`, `tests/test_dify_sidecar.py`, `tests/test_pg_storage.py`
**Scope:** M

### Task 9: `scripts/migrate_to_postgres.py`

**Description:** One-shot import of a file layout into Postgres: read
`graph.json` / `index.faiss` + `metadata.json` / `keyword_index.json` /
`jev_mem_config.json` per namespace, insert all rows in a single transaction
with row-count verification and a summary report. `--source <cache_dir>`
required; refuses to overwrite non-empty namespaces without `--force`.

**Acceptance criteria:**
- [ ] Migrating a real file layout reproduces identical row counts and data (row-level diff for a sample)
- [ ] Re-running on a migrated namespace fails clearly without `--force`

**Verification:**
- [ ] PG-gated round-trip test using a tmp file layout

**Dependencies:** Task 4, Task 6, Task 7
**Files:** `scripts/migrate_to_postgres.py` (new), `tests/test_pg_storage.py`
**Scope:** M

### Task 10: e2e `JevMemSystem` on Postgres

**Description:** PG-gated end-to-end test: full `JevMemSystem` with
`JEV_MEM_STORAGE=postgres` and `JEV_MEM_MOCK=1` (no model calls), running
ingest → query across all four retrieval views → consolidate, asserting the
answers/citations come from Postgres-backed stores.

**Acceptance criteria:**
- [ ] Ingest stores (graph + vectors + keywords in PG); query returns generated answers; stats reflect PG rows
- [ ] Same assertions pass in file mode (test is backend-agnostic)

**Verification:**
- [ ] Run with PG container; run without (file mode) — both green

**Dependencies:** Task 4, Task 6, Task 7, Task 8
**Files:** `tests/test_pg_storage.py`
**Scope:** S

### Task 11: CI — ubuntu pgvector service container

**Description:** Add a `pgvector/pgvector:pg18` service to the ubuntu job in
`.github/workflows/ci.yml` (port 15432, `POSTGRES_PASSWORD=postgres`), export
`JEV_MEM_PG_DSN` for that job so PG-gated tests execute; macos job unchanged
(PG tests skip there). Keep `scripts/check_public_release.py` green.

**Acceptance criteria:**
- [ ] GitHub ubuntu run: PG tests execute (not skipped) and pass
- [ ] No secrets in the workflow or diff

**Verification:**
- [ ] `scripts/check_public_release.py`
- [ ] CI run (push or `act`-style local check)

**Dependencies:** Tasks 5, 10
**Files:** `.github/workflows/ci.yml`
**Scope:** S

### Checkpoint: Complete
- [ ] All SPEC.md success criteria met
- [ ] `scripts/check_public_release.py` clean
- [ ] Review with human; commit (file mode default untouched throughout)
