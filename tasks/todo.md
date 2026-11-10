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
- [x] `psycopg` added to `requirements.txt`
- [x] DDL executes cleanly on a stock `pgvector/pgvector:pg18` instance
- [x] Dimension mismatch raises a clear error before any data is written

**Verification:**
- [x] Unit: DDL string sanity + config wiring (no PG)
- [x] PG-gated smoke: connect + `initialize()` creates all 5 tables
      (docker jev-mem-pg, pgvector 0.8.7, `halfvec(1024)` confirmed,
      idempotent second call)

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
- [x] All protocol methods work against a live `pgvector/pgvector:pg18`
- [x] `storage=file` never imports psycopg at runtime (lazy import)

**Verification:**
- [x] PG-gated unit tests for add/search/delete/size
- [x] `.venv/bin/python -m pytest -q` (file-mode suite still green)

**Done (2026-11-09):** `PgVectorStore` in `memory/pg_vector_store.py` (exact L2
`<->` search with namespace filter, `ON CONFLICT` upserts, namespace-scoped
delete/clear, `save`/`load` no-ops). Wiring: `create_vector_db(backend="postgres")`
in `vector_db.py` (lazy import), `trg_memory.py` selects `PgVectorStore` when
`jev_config.storage == "postgres"`, `memory_builder.py` passes `jev_config`
down, `jev_mem_config.py` gained `namespace` (env `JEV_MEM_NAMESPACE`).
5 PG-gated tests pass with DSN (halfvec float16 tolerance 2e-3); file-mode
suite: 239 passed, 5 skipped.

**Dependencies:** Task 2, Task 3
**Files:** `memory/pg_vector_store.py` (new), `memory/vector_db.py` or `jev_mem/system.py` (factory), `tests/test_pg_storage.py`
**Scope:** M

### Task 5: vector parity tests

**Description:** PG-gated integration tests proving `PgVectorStore` matches
`FileVectorStore`: same top-k ids for the same queries, L2 distances within
1e-3 relative tolerance (fp16 vs fp32), and metadata round-trip.

**Acceptance criteria:**
- [x] Parity test inserts the same vectors into both backends and compares top-k ids + distances
- [x] Test skips (not fails) when `JEV_MEM_PG_DSN` is unset

**Verification:**
- [x] `docker run -d --name jev-mem-pg -e POSTGRES_PASSWORD=postgres -p 15432:5432 pgvector/pgvector:pg18` then run with `JEV_MEM_PG_DSN=postgresql://postgres:postgres@localhost:15432/jev_mem_test`

Done (2026-11-10): `test_pg_vector_parity_with_file_store` — 30 random vectors in both
backends, identical top-10 ids, similarity scores within abs 2e-3 (halfvec fp16 vs
FAISS flat fp32). Required a real fix in `pg_vector_store.py`: pgvector's `<->` is
Euclidean L2 while FAISS IndexFlatL2 reports **squared** L2, so PgVectorStore now
computes `1 / (1 + dist^2)` to match file-backend similarity exactly. Suite: 6 PG tests
pass with DSN; full suite 239 passed + 6 skipped without DSN.

**Dependencies:** Task 4
**Files:** `tests/test_pg_storage.py`, `memory/pg_vector_store.py`
**Scope:** S

### Checkpoint: Phase 1
- [x] Parity tests pass against `pgvector/pgvector:pg18`
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
- [x] Round-trip: upsert node/link → delete in-memory cache → load → identical object model
- [x] Two connections: a mutation on A invalidates B's cache (version check)
- [x] jsonb payload is the only thing read back into the object model

**Verification:**
- [x] PG-gated tests (round-trip, invalidation, cascade delete)

**Dependencies:** Task 2, Task 3
**Files:** `memory/pg_table_graph_store.py` (new), `memory/pg_common.py`, `tests/test_pg_storage.py`
**Scope:** M

Done (2026-11-10): `PGTableGraphStorage(GraphStore)` in memory/pg_table_graph_store.py
— full protocol (add_node/add_link/get_node/get_link/get_neighbors/update_node/
update_link/delete_node/delete_link/traverse/find_path/get_subgraph/
delete_namespace), write-through to memory_nodes/memory_links with jsonb canonical
payload + scalar denormalization, `graph_updated_at` version check + TTL invalidation
(check_interval throttle). Internal cache is a FileGraphStore (NetworkX) so retrieval
latency is unchanged; node dicts deserialize via the node_type dispatch (EPISODE/
SESSION/else EVENT). pg_common.py: extracted `graph_ddl()` from `namespace_ddl()`.
6 new PG-gated tests (round-trip, write-through, cross-process invalidation,
update, delete cascades); 12/12 pass with DSN, full suite 239 passed + 12 skipped
without DSN. Exported in memory/__init__.py.

### Task 7: keyword posting + `memory_builder` write-path hooks

**Description:** Route `memory_builder.py` writes/updates/deletes through the
protocols: keyword posting rows (`INSERT ... ON CONFLICT DO NOTHING`) in the
same transaction as the node write (mirroring `_index_text_basic`); on text
update, delete old terms + insert new; on node delete, rely on the FK cascade
for links/keywords plus the paired vector delete. The keyword posting dict is
reloaded with the graph cache.

**Acceptance criteria:**
- [x] Keyword posting parity: PG rows == `keyword_index.json` dict for the same writes (incl. length-1 words inside bigrams)
- [x] Update/delete paths remove stale terms/rows
- [x] File-mode write path unchanged (delegation only)

**Verification:**
- [x] PG-gated keyword parity test; full suite (file mode) green

**Dependencies:** Task 4, Task 6
**Files:** `memory/memory_builder.py`, `memory/pg_table_graph_store.py` (keyword table helpers), `tests/test_pg_storage.py`
**Scope:** M

Done (2026-11-10): `PGTableGraphStorage` gained `keyword_index` (loaded with the graph
cache in `_load_from_db`, reset in `delete_namespace`), `reload()`, and
`index_node_terms` / `reindex_node_terms` (executemany INSERT ON CONFLICT DO NOTHING;
reindex deletes the node's old rows first, same transaction; bumps
`graph_updated_at` so other processes reload the posting dict with the cache).
`MemoryBuilder`: `_index_text_basic` now also tracks per-node terms
(`self._node_terms`); `index_event` ends with `_flush_node_terms`, which persists them
through the store's `index_node_terms` when present (file mode: no such attribute,
pure no-op). `load()` copies the store's `keyword_index` into `self.node_index` in PG
mode, mirroring the file-mode keyword_index.json load. 3 new PG-gated tests
(roundtrip incl. multi-node posting, dedup+reindex, FK cascade on node delete); 15/15
PG tests pass with DSN; full file-mode suite 239 passed + 15 skipped.

### Task 8: namespace management

**Description:** Make `memory_namespaces` the registry: lazy create
(`INSERT ... ON CONFLICT DO NOTHING`), existence check, `list()` (new
capability), `DELETE` with cascade, `last_used_at` bumped on load. Update the
sidecar `NamespaceRegistry` to consult the table (not only its in-process
dict) in postgres mode, with `last_used_at`-based eviction/recovery.

**Acceptance criteria:**
- [x] create/list/delete work per namespace; delete cascades to all child rows
- [x] Sidecar on postgres mode recovers the namespace set across restarts
- [x] File mode: sidecar behavior unchanged

**Verification:**
- [x] PG-gated tests (registry + cascade); sidecar tests green

**Dependencies:** Task 6
**Files:** `memory/pg_table_graph_store.py` (or `pg_common.py`), `integrations/dify/service/app.py`, `tests/test_dify_sidecar.py`, `tests/test_pg_storage.py`
**Scope:** M

Done (2026-11-10): `pg_common` gained `namespace_create/exists/list/delete/
touch` (lazy `INSERT ... ON CONFLICT DO NOTHING`, list ordered by
last_used_at DESC, FK cascades on delete). `PGTableGraphStorage._load_from_db`
bumps `last_used_at` in the same connection. `TemporalResonanceGraphMemory`
selects `PGTableGraphStorage` when `jev_config.storage == "postgres"` (graph_db
wiring + vector store). Sidecar `NamespaceRegistry`: `_register_in_db` (lazy
create on first use), `_touch` (throttled last_usedAt bump per get), and
`names()` now merges the in-process dict with `namespace_list` in postgres
mode (restart recovery); `_create` sets the config's namespace to the sidecar
namespace via `dataclasses.replace` in postgres mode. New endpoint
`GET /v1/namespaces` (authenticated). 5 new tests (4 PG-gated + 1 sidecar
file-mode no-op + 1 PG restart-recovery, sidecar test PG-gated); full suite
240 passed + 20 skipped without DSN, 260 passed with DSN.

### Task 9: `scripts/migrate_to_postgres.py`

**Description:** One-shot import of a file layout into Postgres: read
`graph.json` / `index.faiss` + `metadata.json` / `keyword_index.json` /
`jev_mem_config.json` per namespace, insert all rows in a single transaction
with row-count verification and a summary report. `--source <cache_dir>`
required; refuses to overwrite non-empty namespaces without `--force`.

**Acceptance criteria:**
- [x] Migrating a real file layout reproduces identical row counts and data (row-level diff for a sample)
- [x] Re-running on a migrated namespace fails clearly without `--force`

**Verification:**
- [x] PG-gated round-trip test using a tmp file layout

**Dependencies:** Task 4, Task 6, Task 7
**Files:** `scripts/migrate_to_postgres.py` (new), `tests/test_pg_storage.py`
**Scope:** M

Done (2026-11-10): `scripts/migrate_to_postgres.py` — discovers namespaces
(base dir of namespace dirs, or a single namespace dir), reads graph.json /
vectors/metadata.json / keyword_index.json / jev_mem_config.json, imports nodes,
links, vectors, keyword postings and the resolved config in one transaction per
namespace, verifies DB row counts against the file counts, refuses a non-empty
namespace without `--force` (RuntimeError mentioning `--force`, exit 1), and
`--force` deletes the namespace (cascade) then re-imports. Dimension comes from
metadata.json with a clear error when the existing `halfvec(N)` differs
(`_check_existing_dimension`). 5 new PG tests (round-trip incl. row-level sample
diff, refuse-without-force, force-overwrite, discovery, fresh-namespace).
Real-layout check: copied `/data/docker-test` out of the jev-mem container and
migrated it into a scratch DB — 1 node / 1 vector / 20 postings, `halfvec(2560)`
created from the source, Chinese narrative intact; `--force` re-run kept the
same counts. Suite: 24 passed in tests/test_pg_storage.py with DSN; full suite
240 passed + 25 skipped without DSN.

Post-review fix (2026-11-10): row-count verification moved INSIDE the
transaction (a mismatch, e.g. a duplicate node_id deduped by ON CONFLICT, now
rolls back instead of leaving partial data committed — regression test added);
malformed source JSON and a non-integer JEV_MEM_PG_VECTOR_DIM now report a
clean MigrationError instead of a traceback. 26 passed with DSN.

### Task 10: e2e `JevMemSystem` on Postgres

**Description:** PG-gated end-to-end test: full `JevMemSystem` with
`JEV_MEM_STORAGE=postgres` and `JEV_MEM_MOCK=1` (no model calls), running
ingest → query across all four retrieval views → consolidate, asserting the
answers/citations come from Postgres-backed stores.

**Acceptance criteria:**
- [x] Ingest stores (graph + vectors + keywords in PG); query returns generated answers; stats reflect PG rows
- [x] Same assertions pass in file mode (test is backend-agnostic)

**Verification:**
- [x] Run with PG container; run without (file mode) — both green

**Dependencies:** Task 4, Task 6, Task 7, Task 8
**Files:** `tests/test_pg_storage.py`, `tests/test_jev_mem.py`, `memory/pg_table_graph_store.py`
**Scope:** S

Done (2026-11-10): `test_pg_end_to_end_system` (PG-gated, in tests/test_pg_storage.py)
builds a fully offline `JevMemSystem` with `JEV_MEM_STORAGE=postgres` + mock mode
(PgVectorStore + PGTableGraphStorage + a dimension-matched deterministic encoder):
ingest 2 observations → admitted 2 / rejected 0, PG rows populated
(nodes>=2, links>=1, vectors>=2, keywords>=2), all four retrieval-view queries
return non-empty answers, `consolidate` returns decisions, and a second fresh
system instance on the same namespace reloads from PG and queries again
(durability across processes). The same flow runs in file mode via
`test_end_to_end_memory_flow_file_mode` (tests/test_jev_mem.py, always runs,
FileVectorStore + FileGraphStore). Found + fixed one real bug: many consumers
access `graph_db.nodes` / `graph_db.links` instance dicts directly, so
`PGTableGraphStorage` gained `nodes` / `links` collection-view properties
(live cache mappings after `_ensure_loaded()`) plus `export_to_json` delegation.
Suite: 241 passed + 26 skipped without DSN; 267 passed with DSN.

### Task 11: CI — ubuntu pgvector service container

**Description:** Add a `pgvector/pgvector:pg18` service to the ubuntu job in
`.github/workflows/ci.yml` (port 15432, `POSTGRES_PASSWORD=postgres`), export
`JEV_MEM_PG_DSN` for that job so PG-gated tests execute; macos job unchanged
(PG tests skip there). Keep `scripts/check_public_release.py` green.

**Acceptance criteria:**
- [x] GitHub ubuntu run: PG tests execute (not skipped) and pass
- [x] No secrets in the workflow or diff

**Verification:**
- [x] `scripts/check_public_release.py`
- [x] CI run (push or `act`-style local check)

**Dependencies:** Tasks 5, 10
**Files:** `.github/workflows/ci.yml`
**Scope:** S

Done (2026-11-10): split the single matrix job into `test-ubuntu` (with a
`pgvector/pgvector:pg18` service on port 15432 + health check + a
`CREATE DATABASE jev_mem_test` step + job-level `JEV_MEM_PG_DSN`) and
`test-macos` (unchanged, PG tests skip). GitHub macOS runners do not support
`services:`, so a matrix-level split was required. Verified locally: YAML
parses, `check_public_release.py --tracked` clean (129 files), and all 25 PG
tests pass against a freshly created empty database (cold-start simulation of
the CI path).

### Checkpoint: Complete
- [ ] All SPEC.md success criteria met
- [ ] `scripts/check_public_release.py` clean
- [ ] Review with human; commit (file mode default untouched throughout)
