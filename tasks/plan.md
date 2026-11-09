# Implementation Plan: PostgreSQL storage backend (phases 0–2)

Spec: [SPEC.md](../SPEC.md) · Design: [docs/postgresql_storage.md](../docs/postgresql_storage.md)

## Overview

Implement `JEV_MEM_STORAGE=postgres` as an alternative backend for Jev-Mem's
per-namespace stores: `PgVectorStore` (pgvector, exact L2 scan),
`PGTableGraphStorage` (ordinary tables — JSONB + B-tree indexes — with an
in-memory NetworkX cache and version/TTL invalidation), the `memory_keywords`
posting table, namespace management, and a lossless
`scripts/migrate_to_postgres.py`. The file layout stays the default and is
behaviorally untouched. Scope is phases 0–2 of the design document; phase 3
(CTE traversal pushdown) is a follow-up. Task details live in
[tasks/todo.md](todo.md).

## Architecture decisions

- **Driver**: psycopg 3, sync API, raw SQL, no ORM (per spec).
- **Connection pooling**: one in-process `psycopg.ConnectionPool(min=1,
  max=5)` in `memory/pg_common.py`, shared by all stores (resolved decision;
  PgBouncer is documented-only for multi-sidecar deployments).
- **Schema**: exactly as `docs/postgresql_storage.md` specifies; DDL lives
  only in `memory/pg_common.py`. `halfvec(JEV_MEM_PG_VECTOR_DIM)`, default
  2560, fail-fast on mismatch with the resolved embedding dimension.
- **No ANN index in this scope**: exact L2 scan only (recall-identical to
  `IndexFlatL2`); per-namespace HNSW/IVF is a future concern (~1e5+ vectors).
- **Graph cache**: in-memory NetworkX per namespace; every node/link mutation
  bumps `memory_namespaces.graph_updated_at` in the same transaction; cache
  compares the version on access (reload on mismatch, also reloads the
  keyword posting dict); TTL (e.g. 30 s) caps staleness on hot paths.
- **Keyword**: `memory_keywords` mirrors `keyword_index.json` 1:1; loaded
  back into the in-memory posting dict at namespace load; all query logic
  including partial matching stays in Python.
- **Protocols**: phase 0 extracts `VectorStore` / `GraphStore`; existing
  classes are renamed to `FileVectorStore` / `FileGraphStore` (refactor only,
  public API unchanged).
- **Fail-fast**: `JEV_MEM_STORAGE=postgres` with unset `JEV_MEM_PG_DSN`
  raises at import, matching `validate_embedding_backend` behavior.

## Task list (ordered; details in todo.md)

### Phase 0: Foundation
- Task 1: new env vars + fail-fast validation
- Task 2: extract `VectorStore` / `GraphStore` protocols; rename file backends

### Checkpoint: Foundation
- Full suite green; rename is refactor-only

### Phase 1: pgvector
- Task 3: `memory/pg_common.py` (DSN, shared pool, DDL, dimension validation)
- Task 4: `memory/pg_vector_store.py` (`PgVectorStore`, exact L2, halfvec)
- Task 5: vector parity tests (PG-gated)

### Checkpoint: Phase 1
- Parity tests pass against `pgvector/pgvector:pg18`; file mode unaffected

### Phase 2: graph, keyword, namespace, migration
- Task 6: `memory/pg_table_graph_store.py` (`PGTableGraphStorage` + cache invalidation)
- Task 7: keyword posting + `memory_builder` write/update/delete hooks via the protocols
- Task 8: namespace management (create / list / delete cascade, `last_used_at`; sidecar registry consults the table)
- Task 9: `scripts/migrate_to_postgres.py` + round-trip test
- Task 10: e2e `JevMemSystem` on PG (`JEV_MEM_MOCK=1`, all four views)
- Task 11: CI — ubuntu `pgvector/pgvector:pg18` service container + DSN env; macos job unchanged

### Checkpoint: Complete
- All SPEC.md success criteria; `scripts/check_public_release.py` clean

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `psycopg` is a new dependency | Med | Flagged in spec (ask-first); pure-Python wheel, Python 3.13 supported |
| fp16 `halfvec` vs fp32 distance parity | Med | Parity test compares top-k ids and uses a 1e-3 relative distance tolerance |
| Phase 0 renames break imports (core + sidecar) | Med | Grep all references; full suite at the foundation checkpoint |
| Cross-process cache invalidation races | Low | Version check + TTL; tested with two connections |
| Large file layout migrated in one transaction | Low | `executemany` batching inside the transaction; row-count verification |
| CI macos has no PostgreSQL | Low | PG-gated tests skip on unset DSN (existing `skipif` pattern) |

## Open questions

None — all resolved in SPEC.md (pooling, partial matching, `age` knob).
