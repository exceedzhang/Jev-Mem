# PostgreSQL storage (design, not yet implemented)

This document is a design for moving Jev-Mem's per-namespace state — the FAISS
vector index, the NetworkX graph, the keyword index, and the resolved config —
from file persistence to PostgreSQL. It records the constraints, the schema,
the storage abstractions, and a phased migration plan. Nothing in this
document is implemented; the current file-based stores remain the default and
the only supported path.

## Why consider it

Current storage layout per namespace under `JEV_MEM_CACHE_DIR`:

```text
<namespace>/
  graph.json              # NetworkX export (nodes + links)
  vectors/index.faiss     # FAISS binary index (default: IndexFlatL2)
  vectors/metadata.json   # per-vector metadata, parallel to index rows
  keyword_index.json      # keyword -> node id posting lists
  jev_mem_config.json     # per-namespace resolved config
```

This is correct for a single process on one machine. It breaks down when:

- More than one process (or container) serves the same namespace — each keeps
  its own in-memory copy and its own file copy, with no shared authority.
- The deployment already runs a PostgreSQL instance for other services and you
  want one backup/restore surface instead of named Docker volumes.
- Tenancy needs hard SQL-level isolation and auditability.

If you run one container and a handful of namespaces, the file layout is
cheaper and faster; keep it.

## Hard constraints

1. **Embedding dimension.** pgvector can index (HNSW/IVFFlat) `vector`
   columns up to **2000 dimensions**. The recommended OpenAI-compatible
   deployment uses `Qwen/Qwen3-Embedding-4B`, which is **2560 dimensions**.
   Two workable resolutions:
   - Store embeddings as `halfvec` (float16). pgvector >= 0.7 indexes halfvec
     up to 4000 dimensions; quality loss from fp16 for retrieval ranking is
     negligible, storage halves.
   - Keep `vector` and take the exact scan (no ANN index), equivalent to the
     current default `flat` index at this data scale.
2. **One dimension per table.** A pgvector column has a fixed dimension, so
   one deployment can hold one embedding dimension per vector table. Mixing
   embedding models across namespaces in a single database requires either
   separate tables per dimension or a standardization on one model.
3. **Traversal cost.** Query-time beam traversal issues many small graph
   lookups per candidate. In-memory NetworkX calls are nanosecond; per-call
   SQL turns one query into hundreds of round trips. The design therefore
   keeps an in-memory graph cache (write-through to Postgres, with the
   version/TTL invalidation described under "Cache consistency across
   processes") rather than translating every BFS step into SQL.

## Schema

All tables are namespaced by a `namespace_id` column; there is one row-level
isolation boundary, no per-tenant schemas.

```sql
CREATE TABLE memory_namespaces (
  id          text PRIMARY KEY,
  created_at  timestamptz NOT NULL DEFAULT now(),
  config      jsonb NOT NULL DEFAULT '{}'::jsonb,   -- resolved per-ns config
  graph_updated_at timestamptz NOT NULL DEFAULT now()  -- bumped on every node/link mutation
  last_used_at   timestamptz NOT NULL DEFAULT now()   -- bumped on namespace load (cross-process LRU)
);

CREATE TABLE memory_nodes (
  namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
  node_id      text NOT NULL,
  node_type    text NOT NULL,                       -- EVENT|EPISODE|NARRATIVE|ENTITY|SESSION
  node         jsonb NOT NULL,                      -- serialization of the concrete node class
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (namespace_id, node_id)
);

CREATE TABLE memory_links (
  namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
  link_id      text NOT NULL,
  source_id    text NOT NULL,
  target_id    text NOT NULL,
  link_type    text NOT NULL,                       -- TEMPORAL|SEMANTIC|CAUSAL|ENTITY (LinkType)
  link         jsonb NOT NULL,                      -- full Link serialization, incl. properties
  PRIMARY KEY (namespace_id, link_id),
  FOREIGN KEY (namespace_id, source_id) REFERENCES memory_nodes ON DELETE CASCADE,
  FOREIGN KEY (namespace_id, target_id) REFERENCES memory_nodes ON DELETE CASCADE
);
CREATE INDEX memory_links_src  ON memory_links (namespace_id, source_id);
CREATE INDEX memory_links_dst  ON memory_links (namespace_id, target_id);
CREATE INDEX memory_links_type ON memory_links (namespace_id, link_type);

CREATE TABLE memory_keywords (
  namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
  term         text NOT NULL,                      -- single word or bigram
  node_id      text NOT NULL,
  PRIMARY KEY (namespace_id, term, node_id),
  FOREIGN KEY (namespace_id, node_id) REFERENCES memory_nodes ON DELETE CASCADE
);
CREATE INDEX memory_keywords_term ON memory_keywords (namespace_id, term);

CREATE TABLE memory_vectors (
  namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
  vector_id    text NOT NULL,
  embedding    halfvec(2560) NOT NULL,              -- dimension is deployment-fixed
  payload      jsonb NOT NULL,                      -- VectorEntry metadata
  PRIMARY KEY (namespace_id, vector_id)
);
CREATE INDEX memory_vectors_ns ON memory_vectors (namespace_id);
```

Notes:

- `link_type` is indexed so view-scoped scans (e.g. causal links only) stay
  cheap. The four retrieval views are *filter policies* over `link_type` plus
  the per-link subtype (`link.properties.sub_type`, e.g. `SUCCEEDS` /
  `PRECEDES` for temporal chains, `LEADS_TO` / `ENABLES` for causal paths) —
  they are not one-to-one column values. See `query_engine.py`
  (`prefer_link_types` per intent) and `graph_db.py:get_neighbors`.
- The `jsonb` payloads (`node`, `link`, `payload`) are canonical. The scalar
  columns (`node_type`, `source_id`, `target_id`, `link_type`) are
  query-optimization denormalizations, written from the same payload in the
  same transaction; the payload is the only thing ever read back into the
  object model.
- `node jsonb` holds the serialization of the concrete node class
  (`EventNode` / `EpisodeNode` / `SessionNode` — there is no generic `Node`
  object); `node_type` selects the class on load, exactly as
  `graph_db.py:import_from_json` dispatches today.
- No ANN index on `memory_vectors` in phase 1: search is an exact L2 scan
  (see below). A global HNSW/IVFFlat index cannot be combined with a
  namespace filter — ANN selects the top-k candidates *before* non-indexed
  predicates are applied, so a global index can silently return zero rows for
  the requested namespace. ANN becomes a per-namespace partitioning decision
  only at ~1e5+ vectors per namespace.
- `memory_keywords` mirrors the `keyword_index.json` posting structure
  one-to-one and migrates in phase 2 (see the keyword store section below).
- Deletes cascade namespace-wide, which is the same guarantee the current
  "delete the namespace directory" semantics give. Per-node deletion (the
  current `delete_node` / `delete_vector` pairing): links and keyword rows
  fall via the foreign keys; the node's vector is removed with a paired
  statement in the same transaction.

## Vector store

`FileVectorStore` (the FAISS file backend in memory/vector_db.py) becomes a
`PgVectorStore` behind the
same public surface:

```python
class VectorStore(Protocol):
    def add_vector(self, vector_id: str, vector: np.ndarray,
                   metadata: dict) -> None: ...
    def add_vectors(self, items: list[tuple[str, np.ndarray, dict]]) -> int: ...
    def search(self, query_vector: np.ndarray, k: int = 10,
               filter_metadata: dict | None = None) -> list[tuple[str, float, dict]]: ...
    def get_vector(self, vector_id: str) -> ...: ...
    def update_vector(self, vector_id: str, vector: np.ndarray,
                      metadata: dict) -> None: ...
    def delete_vector(self, vector_id: str) -> bool: ...
    def size(self) -> int: ...
    def clear(self) -> bool: ...
    def save(self, path: str | None = None) -> None: ...
    def load(self, path: str | None = None) -> None: ...
```

For `PgVectorStore`, `load()` and `save()` are no-ops: search goes straight
to SQL and there is no in-memory index to load (unlike the FAISS file path,
where `load()` reads the index into memory).

Search is one statement; L2 order is preserved to match the FAISS default:

```sql
SELECT v.vector_id, v.payload,
       v.embedding <-> $1::halfvec AS dist
FROM memory_vectors v
WHERE v.namespace_id = $2
ORDER BY dist
LIMIT $3;
```

The `flat` (current default) → exact-scan mapping holds from day one: with
no ANN index the statement above is the whole search path and its recall is
identical to `IndexFlatL2`. The `ivf`/`hnsw` FAISS options do **not** map to a
global index — they become a per-namespace ANN index only after a namespace
passes ~1e5 vectors and a filtered ANN strategy (partial index or
`namespace_id` partitioning) is in place; below that, exact scan is both
simpler and faster than maintaining an index. The DDL above adopts the
`halfvec` option from constraint 1; a deployment that instead keeps plain
`vector` columns must change the column type and the `$1::halfvec` cast in
the search statement accordingly.

## Graph storage implementations

Two graph storage implementations are specified here. **For new PostgreSQL deployments,
`PGTableGraphStorage` is the recommended `GRAPH_STORAGE` implementation and
supersedes `PGGraphStorage`.**

### PGTableGraphStorage (recommended)

Keeps the entity-relation graph in ordinary tables — JSONB properties plus
B-tree indexes — instead of going through Apache AGE. Two practical
advantages:

- **No extension to install.** `PGGraphStorage` requires the Apache AGE
  extension, which most managed PostgreSQL services (Amazon RDS, Cloud SQL,
  Supabase, Neon) do not offer — so the graph layer frequently could not run
  on the same database as the vector, keyword, and scalar stores.
  `PGTableGraphStorage` runs on any stock PostgreSQL 14+ and creates the
  tables it needs during `initialize()`.
- **One Docker image.** For a container deployment the official
  `pgvector/pgvector:pg18` image is sufficient; the AGE-bundled
  `gzdaniel/postgres-for-rag:pg18-age-pgvector` image is only needed by
  `PGGraphStorage`.

Persistence model: the `memory_nodes` / `memory_links` rows above, with the
full serialization of the concrete node/link classes in `jsonb` columns
(canonical; scalar columns are denormalizations) and B-tree indexes on the
hot lookup keys (namespace, source, target, link_type).

The interface both implementations expose (sketch):

```python
class GraphStore(Protocol):
    def initialize(self) -> None: ...           # create tables if missing
    def load(self, namespace_id: str) -> None: ...    # version/TTL check
    def upsert_node(self, namespace_id: str, node: dict) -> None: ...
    def upsert_link(self, namespace_id: str, link: dict) -> None: ...
    def delete_namespace(self, namespace_id: str) -> None: ...
    def flush(self) -> None: ...
```

In-memory cache. Beam traversal issues many small graph lookups per
candidate; the namespace graph is therefore loaded into NetworkX on first
access and mutations write through to Postgres (one round trip per namespace
load, zero per traversal).

Cache consistency across processes. Write-through gives durability, not
freshness: a second process holding its own cache copy does not see the
first process's writes. Invalidation design:

- Every node/link mutation bumps `memory_namespaces.graph_updated_at` in the
  same transaction.
- Each cache entry records the `graph_updated_at` seen at load; on access,
  one cheap `SELECT graph_updated_at` compares and, on mismatch, reloads
  both the graph and the keyword posting dict (keyword writes share the
  node-write transaction, so no separate version is needed).
- A TTL (e.g. 30 s) caps staleness when the check is skipped for
  latency-sensitive hot paths.
- Optional upgrades: `LISTEN/NOTIFY` for push invalidation, or a documented
  bounded staleness window per deployment.

Net consistency guarantee: no weaker than today's single-process file
semantics; multi-process reads see other processes' writes within the TTL.

Traversal in SQL (when the graph is consumed as data, e.g. reporting).
Causal-path example (mirrors `graph_db.py:find_causal_paths` — `CAUSAL`
links with subtype `LEADS_TO`/`ENABLES`, directed, depth-bounded):

```sql
WITH RECURSIVE reach AS (
  SELECT target_id, ARRAY[source_id] AS path, 1 AS depth
  FROM memory_links
  WHERE namespace_id = $1
    AND source_id = $2
    AND link_type = 'CAUSAL'
    AND link->'properties'->>'sub_type' IN ('LEADS_TO', 'ENABLES')
  UNION
  SELECT l.target_id, r.path || l.target_id, r.depth + 1
  FROM memory_links l
  JOIN reach r ON l.source_id = r.target_id
  WHERE r.depth < $3            -- pass max_depth + 1 (strict bound)
    AND l.namespace_id = $1
    AND l.link_type = 'CAUSAL'
    AND l.link->'properties'->>'sub_type' IN ('LEADS_TO', 'ENABLES')
)
SELECT target_id, path FROM reach;
```

The depth bound terminates recursion; pass `max_depth + 1` as $3 so the CTE
admits the same number of edges as the in-memory DFS (which stops at
`depth >= max_depth`). The DFS also keeps a per-path cycle guard; reporting
queries needing cycle-free paths can add `path`-based pruning.

### PGGraphStorage (Apache AGE, superseded)

openCypher queries over PG. Only relevant for deployments that already run
AGE, or as a reference if a future workload needs unbounded traversal or
graph algorithms (neither fits in bounded-depth SQL). It requires a
custom/AGE-bundled image, which is why it is superseded for new deployments.

## Keyword store

The keyword index migrates with the rest: the `memory_keywords` posting
table mirrors today's `keyword_index.json` structure one-to-one — one row per
(term, node) pair, where a term is a single word (indexed only with length
>= 2 after punctuation stripping) or an adjacent-pair bigram (always
indexed), produced by the same Python tokenization (lowercase, whitespace
split).

- Write: `INSERT ... ON CONFLICT DO NOTHING` per term in the same
  transaction as the node write (mirrors the `_index_text_basic` loop).
- Read: at namespace load the posting table is read back into the same
  in-memory dict the query engine already uses, so the query path — exact
  term lookup, bigram lookup, and the partial/substring match loop — is
  unchanged; partial matching stays in Python rather than in SQL.
- Update: when a node's text changes, its old terms are deleted and the new
  terms inserted in the same transaction.
- Delete: nothing explicit — the foreign key to `memory_nodes` cascades on
  node removal; the namespace cascade covers the rest.

Row counts are large (tens of terms per memory) but ordinary for Postgres at
the ~1e5-memory scale of this design.

Alternatives considered: a per-namespace JSONB blob (minimal code change, but
every write rewrites the whole blob and concurrent writers last-write-wins),
and a `tsvector` column (fast and built in, but it changes tokenization
semantics — the default parser does not segment CJK — which would change
recall, so it is not a drop-in replacement for the current index).

## Namespace management

The file layout has no namespace registry — a namespace exists if its
directory exists, and the sidecar's LRU bound (8 live namespaces) lives in
process memory and is lost on restart. In PostgreSQL, `memory_namespaces`
is the registry:

- Create: `INSERT ... ON CONFLICT DO NOTHING` on first write (lazy, as
  today).
- Existence / load: `SELECT ... WHERE id = $1`. The sidecar's registry
  `get()` should consult the table, not only its in-process dict.
- List: `SELECT id FROM memory_namespaces` — impossible with the file
  layout; a new capability for sidecar administration.
- LRU / restart recovery: bump `last_used_at` on namespace load; eviction
  and restart recovery work off the table instead of process memory.
- Delete: `DELETE FROM memory_namespaces WHERE id = $1` cascades to all
  child rows (today's "delete the namespace directory" semantics).

## Configuration

```text
JEV_MEM_STORAGE=file        # default; also: postgres
JEV_MEM_PG_DSN=postgresql://user:pass@host:5432/jev_mem
JEV_MEM_GRAPH_STORAGE=table # default; age is a reserved knob (not implemented)
```

With `JEV_MEM_STORAGE=postgres`, the graph component uses
`PGTableGraphStorage` (the `table` option). `JEV_MEM_GRAPH_STORAGE=age`
(`PGGraphStorage`) is a reserved, documented knob and is not implemented in
this scope.

`JEV_MEM_STORAGE=postgres` with an unset DSN fails fast at import, matching
the existing `validate_embedding_backend` behavior. The DSN never appears in
checked-in files; it rides in the sidecar `.env` exactly like the current
model endpoints.

## Migration plan

Phase 0 — interfaces. Extract `VectorStore`/`GraphStore` protocols from the
file implementations; rename the existing classes to `FileVectorStore` /
`FileGraphStore`. No behavior change; the full test suite stays green.

Phase 1 — pgvector. Implement `PgVectorStore` (exact L2 scan, no ANN index —
see the vector-index note); `docker run pgvector/pgvector:pg18` integration
test gated on `JEV_MEM_PG_DSN`. Import existing `index.faiss` +
`metadata.json` with `scripts/migrate_to_postgres.py` (one transaction,
row-count verification).

Phase 2 — graph and keywords, `PGTableGraphStorage`. Add node/link upserts
(each bumping `memory_namespaces.graph_updated_at`); the per-namespace
resolved config (`jev_mem_config.json`) moves to `memory_namespaces.config`
in the same import. The keyword index moves to `memory_keywords`
with the same writes (see the keyword store section). Namespace load reads from Postgres and the
in-memory caches are write-through with the version check + TTL invalidation
described above. The file layout is retired entirely; a final
`graph.json` / `index.faiss` / `keyword_index.json` export remains
available, not the source.

Phase 3 — traversal pushdown, only if needed. Rewrite the traversal entry
points to recursive CTEs (see the causal-path example), re-verify depth
budgets and latency budgets end to end (the `CallBudget` in `jev_client.py`
already caps model calls; add a round-trip budget here).

Rough effort: phase 0 ~half a day, phase 1 ~1 day, phase 2 ~1 day, phase 3
~2-3 days.

## Open questions

- Embedding dimension policy: standardize the deployment on one model, or
  allow per-namespace models with a per-dimension vector table?
- Connection pooling: a small pool in-process is fine for one sidecar;
  multiple sidecars should sit behind PgBouncer (transaction mode).
- Retention: the current store has no tombstones; Postgres adds the option of
  `updated_at`-based GC but phase 1 should not add behavior.
