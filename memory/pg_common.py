"""Shared PostgreSQL plumbing for the Postgres storage backends.

Single home for the process-wide connection pool, the schema DDL (mirrored
from docs/postgresql_storage.md), and the small validation helpers shared by
``PgVectorStore`` and ``PGTableGraphStorage``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np
from psycopg_pool import ConnectionPool

_POOL_MIN = 1
_POOL_MAX = 5

_pool: ConnectionPool | None = None
_pool_dsn: str = ""
_pool_lock = threading.Lock()


def resolve_dsn(pg_dsn: str) -> str:
    """Return the stripped DSN, failing fast when it is unset.

    Mirrors the ``JevMemConfig`` validation: Postgres storage is unusable
    without a DSN, so raise here instead of deferring the error to the first
    query.
    """
    dsn = (pg_dsn or "").strip()
    if not dsn:
        raise ValueError(
            "pg_dsn is required when storage is postgres (set JEV_MEM_PG_DSN)"
        )
    return dsn


def get_pool(pg_dsn: str) -> ConnectionPool:
    """Process-wide connection pool shared by all Postgres stores.

    psycopg connections are not thread-safe and the sidecar runs sync code in
    a thread pool, so every store in the process shares one pool (min=1,
    max=5 per the spec). PgBouncer is only relevant for multi-sidecar
    deployments and is not built here.
    """
    global _pool, _pool_dsn
    dsn = resolve_dsn(pg_dsn)
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                dsn, min_size=_POOL_MIN, max_size=_POOL_MAX,
                open=True, timeout=10.0,
            )
            _pool_dsn = dsn
        elif dsn != _pool_dsn:
            raise ValueError(
                "connection pool already created with a different DSN; "
                "one process serves one database"
            )
        return _pool


def reset_pool() -> None:
    """Close the shared pool (test support)."""
    global _pool, _pool_dsn
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_dsn = ""


@contextmanager
def connection(pg_dsn: str):
    """Yield a pooled connection; the caller manages the transaction.

    psycopg 3 auto-begins a transaction on the first statement; use
    ``conn.transaction()`` for an explicit commit/rollback block.
    """
    with get_pool(pg_dsn).connection() as conn:
        yield conn


def graph_ddl() -> list[str]:
    """The graph/keyword part of the schema (namespaces, nodes, links,
    keywords) in dependency order. Used by ``PGTableGraphStorage``, which
    must not create the vector table (it does not know the dimension)."""
    return [
        """
        CREATE TABLE IF NOT EXISTS memory_namespaces (
          id          text PRIMARY KEY,
          created_at  timestamptz NOT NULL DEFAULT now(),
          config      jsonb NOT NULL DEFAULT '{}'::jsonb,
          graph_updated_at timestamptz NOT NULL DEFAULT now(),
          last_used_at   timestamptz NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_nodes (
          namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
          node_id      text NOT NULL,
          node_type    text NOT NULL,
          node         jsonb NOT NULL,
          created_at   timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (namespace_id, node_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_links (
          namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
          link_id      text NOT NULL,
          source_id    text NOT NULL,
          target_id    text NOT NULL,
          link_type    text NOT NULL,
          link         jsonb NOT NULL,
          PRIMARY KEY (namespace_id, link_id),
          FOREIGN KEY (namespace_id, source_id) REFERENCES memory_nodes ON DELETE CASCADE,
          FOREIGN KEY (namespace_id, target_id) REFERENCES memory_nodes ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS memory_links_src "
        "ON memory_links (namespace_id, source_id)",
        "CREATE INDEX IF NOT EXISTS memory_links_dst "
        "ON memory_links (namespace_id, target_id)",
        "CREATE INDEX IF NOT EXISTS memory_links_type "
        "ON memory_links (namespace_id, link_type)",
        """
        CREATE TABLE IF NOT EXISTS memory_keywords (
          namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
          term         text NOT NULL,
          node_id      text NOT NULL,
          PRIMARY KEY (namespace_id, term, node_id),
          FOREIGN KEY (namespace_id, node_id) REFERENCES memory_nodes ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS memory_keywords_term "
        "ON memory_keywords (namespace_id, term)",
    ]


def namespace_ddl(dimension: int) -> list[str]:
    """The five-table schema in dependency order (design doc: "Schema").

    All statements use ``IF NOT EXISTS`` so ``initialize()`` is idempotent.
    The vector column is ``halfvec(dimension)`` per constraint 1 of the
    design doc; deployments keeping plain ``vector`` columns must diverge
    here and in the search cast.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("vector dimension must be a positive integer")
    return graph_ddl() + [
        f"""
        CREATE TABLE IF NOT EXISTS memory_vectors (
          namespace_id text NOT NULL REFERENCES memory_namespaces(id) ON DELETE CASCADE,
          vector_id    text NOT NULL,
          embedding    halfvec({dimension}) NOT NULL,
          payload      jsonb NOT NULL DEFAULT '{{}}'::jsonb,
          PRIMARY KEY (namespace_id, vector_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS memory_vectors_ns "
        "ON memory_vectors (namespace_id)",
    ]


def create_schema(pg_dsn: str, dimension: int) -> None:
    """Create the pgvector extension and all tables if missing.

    Idempotent; runs in one transaction so a partial failure rolls back.
    """
    statements = ["CREATE EXTENSION IF NOT EXISTS vector"]
    statements.extend(namespace_ddl(dimension))
    with connection(pg_dsn) as conn:
        with conn.transaction():
            for statement in statements:
                conn.execute(statement)


def validate_vector_dimension(vector, expected: int) -> None:
    """Fail fast when a vector's dimension differs from the configured one."""
    actual = int(np.asarray(vector).ravel().shape[0])
    if actual != expected:
        raise ValueError(
            f"vector dimension {actual} does not match the configured "
            f"dimension {expected} (JEV_MEM_PG_VECTOR_DIM)"
        )


# --- namespace registry (design doc: "Namespace management") -----------------

def namespace_create(pg_dsn: str, namespace: str) -> bool:
    """Lazily register a namespace; True when a new row was created."""
    with connection(pg_dsn) as conn:
        cur = conn.execute(
            "INSERT INTO memory_namespaces (id) VALUES (%s) "
            "ON CONFLICT DO NOTHING",
            (namespace,),
        )
        return cur.rowcount == 1


def namespace_exists(pg_dsn: str, namespace: str) -> bool:
    with connection(pg_dsn) as conn:
        row = conn.execute(
            "SELECT 1 FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchone()
    return row is not None


def namespace_list(pg_dsn: str) -> list[str]:
    """Every registered namespace, most recently used first."""
    with connection(pg_dsn) as conn:
        rows = conn.execute(
            "SELECT id FROM memory_namespaces ORDER BY last_used_at DESC, id"
        ).fetchall()
    return [row[0] for row in rows]


def namespace_delete(pg_dsn: str, namespace: str) -> bool:
    """Delete a namespace; child rows go with it (ON DELETE CASCADE)."""
    with connection(pg_dsn) as conn:
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM memory_namespaces WHERE id = %s", (namespace,)
            )
        return cur.rowcount == 1


def namespace_touch(pg_dsn: str, namespace: str) -> None:
    """Mark a namespace as used now (cross-process LRU ordering)."""
    with connection(pg_dsn) as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE memory_namespaces SET last_used_at = now() WHERE id = %s",
                (namespace,),
            )
