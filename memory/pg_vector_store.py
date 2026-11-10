"""Postgres implementation of the VectorStore protocol.

Exact L2 search (``ORDER BY embedding <-> $1::halfvec``) with a namespace
filter and no ANN index, per the design doc: a global ANN index cannot be
combined with the namespace predicate, so exact scan is the phase-1 path
(identical recall to the file backend's default flat index).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import pg_common
from .vector_db import VectorEntry, VectorStore


def _parse_halfvec(value: Any) -> np.ndarray:
    """Parse a pgvector halfvec column value into a float32 array.

    psycopg has no built-in pgvector adapter, so the column arrives as the
    text form ``[0.1,0.2,...]`` unless a value is returned as a list.
    """
    if isinstance(value, str):
        return np.fromstring(value.strip().lstrip("[").rstrip("]"),
                             sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32).ravel()


class PgVectorStore(VectorStore):
    """Namespace-scoped pgvector store backed by ``memory_vectors``.

    ``save`` / ``load`` are no-ops: the rows in Postgres are the storage, so
    there is nothing to write or read back (unlike the FAISS file path).
    """

    def __init__(self, dimension: int, dsn: str, namespace: str = "default",
                 persist_path: Optional[str] = None):
        self.dimension = int(dimension)
        self.dsn = pg_common.resolve_dsn(dsn)
        self.namespace = namespace
        self.persist_path = persist_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if missing and upsert this namespace's row."""
        pg_common.create_schema(self.dsn, self.dimension)
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO memory_namespaces (id) VALUES (%s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (self.namespace,),
                )

    @staticmethod
    def _vec(vector, expected_dimension: int) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).ravel()
        pg_common.validate_vector_dimension(arr, expected_dimension)
        return arr

    def add_vector(self, vector_id: str, vector: np.ndarray,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a vector; returns False when the id already exists."""
        arr = self._vec(vector, self.dimension)
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                cur = conn.execute(
                    "INSERT INTO memory_vectors "
                    "(namespace_id, vector_id, embedding, payload) "
                    "VALUES (%s, %s, %s::halfvec, %s::jsonb) "
                    "ON CONFLICT (namespace_id, vector_id) DO NOTHING",
                    (self.namespace, vector_id, arr.tolist(),
                     json.dumps(metadata or {})),
                )
                return cur.rowcount == 1

    def add_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> int:
        """Batch add; duplicates and dimension mismatches are skipped."""
        rows = []
        for vector_id, vector, metadata in vectors:
            try:
                arr = self._vec(vector, self.dimension)
            except ValueError:
                continue
            rows.append((self.namespace, vector_id, arr.tolist(),
                         json.dumps(metadata or {})))
        if not rows:
            return 0
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                cur = conn.executemany(
                    "INSERT INTO memory_vectors "
                    "(namespace_id, vector_id, embedding, payload) "
                    "VALUES (%s, %s, %s::halfvec, %s::jsonb) "
                    "ON CONFLICT (namespace_id, vector_id) DO NOTHING",
                    rows,
                )
                return cur.rowcount

    def search(self, query_vector: np.ndarray, k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Exact L2 search, similarity = 1 / (1 + dist^2), metadata filtered in Python.

        FAISS IndexFlatL2 reports squared L2 distances and the file backend
        turns that into ``1 / (1 + dist)``; pgvector's ``<->`` operator
        reports Euclidean (non-squared) distance, so the square is applied
        here to keep the two backends' similarity scores identical.

        The top-k is taken before the metadata filter, matching the file
        backend's semantics (a filtered search may return fewer than k rows).
        """
        arr = self._vec(query_vector, self.dimension)
        with pg_common.connection(self.dsn) as conn:
            rows = conn.execute(
                "SELECT vector_id, payload, embedding <-> %s::halfvec AS dist "
                "FROM memory_vectors WHERE namespace_id = %s "
                "ORDER BY dist LIMIT %s",
                (arr.tolist(), self.namespace, int(k)),
            ).fetchall()
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for vector_id, payload, dist in rows:
            metadata = payload if isinstance(payload, dict) else (payload or {})
            if filter_metadata:
                if not all(metadata.get(key) == value
                           for key, value in filter_metadata.items()):
                    continue
            results.append((vector_id, 1.0 / (1.0 + float(dist) ** 2), metadata))
        return results

    def get_vector(self, vector_id: str) -> Optional[VectorEntry]:
        with pg_common.connection(self.dsn) as conn:
            row = conn.execute(
                "SELECT embedding::text, payload FROM memory_vectors "
                "WHERE namespace_id = %s AND vector_id = %s",
                (self.namespace, vector_id),
            ).fetchone()
        if row is None:
            return None
        return VectorEntry(
            vector_id=vector_id,
            vector=_parse_halfvec(row[0]),
            metadata=row[1] if isinstance(row[1], dict) else (row[1] or {}),
        )

    def update_vector(self, vector_id: str, vector: np.ndarray,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        arr = self._vec(vector, self.dimension)
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                cur = conn.execute(
                    "UPDATE memory_vectors SET embedding = %s::halfvec, "
                    "payload = %s::jsonb WHERE namespace_id = %s AND vector_id = %s",
                    (arr.tolist(), json.dumps(metadata or {}),
                     self.namespace, vector_id),
                )
                return cur.rowcount == 1

    def delete_vector(self, vector_id: str) -> bool:
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                cur = conn.execute(
                    "DELETE FROM memory_vectors WHERE namespace_id = %s "
                    "AND vector_id = %s",
                    (self.namespace, vector_id),
                )
                return cur.rowcount == 1

    def exists(self, vector_id: str) -> bool:
        with pg_common.connection(self.dsn) as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_vectors WHERE namespace_id = %s "
                "AND vector_id = %s",
                (self.namespace, vector_id),
            ).fetchone()
        return row is not None

    def size(self) -> int:
        with pg_common.connection(self.dsn) as conn:
            row = conn.execute(
                "SELECT count(*) FROM memory_vectors WHERE namespace_id = %s",
                (self.namespace,),
            ).fetchone()
        return int(row[0])

    def clear(self) -> bool:
        """Remove all vectors of this namespace only."""
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM memory_vectors WHERE namespace_id = %s",
                    (self.namespace,),
                )
        return True

    def save(self, path: Optional[str] = None):
        """No-op: the Postgres rows are already the storage."""

    def load(self, path: Optional[str] = None):
        """No-op: there is no in-memory index to load."""
