"""Postgres graph storage on ordinary tables (no Apache AGE).

``PGTableGraphStorage`` keeps the entity-relation graph in ``memory_nodes`` /
``memory_links`` rows (JSONB payloads, B-tree indexes) and serves retrieval
from an in-memory NetworkX cache, exactly like the file backend. Writes are
write-through in the same transaction as a ``memory_namespaces.graph_updated_at``
bump; reads validate that version (optionally throttled) so a second process
sees the first process's writes within the configured window.

This is the recommended ``GRAPH_STORAGE`` for new PostgreSQL deployments and
supersedes ``PGGraphStorage`` (Apache AGE), which needs an extension most
managed services do not offer. See docs/postgresql_storage.md.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from . import pg_common
from .graph_db import (
    EventNode,
    EpisodeNode,
    FileGraphStore,
    GraphStore,
    Link,
    LinkType,
    SessionNode,
)

_CACHE_TTL = 30.0      # seconds; hard cap on staleness
_CHECK_INTERVAL = 0.0  # seconds between version checks (0 = every access)


def _node_from_dict(data: Dict[str, Any]) -> EventNode:
    """Rebuild the concrete node class from its serialized dict.

    Same dispatch as ``FileGraphStore.import_from_json``: there is no generic
    ``Node``; ``node_type`` selects EventNode / EpisodeNode / SessionNode.
    """
    node_type = data.get("node_type", "EVENT")
    if node_type == "EPISODE":
        return EpisodeNode.from_dict(data)
    if node_type == "SESSION":
        return SessionNode.from_dict(data)
    return EventNode.from_dict(data)


class PGTableGraphStorage(GraphStore):
    """GraphStore backed by Postgres tables with an in-memory read cache."""

    def __init__(self, dsn: str, namespace: str = "default",
                 cache_ttl: float = _CACHE_TTL,
                 check_interval: float = _CHECK_INTERVAL):
        self.dsn = pg_common.resolve_dsn(dsn)
        self.namespace = namespace
        self.cache_ttl = cache_ttl
        self.check_interval = check_interval
        self._mem = FileGraphStore()
        self.keyword_index: Dict[str, set] = {}
        self._version = None
        self._loaded_at = 0.0
        self._last_check = 0.0

    # --- lifecycle -----------------------------------------------------

    def initialize(self) -> None:
        """Create the graph tables if missing and register the namespace."""
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                for statement in pg_common.graph_ddl():
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO memory_namespaces (id) VALUES (%s) "
                    "ON CONFLICT DO NOTHING",
                    (self.namespace,),
                )

    def delete_namespace(self, namespace_id: Optional[str] = None) -> None:
        """Delete every row of a namespace (cascades to nodes/links/etc.)."""
        ns = namespace_id or self.namespace
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM memory_namespaces WHERE id = %s", (ns,)
                )
        if ns == self.namespace:
            self._mem = FileGraphStore()
            self.keyword_index = {}
            self._version = None
            self._loaded_at = 0.0
            self._last_check = 0.0

    def flush(self) -> None:
        """No-op: every mutation is already committed (write-through)."""

    # --- collection views (consumers iterate these directly) -----------

    @property
    def nodes(self) -> Dict[str, Any]:
        """Live mapping from the in-memory cache (loaded on first access)."""
        self._ensure_loaded()
        return self._mem.nodes

    @property
    def links(self) -> Dict[str, Any]:
        """Live mapping from the in-memory cache (loaded on first access)."""
        self._ensure_loaded()
        return self._mem.links

    def export_to_json(self, filepath: str) -> None:
        """Write a JSON snapshot (the file layout stays an export, not the source)."""
        self._ensure_loaded()
        self._mem.export_to_json(filepath)

    def save(self, filepath: Optional[str] = None) -> None:
        """No-op: every mutation is already committed (write-through).

        Call export_to_json() for an explicit graph.json snapshot.
        """

    def load(self, filepath: Optional[str] = None) -> None:
        """Postgres is the source of truth; the file argument is ignored."""
        self._ensure_loaded()

    # --- cache ---------------------------------------------------------

    def _read_version(self) -> Any:
        with pg_common.connection(self.dsn) as conn:
            row = conn.execute(
                "SELECT graph_updated_at FROM memory_namespaces WHERE id = %s",
                (self.namespace,),
            ).fetchone()
        return row[0] if row else None

    def _load_from_db(self) -> None:
        with pg_common.connection(self.dsn) as conn:
            node_rows = conn.execute(
                "SELECT node FROM memory_nodes WHERE namespace_id = %s",
                (self.namespace,),
            ).fetchall()
            link_rows = conn.execute(
                "SELECT link FROM memory_links WHERE namespace_id = %s",
                (self.namespace,),
            ).fetchall()
            keyword_rows = conn.execute(
                "SELECT term, node_id FROM memory_keywords WHERE namespace_id = %s",
                (self.namespace,),
            ).fetchall()
            version_row = conn.execute(
                "SELECT graph_updated_at FROM memory_namespaces WHERE id = %s",
                (self.namespace,),
            ).fetchone()
            # A namespace load is a use: bump last_used_at (cross-process LRU).
            with conn.transaction():
                conn.execute(
                    "UPDATE memory_namespaces SET last_used_at = now() "
                    "WHERE id = %s",
                    (self.namespace,),
                )
        mem = FileGraphStore()
        for (node_data,) in node_rows:
            mem.add_node(_node_from_dict(node_data))
        for (link_data,) in link_rows:
            mem.add_link(Link.from_dict(link_data))
        keywords: Dict[str, set] = {}
        for term, node_id in keyword_rows:
            keywords.setdefault(term, set()).add(node_id)
        self._mem = mem
        self.keyword_index = keywords
        self._version = version_row[0] if version_row else None
        now = time.monotonic()
        self._loaded_at = now
        self._last_check = now

    def reload(self) -> None:
        """Force a fresh read of the namespace (nodes, links, keywords)."""
        self._load_from_db()

    def _ensure_loaded(self) -> None:
        """Load on first access; reload when another writer bumped the graph.

        The version check is one cheap primary-key SELECT. ``check_interval``
        throttles it for traversal hot paths; ``cache_ttl`` is the hard cap
        on staleness when checks are skipped.
        """
        now = time.monotonic()
        if self._loaded_at == 0.0:
            self._load_from_db()
            return
        if now - self._loaded_at > self.cache_ttl:
            self._load_from_db()
            return
        if self.check_interval and now - self._last_check < self.check_interval:
            return
        self._last_check = now
        if self._read_version() != self._version:
            self._load_from_db()

    # --- persistence helpers -------------------------------------------

    def _bump_version(self, conn) -> None:
        conn.execute(
            "UPDATE memory_namespaces SET graph_updated_at = now() WHERE id = %s",
            (self.namespace,),
        )

    def _persist_node(self, node: Any) -> None:
        data = node.to_dict()
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO memory_nodes "
                    "(namespace_id, node_id, node_type, node) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (namespace_id, node_id) DO UPDATE "
                    "SET node_type = EXCLUDED.node_type, node = EXCLUDED.node",
                    (self.namespace, data["node_id"], data["node_type"],
                     json.dumps(data, default=str)),
                )
                self._bump_version(conn)

    def _persist_link(self, link: Link) -> None:
        data = link.to_dict()
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO memory_links "
                    "(namespace_id, link_id, source_id, target_id, link_type, link) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (namespace_id, link_id) DO UPDATE "
                    "SET source_id = EXCLUDED.source_id, "
                    "    target_id = EXCLUDED.target_id, "
                    "    link_type = EXCLUDED.link_type, "
                    "    link = EXCLUDED.link",
                    (self.namespace, data["link_id"], data["source_node_id"],
                     data["target_node_id"], data["link_type"],
                     json.dumps(data, default=str)),
                )
                self._bump_version(conn)

    # --- keyword posting --------------------------------------------------

    def index_node_terms(self, node_id: str, terms) -> int:
        """Insert posting rows for a node's terms (no duplicates).

        Mirrors ``MemoryBuilder._index_text_basic``: the caller computes the
        terms with the same tokenization; this method persists them, updates
        the in-memory posting dict, and bumps the namespace version so other
        processes reload it with the graph cache.
        """
        return self._write_terms(node_id, terms, replace=False)

    def reindex_node_terms(self, node_id: str, terms) -> int:
        """Replace a node's posting rows (use when its text changes)."""
        return self._write_terms(node_id, terms, replace=True)

    def _write_terms(self, node_id: str, terms, replace: bool) -> int:
        terms = {term for term in terms if term}
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                if replace:
                    conn.execute(
                        "DELETE FROM memory_keywords "
                        "WHERE namespace_id = %s AND node_id = %s",
                        (self.namespace, node_id),
                    )
                inserted = 0
                if terms:
                    rows = [(self.namespace, term, node_id) for term in terms]
                    with conn.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO memory_keywords "
                            "(namespace_id, term, node_id) "
                            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                            rows,
                        )
                        inserted = cur.rowcount
                if inserted or replace:
                    self._bump_version(conn)
        if replace:
            for term in list(self.keyword_index):
                self.keyword_index[term].discard(node_id)
                if not self.keyword_index[term]:
                    del self.keyword_index[term]
        for term in terms:
            self.keyword_index.setdefault(term, set()).add(node_id)
        return inserted

    # --- GraphStore: writes --------------------------------------------

    def add_node(self, node: EventNode) -> str:
        self._ensure_loaded()
        node_id = self._mem.add_node(node)
        self._persist_node(node)
        return node_id

    def add_link(self, link: Link) -> str:
        self._ensure_loaded()
        link_id = self._mem.add_link(link)
        self._persist_link(link)
        return link_id

    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        self._ensure_loaded()
        if not self._mem.update_node(node_id, updates):
            return False
        self._persist_node(self._mem.get_node(node_id))
        return True

    def update_link(self, link_id: str, updates: Dict[str, Any]) -> bool:
        self._ensure_loaded()
        if not self._mem.update_link(link_id, updates):
            return False
        link = self._mem.get_link(link_id)
        if link is None:
            return False
        self._persist_link(link)
        return True

    def delete_node(self, node_id: str) -> bool:
        self._ensure_loaded()
        if node_id not in self._mem.nodes:
            return False
        self._mem.delete_node(node_id)
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM memory_nodes "
                    "WHERE namespace_id = %s AND node_id = %s",
                    (self.namespace, node_id),
                )
                self._bump_version(conn)
        return True

    def delete_link(self, link_id: str) -> bool:
        self._ensure_loaded()
        if link_id not in self._mem.links:
            return False
        self._mem.delete_link(link_id)
        with pg_common.connection(self.dsn) as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM memory_links "
                    "WHERE namespace_id = %s AND link_id = %s",
                    (self.namespace, link_id),
                )
                self._bump_version(conn)
        return True

    # --- GraphStore: reads ---------------------------------------------

    def get_node(self, node_id: str) -> Optional[EventNode]:
        self._ensure_loaded()
        return self._mem.get_node(node_id)

    def get_link(self, link_id: str) -> Optional[Link]:
        self._ensure_loaded()
        return self._mem.get_link(link_id)

    def get_neighbors(
        self, node_id: str, link_type: Optional[LinkType] = None
    ) -> List[Tuple[EventNode, Link]]:
        self._ensure_loaded()
        return self._mem.get_neighbors(node_id, link_type)

    def traverse(self, start_nodes: List[str], constraints) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._mem.traverse(start_nodes, constraints)

    def find_path(self, source_id: str, target_id: str,
                  link_type: Optional[LinkType] = None) -> Optional[List[str]]:
        self._ensure_loaded()
        return self._mem.find_path(source_id, target_id, link_type)

    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._mem.get_subgraph(node_ids)