#!/usr/bin/env python3
"""One-shot import of a Jev-Mem file layout into PostgreSQL.

Reads the per-namespace layout written by ``MemoryBuilder.save()``

    <namespace>/graph.json
    <namespace>/vectors/index.faiss
    <namespace>/vectors/metadata.json
    <namespace>/keyword_index.json
    <namespace>/jev_mem_config.json

and inserts nodes, links, vectors, keyword postings, and the resolved config
into the tables described by ``docs/postgresql_storage.md`` (Phase 2 of
``tasks/plan.md``). Everything for one namespace is written in a single
transaction; the database is the source of truth afterwards.

Usage::

    python scripts/migrate_to_postgres.py --source /data \\
        --dsn postgresql://postgres@localhost:5432/jev_mem

``--source`` may be either a cache base directory (each immediate
subdirectory that contains the layout is a namespace, named after the
directory) or a single namespace directory (its basename is the namespace).

The vector dimension is taken from ``vectors/metadata.json`` (falling back to
``--dim`` / ``JEV_MEM_PG_VECTOR_DIM`` / 2560 when a namespace has no
vectors). A target table created with a different dimension is rejected with
a clear error: one vector table has one dimension.

Refuses to overwrite a non-empty namespace unless ``--force`` is given; with
``--force`` the existing namespace is deleted (cascading to its rows) and
re-imported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running as a script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory import pg_common  # noqa: E402

_DEFAULT_DIM = 2560


class MigrationError(RuntimeError):
    """A namespace cannot be migrated (non-empty, bad layout, dimension)."""


@dataclass
class MigrationReport:
    namespace: str
    nodes: int
    links: int
    vectors: int
    keywords: int
    config: bool


def discover_namespaces(source: Path) -> dict[str, Path]:
    """Map namespace name -> namespace directory under ``source``.

    A directory "is a namespace" when it contains at least one of the file
    layout artifacts. ``source`` itself may be a single namespace directory.
    """
    source = Path(source)
    markers = ("graph.json", "keyword_index.json", "vectors")
    if any((source / marker).exists() for marker in markers):
        return {source.name: source}
    found: dict[str, Path] = {}
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and any((entry / marker).exists() for marker in markers):
            found[entry.name] = entry
    return found


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def _load_graph(directory: Path) -> tuple[list[dict], list[dict]]:
    data = _read_json(directory / "graph.json", {"nodes": [], "links": []})
    return list(data.get("nodes", [])), list(data.get("links", []))


def _load_vectors(directory: Path) -> list[tuple[str, list[float], dict]]:
    """Vectors from metadata.json (canonical float32 payload + metadata).

    metadata.json already contains the full float32 vector for every entry,
    so the FAISS binary does not need to be parsed (and its index order does
    not have to be reconstructed).
    """
    metadata = _read_json(directory / "vectors" / "metadata.json", None)
    if not metadata:
        return []
    rows = []
    for vector_id, entry in metadata.get("entries", {}).items():
        rows.append((
            vector_id,
            np.asarray(entry["vector"], dtype=np.float32).tolist(),
            entry.get("metadata") or {},
        ))
    return rows


def _load_keywords(directory: Path) -> list[tuple[str, str]]:
    """(term, node_id) posting rows from keyword_index.json."""
    data = _read_json(directory / "keyword_index.json", {})
    return [
        (str(term), str(node_id))
        for term, node_ids in data.items()
        for node_id in node_ids
    ]


def _file_row_counts(directory: Path) -> dict[str, int]:
    nodes, links = _load_graph(directory)
    return {
        "nodes": len(nodes),
        "links": len(links),
        "vectors": len(_load_vectors(directory)),
        "keywords": len(_load_keywords(directory)),
    }


def _db_row_counts(conn, namespace: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, table in (
        ("nodes", "memory_nodes"),
        ("links", "memory_links"),
        ("vectors", "memory_vectors"),
        ("keywords", "memory_keywords"),
    ):
        counts[key] = conn.execute(
            f"SELECT count(*) FROM {table} WHERE namespace_id = %s", (namespace,)
        ).fetchone()[0]
    return counts


def _vector_dimension(directory: Path, fallback: int) -> int:
    metadata = _read_json(directory / "vectors" / "metadata.json", None)
    if metadata and metadata.get("dimension"):
        return int(metadata["dimension"])
    return fallback


def _check_existing_dimension(conn, dimension: int) -> None:
    """Fail clearly when the existing table was created with another dim."""
    row = conn.execute("SELECT to_regclass('memory_vectors')").fetchone()[0]
    if row is None:
        return
    typ = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'memory_vectors'::regclass AND attname = 'embedding'"
    ).fetchone()
    if typ is None:
        return
    existing = int(typ[0].split("(")[1].rstrip(")"))
    if existing != dimension:
        raise MigrationError(
            f"existing memory_vectors.embedding is halfvec({existing}) but the "
            f"source layout has {dimension}-dimensional vectors; one vector "
            "table has one dimension (set JEV_MEM_PG_VECTOR_DIM to match the "
            "embedding model or migrate into a fresh database)"
        )


def migrate_namespace(pg_dsn: str, namespace: str, directory, dimension: int,
                      force: bool = False) -> MigrationReport:
    """Import one namespace directory into PostgreSQL (single transaction)."""
    directory = Path(directory)
    if not namespace:
        raise MigrationError("namespace must be a non-empty name")

    try:
        nodes, links = _load_graph(directory)
        vectors = _load_vectors(directory)
        keywords = _load_keywords(directory)
        config = _read_json(directory / "jev_mem_config.json", {})
        dim = _vector_dimension(directory, dimension)
    except (KeyError, TypeError, ValueError) as exc:
        # Malformed source JSON (missing node_id, bad vector, ...): report the
        # directory and exit cleanly instead of a bare traceback.
        raise MigrationError(f"cannot read {directory}: {exc}") from exc
    pg_common.create_schema(pg_dsn, dim)

    with pg_common.connection(pg_dsn) as conn:
        _check_existing_dimension(conn, dim)
        existing = _db_row_counts(conn, namespace)
        nonempty = any(existing.values())
        if nonempty and not force:
            raise MigrationError(
                f"namespace {namespace!r} already has data "
                f"({existing}); re-run with --force to overwrite it"
            )

        with conn.transaction():
            if force and nonempty:
                conn.execute(
                    "DELETE FROM memory_namespaces WHERE id = %s", (namespace,)
                )

            conn.execute(
                "INSERT INTO memory_namespaces (id, config) "
                "VALUES (%s, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET config = EXCLUDED.config",
                (namespace, json.dumps(config)),
            )

            if nodes:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO memory_nodes "
                        "(namespace_id, node_id, node_type, node) "
                        "VALUES (%s, %s, %s, %s::jsonb) "
                        "ON CONFLICT (namespace_id, node_id) DO NOTHING",
                        [
                            (namespace, node["node_id"],
                             node.get("node_type", "EVENT"), json.dumps(node))
                            for node in nodes
                        ],
                    )

            if links:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO memory_links "
                        "(namespace_id, link_id, source_id, target_id, "
                        " link_type, link) "
                        "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                        "ON CONFLICT (namespace_id, link_id) DO NOTHING",
                        [
                            (namespace, link["link_id"],
                             link["source_node_id"], link["target_node_id"],
                             link.get("link_type", "TEMPORAL"), json.dumps(link))
                            for link in links
                        ],
                    )

            if vectors:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO memory_vectors "
                        "(namespace_id, vector_id, embedding, payload) "
                        "VALUES (%s, %s, %s::halfvec, %s::jsonb) "
                        "ON CONFLICT (namespace_id, vector_id) DO NOTHING",
                        [
                            (namespace, vector_id, vector, json.dumps(payload))
                            for vector_id, vector, payload in vectors
                        ],
                    )

            if keywords:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO memory_keywords (namespace_id, term, node_id) "
                        "VALUES (%s, %s, %s) "
                        "ON CONFLICT (namespace_id, term, node_id) DO NOTHING",
                        [(namespace, term, node_id) for term, node_id in keywords],
                    )

            conn.execute(
                "UPDATE memory_namespaces "
                "SET graph_updated_at = now(), last_used_at = now() WHERE id = %s",
                (namespace,),
            )

            final = _db_row_counts(conn, namespace)

            expected = {
                "nodes": len(nodes),
                "links": len(links),
                "vectors": len(vectors),
                "keywords": len(keywords),
            }
            mismatched = {
                key: (final[key], count)
                for key, count in expected.items()
                if final[key] != count
            }
            if mismatched:
                # Raising inside the transaction rolls everything back: a
                # count mismatch (e.g. a duplicate node_id in the source)
                # must never leave partial data committed.
                raise MigrationError(
                    f"row-count verification failed for {namespace!r}: "
                    + ", ".join(
                        f"{key} in DB={db} expected={file}"
                        for key, (db, file) in mismatched.items()
                    )
                )

    return MigrationReport(
        namespace=namespace,
        nodes=final["nodes"],
        links=final["links"],
        vectors=final["vectors"],
        keywords=final["keywords"],
        config=bool(config),
    )


def _default_dimension() -> int:
    """JEV_MEM_PG_VECTOR_DIM with a clean error instead of a bare ValueError."""
    raw = os.getenv("JEV_MEM_PG_VECTOR_DIM", "").strip()
    if not raw:
        return _DEFAULT_DIM
    try:
        return int(raw)
    except ValueError as exc:
        raise MigrationError(
            f"JEV_MEM_PG_VECTOR_DIM must be an integer (got {raw!r})"
        ) from exc


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Migrate a Jev-Mem file layout into PostgreSQL.")
    parser.add_argument("--source", required=True,
                        help="cache base dir or one namespace directory")
    parser.add_argument("--dsn", default=os.getenv("JEV_MEM_PG_DSN", "").strip(),
                        help="PostgreSQL DSN (default: JEV_MEM_PG_DSN)")
    parser.add_argument("--dim", type=int,
                        default=_default_dimension(),
                        help="vector dimension when the layout has none "
                             "(default: JEV_MEM_PG_VECTOR_DIM or 2560)")
    parser.add_argument("--namespace", action="append", default=None,
                        help="only migrate this namespace (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite namespaces that already have data")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = _parse_args(argv)
        dsn = pg_common.resolve_dsn(args.dsn)
        source = Path(args.source)
        if not source.is_dir():
            raise MigrationError(f"source directory not found: {source}")
        namespaces = discover_namespaces(source)
        if args.namespace:
            selected = set(args.namespace)
            missing = selected - set(namespaces)
            if missing:
                raise MigrationError(
                    f"namespace(s) not found under {source}: "
                    + ", ".join(sorted(missing))
                )
            namespaces = {k: v for k, v in namespaces.items() if k in selected}
        if not namespaces:
            raise MigrationError(
                f"no namespace layout found under {source} "
                "(expected graph.json / vectors/ / keyword_index.json)"
            )

        print(f"migrating {len(namespaces)} namespace(s) from {source}")
        total = MigrationReport("", 0, 0, 0, 0, False)
        for namespace, directory in namespaces.items():
            report = migrate_namespace(dsn, namespace, directory,
                                       dimension=args.dim, force=args.force)
            print(
                f"  {report.namespace}: nodes={report.nodes} "
                f"links={report.links} vectors={report.vectors} "
                f"keywords={report.keywords} config={report.config}"
            )
            total = MigrationReport(
                "<total>",
                total.nodes + report.nodes,
                total.links + report.links,
                total.vectors + report.vectors,
                total.keywords + report.keywords,
                total.config or report.config,
            )
        print(
            f"done: nodes={total.nodes} links={total.links} "
            f"vectors={total.vectors} keywords={total.keywords}"
        )
        return 0
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        pg_common.reset_pool()


if __name__ == "__main__":
    raise SystemExit(main())