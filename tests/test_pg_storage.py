"""PG-gated tests for the Postgres storage backends.

Every test skips (not fails) when JEV_MEM_PG_DSN is unset, so the default
suite runs without a database. With a DSN set (e.g. a local
pgvector/pgvector:pg18 container) the tests exercise PgVectorStore against
a live database, using a throwaway namespace per test.
"""

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memory import pg_common
from memory.graph_db import EventNode, Link, LinkType
from memory.pg_table_graph_store import PGTableGraphStorage
from memory.pg_vector_store import PgVectorStore

_DSN = os.getenv("JEV_MEM_PG_DSN", "").strip()
_DEFAULT_DIM = 16  # used when the test database has no schema yet


@pytest.fixture(scope="session")
def dsn():
    if not _DSN:
        # Skip (never fail) so the default suite runs without a database.
        # Fixture-based skip propagates to every PG test; the file-mode
        # end-to-end test below does not use this fixture and always runs.
        pytest.skip("JEV_MEM_PG_DSN not set (PG-gated tests)")
    return _DSN


@pytest.fixture(scope="session")
def dim(dsn):
    """The table dimension: a fresh DB gets _DEFAULT_DIM, a reused one keeps
    whatever it already has (one dimension per table, per the design doc)."""
    pg_common.create_schema(dsn, _DEFAULT_DIM)
    with pg_common.connection(dsn) as conn:
        typ = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'memory_vectors'::regclass AND attname = 'embedding'"
        ).fetchone()[0]
    return int(typ.split("(")[1].rstrip(")"))


@pytest.fixture(autouse=True)
def _cleanup_system_tmpdirs():
    """Remove the per-test cache dirs the *_system helpers create via mkdtemp."""
    yield
    temp_base = tempfile.gettempdir()
    for name in os.listdir(temp_base):
        if name.startswith("jev-pg-e2e-"):
            shutil.rmtree(os.path.join(temp_base, name), ignore_errors=True)


@pytest.fixture()
def store(dsn, dim):
    namespace = f"t-{os.getpid()}-{np.random.rand():.6f}"
    store = PgVectorStore(dim, dsn=dsn, namespace=namespace)
    yield store, namespace
    with pg_common.connection(dsn) as conn:
        with conn.transaction():
            conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))
    pg_common.reset_pool()


def _vec(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    # halfvec stores float16; allow ~2e-3 absolute error for |x| < 4.
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)


def test_pg_vector_crud(dsn, dim, store):
    store, namespace = store
    assert store.size() == 0
    assert store.add_vector("a", _vec(1, dim), {"k": "v"}) is True
    assert store.add_vector("a", _vec(2, dim)) is False, "duplicate id must not insert"
    assert store.exists("a") and store.size() == 1

    entry = store.get_vector("a")
    assert entry is not None and entry.metadata == {"k": "v"}
    assert entry.vector.shape == (dim,)
    _assert_close(entry.vector, _vec(1, dim))

    assert store.update_vector("b", _vec(3, dim)) is False, "update of missing id"
    assert store.add_vector("b", _vec(3, dim)) is True
    assert store.update_vector("b", _vec(4, dim), {"k": "w"}) is True
    entry = store.get_vector("b")
    assert entry.metadata == {"k": "w"}
    _assert_close(entry.vector, _vec(4, dim))

    assert store.delete_vector("b") is True
    assert store.delete_vector("b") is False
    assert store.size() == 1

    assert store.clear() is True
    assert store.size() == 0


def test_pg_vector_search_ordering(dim, store):
    store, namespace = store
    base = np.zeros(dim, dtype=np.float32)
    for i in range(5):
        v = base.copy()
        v[i] = float(i + 1)
        store.add_vector(f"v{i}", v, {"rank": i})
    results = store.search(np.zeros(dim, dtype=np.float32), k=3)
    assert [r[0] for r in results] == ["v0", "v1", "v2"]
    assert results[0][1] == pytest.approx(0.5), "L2 distance 1 -> similarity 1/2"
    assert all(results[0][1] >= r[1] for r in results[1:]), "similarities descend"


def test_pg_vector_search_metadata_filter(dim, store):
    store, namespace = store
    for i in range(4):
        v = np.zeros(dim, dtype=np.float32)
        v[0] = float(i + 1)
        store.add_vector(f"v{i}", v, {"group": "a" if i % 2 == 0 else "b"})
    results = store.search(np.zeros(dim, dtype=np.float32), k=4, filter_metadata={"group": "a"})
    assert {r[0] for r in results} == {"v0", "v2"}, "filter may return fewer than k"


def test_pg_vector_namespace_isolation(dsn, dim, store):
    store, namespace = store
    other = PgVectorStore(dim, dsn=dsn, namespace=namespace + "-other")
    store.add_vector("x", _vec(9, dim), {"ns": namespace})
    other.add_vector("x", _vec(10, dim), {"ns": namespace + "-other"})
    assert store.size() == 1 and other.size() == 1
    entry = store.get_vector("x")
    assert entry.metadata["ns"] == namespace
    assert other.search(_vec(9, dim), k=1)[0][0] == "x"
    assert other.search(_vec(9, dim), k=1)[0][2]["ns"] == namespace + "-other"


def test_pg_vector_dimension_mismatch_fails_fast(dim, store):
    store, namespace = store
    with pytest.raises(ValueError, match="JEV_MEM_PG_VECTOR_DIM"):
        store.add_vector("bad", _vec(1, dim)[:4])
    assert store.size() == 0


def test_pg_vector_parity_with_file_store(dim, store):
    """Same vectors in both backends: identical top-k ordering; scores agree
    within the halfvec float16 tolerance (FAISS flat is fp32)."""
    from memory.vector_db import FileVectorStore

    pg_store, _namespace = store
    file_store = FileVectorStore(dimension=dim, index_type="flat")
    rng = np.random.default_rng(42)
    for i in range(30):
        v = rng.normal(size=dim).astype(np.float32)
        assert file_store.add_vector(f"p{i}", v, {"rank": i}) is True
        assert pg_store.add_vector(f"p{i}", v, {"rank": i}) is True
    query = rng.normal(size=dim).astype(np.float32)
    file_results = file_store.search(query, k=10)
    pg_results = pg_store.search(query, k=10)
    assert len(file_results) == len(pg_results) == 10
    assert [r[0] for r in file_results] == [r[0] for r in pg_results]
    for (_, fsim, _), (_, psim, _) in zip(file_results, pg_results):
        assert psim == pytest.approx(fsim, abs=2e-3)


# --- PGTableGraphStorage (Task 6) -------------------------------------------------

@pytest.fixture()
def gstore(dsn):
    namespace = f"g-{os.getpid()}-{np.random.rand():.6f}"
    store = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    store.initialize()
    yield store, namespace
    with pg_common.connection(dsn) as conn:
        with conn.transaction():
            conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))
    pg_common.reset_pool()


def _event(node_id: str, narrative: str) -> EventNode:
    return EventNode(node_id=node_id, content_narrative=narrative)


def _causal(source: str, target: str) -> Link:
    return Link(
        link_id=f"{source}->{target}",
        source_node_id=source,
        target_node_id=target,
        link_type=LinkType.CAUSAL,
        properties={"sub_type": "LEADS_TO"},
    )


def test_pg_graph_roundtrip(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "first"))
    store.add_node(_event("n2", "second"))
    store.add_link(_causal("n1", "n2"))
    # A fresh instance (a different "process") sees the rows from Postgres.
    fresh = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    fresh.initialize()
    assert fresh.get_node("n1") is not None
    assert fresh.get_node("n1").content_narrative == "first"
    neighbors = fresh.get_neighbors("n2", LinkType.CAUSAL)
    assert [n[0].node_id for n in neighbors] == ["n1"]
    link = fresh.get_link("n1->n2")
    assert link is not None and link.properties["sub_type"] == "LEADS_TO"


def test_pg_graph_write_through_persists(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("a", "one"))
    store.add_node(_event("b", "two"))
    store.add_link(_causal("a", "b"))
    # Directly in Postgres, not just in the writer's memory.
    with pg_common.connection(dsn) as conn:
        nodes = conn.execute(
            "SELECT node_id FROM memory_nodes WHERE namespace_id = %s",
            (namespace,),
        ).fetchall()
        links = conn.execute(
            "SELECT link_id FROM memory_links WHERE namespace_id = %s",
            (namespace,),
        ).fetchall()
    assert {r[0] for r in nodes} == {"a", "b"}
    assert [r[0] for r in links] == ["a->b"]


def test_pg_graph_external_write_invalidates_reader(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "one"))
    reader = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    reader.initialize()
    assert reader.get_node("n1") is not None, "first read loads the graph"
    store.add_node(_event("n2", "two"))  # external write bumps graph_updated_at
    assert reader.get_node("n2") is not None, "version check must reload"


def test_pg_graph_update_node_persists(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "before"))
    assert store.update_node("n1", {"content_narrative": "after"}) is True
    fresh = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    fresh.initialize()
    assert fresh.get_node("n1").content_narrative == "after"


def test_pg_graph_delete_node_removes_links_in_db(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "one"))
    store.add_node(_event("n2", "two"))
    store.add_link(_causal("n1", "n2"))
    assert store.delete_node("n1") is True
    with pg_common.connection(dsn) as conn:
        links = conn.execute(
            "SELECT link_id FROM memory_links WHERE namespace_id = %s",
            (namespace,),
        ).fetchall()
    assert links == []
    fresh = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    fresh.initialize()
    assert fresh.get_node("n1") is None
    assert fresh.get_node("n2") is not None


def test_pg_graph_delete_namespace_cascades(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "one"))
    store.delete_namespace(namespace)
    with pg_common.connection(dsn) as conn:
        rows = conn.execute(
            "SELECT id FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchall()
    assert rows == []


# --- keyword posting (Task 7) ------------------------------------------------------

def test_pg_graph_keyword_roundtrip(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "hello world"))
    store.index_node_terms("n1", ["hello", "hello world"])
    store.add_node(_event("n2", "another"))
    store.index_node_terms("n2", ["hello"])
    fresh = PGTableGraphStorage(dsn=dsn, namespace=namespace)
    fresh.reload()
    assert fresh.keyword_index == {
        "hello": {"n1", "n2"},
        "hello world": {"n1"},
    }, "the posting dict reloads together with the graph cache"


def test_pg_graph_keyword_dedup_and_reindex(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "a b c"))
    store.index_node_terms("n1", ["a", "b"])
    store.index_node_terms("n1", ["b", "c"])
    with pg_common.connection(dsn) as conn:
        rows = conn.execute(
            "SELECT term FROM memory_keywords WHERE node_id = %s ORDER BY term",
            ("n1",),
        ).fetchall()
    assert [r[0] for r in rows] == ["a", "b", "c"], "duplicate terms must not insert twice"

    store.reindex_node_terms("n1", ["z"])
    with pg_common.connection(dsn) as conn:
        rows = conn.execute(
            "SELECT term FROM memory_keywords WHERE node_id = %s", ("n1",)
        ).fetchall()
    assert [r[0] for r in rows] == ["z"], "reindex replaces the node's old terms"
    assert store.keyword_index == {"z": {"n1"}}


def test_pg_graph_keyword_cascade_on_node_delete(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("n1", "x"))
    store.index_node_terms("n1", ["x"])
    store.delete_node("n1")
    with pg_common.connection(dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM memory_keywords WHERE node_id = %s", ("n1",)
        ).fetchone()[0]
    assert count == 0, "node deletion must cascade to keyword rows"


# --- namespace management (Task 8) ---------------------------------------------

def test_pg_namespace_register_exists_list(dsn):
    ns = f"ns-{os.getpid()}-{np.random.rand():.6f}"
    assert pg_common.namespace_create(dsn, ns) is True, "first create registers"
    assert pg_common.namespace_create(dsn, ns) is False, "second create is a no-op"
    assert pg_common.namespace_exists(dsn, ns)
    assert ns in pg_common.namespace_list(dsn)


def test_pg_namespace_delete_cascades(dsn, gstore):
    store, namespace = gstore
    store.add_node(_event("c1", "cascade"))
    store.index_node_terms("c1", ["kw"])
    assert pg_common.namespace_delete(dsn, namespace) is True
    assert not pg_common.namespace_exists(dsn, namespace)
    with pg_common.connection(dsn) as conn:
        nodes = conn.execute(
            "SELECT count(*) FROM memory_nodes WHERE namespace_id = %s", (namespace,)
        ).fetchone()[0]
        kws = conn.execute(
            "SELECT count(*) FROM memory_keywords WHERE namespace_id = %s", (namespace,)
        ).fetchone()[0]
    assert nodes == 0 and kws == 0, "namespace delete must cascade to child rows"


def test_pg_namespace_touch_bumps_last_used(dsn, gstore):
    import time

    store, namespace = gstore
    with pg_common.connection(dsn) as conn:
        before = conn.execute(
            "SELECT last_used_at FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchone()[0]
    time.sleep(0.02)
    pg_common.namespace_touch(dsn, namespace)
    with pg_common.connection(dsn) as conn:
        after = conn.execute(
            "SELECT last_used_at FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchone()[0]
    assert after > before, "touch must bump last_used_at"


def test_pg_graph_load_bumps_last_used(dsn, gstore):
    import time

    store, namespace = gstore
    with pg_common.connection(dsn) as conn:
        before = conn.execute(
            "SELECT last_used_at FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchone()[0]
    time.sleep(0.02)
    store.reload()  # forces _load_from_db, which must bump last_usedAt
    with pg_common.connection(dsn) as conn:
        after = conn.execute(
            "SELECT last_used_at FROM memory_namespaces WHERE id = %s", (namespace,)
        ).fetchone()[0]
    assert after > before, "a namespace load must bump last_usedAt"


# --- migration script (Task 9) ---------------------------------------------------

def _migration_module():
    """Load scripts/migrate_to_postgres.py as a module (it is not a package).

    The module is registered in sys.modules before execution: dataclasses
    resolve string annotations through the module registry.
    """
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_to_postgres.py"
    spec = importlib.util.spec_from_file_location("migrate_to_postgres", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_file_layout(tmp_path, dimension):
    """Write a real per-namespace file layout using the file backends."""
    from memory.vector_db import FileVectorStore
    from memory.graph_db import FileGraphStore, EventNode, Link, LinkType

    import json

    ns = tmp_path / "ns1"
    ns.mkdir(parents=True, exist_ok=True)

    graph = FileGraphStore()
    n1 = EventNode(content_narrative="first memory",
                   embedding_vector=[0.1] * dimension)
    n2 = EventNode(content_narrative="second memory",
                   embedding_vector=[0.2] * dimension)
    graph.add_node(n1)
    graph.add_node(n2)
    link = Link(source_node_id=n1.node_id, target_node_id=n2.node_id,
                link_type=LinkType.CAUSAL, properties={"sub_type": "LEADS_TO"})
    graph.add_link(link)
    graph.export_to_json(str(ns / "graph.json"))

    vecdb = FileVectorStore(dimension=dimension, index_type="flat")
    vecdb.add_vector(n1.node_id, np.full(dimension, 0.1, np.float32),
                     {"memory_type": "semantic"})
    vecdb.add_vector(n2.node_id, np.full(dimension, 0.2, np.float32),
                     {"memory_type": "episodic"})
    vecdb.save(str(ns / "vectors"))

    with open(ns / "keyword_index.json", "w") as f:
        json.dump({"first": [n1.node_id], "memory": [n1.node_id, n2.node_id]}, f)
    with open(ns / "jev_mem_config.json", "w") as f:
        json.dump({"profile": "default", "decision_backend": "magma"}, f)

    return ns, n1, n2, link


def test_pg_migrate_file_layout_roundtrip(dsn, dim, tmp_path):
    mig = _migration_module()
    ns, n1, n2, link = _build_file_layout(tmp_path, dim)
    namespace = ns.name

    report = mig.migrate_namespace(dsn, namespace, ns, dimension=dim)

    assert report.nodes == 2
    assert report.links == 1
    assert report.vectors == 2
    assert report.keywords == 3  # "first"->1, "memory"->2
    assert report.config is True

    # row-level sample diff
    with pg_common.connection(dsn) as conn:
        node_row = conn.execute(
            "SELECT node FROM memory_nodes WHERE namespace_id=%s AND node_id=%s",
            (namespace, n1.node_id),
        ).fetchone()
        assert node_row[0]["content_narrative"] == "first memory"
        assert node_row[0]["node_type"] == "EVENT"

        link_row = conn.execute(
            "SELECT link FROM memory_links WHERE namespace_id=%s", (namespace,)
        ).fetchone()
        assert link_row[0]["link_type"] == "CAUSAL"
        assert link_row[0]["properties"]["sub_type"] == "LEADS_TO"

        vec_row = conn.execute(
            "SELECT embedding::text, payload FROM memory_vectors "
            "WHERE namespace_id=%s AND vector_id=%s", (namespace, n1.node_id)
        ).fetchone()
        assert vec_row[1] == {"memory_type": "semantic"}

        kw = conn.execute(
            "SELECT count(*) FROM memory_keywords WHERE namespace_id=%s", (namespace,)
        ).fetchone()[0]
        assert kw == 3

    with pg_common.connection(dsn) as conn:
        conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))


def test_pg_migrate_refuses_nonempty_without_force(dsn, dim, tmp_path):
    mig = _migration_module()
    ns, *_ = _build_file_layout(tmp_path, dim)
    namespace = ns.name
    mig.migrate_namespace(dsn, namespace, ns, dimension=dim)
    try:
        with pytest.raises(RuntimeError, match="--force"):
            mig.migrate_namespace(dsn, namespace, ns, dimension=dim)
    finally:
        with pg_common.connection(dsn) as conn:
            conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))


def test_pg_migrate_force_overwrites(dsn, dim, tmp_path):
    mig = _migration_module()
    ns, *_ = _build_file_layout(tmp_path, dim)
    namespace = ns.name
    mig.migrate_namespace(dsn, namespace, ns, dimension=dim)
    report2 = mig.migrate_namespace(dsn, namespace, ns, dimension=dim, force=True)
    # idempotent overwrite: still exactly the source row counts
    assert report2.nodes == 2 and report2.links == 1 and report2.vectors == 2
    with pg_common.connection(dsn) as conn:
        conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))


def test_pg_migrate_discover_namespaces(dsn, dim, tmp_path):
    mig = _migration_module()
    ns, *_ = _build_file_layout(tmp_path, dim)
    assert mig.discover_namespaces(tmp_path) == {"ns1": ns}
    assert mig.discover_namespaces(ns) == {"ns1": ns}, "single namespace dir"


def test_pg_migrate_empty_namespace_no_force_needed(dsn, dim, tmp_path):
    mig = _migration_module()
    ns, *_ = _build_file_layout(tmp_path, dim)
    namespace = f"{ns.name}-fresh"
    # fresh (absent) namespace needs no --force
    report = mig.migrate_namespace(dsn, namespace, ns, dimension=dim)
    assert report.nodes == 2
    with pg_common.connection(dsn) as conn:
        conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))


def test_pg_migrate_duplicate_node_id_rolls_back(dsn, dim, tmp_path):
    """A row-count mismatch (duplicate node_id deduped by ON CONFLICT) must
    roll back the whole transaction, not leave partial data committed."""
    import json as _json

    mig = _migration_module()
    ns, *_ = _build_file_layout(tmp_path, dim)
    graph = _json.loads((ns / "graph.json").read_text())
    graph["nodes"].append(dict(graph["nodes"][0]))
    (ns / "graph.json").write_text(_json.dumps(graph))

    namespace = f"{ns.name}-dup"
    with pg_common.connection(dsn) as conn:
        conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))
    with pytest.raises(mig.MigrationError, match="row-count verification"):
        mig.migrate_namespace(dsn, namespace, ns, dimension=dim)

    counts = _pg_counts(dsn, namespace)
    assert counts == {
        "memory_nodes": 0,
        "memory_links": 0,
        "memory_vectors": 0,
        "memory_keywords": 0,
    }
    with pg_common.connection(dsn) as conn:
        conn.execute("DELETE FROM memory_namespaces WHERE id = %s", (namespace,))

# --- end-to-end system on Postgres (Task 10)


class _DeterministicEncoder:
    """Token-hash encoder sized to the test table's vector dimension."""

    def __init__(self, dimension):
        self.dimension = dimension

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                result[i, int.from_bytes(digest[:4], "big") % self.dimension] += 1
        return result


def _pg_system(dsn_value, dim_value, namespace):
    """Fully offline JevMemSystem on the PG stores (mock JEV, no LLM)."""
    import tempfile

    from jev_mem.system import JevMemSystem
    from memory.jev_mem_config import JevMemConfig
    from memory.memory_builder import MemoryBuilder
    from memory.trg_memory import TemporalResonanceGraphMemory

    config = JevMemConfig(
        storage="postgres", pg_dsn=dsn_value, pg_vector_dim=dim_value,
        namespace=namespace, jev_mock=True, write_enabled=True,
        read_enabled=True)
    trg = TemporalResonanceGraphMemory(
        vector_db=PgVectorStore(dim_value, dsn=dsn_value, namespace=namespace),
        graph_db=PGTableGraphStorage(dsn=dsn_value, namespace=namespace),
        encoder=_DeterministicEncoder(dim_value), llm_backend=None)
    builder = MemoryBuilder(
        tempfile.mkdtemp(prefix="jev-pg-e2e-"), jev_config=config,
        trg_memory=trg, llm_enabled=False)
    return JevMemSystem(cache_dir=builder.cache_dir, memory_builder=builder)


def _pg_counts(dsn_value, namespace):
    with pg_common.connection(dsn_value) as conn:
        return {
            table: conn.execute(
                f"SELECT count(*) FROM {table} WHERE namespace_id = %s",
                (namespace,)).fetchone()[0]
            for table in ("memory_nodes", "memory_links",
                          "memory_vectors", "memory_keywords")
        }


def test_pg_end_to_end_system(dsn, dim):
    """ingest -> query (all intents) -> consolidate -> reload, all in PG."""
    namespace = f"e2e-{os.getpid()}-{os.urandom(4).hex()}"
    try:
        system = _pg_system(dsn, dim, namespace)
        result = system.build_memory_from_conversation([
            "Anna booked a flight to Tokyo on March 3rd.",
            "Anna's flight was delayed because of a storm that day.",
        ])
        assert result == {"admitted": 2, "rejected": 0}

        counts = _pg_counts(dsn, namespace)
        assert counts["memory_nodes"] >= 2
        assert counts["memory_links"] >= 1
        assert counts["memory_vectors"] >= 2
        assert counts["memory_keywords"] >= 2

        questions = [
            "When did Anna book her flight?",   # temporal view
            "Where did Anna fly to?",           # semantic/entity view
            "Why was Anna's flight delayed?",   # causal view
            "Who booked the flight to Tokyo?",  # entity view
        ]
        for question in questions:
            answer = system.query(question)
            assert isinstance(answer, str) and answer.strip()

        with pg_common.connection(dsn) as conn:
            memory_id = conn.execute(
                "SELECT node_id FROM memory_nodes WHERE namespace_id = %s "
                "ORDER BY node_id LIMIT 1", (namespace,)).fetchone()[0]
        decisions = system.memory_builder.consolidate(memory_id)
        assert isinstance(decisions, list)

        # A second process (fresh instance, no cache) recovers from PG alone.
        reloaded = _pg_system(dsn, dim, namespace)
        reloaded.load_memory()
        assert reloaded.query(questions[0]).strip()
        assert _pg_counts(dsn, namespace)["memory_nodes"] == counts["memory_nodes"]
    finally:
        with pg_common.connection(dsn) as conn:
            conn.execute("DELETE FROM memory_namespaces WHERE id = %s",
                         (namespace,))
            conn.commit()
        pg_common.reset_pool()


# --- stale file layout next to Postgres ----------------------------------


def test_pg_load_ignores_stale_file_layout(dsn, dim):
    """A leftover per-namespace file layout must not break or shadow a PG load."""
    ns = f"stale-{os.getpid()}-{os.urandom(4).hex()}"
    try:
        first = _pg_system(dsn, dim, ns)
        first.build_memory_from_conversation(["Anna booked a flight to Lisbon."])
        assert _pg_counts(dsn, ns)["memory_nodes"] >= 1

        fresh = _pg_system(dsn, dim, ns)
        cache = Path(fresh.cache_dir)
        (cache / "graph.json").write_text("{not json")
        (cache / "keyword_index.json").write_text('{"ghost": ["nope"]}')
        fresh.load_memory()  # previously: AttributeError, no attribute 'load'

        assert "ghost" not in fresh.memory_builder.node_index
        assert len(fresh.graph_db.nodes) >= 1
    finally:
        pg_common.namespace_delete(dsn, ns)
        pg_common.reset_pool()
