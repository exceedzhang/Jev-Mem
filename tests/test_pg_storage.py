"""PG-gated tests for the Postgres storage backends.

Every test skips (not fails) when JEV_MEM_PG_DSN is unset, so the default
suite runs without a database. With a DSN set (e.g. a local
pgvector/pgvector:pg18 container) the tests exercise PgVectorStore against
a live database, using a throwaway namespace per test.
"""

import os

import numpy as np
import pytest

from memory import pg_common
from memory.pg_vector_store import PgVectorStore

_DSN = os.getenv("JEV_MEM_PG_DSN", "").strip()
_DEFAULT_DIM = 16  # used when the test database has no schema yet

pytestmark = pytest.mark.skipif(
    not _DSN, reason="JEV_MEM_PG_DSN not set (PG-gated tests)"
)


@pytest.fixture(scope="session")
def dsn():
    assert _DSN, "JEV_MEM_PG_DSN is required for PG-gated tests"
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
