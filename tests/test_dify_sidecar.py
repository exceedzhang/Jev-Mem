"""Tests for the Dify sidecar embedding backend selection (API vs local)."""
import importlib.util
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

SERVICE_APP = Path(__file__).resolve().parents[1] / "integrations" / "dify" / "service" / "app.py"


def _prep_env(monkeypatch, *, mock="0", jev_mem_embedding="", openai_embedding="",
              openai_api_key=None):
    monkeypatch.setenv("JEV_MEM_MOCK", mock)
    for name in ("JEV_MEM_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL",
                 "OPENAI_EMBEDDING_DIMENSIONS", "OPENAI_API_KEY",
                 "JEV_BASE_URL", "JEV_API_KEY", "JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)
    if jev_mem_embedding:
        monkeypatch.setenv("JEV_MEM_EMBEDDING_MODEL", jev_mem_embedding)
    if openai_embedding:
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", openai_embedding)
    if openai_api_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)


@contextmanager
def _isolated_environ():
    """Restore os.environ after import: load_dotenv() mutates it globally."""
    original = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _stage_service(tmp_path, env_text=None):
    """Copy app.py into a temp service dir so the repo's real .env is not read."""
    service = tmp_path / "service"
    service.mkdir()
    shutil.copy(SERVICE_APP, service / "app.py")
    if env_text is not None:
        (service / ".env").write_text(env_text)
    return service / "app.py"


def _load_sidecar(monkeypatch, module_name, app_path, tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("JEV_MEM_HOME", str(home))
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    with _isolated_environ():
        spec.loader.exec_module(module)
        # Snapshot for assertions after os.environ is restored.
        module._test_environ = dict(os.environ)
    return module


def test_select_embedding_backend_matrix(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, mock="1")
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_select", app_file, tmp_path)
    select = app.select_embedding_backend
    assert select("", "") == "minilm"
    assert select("", "text-embedding-3-small") == "openai"
    assert select("openai", "") == "openai"
    assert select("minilm", "text-embedding-3-small") == "minilm"
    assert select(" MINILM ", "") == "minilm"
    with pytest.raises(ValueError, match="JEV_MEM_EMBEDDING_MODEL"):
        select("bogus", "")


def test_openai_embedding_env_selects_api_backend(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, openai_embedding="text-embedding-3-small", openai_api_key="test-key")
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_api", app_file, tmp_path)
    assert app.EMBEDDING_BACKEND == "openai"
    assert app.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"


def test_openai_embedding_without_key_fails_fast(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, openai_embedding="text-embedding-3-small")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _load_sidecar(monkeypatch, "jev_mem_sidecar_no_key", app_file, tmp_path)


def test_explicit_minilm_override_keeps_local(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, jev_mem_embedding="minilm",
              openai_embedding="text-embedding-3-small", openai_api_key="test-key")
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_local_override", app_file, tmp_path)
    assert app.EMBEDDING_BACKEND == "minilm"


def test_invalid_explicit_backend_rejected_at_start(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, jev_mem_embedding="bogus", openai_api_key="test-key")
    with pytest.raises(ValueError, match="JEV_MEM_EMBEDDING_MODEL"):
        _load_sidecar(monkeypatch, "jev_mem_sidecar_invalid", app_file, tmp_path)


def test_service_env_file_supplies_missing_keys(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path, env_text=(
        "OPENAI_EMBEDDING_MODEL=from-file\n"
        "OPENAI_API_KEY=file-key\n"
        "JEV_BASE_URL=http://127.0.0.1:8900\n"
        "JEV_MODEL=agent-model\n"
    ))
    _prep_env(monkeypatch)
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_env_file", app_file, tmp_path)
    assert app.EMBEDDING_BACKEND == "openai"
    assert app.OPENAI_EMBEDDING_MODEL == "from-file"
    assert app._test_environ["JEV_BASE_URL"] == "http://127.0.0.1:8900"
    assert app._test_environ["JEV_MODEL"] == "agent-model"


def test_shell_env_beats_service_env_file(monkeypatch, tmp_path):
    app_file = _stage_service(tmp_path, env_text=(
        "OPENAI_EMBEDDING_MODEL=from-file\n"
        "OPENAI_API_KEY=file-key\n"
    ))
    _prep_env(monkeypatch, openai_embedding="from-shell", openai_api_key="shell-key")
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_env_shell", app_file, tmp_path)
    assert app.EMBEDDING_BACKEND == "openai"
    assert app.OPENAI_EMBEDDING_MODEL == "from-shell"


def test_vector_encoder_resolves_openai_embedding_model_env(monkeypatch):
    from memory.vector_db import VectorEncoder
    # An explicit dimension keeps construction offline: without one the encoder
    # probes the endpoint for the model's native dimension.
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-embed")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    encoder = VectorEncoder(use_openai=True)
    assert encoder.use_openai
    assert encoder.encoder.model_name == "custom-embed"
    assert encoder.encoder.send_dimensions is True
    assert encoder.dimension == 1536
_PG_DSN = os.getenv("JEV_MEM_PG_DSN", "").strip()


def test_sidecar_registry_file_mode_ignores_db(monkeypatch, tmp_path):
    """File mode: registry registration is a no-op and never touches a DB."""
    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, mock="1")
    monkeypatch.delenv("JEV_MEM_STORAGE", raising=False)
    monkeypatch.delenv("JEV_MEM_PG_DSN", raising=False)
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_fileRegistry", app_file, tmp_path)
    assert app.STORAGE == "file"
    registry = app.NamespaceRegistry()
    registry._register_in_db("some-ns")  # no-op in filemode
    assert registry.names() == []


@pytest.mark.skipif(not _PG_DSN, reason="JEV_MEM_PG_DSN not set (PG-gated test)")
def test_sidecar_registry_recovers_namespaces_from_postgres(monkeypatch, tmp_path):
    """Postgres mode: the table is the registry, so a fresh process (empty
    in-process dict) still lists the namespaces it created before the restart."""
    import uuid

    from memory import pg_common

    app_file = _stage_service(tmp_path)
    _prep_env(monkeypatch, mock="1")
    monkeypatch.setenv("JEV_MEM_STORAGE", "postgres")
    monkeypatch.setenv("JEV_MEM_PG_DSN", _PG_DSN)
    app = _load_sidecar(monkeypatch, "jev_mem_sidecar_pgRegistry", app_file, tmp_path)
    assert app.STORAGE == "postgres"

    ns = f"sidecar-{uuid.uuid4().hex[:8]}"
    try:
        registry = app.NamespaceRegistry()
        registry._register_in_db(ns)
        assert pg_common.namespace_exists(_PG_DSN, ns)

        fresh = app.NamespaceRegistry()  # simulate restart
        assert not fresh._systems
        assert ns in fresh.names(), "names() must come from the table"
    finally:
        pg_common.namespace_delete(_PG_DSN, ns)
        pg_common.reset_pool()


def test_query_endpoint_passes_sort_order(monkeypatch, tmp_path):
    """The /v1/memory/query endpoint forwards a validated, normalized sort."""
    _prep_env(monkeypatch, mock="1")
    app_module = _load_sidecar(monkeypatch, "sidecar_sort_app", _stage_service(tmp_path), tmp_path)
    monkeypatch.setattr(app_module, "API_KEY", "")  # auth off

    calls = []

    class _FakeQueryEngine:
        def query(self, question, top_k=15, sort="time"):
            # Mirrors the real engine: order_evidence_nodes rejects unknown orders.
            if sort not in ("time", "similarity"):
                raise ValueError(f"sort must be one of time, similarity; got {sort!r}")
            calls.append((question, top_k, sort))

            class _Ctx:
                metadata = {}

            return _Ctx(), "evidence text"

    class _FakeSystem:
        query_engine = _FakeQueryEngine()
        llm_controller = None

    class _FakeRegistry:
        def get(self, namespace):
            return _FakeSystem()

        def names(self):
            return []

    monkeypatch.setattr(app_module, "registry", _FakeRegistry())

    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)

    r = client.post("/v1/memory/query", json={"question": "q"})
    assert r.status_code == 200
    assert r.json()["evidence"] == "evidence text"
    assert calls == [("q", 15, "time")]

    r = client.post("/v1/memory/query", json={"question": "q", "sort": "SIMILARITY"})
    assert r.status_code == 200
    assert calls[-1] == ("q", 15, "similarity")

    r = client.post("/v1/memory/query", json={"question": "q", "sort": "bogus"})
    assert r.status_code == 422
    assert "sort" in r.json()["detail"]


def test_make_summarizer_returns_none_without_llm(monkeypatch, tmp_path):
    """Mock mode (no System-Two LLM) keeps the decision-only consolidate behavior."""
    _prep_env(monkeypatch, mock="1")
    app_module = _load_sidecar(monkeypatch, "sidecar_summarizer_app", _stage_service(tmp_path), tmp_path)

    class _System:
        llm_controller = None

    assert app_module._make_summarizer(_System()) is None


def test_make_summarizer_calls_llm_and_strips_output(monkeypatch, tmp_path):
    _prep_env(monkeypatch, mock="1")
    app_module = _load_sidecar(monkeypatch, "sidecar_summarizer_app2", _stage_service(tmp_path), tmp_path)

    calls = []

    class _FakeLLM:
        def get_completion(self, prompt, response_format=None, temperature=None):
            calls.append((prompt, response_format, temperature))
            return "  Anna plans her week in Notion every Monday.  "

    class _FakeLLMController:
        llm = _FakeLLM()

    class _System:
        llm_controller = _FakeLLMController()

    summarize = app_module._make_summarizer(_System())
    result = summarize(["Anna plans in Notion.", "Every Monday, Notion, Anna."])
    assert result == "Anna plans her week in Notion every Monday."
    prompt, response_format, temperature = calls[0]
    assert "Anna plans in Notion." in prompt
    assert "Every Monday, Notion, Anna." in prompt
    assert response_format == {"type": "text"}
    assert temperature == 0.0


def test_make_summarizer_raises_on_empty_completion(monkeypatch, tmp_path):
    _prep_env(monkeypatch, mock="1")
    app_module = _load_sidecar(monkeypatch, "sidecar_summarizer_app3", _stage_service(tmp_path), tmp_path)

    class _FakeLLM:
        def get_completion(self, prompt, response_format=None, temperature=None):
            return "   "

    class _FakeLLMController:
        llm = _FakeLLM()

    class _System:
        llm_controller = _FakeLLMController()

    summarize = app_module._make_summarizer(_System())
    with pytest.raises(RuntimeError, match="empty"):
        summarize(["a", "b"])


def test_registry_eviction_stops_the_consolidation_worker(tmp_path, monkeypatch):
    """An evicted namespace's background consolidation worker is stopped."""
    _prep_env(monkeypatch, mock="1")
    app_module = _load_sidecar(monkeypatch, "sidecar_evict_app", _stage_service(tmp_path), tmp_path)
    monkeypatch.setattr(app_module, "PROFILE_PATH",
                        str(Path(__file__).resolve().parents[1] / "config" / "jev_mem.json"))
    monkeypatch.setattr(app_module, "MAX_NAMESPACES", 2)
    registry = app_module.NamespaceRegistry()
    first = registry.get("ns-a")
    first.memory_builder._ensure_worker()
    worker = first.memory_builder._worker
    assert worker is not None and worker.is_alive()
    try:
        registry.get("ns-b")
        registry.get("ns-c")  # evicts ns-a
    finally:
        first.memory_builder.stop_worker()
    assert first.memory_builder._worker is None
    assert not worker.is_alive()
