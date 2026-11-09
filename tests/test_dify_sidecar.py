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