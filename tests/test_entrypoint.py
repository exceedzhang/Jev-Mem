"""Unit tests for the dual-service container entrypoint (no real processes)."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SERVICE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "dify" / "service"


def _load_entrypoint():
    if "entrypoint" in sys.modules:
        return sys.modules["entrypoint"]
    spec = importlib.util.spec_from_file_location(
        "entrypoint", _SERVICE_DIR / "entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["entrypoint"] = module
    spec.loader.exec_module(module)
    return module


entrypoint = _load_entrypoint()


class _FakeProcess:
    """A finished process: poll() immediately reports the stored return code."""

    def __init__(self, rc: int):
        self.returncode = rc

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return self.returncode


class _FlakyProcess:
    """Stays "alive" for a number of poll() calls, then reports rc."""

    def __init__(self, alive_polls: int, rc: int):
        self._remaining = alive_polls
        self._rc = rc
        self.terminated = False

    @property
    def returncode(self):
        return self._rc

    def poll(self):
        if self._remaining > 0:
            self._remaining -= 1
            return None
        return self._rc

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self._rc


@pytest.fixture()
def _clean_mcp_env(monkeypatch):
    # main() loads the real service .env through python-dotenv, which mutates
    # os.environ behind monkeypatch's back and would leak into later tests.
    # Neither the dotenv load nor the co-located defaults are what these tests
    # exercise, so both are contained here.
    monkeypatch.setattr(entrypoint, "load_service_env", lambda: None)
    monkeypatch.delenv("JEV_MEM_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MEM_MCP_BASE_URL", raising=False)
    monkeypatch.delenv("JEV_MEM_MCP_API_KEY", raising=False)
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_start_services_without_key_skips_mcp(monkeypatch, _clean_mcp_env):
    calls = []

    class _Recorder(_FlakyProcess):
        def __init__(self, args, cwd):
            super().__init__(0, 0)
            calls.append((args, cwd))

    monkeypatch.setattr(entrypoint.subprocess, "Popen", _Recorder)
    sidecar, mcp = entrypoint.start_services()
    assert mcp is None
    assert len(calls) == 1
    assert "uvicorn" in calls[0][0]
    assert sidecar is not None


def test_start_services_with_key_starts_both(monkeypatch, _clean_mcp_env):
    monkeypatch.setenv("JEV_MEM_MCP_API_KEY", "secret")
    calls = []

    class _Recorder(_FlakyProcess):
        def __init__(self, args, cwd):
            super().__init__(0, 0)
            calls.append((args, cwd))

    monkeypatch.setattr(entrypoint.subprocess, "Popen", _Recorder)
    _sidecar, mcp = entrypoint.start_services()
    assert mcp is not None
    assert len(calls) == 2
    assert calls[1][0][1].endswith(os.path.join("integrations", "mcp", "server.py"))
    assert calls[1][1] == str(entrypoint.APP_ROOT)


def test_main_returns_zero_when_sidecar_exits_cleanly(monkeypatch, _clean_mcp_env):
    monkeypatch.setattr(entrypoint, "start_services",
                        lambda: (_FakeProcess(0), None))
    assert entrypoint.main() == 0


def test_main_returns_nonzero_when_sidecar_crashes(monkeypatch, _clean_mcp_env):
    monkeypatch.setattr(entrypoint, "start_services",
                        lambda: (_FakeProcess(1), None))
    assert entrypoint.main() == 1


def test_main_keeps_running_when_mcp_fails(monkeypatch, _clean_mcp_env, capsys):
    mcp = _FlakyProcess(0, 1)  # dies immediately
    sidecar = _FlakyProcess(3, 0)  # outlives the MCP failure
    monkeypatch.setattr(entrypoint, "start_services", lambda: (sidecar, mcp))
    monkeypatch.setattr(entrypoint.time, "sleep", lambda _s: None)
    assert entrypoint.main() == 0
    out = capsys.readouterr().out
    assert "MCP server exited with code 1" in out
    assert "keeps running" in out