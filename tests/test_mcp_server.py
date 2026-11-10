"""Unit tests for the Jev-Mem MCP integration (no sidecar required)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1] / "integrations" / "mcp"


def _load(name: str):
    """Import a module from integrations/mcp without packaging it."""
    spec = importlib.util.spec_from_file_location(
        f"_mcp_{name}", _MCP_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


client = _load("jmem_client")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Recorder:
    """Captures the outgoing request and returns a canned response."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {"ok": True}
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return _FakeResponse(self.payload)

    @property
    def last(self):
        request, _timeout = self.requests[-1]
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        return {
            "method": request.get_method(),
            "url": request.full_url,
            "headers": dict(request.header_items()),
            "body": body,
        }


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(client, "_open", rec)
    return rec


def _config(**overrides):
    values = dict(base_url="http://localhost:8700", api_key="secret",
                  namespace="opencode")
    values.update(overrides)
    return client.JMemConfig(**values)


def test_config_fails_fast_without_required_values(monkeypatch):
    monkeypatch.delenv("JEV_MEM_MCP_BASE_URL", raising=False)
    monkeypatch.delenv("JEV_MEM_MCP_API_KEY", raising=False)
    with pytest.raises(client.JMemConfigError, match="JEV_MEM_MCP_BASE_URL"):
        client.load_config()


def test_config_defaults_namespace_and_reads_env(monkeypatch):
    monkeypatch.setenv("JEV_MEM_MCP_BASE_URL", "http://sidecar:8700/")
    monkeypatch.setenv("JEV_MEM_MCP_API_KEY", "k")
    monkeypatch.delenv("JEV_MEM_MCP_NAMESPACE", raising=False)
    config = client.load_config()
    assert config.base_url == "http://sidecar:8700"
    assert config.namespace == "opencode"


def test_config_rejects_non_http_base_url():
    with pytest.raises(client.JMemConfigError, match="http"):
        _config(base_url="sidecar:8700").validate()


def test_ingest_posts_text_namespace_and_source(recorder):
    result = client.JMemClient(_config()).ingest(
        "I plan the week on Monday 9am with Notion", source="opencode"
    )
    assert result == {"ok": True}
    sent = recorder.last
    assert sent["method"] == "POST"
    assert sent["url"] == "http://localhost:8700/v1/memory/ingest"
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["body"] == {
        "text": "I plan the week on Monday 9am with Notion",
        "namespace": "opencode",
        "metadata": {"source": "opencode"},
    }


def test_query_and_consolidate_carry_namespace(recorder):
    jm = client.JMemClient(_config(namespace="team-a"))
    jm.query("when do I plan the week?", top_k=5)
    assert recorder.last["url"].endswith("/v1/memory/query")
    assert recorder.last["body"] == {
        "question": "when do I plan the week?",
        "namespace": "team-a",
        "top_k": 5,
        "answer": True,
        "sort": "time",
    }
    jm.query("q", sort="similarity", answer=False)
    assert recorder.last["body"]["sort"] == "similarity"
    assert recorder.last["body"]["answer"] is False
    jm.consolidate("mem-1")
    assert recorder.last["url"].endswith("/v1/memory/consolidate")
    assert recorder.last["body"] == {"memory_id": "mem-1",
                                     "namespace": "team-a"}


def test_stats_uses_quoted_query_parameter(recorder):
    client.JMemClient(_config(namespace="team a/b")).stats()
    sent = recorder.last
    assert sent["method"] == "GET"
    assert sent["url"] == (
        "http://localhost:8700/v1/memory/stats?namespace=team%20a%2Fb"
    )
    assert sent["body"] is None


def test_http_error_is_mapped_to_client_error(monkeypatch):
    def _raise(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, None
        )

    monkeypatch.setattr(client, "_open", _raise)
    with pytest.raises(client.JMemClientError, match="401"):
        client.JMemClient(_config()).stats()


def test_connection_error_is_mapped_with_hint(monkeypatch):
    def _raise(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(client, "_open", _raise)
    with pytest.raises(client.JMemClientError, match="sidecar"):
        client.JMemClient(_config()).query("hi")

# --- MCP server tools (Task 13): fake client, no sidecar

# server.py imports the sibling module by its plain name; reuse the
# already-loaded copy so exception classes match across both modules.
sys.modules.setdefault("jmem_client", client)
server = _load("server")


class _FakeMcpClient:
    """Records calls and returns canned payloads."""

    def __init__(self):
        self.calls = []
        self.fail_with = None

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        return {"name": name}

    def ingest(self, text, timestamp=None, source=None):
        return self._record("ingest", text, timestamp=timestamp, source=source)

    def query(self, question, top_k=15, answer=True, sort="time"):
        return self._record("query", question, top_k=top_k, answer=answer, sort=sort)

    def consolidate(self, memory_id):
        return self._record("consolidate", memory_id)

    def stats(self):
        return self._record("stats")


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeMcpClient()
    monkeypatch.setattr(server, "_get_client", lambda: client)
    return client


def test_mcp_tool_record_memory(fake_client):
    out = server.record_memory("text A", timestamp="2026-11-10", source="opencode")
    payload = json.loads(out)
    assert fake_client.calls == [("ingest", ("text A",),
                                  {"timestamp": "2026-11-10", "source": "opencode"})]
    assert payload["name"] == "ingest"


def test_mcp_tool_recall_memory_defaults_top_k(fake_client):
    out = json.loads(server.recall_memory("what happened?"))
    assert fake_client.calls == [("query", ("what happened?",),
                                  {"top_k": 15, "answer": True, "sort": "time"})]
    assert out["name"] == "query"


def test_mcp_tool_recall_memory_mode_memory_skips_synthesis(fake_client):
    server.recall_memory("what happened?", result_mode="memory")
    assert fake_client.calls == [("query", ("what happened?",),
                                  {"top_k": 15, "answer": False, "sort": "time"})]


def test_mcp_tool_recall_memory_rejects_unknown_mode(fake_client):
    out = json.loads(server.recall_memory("x", result_mode="xml"))
    assert out["error"] == "invalid parameter"
    assert "result_mode" in out["detail"]
    assert fake_client.calls == []


def test_mcp_tool_recall_memory_passes_sort_order(fake_client):
    server.recall_memory("q", sort="SIMILARITY")
    assert fake_client.calls == [("query", ("q",),
                                  {"top_k": 15, "answer": True, "sort": "similarity"})]


def test_mcp_tool_recall_memory_rejects_unknown_sort(fake_client):
    out = json.loads(server.recall_memory("q", sort="bogus"))
    assert out["error"] == "invalid parameter"
    assert "sort" in out["detail"]
    assert fake_client.calls == []


def test_mcp_tool_consolidate_and_stats(fake_client):
    server.consolidate_memory("mid-1")
    server.memory_stats()
    assert fake_client.calls == [("consolidate", ("mid-1",), {}), ("stats", (), {})]


def test_mcp_tool_error_returns_structured_json(fake_client):
    fake_client.fail_with = client.JMemClientError("Jev-Mem service error 401: unauthorized")
    out = json.loads(server.record_memory("x"))
    assert out["error"] == "sidecar error"
    assert "401" in out["detail"]


def test_mcp_server_config_port_validation(monkeypatch):
    monkeypatch.setenv("JEV_MEM_MCP_BASE_URL", "https://h")
    monkeypatch.setenv("JEV_MEM_MCP_API_KEY", "k")
    monkeypatch.setenv("JEV_MEM_MCP_PORT", "not-a-port")
    with pytest.raises(client.JMemConfigError, match="JEV_MEM_MCP_PORT"):
        server.load_server_config()


_LIVE_URL = os.environ.get("JEV_MEM_MCP_TEST_URL", "").strip()


@pytest.mark.skipif(not _LIVE_URL, reason="JEV_MEM_MCP_TEST_URL not set (live MCP smoke)")
def test_mcp_live_sse_roundtrip():
    """Live smoke: connect to a running MCP server over SSE and call stats.

    Skipped by default; set JEV_MEM_MCP_TEST_URL=http://localhost:8701/sse
    with the server running (and NO_PROXY=localhost if an HTTP proxy is set).
    """
    import asyncio

    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async def run():
        async with sse_client(_LIVE_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                assert {"record_memory", "recall_memory",
                        "consolidate_memory", "memory_stats"} <= set(names)
                out = await session.call_tool("memory_stats", {})
                payload = json.loads(out.content[0].text)
                assert "namespace" in payload

    asyncio.run(run())
