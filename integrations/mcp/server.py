"""Jev-Mem MCP server: expose the sidecar's memory API to MCP agents.

Thin client — all memory logic (models, namespaces, storage) stays in the
Jev-Mem sidecar; this process only translates MCP tool calls into HTTP
requests (see SPEC-mcp.md).

Run:

    JEV_MEM_MCP_BASE_URL=http://localhost:8700 \
    JEV_MEM_MCP_API_KEY=... \
    python server.py

Then point an agent (e.g. OpenCode) at ``http://<host>:8701/sse``.
"""

from __future__ import annotations

import json
import os

try:
    from . import jmem_client
except ImportError:  # loaded by file path or run as a script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jmem_client

from mcp.server.mcpserver import MCPServer

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8701
RESULT_MODES = ("answer", "memory")


def load_server_config() -> tuple:
    """Client config plus the SSE bind address (fails fast on bad values)."""
    config = jmem_client.load_config()
    host = (os.getenv("JEV_MEM_MCP_HOST", "") or "").strip() or DEFAULT_HOST
    raw_port = (os.getenv("JEV_MEM_MCP_PORT", "") or "").strip() or str(DEFAULT_PORT)
    try:
        port = int(raw_port)
    except ValueError:
        raise jmem_client.JMemConfigError(
            f"JEV_MEM_MCP_PORT must be an integer (got {raw_port!r})"
        )
    return config, host, port


def _get_client() -> "jmem_client.JMemClient":
    return jmem_client.JMemClient(jmem_client.load_config())


def _to_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _call(op) -> str:
    """Run one client call and return a single JSON string.

    Failures are returned as ``{"error": ..., "detail": ...}`` so the agent
    always gets structured output instead of a traceback.
    """
    try:
        return _to_json(op(_get_client()))
    except jmem_client.JMemClientError as exc:
        return _to_json({"error": "sidecar error", "detail": str(exc)})
    except jmem_client.JMemConfigError as exc:
        return _to_json({"error": "configuration error", "detail": str(exc)})


def _answer_synthesis(result_mode) -> bool:
    """Map the ``result_mode`` tool parameter to the sidecar's answer flag."""
    mode = str(result_mode if result_mode is not None else "answer").strip().lower()
    if mode not in RESULT_MODES:
        raise ValueError(
            "result_mode must be one of "
            f"{', '.join(RESULT_MODES)}; got {result_mode!r}"
        )
    return mode == "answer"


app = MCPServer("jev-mem")


@app.tool(description="Record a memory in the Jev-Mem namespace.")
def record_memory(text: str, timestamp: str | None = None,
                  source: str | None = None) -> str:
    return _call(lambda c: c.ingest(text, timestamp=timestamp, source=source))


@app.tool(description=(
    "Recall from long-term memory. result_mode='answer' (default) returns a "
    "synthesized answer; result_mode='memory' returns the raw memory records "
    "so your own model can combine them with the current context."
))
def recall_memory(question: str, top_k: int = 15,
                  result_mode: str = "answer") -> str:
    try:
        answer = _answer_synthesis(result_mode)
    except ValueError as exc:
        return _to_json({"error": "invalid parameter", "detail": str(exc)})
    return _call(lambda c: c.query(question, top_k=top_k, answer=answer))


@app.tool(description="Consolidate one stored memory (merge/relate it to related memories).")
def consolidate_memory(memory_id: str) -> str:
    return _call(lambda c: c.consolidate(memory_id))


@app.tool(description="Return storage statistics (nodes, events, links, vectors) for the namespace.")
def memory_stats() -> str:
    return _call(lambda c: c.stats())


def main() -> None:
    _config, host, port = load_server_config()
    app.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
