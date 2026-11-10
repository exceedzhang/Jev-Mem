# Jev-Mem MCP server

Exposes the Jev-Mem sidecar to MCP agents (OpenCode, Claude, any MCP client)
over **HTTP/SSE**. The server is a thin client: every tool call is translated
into one HTTP request against the sidecar's `/v1/memory/*` endpoints, so all
models, storage backends, namespace handling, and authentication stay in the
sidecar process.

```
MCP agent (OpenCode, ...)
   │  SSE  http://<host>:8701/sse
   ▼
integrations/mcp/server.py          (this directory)
   │  HTTP + Bearer
   ▼
Jev-Mem sidecar  /v1/memory/*       (file or PostgreSQL storage)
```

## Tools

| Tool | Sidecar endpoint | Notes |
|---|---|---|
| `record_memory(text, timestamp?, source?)` | `POST /v1/memory/ingest` | records one memory; returns `{stored, memory_id, ...}` |
| `recall_memory(question, top_k=15)` | `POST /v1/memory/query` | retrieves and synthesizes an answer |
| `consolidate_memory(memory_id)` | `POST /v1/memory/consolidate` | runs consolidation for one memory |
| `memory_stats()` | `GET /v1/memory/stats` | node/link/vector counts for the namespace |

Every tool returns a single JSON string. Failures are returned as
`{"error": "...", "detail": "..."}` rather than raised.

## Run

Requires the sidecar to be reachable (real or mock mode).

```bash
pip install -r integrations/mcp/requirements.txt   # mcp[cli]

export JEV_MEM_MCP_BASE_URL=http://localhost:8700   # the sidecar
export JEV_MEM_MCP_API_KEY=<JEV_MEM_API_KEY>        # sidecar API key
python integrations/mcp/server.py                    # SSE on 0.0.0.0:8701
```

Environment variables (see `.env.example` in this directory):

| Variable | Default | Meaning |
|---|---|---|
| `JEV_MEM_MCP_BASE_URL` | — (required) | Sidecar base URL, no trailing slash needed |
| `JEV_MEM_MCP_API_KEY` | — (required) | Bearer key accepted by the sidecar |
| `JEV_MEM_MCP_NAMESPACE` | `opencode` | One namespace per deployment |
| `JEV_MEM_MCP_HOST` | `0.0.0.0` | Bind address for the SSE server |
| `JEV_MEM_MCP_PORT` | `8701` | SSE port (the sidecar stays on 8700) |
| `JEV_MEM_MCP_TEST_URL` | — (optional) | If set, the live smoke test runs against it |

Missing `JEV_MEM_MCP_BASE_URL` / `JEV_MEM_MCP_API_KEY` fail fast with a clear
message. Behind an HTTP proxy, set `NO_PROXY=localhost,127.0.0.1` so clients
don't route the SSE connection through the proxy.

## Register in OpenCode

`~/.config/opencode/opencode.json` (or project `.opencode/`):

```json
{
  "mcp": {
    "jev-mem": {
      "type": "remote",
      "url": "http://localhost:8701/sse",
      "enabled": true
    }
  }
}
```

Other MCP clients: point them at the same `http://<host>:8701/sse` endpoint.

## Tests

```bash
python -m pytest tests/test_mcp_server.py -q
```

Unit tests run without a sidecar (a fake transport records the HTTP calls the
tools would make). One live smoke test is skipped unless
`JEV_MEM_MCP_TEST_URL` points at a running server.
