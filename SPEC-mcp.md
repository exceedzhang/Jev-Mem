# Spec: MCP server for Jev-Mem memory tools

## Objective

Explain to any MCP-capable agent (OpenCode and similar) how to operate a
Jev-Mem namespace: **record** a memory, **recall** memories, **consolidate** a
memory, and **query memory statistics**. The agent gets four tools with the
same semantics as the Dify plugin, so behavior can be cross-checked between
the two integrations.

Architecture: a **thin MCP client** — the MCP server performs protocol
translation only and calls the existing sidecar's `/v1/memory/*` endpoints
over HTTP. Model calls, namespaces, and authentication all stay in the
sidecar; the MCP process stays lightweight (no torch / sentence-transformers).

Transport: **HTTP/SSE** — the MCP server runs as a standalone service the
agent connects to over the network (remote MCP), deployable on another host
than the agent.

Consumers: OpenCode (primary, local or remote), any other MCP client.

Acceptance criteria:
1. `python integrations/mcp/server.py` starts a service exposing an SSE MCP
   endpoint; an MCP client can list the 4 tools.
2. Each tool hits the correct sidecar endpoint with the correct payload and
   maps results/errors to readable tool output.
3. All tests pass with no sidecar and no Postgres (unit level uses a fake
   client); the existing full suite stays green.
4. Documented OpenCode registration works end-to-end (manual check).

## Tech Stack

- Python 3.13 (same `.venv` as the repo), sync code only.
- `mcp` (official Python SDK, FastMCP) with SSE transport — new dependency
  (`mcp[cli]` in `integrations/mcp/requirements.txt`).
- Sidecar HTTP client: stdlib `urllib` (same style as the Dify plugin's
  `jmem_client.py`), no new HTTP dependency.
- Sidecar endpoints (unchanged): `POST /v1/memory/ingest`,
  `POST /v1/memory/query`, `POST /v1/memory/consolidate`,
  `GET /v1/memory/stats` — all `Authorization: Bearer <JEV_MEM_MCP_API_KEY>`.

## Commands

```bash
# run the MCP server (env config)
JEV_MEM_MCP_BASE_URL=http://localhost:8700 JEV_MEM_MCP_API_KEY=... \
  .venv/bin/python integrations/mcp/server.py

# tests (unit level; no sidecar needed)
.venv/bin/python -m pytest -q

# release gate
.venv/bin/python scripts/check_public_release.py --tracked
```

Configuration (env vars, fail fast if base URL/key missing):

| Var | Default | Meaning |
| --- | --- | --- |
| `JEV_MEM_MCP_BASE_URL` | — (required) | Sidecar base URL, e.g. `http://localhost:8700` |
| `JEV_MEM_MCP_API_KEY` | — (required) | Bearer key, same as the sidecar's `JEV_MEM_API_KEY` |
| `JEV_MEM_MCP_NAMESPACE` | `opencode` | Namespace the server operates on |
| `JEV_MEM_MCP_HOST` | `0.0.0.0` | Bind address |
| `JEV_MEM_MCP_PORT` | `8701` | SSE port (sidecar stays on 8700) |

## Project Structure

```
integrations/mcp/
  server.py           # FastMCP app + 4 tools + main() (SSE transport)
  jmem_client.py      # thin urllib client: ingest/query/consolidate/stats
  requirements.txt    # mcp[cli]
  README.md           # run instructions + OpenCode registration example
  .env.example        # the 5 vars above, placeholders only
tests/test_mcp_server.py   # unit tests (fake client), live smoke gated on env
```

No changes to the sidecar, the core library, or existing integrations.

## Code Style

Match the Dify plugin client: small functions, stdlib `urllib.request`,
explicit error types. Sketch:

```python
def query(base_url: str, api_key: str, namespace: str,
          question: str, top_k: int = 15) -> dict:
    payload = json.dumps({"question": question, "top_k": top_k}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/memory/query", data=payload, method="POST",
        headers={**auth_headers(api_key), "Namespace": namespace,
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())
```

Tools are thin wrappers: validate inputs → call client → return a single
JSON string (structured payload, same fields the Dify plugin's `json`
output carries). Failures return `{"error": "...", "detail": "..."}` as tool
output (never a traceback to the agent).

## Testing Strategy

1. **Unit (always run, no sidecar/PG needed):** import the server module with
   a fake client (module-level monkeypatch of `jmem_client` functions) and
   assert per tool: correct endpoint, payload fields, namespace header,
   output JSON shape; error mapping (HTTP 401/404/500 → readable
   `{"error": ...}`). Also: server construction fails fast without
   `JEV_MEM_MCP_BASE_URL`/`API_KEY`.
2. **Live smoke (env-gated, skip if unset):** with `JEV_MEM_MCP_TEST_URL`
   pointing at a running sidecar, one test per tool round-trips against it
   (mirrors how `tests/test_dify_sidecar.py` gates on a real app).
3. **Manual (documented in the MCP README):** register in OpenCode and call
   all 4 tools from an agent session.

OpenCode registration example (documented):

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

## Boundaries

- **Always:**
  - Run the full suite before committing; file mode and existing
    integrations stay untouched.
  - Tool names/semantics stay 1:1 with the Dify plugin
    (`record_memory`/`recall_memory`/`consolidate_memory`/`memory_stats`).
  - All sidecar credentials via env; fail fast on missing config.
- **Ask first:**
  - Adding the `mcp` dependency to `integrations/mcp/requirements.txt`.
  - Any change to the sidecar itself (expected: none).
  - Multi-tenant namespace routing (currently: one namespace per server
    deployment, by design).
- **Never:**
  - Commit real keys or a filled `.env`.
  - Embed `JevMemSystem` / model code in the MCP process (thin client only).
  - Add a database to the MCP server (stateless).
  - Remove/skip failing tests to get green.

## Success Criteria

- [ ] `server.py` starts, exposes SSE MCP; an MCP client lists the 4 tools.
- [ ] Each tool → correct sidecar endpoint + payload; errors mapped to
      readable JSON output.
- [ ] Unit tests pass with no sidecar; full suite green (baseline 241 passed
      + 27 skipped without DSN).
- [ ] OpenCode registration documented and manually verified.
- [ ] `scripts/check_public_release.py --tracked` clean.

## Open Questions

- Port 8701 for the MCP SSE endpoint (sidecar keeps 8700) — ok?
- One namespace per MCP deployment (set via `JEV_MEM_MCP_NAMESPACE`);
  multi-tenant = multiple deployments. Acceptable for now?
