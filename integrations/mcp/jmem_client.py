"""Thin HTTP client for the Jev-Mem sidecar service (stdlib only).

All model calls, storage and namespace management stay in the sidecar; this
module is only protocol translation for the MCP server. Mirrors the Dify
plugin's client contract: namespace travels in the JSON body (query string
for stats), auth is a Bearer token, and every failure becomes a
``JMemClientError`` with a message a tool caller can act on.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_TIMEOUT = 60
DEFAULT_NAMESPACE = "opencode"


class JMemConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


class JMemClientError(Exception):
    """Friendly error surfaced to the MCP tool caller."""


class JMemConfig:
    """Connection settings, from the environment unless overridden."""

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 namespace: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT):
        base_url = base_url if base_url is not None else os.getenv("JEV_MEM_MCP_BASE_URL", "")
        api_key = api_key if api_key is not None else os.getenv("JEV_MEM_MCP_API_KEY", "")
        namespace = namespace if namespace is not None else os.getenv("JEV_MEM_MCP_NAMESPACE", "")
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.namespace = (namespace or "").strip() or DEFAULT_NAMESPACE
        self.timeout = timeout

    def validate(self) -> None:
        missing = [
            name
            for name, value in (("JEV_MEM_MCP_BASE_URL", self.base_url),
                                ("JEV_MEM_MCP_API_KEY", self.api_key))
            if not value
        ]
        if missing:
            raise JMemConfigError(
                "missing required configuration: " + ", ".join(missing)
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise JMemConfigError(
                "JEV_MEM_MCP_BASE_URL must start with http:// or https://"
            )


def load_config() -> JMemConfig:
    """Environment-backed config; fails fast when required values are absent."""
    config = JMemConfig()
    config.validate()
    return config


def _open(request: urllib.request.Request, timeout: int):
    """Indirection over urlopen so tests can substitute a fake transport."""
    return urllib.request.urlopen(request, timeout=timeout)


class JMemClient:
    def __init__(self, config: JMemConfig, timeout: Optional[int] = None):
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.namespace = config.namespace
        self.timeout = timeout if timeout is not None else config.timeout

    def _request(self, method: str, path: str,
                 body: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method,
                                         headers=headers)
        try:
            with _open(request, self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise JMemClientError(
                f"Jev-Mem service error {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise JMemClientError(
                f"Cannot connect to the Jev-Mem service at {self.base_url}. "
                "Check JEV_MEM_MCP_BASE_URL and that the sidecar is running."
            ) from exc
        except TimeoutError as exc:
            raise JMemClientError(
                f"The Jev-Mem service timed out after {self.timeout}s."
            ) from exc
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    def ingest(self, text: str, timestamp: Optional[str] = None,
               source: Optional[str] = None) -> dict:
        body: Dict[str, Any] = {"text": text, "namespace": self.namespace}
        if timestamp:
            body["timestamp"] = timestamp
        body["metadata"] = {"source": source} if source else {}
        return self._request("POST", "/v1/memory/ingest", body)

    def query(self, question: str, top_k: int = 15) -> dict:
        return self._request("POST", "/v1/memory/query", {
            "question": question,
            "namespace": self.namespace,
            "top_k": top_k,
            "answer": True,
        })

    def consolidate(self, memory_id: str) -> dict:
        return self._request("POST", "/v1/memory/consolidate", {
            "memory_id": memory_id,
            "namespace": self.namespace,
        })

    def stats(self) -> dict:
        namespace = urllib.parse.quote(self.namespace, safe="")
        return self._request(
            "GET", f"/v1/memory/stats?namespace={namespace}"
        )