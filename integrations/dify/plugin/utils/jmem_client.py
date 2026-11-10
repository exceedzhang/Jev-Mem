"""Minimal HTTP client for the Jev-Mem sidecar service."""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class JMemClientError(Exception):
    """Friendly error surfaced to the tool caller."""


class JMemClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 namespace: str = "default", timeout: int = 60):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.namespace = namespace or "default"
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(method, url, json=json_body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise JMemClientError(
                f"Cannot connect to the Jev-Mem service at {self.base_url}. "
                "Check the Service URL and that the service is running."
            ) from exc
        except requests.Timeout as exc:
            raise JMemClientError(f"The Jev-Mem service timed out after {self.timeout}s.") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise JMemClientError(f"Jev-Mem service error {resp.status_code}: {detail}")
        if not resp.content:
            return {}
        return resp.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ingest(self, text: str, timestamp: Optional[str] = None,
               metadata: Optional[dict] = None) -> dict:
        body: Dict[str, Any] = {
            "text": text,
            "namespace": self.namespace,
            "metadata": metadata or {},
        }
        if timestamp:
            body["timestamp"] = timestamp
        return self._request("POST", "/v1/memory/ingest", body)

    def query(self, question: str, top_k: int = 15, answer: bool = True,
          sort: str = "time") -> dict:
        return self._request("POST", "/v1/memory/query", {
            "question": question,
            "namespace": self.namespace,
            "top_k": top_k,
            "answer": answer,
            "sort": sort,
        })

    def consolidate(self, memory_id: str) -> dict:
        return self._request("POST", "/v1/memory/consolidate", {
            "memory_id": memory_id,
            "namespace": self.namespace,
        })

    def stats(self) -> dict:
        from urllib.parse import quote

        return self._request("GET", f"/v1/memory/stats?namespace={quote(self.namespace)}")
