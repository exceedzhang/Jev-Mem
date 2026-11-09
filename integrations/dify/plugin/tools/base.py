from typing import Any

from utils.jmem_client import JMemClient, JMemClientError


class JMemBaseTool:
    def _client(self) -> JMemClient:
        credentials: dict[str, Any] = getattr(self.runtime, "credentials", None) or {}
        base_url = (credentials.get("base_url") or "").strip()
        if not base_url:
            raise JMemClientError("Service URL is not configured for the Jev-Mem provider.")
        return JMemClient(
            base_url,
            api_key=credentials.get("api_key"),
            namespace=credentials.get("namespace") or "default",
        )
