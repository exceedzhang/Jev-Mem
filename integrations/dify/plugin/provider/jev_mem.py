from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from utils.jmem_client import JMemClient, JMemClientError


class JevMemProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        base_url = (credentials.get("base_url") or "").strip()
        if not base_url:
            raise ToolProviderCredentialValidationError("Service URL is required")
        try:
            client = JMemClient(base_url, api_key=credentials.get("api_key"))
            client.health()
        except JMemClientError as exc:
            raise ToolProviderCredentialValidationError(str(exc))
        except Exception as exc:
            raise ToolProviderCredentialValidationError(f"Cannot reach the Jev-Mem service: {exc}")
