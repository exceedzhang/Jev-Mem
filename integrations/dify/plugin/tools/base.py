from typing import Any

from utils.jmem_client import JMemClient, JMemClientError
from utils import params as _params

RESPONSE_FORMATS = _params.RESPONSE_FORMATS
RESULT_MODES = _params.RESULT_MODES
SORT_ORDERS = _params.SORT_ORDERS

__all__ = [
    "JMemBaseTool",
    "RESPONSE_FORMATS",
    "RESULT_MODES",
    "SORT_ORDERS",
    "normalize_response_format",
    "normalize_result_mode",
    "normalize_sort_order",
]


def normalize_result_mode(value: Any) -> str:
    """Validate ``result_mode``; invalid values raise ``JMemClientError``."""
    try:
        return _params.normalize_result_mode(value)
    except ValueError as exc:
        raise JMemClientError(str(exc)) from exc


def normalize_response_format(value: Any) -> str:
    """Validate ``response_format``; invalid values raise ``JMemClientError``."""
    try:
        return _params.normalize_response_format(value)
    except ValueError as exc:
        raise JMemClientError(str(exc)) from exc


def normalize_sort_order(value: Any) -> str:
    """Validate ``sort``; invalid values raise ``JMemClientError``."""
    try:
        return _params.normalize_sort_order(value)
    except ValueError as exc:
        raise JMemClientError(str(exc)) from exc


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