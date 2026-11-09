from typing import Any

from utils.jmem_client import JMemClient, JMemClientError

RESPONSE_FORMATS = ("both", "json", "text")


def normalize_response_format(value: Any) -> str:
    """Validate the ``response_format`` tool parameter.

    ``both`` (default) yields a JSON message and a text message, ``json``
    yields only the structured ``json`` output, ``text`` only the ``text``
    output.
    """
    fmt = str(value if value is not None else "both").strip().lower()
    if fmt not in RESPONSE_FORMATS:
        raise JMemClientError(
            "Parameter 'response_format' must be one of "
            f"{', '.join(RESPONSE_FORMATS)}; got {value!r}."
        )
    return fmt


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