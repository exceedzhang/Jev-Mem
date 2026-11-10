from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.base import JMemBaseTool, normalize_response_format
from utils.jmem_client import JMemClientError


class GetMemoryCountTool(JMemBaseTool, Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            response_format = normalize_response_format(tool_parameters.get("response_format"))
        except JMemClientError as exc:
            yield self.create_text_message(str(exc))
            return
        try:
            result = self._client().stats()
        except JMemClientError as exc:
            yield self.create_text_message(f"Jev-Mem stats failed: {exc}")
            return
        # Count = ingested EVENT nodes (entities are byproducts of extraction,
        # not memories themselves), matching the sidecar's `events` stat.
        count = int(result.get("events") or 0)
        payload = {
            "count": count,
            "nodes": result.get("nodes"),
            "events": result.get("events"),
            "links": result.get("links"),
            "vectors": result.get("vectors"),
        }
        if response_format in ("both", "json"):
            yield self.create_json_message(payload)
        if response_format in ("both", "text"):
            # Text output: the count, ready to feed into a code/if-else node.
            yield self.create_text_message(str(count))
