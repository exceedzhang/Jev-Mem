import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.base import JMemBaseTool
from utils.jmem_client import JMemClientError


class ConsolidateMemoryTool(JMemBaseTool, Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        memory_id = (tool_parameters.get("memory_id") or "").strip()
        if not memory_id:
            yield self.create_text_message(
                "Parameter 'memory_id' is required. Use the id returned by ingest_memory."
            )
            return
        try:
            result = self._client().consolidate(memory_id)
        except JMemClientError as exc:
            yield self.create_text_message(f"Jev-Mem consolidation failed: {exc}")
            return
        payload = result if isinstance(result, dict) else {"result": result}
        yield self.create_json_message(payload)
        yield self.create_text_message(json.dumps(payload, ensure_ascii=False, default=str))
