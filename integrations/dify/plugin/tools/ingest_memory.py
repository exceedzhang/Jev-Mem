from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.base import JMemBaseTool
from utils.jmem_client import JMemClientError


class IngestMemoryTool(JMemBaseTool, Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        text = (tool_parameters.get("text") or "").strip()
        if not text:
            yield self.create_text_message("Parameter 'text' is required.")
            return
        timestamp = (tool_parameters.get("timestamp") or "").strip() or None
        source = (tool_parameters.get("source") or "dify").strip() or "dify"
        try:
            result = self._client().ingest(text, timestamp=timestamp, metadata={"source": source})
        except JMemClientError as exc:
            yield self.create_text_message(f"Jev-Mem ingest failed: {exc}")
            return
        if result.get("stored"):
            payload = {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "controller": (result.get("node") or {}).get("controller"),
                "memory_type": (result.get("node") or {}).get("memory_type"),
            }
            yield self.create_json_message(payload)
            # Text message: the memory_id, ready to feed into consolidate_memory.
            yield self.create_text_message(str(result.get("memory_id") or ""))
        else:
            reason = result.get("reason") or "rejected"
            yield self.create_json_message({"stored": False, "reason": reason})
            yield self.create_text_message(f"rejected: {reason}")
