import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.base import JMemBaseTool
from utils.jmem_client import JMemClientError


class QueryMemoryTool(JMemBaseTool, Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        question = (tool_parameters.get("question") or "").strip()
        if not question:
            yield self.create_text_message("Parameter 'question' is required.")
            return
        try:
            top_k = int(tool_parameters.get("top_k") or 15)
        except (TypeError, ValueError):
            top_k = 15
        try:
            result = self._client().query(question, top_k=top_k)
        except JMemClientError as exc:
            yield self.create_text_message(f"Jev-Mem query failed: {exc}")
            return
        trace = result.get("trace") or {}
        payload = {
            "answer": result.get("answer"),
            "evidence": result.get("evidence"),
            "generated": result.get("generated"),
            "controller": result.get("controller"),
            "stopping_decision": result.get("stopping_decision"),
            "nodes_visited": result.get("nodes_visited"),
            "edges_examined": result.get("edges_examined"),
            "jev_calls": result.get("jev_calls"),
            "retrieval_depth": result.get("retrieval_depth"),
            "query_id": trace.get("query_id"),
        }
        yield self.create_json_message(payload)
        # Text message so the tool node's ``text`` output is usable directly in
        # downstream nodes (Dify 1.18 exposes JSON messages as ``json`` only).
        answer = (result.get("answer") or "").strip()
        yield self.create_text_message(
            answer or json.dumps(payload, ensure_ascii=False, default=str)
        )
