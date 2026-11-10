import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.base import (
    JMemBaseTool,
    normalize_response_format,
    normalize_result_mode,
)
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
            result_mode = normalize_result_mode(tool_parameters.get("result_mode"))
        except JMemClientError as exc:
            yield self.create_text_message(str(exc))
            return
        try:
            response_format = normalize_response_format(tool_parameters.get("response_format"))
        except JMemClientError as exc:
            yield self.create_text_message(str(exc))
            return
        try:
            # ``memory`` mode skips the sidecar's answer synthesis so the raw
            # memory records come back for the caller's own model to use.
            result = self._client().query(
                question, top_k=top_k, answer=(result_mode == "answer")
            )
        except JMemClientError as exc:
            yield self.create_text_message(f"Jev-Mem query failed: {exc}")
            return
        trace = result.get("trace") or {}
        payload = {
            "result_mode": result_mode,
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
        # ``json`` -> structured output (tool node ``json`` variable,
        # Array[Object]); ``text`` -> plain text (tool node ``text`` variable).
        if response_format in ("both", "json"):
            yield self.create_json_message(payload)
        if response_format in ("both", "text"):
            # ``memory`` mode hands the raw memory records to the caller so its
            # own model can post-process them; ``answer`` mode keeps the
            # synthesized answer (falling back to the evidence text).
            if result_mode == "memory":
                text = (result.get("evidence") or "").strip()
            else:
                text = (result.get("answer") or "").strip()
            yield self.create_text_message(
                text or json.dumps(payload, ensure_ascii=False, default=str)
            )