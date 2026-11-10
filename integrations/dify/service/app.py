"""Jev-Mem sidecar service.

Exposes the Jev-Mem long-term memory (write path, graph-memory plane,
retrieve path) as a small HTTP API that the Dify tool plugin calls.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8700
or:
    python app.py

Environment:
    JEV_MEM_HOME            repo root (default: three levels above this file)
    JEV_MEM_PROFILE         config profile JSON (default: $JEV_MEM_HOME/config/jev_mem.json)
    JEV_MEM_CACHE_DIR       base cache dir (default: $JEV_MEM_HOME/cache/dify)
    JEV_MEM_MOCK            "1" -> fully offline mock Jev + mock encoder
    JEV_MEM_API_KEY         if set, /v1/* requires "Authorization: Bearer <key>"
    LLM_MODEL               System-Two / extraction model (default gpt-4o-mini)
    OPENAI_BASE_URL         OpenAI-compatible endpoint for the LLM (and embeddings)
    OPENAI_API_KEY          API key for that endpoint
    OPENAI_EMBEDDING_MODEL  embedding model name on that endpoint (API embeddings)
    OPENAI_EMBEDDING_DIMENSIONS  optional; unset = auto (probe the endpoint's
                             native dimension, omit the `dimensions` request
                             param — vLLM/Qwen models reject it)
    JEV_BASE_URL            System-One Jev-compatible endpoint (legacy: TYPESAFE_BASE_URL)
    JEV_API_KEY             its API key (legacy: TYPESAFE_API_KEY)
    JEV_MODEL               its decision model name (legacy: TYPESAFE_DEFAULT_MODEL)
    JEV_MEM_EMBEDDING_MODEL optional explicit backend: "openai" (API) or "minilm"
                            (local); unset = API when OPENAI_EMBEDDING_MODEL is set,
                            otherwise local minilm
    JEV_MEM_PORT            default port (default 8700)
    JEV_MEM_STORAGE          "file" (default) or "postgres"
    JEV_MEM_PG_DSN          Postgres connection string (required when
                             JEV_MEM_STORAGE=postgres)
"""
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SERVICE_DIR = Path(__file__).resolve().parent
_HOME = Path(os.getenv("JEV_MEM_HOME") or _SERVICE_DIR.parents[2])
if str(_HOME) not in sys.path:
    sys.path.insert(0, str(_HOME))

from dotenv import load_dotenv  # noqa: E402
from fastapi import Depends, FastAPI, Header, HTTPException, Query  # noqa: E402

# .env support (real environment variables always win): the service-local
# file next to this app is most specific and is loaded first, then the
# repo-root .env fills any keys it does not define.
load_dotenv(_SERVICE_DIR / ".env", override=False)
load_dotenv(_HOME / ".env", override=False)
from pydantic import BaseModel, Field  # noqa: E402

from memory.jev_mem_config import JevMemConfig  # noqa: E402
from memory.memory_builder import MemoryBuilder  # noqa: E402
from memory.trg_memory import TemporalResonanceGraphMemory  # noqa: E402
from memory.vector_db import NumpyVectorDB  # noqa: E402
from memory.mock_encoder import MockEncoder  # noqa: E402
from jev_mem.system import JevMemSystem  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("jev-mem-dify")

PROFILE_PATH = os.getenv("JEV_MEM_PROFILE", str(_HOME / "config" / "jev_mem.json"))
CACHE_BASE = Path(os.getenv("JEV_MEM_CACHE_DIR", str(_HOME / "cache" / "dify")))
MOCK = os.getenv("JEV_MEM_MOCK", "").lower() in ("1", "true", "yes")
API_KEY = os.getenv("JEV_MEM_API_KEY", "")
STORAGE = (os.getenv("JEV_MEM_STORAGE") or "file").strip().lower()
PG_DSN = os.getenv("JEV_MEM_PG_DSN", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("JEV_MEM_EMBEDDING_MODEL", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "")
MAX_NAMESPACES = int(os.getenv("JEV_MEM_MAX_NAMESPACES", "8"))


def select_embedding_backend(jev_mem_embedding_model: str = "", openai_embedding_model: str = "") -> str:
    """Resolve the embedding backend: "openai" (API) or "minilm" (local).

    An explicit JEV_MEM_EMBEDDING_MODEL wins; otherwise setting
    OPENAI_EMBEDDING_MODEL (the API model name) selects the
    OpenAI-compatible API endpoint, keeping minilm as the offline default.
    """
    backend = (jev_mem_embedding_model or "").strip().lower()
    if backend:
        if backend not in ("minilm", "openai"):
            raise ValueError("JEV_MEM_EMBEDDING_MODEL must be 'openai' or 'minilm'")
        return backend
    return "openai" if (openai_embedding_model or "").strip() else "minilm"


def validate_embedding_backend(backend: str, openai_api_key: str = "") -> None:
    """API embeddings require OPENAI_API_KEY; fail fast with a clear message."""
    if backend == "openai" and not (openai_api_key or "").strip():
        raise RuntimeError(
            "API embedding requires OPENAI_API_KEY "
            "(or set JEV_MEM_EMBEDDING_MODEL=minilm to use the local encoder)")


EMBEDDING_BACKEND = "mock" if MOCK else select_embedding_backend(EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL)
if not MOCK:
    validate_embedding_backend(EMBEDDING_BACKEND, os.getenv("OPENAI_API_KEY", ""))

app = FastAPI(title="Jev-Mem Service", version="0.1.0")


class NamespaceRegistry:
    """One JevMemSystem per namespace (Dify app / tenant), LRU-bounded."""

    _TOUCH_INTERVAL = 5.0  # seconds between last_used_at bumps per namespace

    def __init__(self):
        self._lock = threading.RLock()
        self._systems: "OrderedDict[str, JevMemSystem]" = OrderedDict()
        self._last_touch: Dict[str, float] = {}

    def _create(self, namespace: str) -> JevMemSystem:
        cache_dir = CACHE_BASE / namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        config = JevMemConfig.load(PROFILE_PATH)
        if MOCK:
            encoder = MockEncoder()
            vector_db = NumpyVectorDB(encoder.dimension, persist_path=str(cache_dir / "vectors"))
            trg = TemporalResonanceGraphMemory(vector_db=vector_db, encoder=encoder, llm_backend=None)
            mock_config = JevMemConfig.load(PROFILE_PATH, jev_mock=True, write_enabled=True, read_enabled=True)
            builder = MemoryBuilder(str(cache_dir), jev_config=mock_config, trg_memory=trg, llm_enabled=False)
            system = JevMemSystem(cache_dir=str(cache_dir), memory_builder=builder)
        else:
            if STORAGE == "postgres":
                # The sidecar namespace is the tenant: point the PG stores at
                # it instead of the profile's default namespace.
                config = replace(config, namespace=namespace)
            system = JevMemSystem(model=LLM_MODEL, embedding_model=EMBEDDING_BACKEND,
                                  cache_dir=str(cache_dir), jev_config=config)
        system.load_memory()
        logger.info("created namespace %s (mock=%s) at %s", namespace, MOCK, cache_dir)
        return system

    def get(self, namespace: str) -> JevMemSystem:
        if not namespace:
            namespace = "default"
        with self._lock:
            system = self._systems.get(namespace)
            if system is None:
                if len(self._systems) >= MAX_NAMESPACES:
                    evicted_name, evicted = self._systems.popitem(last=False)
                    try:
                        evicted.save_memory()
                    except Exception:
                        logger.exception("failed to save evicted namespace %s", evicted_name)
                    logger.info("evicted namespace %s (limit %d)", evicted_name, MAX_NAMESPACES)
                self._register_in_db(namespace)
                system = self._create(namespace)
                self._systems[namespace] = system
            else:
                self._systems.move_to_end(namespace)
            self._touch(namespace)
            return system

    def _register_in_db(self, namespace: str) -> None:
        """Lazily register the namespace row (postgres mode only)."""
        if STORAGE != "postgres":
            return
        from memory import pg_common
        pg_common.namespace_create(PG_DSN, namespace)

    def _touch(self, namespace: str) -> None:
        """Bump last_used_at, throttled (postgres mode only)."""
        if STORAGE != "postgres":
            return
        now = time.monotonic()
        if now - self._last_touch.get(namespace, 0.0) < self._TOUCH_INTERVAL:
            return
        from memory import pg_common
        pg_common.namespace_touch(PG_DSN, namespace)
        self._last_touch[namespace] = now

    def names(self) -> List[str]:
        with self._lock:
            if STORAGE == "postgres":
                from memory import pg_common
                table = set(pg_common.namespace_list(PG_DSN))
                return sorted(set(self._systems) | table)
            return list(self._systems.keys())


registry = NamespaceRegistry()


def require_api_key(authorization: Optional[str] = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class IngestRequest(BaseModel):
    text: str
    namespace: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    items: List[Any]
    namespace: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    namespace: Optional[str] = None
    top_k: int = 15
    answer: bool = True
    sort: str = "time"


class ConsolidateRequest(BaseModel):
    memory_id: str
    namespace: Optional[str] = None


class SaveRequest(BaseModel):
    namespace: Optional[str] = None


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _node_summary(node) -> Dict[str, Any]:
    attributes = node.attributes or {}
    jev = attributes.get("jev_mem", {})
    return {
        "memory_id": node.node_id,
        "content": node.content_narrative,
        "timestamp": node.timestamp.isoformat() if node.timestamp else None,
        "entities": attributes.get("entities", []),
        "keywords": attributes.get("keywords", []),
        "controller": jev.get("controller"),
        "memory_type": jev.get("memory_type"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK, "profile": PROFILE_PATH,
            "cache_base": str(CACHE_BASE), "namespaces": registry.names()}


@app.get("/v1/namespaces", dependencies=[Depends(require_api_key)])
def list_namespaces():
    """Namespaces registered in this deployment (table-backed in postgres mode)."""
    return {"namespaces": registry.names()}


@app.post("/v1/memory/ingest", dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must be non-empty")
    system = registry.get(req.namespace or "default")
    try:
        node = system.memory_builder.build(req.text, _parse_timestamp(req.timestamp), req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if node is None:
        return {"stored": False, "memory_id": None, "reason": "rejected by admission policy"}
    system.save_memory()
    return {"stored": True, "memory_id": node.node_id, "node": _node_summary(node)}


@app.post("/v1/memory/ingest/batch", dependencies=[Depends(require_api_key)])
def ingest_batch(req: BatchIngestRequest):
    system = registry.get(req.namespace or "default")
    try:
        result = system.build_memory_from_conversation(req.items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    system.save_memory()
    return result


@app.post("/v1/memory/query", dependencies=[Depends(require_api_key)])
def query_memory(req: QueryRequest):
    system = registry.get(req.namespace or "default")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must be non-empty")
    try:
        context, evidence = system.query_engine.query(
            req.question, top_k=req.top_k, sort=(req.sort or "time").strip().lower())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    answer = None
    if req.answer and system.llm_controller is not None:
        prompt = system.answer_formatter.build_qa_prompt(evidence, req.question)
        raw = system.llm_controller.llm.get_completion(
            prompt, response_format={"type": "text"}, temperature=0.0)
        answer = system.answer_formatter.extract_answer(raw, req.question)
        system.memory_builder.jev.audit.emit(
            "system_two_answer", query_id=context.metadata.get("query_id"), llm_calls=1)
    trace = context.metadata or {}
    return {
        "answer": answer or evidence or "Information not found",
        "evidence": evidence,
        "generated": answer is not None,
        "nodes_visited": trace.get("nodes_visited"),
        "edges_examined": trace.get("edges_examined"),
        "jev_calls": trace.get("jev_calls"),
        "retrieval_depth": trace.get("retrieval_depth"),
        "stopping_decision": trace.get("stopping_decision"),
        "controller": trace.get("controller"),
        "trace": trace,
    }


@app.post("/v1/memory/consolidate", dependencies=[Depends(require_api_key)])
def consolidate(req: ConsolidateRequest):
    system = registry.get(req.namespace or "default")
    try:
        decisions = system.memory_builder.consolidate(req.memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    system.save_memory()
    return {"memory_id": req.memory_id, "decisions": decisions}


@app.post("/v1/memory/save", dependencies=[Depends(require_api_key)])
def save_memory(req: SaveRequest):
    system = registry.get(req.namespace or "default")
    system.save_memory()
    return {"saved": True, "namespace": req.namespace or "default"}


@app.get("/v1/memory/stats", dependencies=[Depends(require_api_key)])
def stats(namespace: str = Query(default="default")):
    system = registry.get(namespace)
    graph = system.graph_db
    nodes = list(graph.nodes.values())
    links = list(graph.links.values()) if isinstance(graph.links, dict) else list(graph.links)
    by_type: Dict[str, int] = {}
    for link in links:
        key = str(getattr(link.link_type, "name", link.link_type))
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "namespace": namespace,
        "nodes": len(nodes),
        "events": sum(n.node_type.name == "EVENT" for n in nodes),
        "links": len(links),
        "links_by_type": by_type,
        "vectors": system.vector_db.size(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("JEV_MEM_PORT", "8700")))
