"""Exercise local decisions through the same budget/validation/cache as Jev."""

import sys
import time
from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, Noul

from memory.jev_client import CallBudget, JevClient, JevUnavailable
from memory.jev_mem_config import JevMemConfig


def install_fake_laya(monkeypatch, predict=None):
    loads, calls = [], []

    def default_predict(state, questions):
        calls.append((state, questions))
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.8}
            else:
                keys = list(question["criteria"])
                answers[name] = {
                    "type": "choice", "choice": keys[0], "confidence": 0.6,
                    "probabilities": {k: float(k == keys[0]) for k in keys},
                    "action": {"act_probability": 0.9},
                }
        return {"answers": answers, "usage": {"input_tokens": 7, "output_tokens": 0}}

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return SimpleNamespace(predict=predict or default_predict)

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules, "laya_mlx", SimpleNamespace(load=load))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return loads, calls


def config(**kwargs):
    return JevMemConfig(decision_backend=kwargs.pop("decision_backend", "laya"), max_retries=0, **kwargs)


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_laya_mixed_questions_without_key_and_cached(monkeypatch, backend):
    loads, calls = install_fake_laya(monkeypatch)
    monkeypatch.setattr("memory.jev_client.TypeSafeClient", lambda **kw: pytest.fail("Jev called"))
    client = JevClient(config(decision_backend=backend, laya_batch_size=1, laya_device="cpu", laya_subfolder="multilingual"))
    expected = {"device": "cpu", "subfolder": "multilingual"}
    if backend == "laya-mlx":
        expected["batch_size"] = 1
    assert loads == [(("convaiinnovations/laya",), expected)]
    questions = {"keep": Noul(instructions="Is this useful?"),
                 "kind": Choice(instructions="Which kind?", criteria={"fact": "a fact", "event": "an event"})}
    budget = CallBudget(1, time.monotonic() + 10)
    result = client.evaluate("write", {"text": "Alice moved."}, questions, budget=budget)
    assert result.source == backend
    assert result.model == "convaiinnovations/laya/multilingual"
    assert result.values == {"keep": 0.8}
    assert result.choices["kind"].choice == "fact"
    assert result.choices["kind"].probabilities == {"fact": 1.0, "event": 0.0}
    assert result.usage == {"input_tokens": 14, "output_tokens": 0}
    assert len(calls) == 2 and budget.calls == 1
    cached = client.evaluate("write", {"text": "Alice moved."}, questions, budget=budget)
    assert cached.source == "cache" and cached.usage == {}
    assert budget.cache_hits == 1 and len(calls) == 2
    client.close()
    assert client._laya.agent is None


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
@pytest.mark.parametrize("fallback", [True, False])
def test_laya_invalid_probability_uses_existing_failure_policy(monkeypatch, fallback, backend):
    install_fake_laya(monkeypatch, lambda *a: {"answers": {"q": {"type": "noul", "noul": float("nan")}}})
    client = JevClient(config(decision_backend=backend, fallback_to_magma=fallback))
    if fallback:
        assert client.probabilities("test", "state", {"q": "True?"}) is None
    else:
        with pytest.raises(JevUnavailable, match="ValueError"):
            client.probabilities("test", "state", {"q": "True?"})


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_laya_deadline_does_not_cache_late_results(monkeypatch, backend):
    install_fake_laya(monkeypatch)
    client = JevClient(config(decision_backend=backend, fallback_to_magma=False, laya_batch_size=1))
    ticks = iter([0, 0, 2])
    monkeypatch.setattr("memory.laya_backend.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(JevUnavailable, match="laya_deadline"):
        client.probabilities("test", "state", {"a": "True?", "b": "False?"},
                             budget=CallBudget(1, time.monotonic() + 1))
    assert not client.cache


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_laya_exhausted_budget_never_predicts(monkeypatch, backend):
    _, calls = install_fake_laya(monkeypatch)
    client = JevClient(config(decision_backend=backend, fallback_to_magma=False))
    with pytest.raises(JevUnavailable, match="budget_exhausted"):
        client.probabilities("test", "state", {"a": "True?"},
                             budget=CallBudget(0, time.monotonic() + 1))
    assert not calls


def test_mock_and_default_jev_do_not_load_laya(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)
    client = JevClient(config(jev_mock=True))
    assert client.probabilities("test", "state", {"q": "True?"}, mock_values={"q": 0.4}).source == "mock"
    assert JevClient(JevMemConfig())._laya is None


def test_laya_auto_uses_cpu_on_macos(monkeypatch):
    from memory import local_device
    monkeypatch.setattr(local_device, "sys", SimpleNamespace(platform="darwin"))
    loads, _ = install_fake_laya(monkeypatch)
    with JevClient(config()):
        assert loads[0][1]["device"] == "cpu"


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_retrievals_queue_before_starting_deadlines(monkeypatch, backend):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from memory.jev_mem_retrieval import RetrievalController
    install_fake_laya(monkeypatch)
    client = JevClient(config(decision_backend=backend))
    controller = RetrievalController(None, client, client.config)
    entered, release, second_queued = Event(), Event(), Event()
    calls = []

    def retrieve(question, top_k, sort="time"):
        calls.append(question)
        if question == "first":
            entered.set()
            assert release.wait(5)
        # _query creates CallBudget; it must not run while the first call owns
        # the local model slot, even if another worker has submitted a query.
        return SimpleNamespace(metadata={}), "evidence"

    monkeypatch.setattr(controller, "_query", retrieve)
    lock = client._retrieval_lock

    class ObservedLock:
        def __enter__(self):
            if entered.is_set():
                second_queued.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    client._retrieval_lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(controller.query, "first", 1)
        try:
            assert entered.wait(5)
            second = pool.submit(controller.query, "second", 1)
            assert second_queued.wait(5)
            assert calls == ["first"]
        finally:
            release.set()
        assert first.result()[0].metadata["retrieval_queue_seconds"] >= 0
        assert second.result()[0].metadata["retrieval_queue_seconds"] >= 0
    assert calls == ["first", "second"]


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_retrieval_slot_released_after_failure(monkeypatch, backend):
    from memory.jev_mem_retrieval import RetrievalController
    install_fake_laya(monkeypatch)
    client = JevClient(config(decision_backend=backend))
    controller = RetrievalController(None, client, client.config)
    def fail(*args):
        raise JevUnavailable("laya_deadline")
    monkeypatch.setattr(controller, "_query", fail)
    with pytest.raises(JevUnavailable):
        controller.query("test", 1)
    assert client._retrieval_lock.acquire(blocking=False)
    client._retrieval_lock.release()


def test_missing_optional_dependency_has_install_instruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)
    with pytest.raises(ImportError, match=r"\.\[laya\]"):
        JevClient(config())


@pytest.mark.parametrize("settings", [{"decision_backend": "typo"}, {"laya_batch_size": 0},
                                       {"laya_batch_size": True}, {"laya_model": ""},
                                       {"laya_device": None}, {"laya_subfolder": None}])
def test_invalid_backend_settings(settings):
    with pytest.raises(ValueError):
        JevMemConfig(**settings)


def test_laya_config_roundtrip(tmp_path):
    import json
    cfg = JevMemConfig.load("config/laya_mem.json")
    assert cfg.decision_backend == "laya" and not cfg.fallback_to_magma
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert json.dumps(JevMemConfig.load(path).to_dict(), sort_keys=True) == json.dumps(cfg.to_dict(), sort_keys=True)


@pytest.mark.parametrize("device, expected", [("auto", None), ("gpu", "gpu"), ("cpu", "cpu")])
def test_mlx_device_policy_and_no_torch_import(monkeypatch, device, expected):
    loads, _ = install_fake_laya(monkeypatch)
    monkeypatch.setitem(sys.modules, "laya", None)
    with JevClient(config(decision_backend="laya-mlx", laya_device=device)):
        assert loads[0][1] == {"device": expected, "subfolder": None, "batch_size": 8}


def test_mlx_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya_mlx", None)
    with pytest.raises(ImportError, match=r"\.\[mlx\]"):
        JevClient(config(decision_backend="laya-mlx"))


def test_mlx_config_and_validation(tmp_path):
    import json
    cfg = JevMemConfig.load("config/laya_mlx_mem.json")
    assert cfg.decision_backend == "laya-mlx"
    assert cfg.laya_model == "aac6fef/laya-mlx"
    assert cfg.laya_device == "gpu"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert json.dumps(JevMemConfig.load(path).to_dict(), sort_keys=True) == json.dumps(cfg.to_dict(), sort_keys=True)
    with pytest.raises(ValueError, match="device"):
        config(decision_backend="laya-mlx", laya_device="cuda")


def test_mlx_mock_does_not_import_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya_mlx", None)
    with JevClient(config(decision_backend="laya-mlx", jev_mock=True)) as client:
        assert client.probabilities("test", "state", {"q": "True?"},
                                    mock_values={"q": 0.4}).source == "mock"


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
def test_local_backend_build_save_reload_and_query(monkeypatch, tmp_path, backend):
    from datetime import datetime
    from memory.memory_builder import MemoryBuilder
    from memory.mock_encoder import MockEncoder
    from memory.query_engine import QueryEngine
    from memory.trg_memory import TemporalResonanceGraphMemory
    from memory.vector_db import NumpyVectorDB

    _, calls = install_fake_laya(monkeypatch)
    monkeypatch.setattr("memory.jev_client.TypeSafeClient", lambda **kw: pytest.fail("Jev called"))
    cfg = config(decision_backend=backend, write_enabled=True, read_enabled=True,
                 fallback_to_magma=False, anchor_count=1, maximum_depth=1)

    def builder():
        encoder = MockEncoder()
        trg = TemporalResonanceGraphMemory(encoder=encoder, vector_db=NumpyVectorDB(encoder.dimension),
                                          llm_backend=None)
        return MemoryBuilder(str(tmp_path), jev_config=cfg, trg_memory=trg, llm_enabled=False)

    first = builder()
    try:
        for day, content in enumerate(["Mira planted basil.", "Mira wants a weekly reminder."], 1):
            node = first.build(content, datetime(2026, 9, day), {"entities": ["Mira"]})
            assert node.attributes["jev_mem"]["controller"] == backend
        first.save()
    finally:
        first.jev.close()
    restored = builder()
    try:
        restored.load()
        engine = QueryEngine(restored.trg, restored.node_index, jev_config=cfg, jev_client=restored.jev)
        context, evidence = engine.query("What reminder does Mira prefer?")
        assert evidence and context.anchor_nodes and calls
        assert context.metadata["decision_backend"] == backend
        assert not context.metadata["fallback_events"]
        assert restored.jev._sdk is None
    finally:
        restored.jev.close()
