"""Measure retrieval seed coverage using a saved graph and local MiniLM only.

Gold dialogue IDs are used only for scoring, never to rank candidates.
This is an anchor diagnostic, not an end-to-end Jev/answer accuracy benchmark.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from jev_mem.datasets.locomo import load_locomo_dataset
    from memory.graph_db import FileGraphStore
    from memory.mock_encoder import MockEncoder
    from memory.query_engine import QueryEngine
    from memory.trg_memory import TemporalResonanceGraphMemory
    from memory.vector_db import NumpyVectorDB

    cache = Path(args.memory)
    graph = FileGraphStore()
    graph.load(str(cache / "graph.json"))
    trg = TemporalResonanceGraphMemory(graph_db=graph, encoder=MockEncoder(),
        vector_db=NumpyVectorDB(384), llm_backend=None)
    index = {k: set(v) for k, v in json.loads((cache / "keyword_index.json").read_text()).items()}
    engine = QueryEngine(trg, index)
    with contextlib.redirect_stdout(io.StringIO()):
        sample = load_locomo_dataset(args.dataset)[args.sample]
    questions = [q for q in sample.qa if q.category != 5 and q.evidence]
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    embeddings = model.encode([trg.keyword_enricher.enrich_query(q.question) for q in questions],
                              convert_to_numpy=True, show_progress_bar=False)
    nodes = list(graph.nodes.values())
    vectors = np.array([n.embedding_vector for n in nodes])
    if vectors.shape[1] != embeddings.shape[1]:
        parser.error("This diagnostic requires a graph built with all-MiniLM-L6-v2")
    rankings = []
    for q, embedding in zip(questions, embeddings):
        vector_nodes = [nodes[i] for i in np.argsort(np.sum((vectors - embedding) ** 2, axis=1))]
        keyword_nodes = engine._keyword_search(q.question, limit=60)
        rankings.append((q, vector_nodes, keyword_nodes))
    experiments = []
    for anchors, normal_k, multihop_k, maximum_nodes in [(5, 15, 30, 30), (10, 30, 40, 60),
            (15, 30, 40, 60), (20, 30, 40, 60), (30, 30, 40, 60)]:
        records = []
        for q, vector_nodes, keyword_nodes in rankings:
            top_k = multihop_k if q.category == 1 else normal_k
            pool = min(maximum_nodes, max(top_k, anchors))
            fused = engine._rrf_fusion([vector_nodes[:pool], keyword_nodes[:maximum_nodes]])
            selected = {node.attributes.get("dia_id") for node, _ in fused[:min(anchors, maximum_nodes, top_k)]}
            gold = set(q.evidence)
            records.append({"category": q.category, "recall": len(gold & selected) / len(gold),
                            "complete": gold <= selected})
        experiments.append({"anchor_count": anchors, "answer_top_k": normal_k,
            "multihop_top_k": multihop_k, "maximum_nodes": maximum_nodes,
            "mean_evidence_recall": float(np.mean([r["recall"] for r in records])),
            "all_evidence_fraction": float(np.mean([r["complete"] for r in records]))})
    report = {"sample": args.sample, "memory": str(cache), "questions_with_evidence": len(questions),
        "method": "Offline MiniLM + current keyword search + RRF anchors; no Jev traversal or answer generation",
        "experiments": experiments}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
