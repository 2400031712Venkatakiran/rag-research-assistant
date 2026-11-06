import pytest

from app.evaluation.benchmark import EvalItem, run_benchmark, summarize
from app.retrieval.base import RetrievedChunk, Retriever


class FixedRetriever(Retriever):
    name = "fixed"

    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, k, filter=None):
        return [
            RetrievedChunk(chunk_id=cid, content="", rank=i)
            for i, cid in enumerate(self.ids[:k], start=1)
        ]


def test_benchmark_scores_any_retriever_and_skips_unanswerable():
    items = [
        EvalItem(id="q1", type="conceptual", question="x", relevant_chunk_ids=["b"]),
        EvalItem(id="q2", type="unanswerable", question="y", relevant_chunk_ids=[]),
    ]
    result = run_benchmark(FixedRetriever(["x", "b", "a"]), items, ks=[1, 3])
    assert result["skipped_unanswerable"] == 1
    assert len(result["per_question"]) == 1

    scores = result["per_question"][0]["scores"]
    assert scores["mrr"] == 0.5
    assert scores["p@1"] == 0.0
    assert scores["p@3"] == pytest.approx(1 / 3)
    assert scores["r@3"] == 1.0

    summary = summarize(result)
    assert summary["n_questions"] == 1
    assert summary["by_type"]["conceptual"]["mrr"] == 0.5