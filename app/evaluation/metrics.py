"""Retrieval metrics with binary relevance. Pure functions: no I/O, no LangChain.

`retrieved` is a ranked list of chunk IDs (best first); `relevant` is the set of
chunk IDs labeled as correct for the question.
"""
import math


def _check(relevant: set[str], k: int | None = None) -> None:
    if not relevant:
        raise ValueError("relevant set is empty; skip unanswerable questions")
    if k is not None and k < 1:
        raise ValueError("k must be >= 1")


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    _check(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg