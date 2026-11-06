import math

import pytest

from app.evaluation import metrics

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


def test_precision_and_recall():
    assert metrics.precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5
    assert metrics.precision_at_k(RETRIEVED, RELEVANT, 4) == 0.5
    assert metrics.recall_at_k(RETRIEVED, RELEVANT, 2) == 0.5
    assert metrics.recall_at_k(RETRIEVED, RELEVANT, 4) == 1.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert metrics.reciprocal_rank(["a", "c"], RELEVANT) == 0.0
    assert metrics.reciprocal_rank(["b"], RELEVANT) == 1.0


def test_ndcg_known_value():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_and_empty():
    assert metrics.ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["a", "c"], RELEVANT, 2) == 0.0


def test_ordering_matters_for_ndcg_but_not_recall():
    early = ["b", "d", "a", "c"]
    late = ["a", "c", "b", "d"]
    assert metrics.recall_at_k(early, RELEVANT, 4) == metrics.recall_at_k(late, RELEVANT, 4)
    assert metrics.ndcg_at_k(early, RELEVANT, 4) > metrics.ndcg_at_k(late, RELEVANT, 4)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        metrics.precision_at_k(RETRIEVED, set(), 3)
    with pytest.raises(ValueError):
        metrics.recall_at_k(RETRIEVED, RELEVANT, 0)