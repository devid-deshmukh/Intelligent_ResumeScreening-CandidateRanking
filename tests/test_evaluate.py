import math

from src.evaluate import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_recall_and_mrr():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert mean_reciprocal_rank(ranked, relevant) == 0.5


def test_ndcg_and_no_hit_cases():
    assert ndcg_at_k(["b", "a"], {"b"}, 2) == 1.0
    assert precision_at_k(["a"], {"b"}, 10) == 0.0
    assert recall_at_k(["a"], {"b"}, 30) == 0.0
    assert ndcg_at_k(["a"], {"b"}, 10) == 0.0
    assert mean_reciprocal_rank(["a"], {"b"}) == 0.0
    assert math.isfinite(ndcg_at_k([], set(), 10))