from evaluation.metrics.retrieval import mrr, precision_at_k, recall_at_k


def test_recall_precision_mrr() -> None:
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b", "y"]
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert precision_at_k(relevant, retrieved, 3) == 2 / 3
    assert mrr(relevant, retrieved) == 0.5
