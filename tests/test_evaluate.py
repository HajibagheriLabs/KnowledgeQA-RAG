import math

import numpy as np
import pytest

from knowledgeqa.evaluate import evaluate_run, max_pool, rank


def test_metrics_match_hand_computed_example():
    # six chunks of five documents, d1 split across two of them
    chunk_doc_ids = ["d2", "d1", "d3", "d1", "d5", "d4"]
    hits = np.array([[3, 0, 1, 2], [4, 5, 0, 1]])
    scores = np.array([[0.5, 0.9, 0.8, 0.4], [0.7, 0.7, -np.inf, -np.inf]])
    run = max_pool(["q1", "q2"], hits, scores, chunk_doc_ids, depth=3)
    assert run == {
        "q1": {"d1": 0.8, "d2": 0.9, "d3": 0.4},
        "q2": {"d5": 0.7, "d4": 0.7},
    }
    # a tie falls back to trec_eval's order, document id descending
    assert rank(run["q2"]) == ["d5", "d4"]

    # q1 ranks d2 (rel 0), d1 (rel 2), d3 (rel 1); q2 ranks d5 (unjudged), d4 (rel 1)
    qrels = {"q1": {"d1": 2, "d2": 0, "d3": 1}, "q2": {"d4": 1}}
    ndcg_q1 = (2 / math.log2(3) + 1 / math.log2(4)) / (
        2 / math.log2(2) + 1 / math.log2(3)
    )
    ndcg_q2 = (1 / math.log2(3)) / (1 / math.log2(2))
    metrics = evaluate_run(run, qrels, ndcg_k=10, recall_k=2, mrr_k=10)
    assert metrics == pytest.approx(
        {
            "ndcg_at_10": (ndcg_q1 + ndcg_q2) / 2,
            "recall_at_2": (1 / 2 + 1 / 1) / 2,
            "mrr_at_10": (1 / 2 + 1 / 2) / 2,
        }
    )
    assert metrics["ndcg_at_10"] == pytest.approx(0.6503, abs=1e-4)

    # four of six chunks retrieved, three documents covered: a top-4 cut-off cannot be trusted
    with pytest.raises(ValueError, match="retrieve more"):
        max_pool(["q1"], hits[:1], scores[:1], chunk_doc_ids, depth=4)
