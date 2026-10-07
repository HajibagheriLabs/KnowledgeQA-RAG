import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from knowledgeqa.ingest import Qrels

Run = dict[str, dict[str, float]]


def max_pool(
    query_ids: Sequence[str],
    chunk_hits: np.ndarray,
    chunk_scores: np.ndarray,
    chunk_doc_ids: Sequence[str],
    depth: int,
) -> Run:
    """Score each document by its best chunk hit; hits scored -inf are padding."""
    if not len(query_ids) == len(chunk_hits) == len(chunk_scores):
        raise ValueError("query_ids, chunk_hits and chunk_scores differ in length")
    run: Run = {}
    for query_id, hits, scores in zip(query_ids, chunk_hits, chunk_scores, strict=True):
        docs: dict[str, float] = {}
        for hit, score in zip(hits.tolist(), scores.tolist(), strict=True):
            if score == -math.inf:
                continue
            doc_id = chunk_doc_ids[hit]
            if score > docs.get(doc_id, -math.inf):
                docs[doc_id] = score
        # unretrieved chunks score below every hit, so only a short truncated list is unsafe
        truncated = len(hits) < len(chunk_doc_ids) and bool(np.isfinite(scores).all())
        if truncated and len(docs) < depth:
            raise ValueError(
                f"query {query_id}: {len(hits)} chunk hits cover only {len(docs)} documents, "
                f"fewer than {depth}; retrieve more candidate chunks"
            )
        run[query_id] = docs
    return run


def rank(scores: Mapping[str, float]) -> list[str]:
    # trec_eval's order: score descending, ties broken by document id descending
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [doc_id for doc_id, _ in ordered]


def write_trec_run(path: Path, run: Run, depth: int, tag: str) -> None:
    """Save the top `depth` documents per query in the standard TREC run format."""
    with path.open("w") as fh:
        for query_id in sorted(run):
            ranking = rank(run[query_id])[:depth]
            for position, doc_id in enumerate(ranking, start=1):
                # repr round-trips the float exactly, so a re-read run ranks identically
                score = repr(run[query_id][doc_id])
                fh.write(f"{query_id} Q0 {doc_id} {position} {score} {tag}\n")


def read_trec_run(path: Path) -> Run:
    run: Run = {}
    for line in path.read_text().splitlines():
        query_id, _, doc_id, _, score, _ = line.split()
        run.setdefault(query_id, {})[doc_id] = float(score)
    return run


def ndcg_at_k(ranking: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """nDCG with linear gains, as computed by trec_eval and reported by BEIR."""
    gains = [judged.get(doc_id, 0) for doc_id in ranking[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains) if gain > 0)
    ideal = sorted((rel for rel in judged.values() if rel > 0), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in judged.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    for i, doc_id in enumerate(ranking[:k]):
        if judged.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(
    run: Run, qrels: Qrels, ndcg_k: int, recall_k: int, mrr_k: int
) -> dict[str, float]:
    """Mean metrics over every judged query; a judged query absent from the run is an error."""
    missing = sorted(set(qrels) - set(run))
    if missing:
        raise ValueError(
            f"{len(missing)} judged queries have no run, e.g. {missing[:3]}"
        )
    totals = {
        f"ndcg_at_{ndcg_k}": 0.0,
        f"recall_at_{recall_k}": 0.0,
        f"mrr_at_{mrr_k}": 0.0,
    }
    for query_id, judged in qrels.items():
        ranking = rank(run[query_id])
        totals[f"ndcg_at_{ndcg_k}"] += ndcg_at_k(ranking, judged, ndcg_k)
        totals[f"recall_at_{recall_k}"] += recall_at_k(ranking, judged, recall_k)
        totals[f"mrr_at_{mrr_k}"] += mrr_at_k(ranking, judged, mrr_k)
    return {name: total / len(qrels) for name, total in totals.items()}


def latency_ms(
    answer: Callable[[str], object], queries: Sequence[str], warmup: int
) -> dict[str, float]:
    """p50 and p95 wall-clock milliseconds to answer each query on its own, after warm-up."""
    for query in queries[:warmup]:
        answer(query)
    timings = []
    for query in queries:
        start = time.perf_counter()
        answer(query)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
    }
