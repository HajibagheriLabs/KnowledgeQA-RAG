import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from knowledgeqa import dense, sparse
from knowledgeqa.chunking import Chunk
from knowledgeqa.config import Config
from knowledgeqa.evaluate import Run, max_pool, reciprocal_rank_fusion


@dataclass(frozen=True)
class Stage:
    run: Run
    # answers one query on its own, end to end, for latency measurement
    answer: Callable[[str], dict[str, float]]
    index_seconds: float


def bm25_stage(
    config: Config,
    device: torch.device,
    chunks: Sequence[Chunk],
    query_ids: Sequence[str],
    texts: Sequence[str],
) -> Stage:
    doc_ids = [chunk.doc_id for chunk in chunks]
    depth, k = config.evaluation.recall_k, config.retrieval.candidate_chunks
    started = time.perf_counter()
    index = sparse.build_index([chunk.text for chunk in chunks], config.bm25, device)
    index_seconds = time.perf_counter() - started
    hits = sparse.search(index, texts, k, config.retrieval.query_batch_size)
    run = max_pool(query_ids, *hits, doc_ids, depth)

    def answer(query: str) -> dict[str, float]:
        one = sparse.search(index, [query], k, 1)
        return max_pool([query], *one, doc_ids, depth)[query]

    return Stage(run, answer, index_seconds)


def dense_stage(
    config: Config,
    encoder: SentenceTransformer,
    chunks: Sequence[Chunk],
    query_ids: Sequence[str],
    texts: Sequence[str],
    cache: Path,
) -> Stage:
    doc_ids = [chunk.doc_id for chunk in chunks]
    depth, k = config.evaluation.recall_k, config.retrieval.candidate_chunks
    # an empty chunk carries no content to match, and its embedding would be noise
    rows = np.array([i for i, chunk in enumerate(chunks) if chunk.text], dtype=np.int64)
    vectors, encode_seconds = dense.cached_encode(
        encoder, [chunks[i].text for i in rows], config.dense, cache
    )
    started = time.perf_counter()
    index = dense.build_index(vectors, rows, len(chunks))
    index_seconds = encode_seconds + time.perf_counter() - started
    prefixed = [config.dense.query_prefix + text for text in texts]
    queries = dense.encode(encoder, prefixed, config.dense.batch_size)
    run = max_pool(query_ids, *dense.search(index, queries, k), doc_ids, depth)

    def answer(query: str) -> dict[str, float]:
        vector = dense.encode(encoder, [config.dense.query_prefix + query], 1)
        return max_pool([query], *dense.search(index, vector, k), doc_ids, depth)[query]

    return Stage(run, answer, index_seconds)


def hybrid_stage(config: Config, bm25: Stage, semantic: Stage) -> Stage:
    rrf_k = config.hybrid.rrf_k
    run = reciprocal_rank_fusion([bm25.run, semantic.run], rrf_k)

    def answer(query: str) -> dict[str, float]:
        both = [{query: bm25.answer(query)}, {query: semantic.answer(query)}]
        return reciprocal_rank_fusion(both, rrf_k)[query]

    return Stage(run, answer, bm25.index_seconds + semantic.index_seconds)
