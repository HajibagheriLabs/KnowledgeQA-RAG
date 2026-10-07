import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from knowledgeqa import dense, sparse
from knowledgeqa.chunking import Chunk, chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import ChunkingConfig, Config
from knowledgeqa.evaluate import Run, max_pool, rank, reciprocal_rank_fusion
from knowledgeqa.ingest import Document, squad_paragraphs
from knowledgeqa.reader import Reader, read
from knowledgeqa.rerank import candidate_pairs, rerank_run

# (query ids, query texts, batch size) -> run
Search = Callable[[Sequence[str], Sequence[str], int], Run]


@dataclass(frozen=True)
class Stage:
    run: Run
    # answers one query on its own, end to end, for latency measurement
    answer: Callable[[str], dict[str, float]]
    index_seconds: float


def bm25_search(
    config: Config, device: torch.device, chunks: Sequence[Chunk]
) -> tuple[Search, float]:
    """Index the chunks once; return a search over them and the indexing seconds."""
    doc_ids = [chunk.doc_id for chunk in chunks]
    depth, k = config.evaluation.recall_k, config.retrieval.candidate_chunks
    started = time.perf_counter()
    index = sparse.build_index([chunk.text for chunk in chunks], config.bm25, device)
    index_seconds = time.perf_counter() - started

    def search(query_ids: Sequence[str], texts: Sequence[str], batch_size: int) -> Run:
        hits = sparse.search(index, texts, k, batch_size)
        return max_pool(query_ids, *hits, doc_ids, depth)

    return search, index_seconds


def dense_search(
    config: Config,
    encoder: SentenceTransformer,
    chunks: Sequence[Chunk],
    cache: Path,
) -> tuple[Search, float]:
    """Embed and index the chunks once; return a search over them and the indexing seconds."""
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

    def search(query_ids: Sequence[str], texts: Sequence[str], batch_size: int) -> Run:
        prefixed = [config.dense.query_prefix + text for text in texts]
        queries = dense.encode(encoder, prefixed, batch_size)
        return max_pool(query_ids, *dense.search(index, queries, k), doc_ids, depth)

    return search, index_seconds


def _stage(
    search: Search,
    index_seconds: float,
    query_ids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
) -> Stage:
    run = search(query_ids, texts, batch_size)
    return Stage(run, lambda query: search([query], [query], 1)[query], index_seconds)


def bm25_stage(
    config: Config,
    device: torch.device,
    chunks: Sequence[Chunk],
    query_ids: Sequence[str],
    texts: Sequence[str],
) -> Stage:
    search, index_seconds = bm25_search(config, device, chunks)
    batch_size = config.retrieval.query_batch_size
    return _stage(search, index_seconds, query_ids, texts, batch_size)


def dense_stage(
    config: Config,
    encoder: SentenceTransformer,
    chunks: Sequence[Chunk],
    query_ids: Sequence[str],
    texts: Sequence[str],
    cache: Path,
) -> Stage:
    search, index_seconds = dense_search(config, encoder, chunks, cache)
    batch_size = config.dense.batch_size
    return _stage(search, index_seconds, query_ids, texts, batch_size)


def hybrid_stage(config: Config, bm25: Stage, semantic: Stage) -> Stage:
    rrf_k = config.hybrid.rrf_k
    run = reciprocal_rank_fusion([bm25.run, semantic.run], rrf_k)

    def answer(query: str) -> dict[str, float]:
        both = [{query: bm25.answer(query)}, {query: semantic.answer(query)}]
        return reciprocal_rank_fusion(both, rrf_k)[query]

    return Stage(run, answer, bm25.index_seconds + semantic.index_seconds)


def squad_corpus(config: Config) -> tuple[list[Document], list[Chunk]]:
    """SQuAD paragraphs, each indexed whole: they are already short and self-contained."""
    docs = squad_paragraphs(config)
    whole = ChunkingConfig("none", None, 0, config.chunking.default.prepend_title)
    offsets = corpus_offsets(config, "squad", docs, load_tokenizer(config))
    return docs, chunk_corpus(docs, offsets, whole)


@dataclass(frozen=True)
class CitedChunk:
    doc_id: str
    title: str
    text: str
    retrieval_score: float
    rerank_score: float | None


@dataclass(frozen=True)
class Response:
    # None when the reader abstains
    answer: str | None
    abstained: bool
    # reader margin over its no-answer score; None when nothing was retrieved
    margin: float | None
    chunks: list[CitedChunk]
    # position in chunks of the paragraph the answer was read from, None on abstention
    source: int | None
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class QAPipeline:
    """One question in, a cited answer or an abstention out."""

    search: Search
    chunks: Sequence[Chunk]
    by_doc: Mapping[str, list[int]]
    docs: Mapping[str, Document]
    reranker: CrossEncoder | None
    reader: Reader
    threshold: float
    rerank_depth: int
    rerank_batch_size: int
    top_k: int

    def answer(self, question: str) -> Response:
        timings = {}
        started = time.perf_counter()
        scores = self.search([question], [question], 1)[question]
        timings["retrieval"] = (time.perf_counter() - started) * 1000
        reranked: dict[str, float] = {}
        if self.reranker is not None:
            started = time.perf_counter()
            pairs = candidate_pairs(
                {question: scores}, [question], self.by_doc, self.rerank_depth
            )
            reranked = rerank_run(
                self.reranker,
                [question],
                pairs,
                {question: question},
                self.chunks,
                self.rerank_batch_size,
            )[question]
            timings["rerank"] = (time.perf_counter() - started) * 1000
        top = rank(reranked or scores)[: self.top_k]
        started = time.perf_counter()
        (found,) = read(self.reader, [question], [[self.docs[d].text for d in top]])
        timings["reader"] = (time.perf_counter() - started) * 1000
        cited = [
            CitedChunk(
                d,
                self.docs[d].title,
                self.docs[d].text,
                scores[d],
                reranked[d] if reranked else None,
            )
            for d in top
        ]
        answered = found.margin > self.threshold
        return Response(
            answer=found.text if answered else None,
            abstained=not answered,
            margin=found.margin if np.isfinite(found.margin) else None,
            chunks=cited,
            source=found.passage if answered else None,
            timings_ms=timings,
        )
