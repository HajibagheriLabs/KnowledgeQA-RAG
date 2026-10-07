from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from knowledgeqa.chunking import Chunk
from knowledgeqa.config import Config
from knowledgeqa.evaluate import Run, rank


def load_reranker(config: Config, name: str, device: torch.device) -> CrossEncoder:
    path = config.paths.model_dir(config.rerank.candidates[name])
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"{path} has no weights; run `make data` first")
    reranker = CrossEncoder(
        str(path),
        device=str(device),
        max_length=config.rerank.max_length,
        local_files_only=True,
    )
    if config.rerank.fp16 and device.type == "cuda":
        reranker.model.half()
        # fp16 logits are coarse enough to tie thousands of candidates, so score in fp32
        head = reranker.model.classifier.float()
        head.register_forward_pre_hook(
            lambda _, inputs: tuple(x.float() for x in inputs)
        )
    return reranker


def chunks_by_doc(chunks: Sequence[Chunk]) -> dict[str, list[int]]:
    """Positions of the non-empty chunks of each document."""
    grouped: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        if chunk.text:
            grouped.setdefault(chunk.doc_id, []).append(position)
    return grouped


def candidate_pairs(
    run: Run,
    query_ids: Sequence[str],
    grouped: Mapping[str, list[int]],
    top_k: int,
) -> list[tuple[str, int]]:
    """(query id, chunk position) for every chunk of each query's top-k documents."""
    return [
        (query_id, position)
        for query_id in query_ids
        for doc_id in rank(run[query_id])[:top_k]
        for position in grouped.get(doc_id, [])
    ]


def rerank_run(
    reranker: CrossEncoder,
    query_ids: Sequence[str],
    pairs: Sequence[tuple[str, int]],
    queries: Mapping[str, str],
    chunks: Sequence[Chunk],
    batch_size: int,
) -> Run:
    """Score every (query, chunk) pair with the cross-encoder and keep each document's best chunk."""
    texts = [(queries[query_id], chunks[position].text) for query_id, position in pairs]
    scores = reranker.predict(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    run: Run = {query_id: {} for query_id in query_ids}
    for (query_id, position), score in zip(
        pairs, np.asarray(scores).tolist(), strict=True
    ):
        docs = run[query_id]
        doc_id = chunks[position].doc_id
        docs[doc_id] = max(score, docs.get(doc_id, -np.inf))
    return run
