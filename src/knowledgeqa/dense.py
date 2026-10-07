import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from knowledgeqa.config import Config, DenseConfig


@dataclass(frozen=True)
class DenseIndex:
    index: faiss.Index
    # faiss row -> chunk position; empty chunks are never embedded, so the two differ
    rows: np.ndarray
    n_chunks: int
    # gpu indexes do not own their resources; dropping this would free memory under them
    resources: object | None


def load_encoder(config: Config, device: torch.device) -> SentenceTransformer:
    path = config.paths.model_dir(config.dense.model)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"{path} has no weights; run `make data` first")
    encoder = SentenceTransformer(str(path), device=str(device), local_files_only=True)
    encoder.max_seq_length = config.dense.max_length
    if config.dense.fp16 and device.type == "cuda":
        encoder.half()
    return encoder


def encode(
    encoder: SentenceTransformer, texts: Sequence[str], batch_size: int
) -> np.ndarray:
    """Unit-length float32 embeddings, so inner product equals cosine similarity."""
    if any(not text for text in texts):
        raise ValueError("refusing to embed an empty text")
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def cached_encode(
    encoder: SentenceTransformer, texts: Sequence[str], config: DenseConfig, path: Path
) -> tuple[np.ndarray, float]:
    """encode() memoised on disk, keyed by the texts and encoder; also the original encode time."""
    settings = (
        f"{config.model.repo}@{config.model.revision}|{config.max_length}|{config.fp16}"
    )
    hasher = hashlib.sha256(settings.encode())
    for text in texts:
        hasher.update(text.encode())
        hasher.update(b"\0")
    digest = hasher.hexdigest()
    if path.is_file():
        with np.load(path) as cached:
            if str(cached["digest"]) == digest:
                return cached["vectors"], float(cached["seconds"])
    started = time.perf_counter()
    vectors = encode(encoder, texts, config.batch_size)
    seconds = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, digest=digest, vectors=vectors, seconds=seconds)
    return vectors, seconds


def build_index(vectors: np.ndarray, rows: np.ndarray, n_chunks: int) -> DenseIndex:
    """Exact inner-product index, on the GPU when faiss sees one."""
    index, resources = faiss.IndexFlatIP(vectors.shape[1]), None
    if faiss.get_num_gpus() > 0:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    index.add(vectors)
    return DenseIndex(index=index, rows=rows, n_chunks=n_chunks, resources=resources)


def search(
    index: DenseIndex, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k chunk positions and scores per query, best first; -inf marks padding."""
    scores, ids = index.index.search(queries, min(k, index.index.ntotal))
    hits = np.where(ids >= 0, index.rows[np.maximum(ids, 0)], 0)
    return hits, np.where(ids >= 0, scores, -np.inf).astype(np.float32)
