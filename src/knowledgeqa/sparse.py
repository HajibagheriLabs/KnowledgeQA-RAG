import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import Stemmer
import torch

from knowledgeqa.config import BM25Config

# lucene's english stopword set, the one the anserini/pyserini beir baselines index with
LUCENE_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
        "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "was", "will", "with",
    }
)  # fmt: skip
# approximates lucene's uax#29 tokenizer, which keeps "3.5", "u.s" and "don't" as single tokens
_TOKEN = re.compile(r"\w+(?:[.'’]\w+)*")


@dataclass(frozen=True)
class BM25Index:
    vocab: dict[str, int]
    # (n_chunks, vocab) sparse matrix of per-term bm25 contributions, so scoring is one matmul
    weights: torch.Tensor
    stemmer: str


def analyze(texts: Sequence[str], stemmer: Stemmer.Stemmer) -> list[list[str]]:
    """Lowercase, tokenise, drop lucene stopwords and possessives, then porter-stem."""
    analyzed = []
    for text in texts:
        tokens = []
        for token in _TOKEN.findall(text.lower()):
            if token.endswith(("'s", "’s")):
                token = token[:-2]
            if token and token not in LUCENE_STOPWORDS:
                tokens.append(token)
        analyzed.append(stemmer.stemWords(tokens))
    return analyzed


def build_index(
    texts: Sequence[str], config: BM25Config, device: torch.device
) -> BM25Index:
    """Lucene's BM25: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), tf saturated by k1 and b."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for row, terms in enumerate(analyze(texts, Stemmer.Stemmer(config.stemmer))):
        ids = [vocab.setdefault(term, len(vocab)) for term in terms]
        cols.append(np.asarray(ids, dtype=np.int64))
        rows.append(np.full(len(ids), row, dtype=np.int64))
    pairs = np.stack([np.concatenate(rows), np.concatenate(cols)])
    indices = torch.from_numpy(pairs).to(device)
    shape = (len(texts), len(vocab))
    ones = torch.ones(indices.shape[1], device=device)
    tf = torch.sparse_coo_tensor(indices, ones, shape, check_invariants=True).coalesce()

    row, col = tf.indices()
    freq = tf.values()
    length = torch.bincount(indices[0], minlength=shape[0]).float()
    # like lucene, documents without any indexed term count neither in N nor in avgdl
    n_docs = int((length > 0).sum())
    if n_docs == 0:
        raise ValueError("nothing to index: every text is empty after analysis")
    avg_length = length.sum() / n_docs
    df = torch.bincount(col, minlength=shape[1]).float()
    idf = torch.log1p((n_docs - df + 0.5) / (df + 0.5))
    norm = config.k1 * (1 - config.b + config.b * length / avg_length)
    weight = idf[col] * freq / (freq + norm[row])
    weights = torch.sparse_coo_tensor(
        tf.indices(), weight, shape, check_invariants=True
    ).coalesce()
    return BM25Index(vocab=vocab, weights=weights, stemmer=config.stemmer)


def search(
    index: BM25Index, queries: Sequence[str], k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k chunk indices and scores per query, best first; -inf marks non-matching chunks."""
    stemmer = Stemmer.Stemmer(index.stemmer)
    device = index.weights.device
    n_chunks, n_terms = index.weights.shape
    k = min(k, n_chunks)
    top_ids, top_scores = [], []
    for start in range(0, len(queries), batch_size):
        batch = analyze(queries[start : start + batch_size], stemmer)
        # a repeated query term counts once per occurrence, as in anserini's bag-of-words queries
        query_terms = torch.zeros(len(batch), n_terms, device=device)
        for i, terms in enumerate(batch):
            ids = [index.vocab[term] for term in terms if term in index.vocab]
            if ids:
                ids_t = torch.tensor(ids, device=device)
                query_terms[i].index_add_(0, ids_t, torch.ones(len(ids), device=device))
        scores = torch.sparse.mm(index.weights, query_terms.T).T
        values, ids = torch.topk(scores, k, dim=1)
        # lucene never retrieves a document without a query term; ranking zeros adds chance hits
        values = values.masked_fill(values <= 0, float("-inf"))
        top_ids.append(ids.cpu().numpy())
        top_scores.append(values.cpu().numpy())
    return np.concatenate(top_ids), np.concatenate(top_scores)
