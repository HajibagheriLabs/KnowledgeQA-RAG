import re
from collections import Counter
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
    # term-major postings: the chunks of term t are chunk_ids[offsets[t] : offsets[t + 1]]
    offsets: np.ndarray
    chunk_ids: torch.Tensor
    # each posting's bm25 contribution, so a query score is a sum over its terms' postings
    weights: torch.Tensor
    n_chunks: int
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
    n_chunks = len(texts)
    row = torch.from_numpy(np.concatenate(rows)).to(device)
    col = torch.from_numpy(np.concatenate(cols)).to(device)
    # one key per (term, chunk) pair; sorting groups postings by term, chunks ascending
    keys, freq = torch.unique(col * n_chunks + row, sorted=True, return_counts=True)
    term, chunk = keys // n_chunks, keys % n_chunks
    length = torch.bincount(row, minlength=n_chunks).float()
    # like lucene, documents without any indexed term count neither in N nor in avgdl
    n_docs = int((length > 0).sum())
    if n_docs == 0:
        raise ValueError("nothing to index: every text is empty after analysis")
    avg_length = length.sum() / n_docs
    df = torch.bincount(term, minlength=len(vocab))
    idf = torch.log1p((n_docs - df + 0.5) / (df + 0.5))
    freq = freq.float()
    norm = config.k1 * (1 - config.b + config.b * length / avg_length)
    weights = idf[term] * freq / (freq + norm[chunk])
    offsets = np.concatenate([[0], torch.cumsum(df, 0).cpu().numpy()])
    return BM25Index(vocab, offsets, chunk, weights, n_chunks, config.stemmer)


def search(
    index: BM25Index, queries: Sequence[str], k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k chunk indices and scores per query, best first; -inf marks non-matching chunks."""
    stemmer = Stemmer.Stemmer(index.stemmer)
    k = min(k, index.n_chunks)
    top_ids, top_scores = [], []
    for start in range(0, len(queries), batch_size):
        batch = analyze(queries[start : start + batch_size], stemmer)
        scores = torch.zeros(len(batch), index.n_chunks, device=index.weights.device)
        for row, terms in enumerate(batch):
            # a repeated query term counts per occurrence, as in anserini's bag-of-words queries
            counts = Counter(index.vocab[term] for term in terms if term in index.vocab)
            # fixed term order and no chunk repeated within a posting list: bitwise stable sums
            for term in sorted(counts):
                lo, hi = int(index.offsets[term]), int(index.offsets[term + 1])
                postings = index.chunk_ids[lo:hi]
                scores[row].index_add_(
                    0, postings, index.weights[lo:hi], alpha=counts[term]
                )
        values, ids = torch.topk(scores, k, dim=1)
        # lucene never retrieves a document without a query term; ranking zeros adds chance hits
        values = values.masked_fill(values <= 0, float("-inf"))
        top_ids.append(ids.cpu().numpy())
        top_scores.append(values.cpu().numpy())
    return np.concatenate(top_ids), np.concatenate(top_scores)
