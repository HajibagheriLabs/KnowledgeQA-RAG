import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from knowledgeqa.config import ChunkingConfig, Config
from knowledgeqa.ingest import Document

# a period after one of these words, or after a single letter, does not end a sentence
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "ca",
        "cf",
        "dr",
        "e.g",
        "eq",
        "eqs",
        "et",
        "fig",
        "figs",
        "i.e",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "nos",
        "prof",
        "ref",
        "refs",
        "resp",
        "sr",
        "st",
        "u.k",
        "u.s",
        "viz",
        "vol",
        "vs",
    }
)
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_index: int
    doc_id: str
    ordinal: int
    char_start: int
    char_end: int
    n_tokens: int
    text: str


def load_tokenizer(config: Config) -> Tokenizer:
    path = config.paths.model_dir(config.chunking.tokenizer) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    tokenizer = Tokenizer.from_file(str(path))
    # chunk lengths are measured on whole documents, so nothing may be cut or padded
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def token_offsets(
    texts: Sequence[str], tokenizer: Tokenizer, batch_size: int
) -> list[np.ndarray]:
    """Character (start, end) of every token in each text, as an (n_tokens, 2) int array."""
    offsets = []
    for i in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(
            list(texts[i : i + batch_size]), add_special_tokens=False
        )
        offsets += [
            np.asarray(e.offsets, dtype=np.int32).reshape(-1, 2) for e in encodings
        ]
    return offsets


def cached_token_offsets(
    texts: Sequence[str], tokenizer: Tokenizer, batch_size: int, path: Path
) -> list[np.ndarray]:
    """token_offsets() memoised on disk under a hash of the exact texts and tokenizer."""
    hasher = hashlib.sha256(tokenizer.to_str().encode())
    for text in texts:
        hasher.update(text.encode())
        hasher.update(b"\0")
    digest = hasher.hexdigest()
    if path.is_file():
        with np.load(path) as cached:
            if str(cached["digest"]) == digest:
                return np.split(cached["offsets"], cached["bounds"])
    offsets = token_offsets(texts, tokenizer, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds = np.cumsum([len(o) for o in offsets], dtype=np.int64)[:-1]
    flat = np.concatenate(offsets) if offsets else np.empty((0, 2), dtype=np.int32)
    np.savez(path, digest=digest, offsets=flat, bounds=bounds)
    return offsets


def corpus_offsets(
    config: Config, name: str, docs: Sequence[Document], tokenizer: Tokenizer
) -> list[np.ndarray]:
    path = config.paths.cache / f"{name}-token-offsets.npz"
    texts = [doc.text for doc in docs]
    return cached_token_offsets(texts, tokenizer, config.chunking.batch_size, path)


def fixed_windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Token windows of `size` where consecutive windows share exactly `overlap` tokens."""
    windows, start = [], 0
    while True:
        end = min(start + size, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            return windows
        start += size - overlap


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in `text`, split after terminal punctuation."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start(1)
        last_word = (
            text[start:end].rsplit(maxsplit=1)[-1].lstrip("\"'([").rstrip(".!?\"')]")
        )
        if last_word.lower() in _ABBREVIATIONS or (
            len(last_word) == 1 and last_word.isalpha()
        ):
            continue
        spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def sentence_windows(
    sentences: Sequence[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int]]:
    """Pack whole sentences into windows of at most `size` tokens, repeating up to `overlap`."""
    if not sentences:
        return [(0, 0)]
    length = [end - start for start, end in sentences]
    windows, i = [], 0
    while i < len(sentences):
        j, used = i, 0
        while j < len(sentences) and used + length[j] <= size:
            used += length[j]
            j += 1
        if j == i:
            # a sentence longer than the whole budget falls back to fixed windows
            start, end = sentences[i]
            windows += [
                (start + a, start + b)
                for a, b in fixed_windows(end - start, size, overlap)
            ]
            i += 1
            continue
        windows.append((sentences[i][0], sentences[j - 1][1]))
        if j == len(sentences):
            break
        # carry sentences back only while sentence j still fits, and never to i, so i advances
        k, carried = j, 0
        while (
            k - 1 > i
            and carried + length[k - 1] <= overlap
            and carried + length[k - 1] + length[j] <= size
        ):
            carried += length[k - 1]
            k -= 1
        i = k
    return windows


def token_windows(
    text: str, offsets: np.ndarray, config: ChunkingConfig
) -> list[tuple[int, int]]:
    n_tokens = len(offsets)
    if config.strategy == "none":
        return [(0, n_tokens)]
    if config.strategy == "fixed":
        return fixed_windows(n_tokens, config.chunk_size, config.overlap)
    if n_tokens == 0:
        return [(0, 0)]
    # each token belongs to the sentence its first character falls in
    starts = np.searchsorted(
        offsets[:, 0], [start for start, _ in sentence_spans(text)]
    )
    starts[0] = 0
    bounds = list(zip(starts.tolist(), [*starts[1:].tolist(), n_tokens]))
    return sentence_windows(
        [b for b in bounds if b[1] > b[0]], config.chunk_size, config.overlap
    )


def chunk_corpus(
    docs: Sequence[Document], offsets: Sequence[np.ndarray], config: ChunkingConfig
) -> list[Chunk]:
    """Split every document into chunks; every document yields at least one chunk."""
    if len(docs) != len(offsets):
        raise ValueError(f"{len(docs)} documents but {len(offsets)} offset arrays")
    chunks = []
    for doc_index, (doc, tokens) in enumerate(zip(docs, offsets, strict=True)):
        for ordinal, (start, end) in enumerate(token_windows(doc.text, tokens, config)):
            char_start, char_end = (
                (tokens[start, 0], tokens[end - 1, 1]) if end > start else (0, 0)
            )
            body = doc.text[char_start:char_end]
            text = (
                f"{doc.title} {body}".strip()
                if config.prepend_title and doc.title
                else body
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_index=doc_index,
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    char_start=int(char_start),
                    char_end=int(char_end),
                    n_tokens=end - start,
                    text=text,
                )
            )
    return chunks
