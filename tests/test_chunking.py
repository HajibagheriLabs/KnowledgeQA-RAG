import re
from itertools import pairwise

import numpy as np

from knowledgeqa.chunking import (
    chunk_corpus,
    fixed_windows,
    sentence_spans,
    sentence_windows,
)
from knowledgeqa.config import ChunkingConfig
from knowledgeqa.ingest import Document

DOCS = [
    Document(
        "MED-10",
        "Statins and survival",
        " ".join(f"Sentence {i} ends here." for i in range(9)),
        "nfcorpus",
    ),
    Document("10", "", "A short answer about index funds.", "fiqa"),
    Document("010", "", "", "fiqa"),
    Document("a#1", "Title only", "", "scifact"),
]


def whitespace_offsets(text: str) -> np.ndarray:
    return np.array(
        [m.span() for m in re.finditer(r"\S+", text)], dtype=np.int32
    ).reshape(-1, 2)


def test_chunk_to_document_mapping_round_trips():
    offsets = [whitespace_offsets(doc.text) for doc in DOCS]
    configs = [
        ChunkingConfig("none", None, 0, True),
        ChunkingConfig("fixed", 7, 3, True),
        ChunkingConfig("sentence", 10, 4, False),
    ]
    for config in configs:
        chunks = chunk_corpus(DOCS, offsets, config)
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        by_doc: dict[int, list] = {}
        for chunk in chunks:
            doc = DOCS[chunk.doc_index]
            assert chunk.doc_id == doc.doc_id
            body = doc.text[chunk.char_start : chunk.char_end]
            expected = (
                f"{doc.title} {body}".strip()
                if config.prepend_title and doc.title
                else body
            )
            assert chunk.text == expected
            by_doc.setdefault(chunk.doc_index, []).append(chunk)

        # no document is lost, not even empty ones, and chunks of a document jointly cover all of it
        assert sorted(by_doc) == list(range(len(DOCS)))
        for doc_index, doc_chunks in by_doc.items():
            assert [chunk.ordinal for chunk in doc_chunks] == list(
                range(len(doc_chunks))
            )
            covered = {
                token
                for chunk in doc_chunks
                for token, (start, end) in enumerate(offsets[doc_index])
                if chunk.char_start <= start and end <= chunk.char_end
            }
            assert covered == set(range(len(offsets[doc_index])))


def test_chunk_boundaries_respect_configured_overlap():
    windows = fixed_windows(23, size=8, overlap=3)
    assert windows == [(0, 8), (5, 13), (10, 18), (15, 23)]
    assert all(end - start == 8 for start, end in windows)
    assert all(prev[1] - nxt[0] == 3 for prev, nxt in pairwise(windows))
    assert fixed_windows(5, 8, 3) == [(0, 5)]
    assert fixed_windows(0, 8, 3) == [(0, 0)]

    # sentences of 4, 3, 5, 2 and 9 tokens; the last one alone exceeds the budget of 8
    sentences = [(0, 4), (4, 7), (7, 12), (12, 14), (14, 23)]
    windows = sentence_windows(sentences, size=8, overlap=3)
    assert windows == [(0, 7), (4, 12), (12, 14), (14, 22), (19, 23)]
    sentence_starts = {start for start, _ in sentences}
    for prev, nxt in pairwise(windows):
        assert nxt[1] - nxt[0] <= 8
        assert max(0, prev[1] - nxt[0]) <= 3
        assert nxt[0] in sentence_starts or nxt[0] > sentences[-1][0]

    text = "Dr. Smith arrived. He left at 5 p.m. today. Fig. 2 shows it, see Smith et al. 2010."
    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "Dr. Smith arrived.",
        "He left at 5 p.m. today.",
        "Fig. 2 shows it, see Smith et al. 2010.",
    ]
