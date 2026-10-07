# KnowledgeQA-RAG

Retrieval-augmented question answering, built and evaluated one stage at a time: a BM25 baseline
first, dense retrieval measured against it, a cross-encoder reranker measured against that, and an
extractive reader whose answers, and whose abstentions on unanswerable questions, are scored end to
end. Everything runs locally from pinned public datasets and open weights, with no API keys and no
network calls at inference.

## Results

Retrieval is evaluated on three BEIR subsets (SciFact, NFCorpus, FiQA-2018) against the official
qrels with nDCG@10, Recall@100 and MRR@10. Answering is evaluated on SQuAD 2.0: EM and F1 on
answerable questions, reported separately from abstention accuracy on unanswerable ones. Latency is
reported as p50/p95 per stage, with the corpus size stated.

| Retriever | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 |
|-----------|-----------------|------------------|--------------|
| BM25      | TBD             | TBD              | TBD          |

Every number in this README is produced by a script in this repository; anything not yet run is
`TBD`.

### Corpora

Lengths are in bert-base-uncased WordPiece tokens (title + text). Chunks are for the default
configuration, fixed 256-token windows with 64 tokens of overlap; `make stats` writes the full
grid to [`reports/corpus_stats.csv`](reports/corpus_stats.csv).

| Corpus   | Documents | Mean tokens | Median | p99 | Max   | Chunks | Chunks per doc |
|----------|-----------|-------------|--------|-----|-------|--------|----------------|
| SciFact  | 5,183     | 335         | 314    | 759 | 1,937 | 9,245  | 1.78           |
| NFCorpus | 3,633     | 355         | 355    | 689 | 2,375 | 6,944  | 1.91           |
| FiQA     | 57,638    | 175         | 118    | 836 | 3,597 | 74,688 | 1.30           |

FiQA includes 38 empty documents that the official qrels still judge relevant; they are kept so the
corpus and qrels match the official release (see DECISIONS.md).

## Setup

Requirements: Linux, Python 3.11 via [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with a
driver that supports CUDA 12.

```bash
make setup   # .venv with pinned dependencies (PyTorch from the CUDA 12.6 wheel index)
make data    # BEIR SciFact, NFCorpus, FiQA and SQuAD 2.0 into data/raw at pinned revisions
```

## Usage

```bash
make lint
make test
make stats   # corpus sizes, token lengths and chunk counts per chunking configuration
```

Evaluation and reporting targets are added as each stage lands.

## Layout

```
configs/default.yaml     every path, revision and hyperparameter
scripts/                 runnable entry points (download, inspection, experiments)
src/knowledgeqa/         library code: ingest, chunking, sparse, dense, rerank, reader, evaluate, serve
tests/                   small pytest suite
reports/                 committed result tables
DECISIONS.md             non-obvious choices and why
```

## License

MIT
