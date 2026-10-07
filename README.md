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

| Retriever                     | Chunking        | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 |
|-------------------------------|-----------------|-----------------|------------------|--------------|
| BM25                          | whole documents | 0.678           | 0.320            | 0.235        |
| BM25                          | fixed 256 / 64  | 0.684           | 0.316            | 0.232        |
| Dense                         |                 | TBD             | TBD              | TBD          |
| Dense + cross-encoder rerank  |                 | TBD             | TBD              | TBD          |
| *Published BM25 (Pyserini)*   | whole documents | *0.679*         | *0.322*          | *0.236*      |

Every number in this README is produced by a script in this repository; anything not yet run is
`TBD`. The published row is the [Pyserini BEIR reproduction](https://castorini.github.io/pyserini/2cr/beir.html)
of BM25 over concatenated title and text ("flat"), the setting our whole-document row mirrors.

### Retrieval: BM25 baseline

BEIR test splits (300, 323 and 648 queries), Lucene-style BM25 with k1 = 0.9, b = 0.4, untuned.
Documents are ranked by their best chunk. Latency is per query, answered on its own on one
RTX 3090: query analysis, scoring every chunk, top-1000 selection and max-pooling to documents.
Produced by `make eval`; full table in [`reports/retrieval_bm25.csv`](reports/retrieval_bm25.csv).

| Chunking        | Corpus   | Documents | Chunks | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
|-----------------|----------|-----------|--------|---------|------------|--------|--------|--------|
| whole documents | SciFact  | 5,183     | 5,183  | 0.678   | 0.925      | 0.643  | 0.88   | 1.07   |
| fixed 256 / 64  | SciFact  | 5,183     | 9,245  | 0.684   | 0.924      | 0.649  | 0.95   | 1.59   |
| whole documents | NFCorpus | 3,633     | 3,633  | 0.320   | 0.246      | 0.521  | 0.85   | 2.06   |
| fixed 256 / 64  | NFCorpus | 3,633     | 6,944  | 0.316   | 0.243      | 0.514  | 0.86   | 1.92   |
| whole documents | FiQA     | 57,638    | 57,638 | 0.235   | 0.541      | 0.295  | 1.26   | 3.57   |
| fixed 256 / 64  | FiQA     | 57,638    | 74,688 | 0.232   | 0.530      | 0.293  | 1.18   | 1.46   |

Whole-document BM25 lands within 0.003 of the published nDCG@10 and Recall@100 (SciFact
0.679 / 0.925, NFCorpus 0.322 / 0.246, FiQA 0.236 / 0.539) on every subset, which validates the
qrels handling end to end; our metric code also agrees with pytrec_eval, the implementation BEIR
reports with, to 1e-15 on all six runs (`scripts/crosscheck_metrics.py`). Chunking moves BM25 by
less than 0.01 nDCG@10 in either direction.

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
make eval    # BM25 on the three BEIR test splits -> reports/retrieval_bm25.csv, MLflow, runs/
make report  # collect every stage's table into reports/results.csv
```

Each `make eval` writes the exact config and the TREC run files to a timestamped `runs/`
directory and logs to a local MLflow store (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).
To recompute the metrics of a run with pytrec_eval:

```bash
uv pip install --python .venv/bin/python -e ".[crosscheck]"
.venv/bin/python scripts/crosscheck_metrics.py runs/<timestamp>-bm25
```

`make train` is a no-op until a stage with learned parameters is added.

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
