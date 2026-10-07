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
