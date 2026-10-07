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

Every number in this README is produced by a script in this repository and the tables between
`table` markers are written by `make report`; anything not yet run is `TBD`.

### Retrieval

nDCG@10 on the BEIR test splits (300, 323 and 648 queries). Documents are ranked by their best
chunk; "whole documents" indexes each document as a single unit, the setting published numbers use.

<!-- table:summary -->
| Retriever | Chunking | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 |
|---|---|---|---|---|
| BM25 | whole documents | 0.678 | 0.320 | 0.235 |
| BM25 | fixed 256 / 64 | 0.684 | 0.316 | 0.232 |
| Dense (bge-base) | whole documents | 0.742 | 0.374 | 0.406 |
| Dense (bge-base) | fixed 256 / 64 | 0.744 | 0.376 | 0.407 |
| Hybrid (RRF) | whole documents | 0.740 | 0.371 | 0.370 |
| Hybrid (RRF) | fixed 256 / 64 | 0.741 | 0.374 | 0.356 |
| Dense + rerank (bge-v2-m3) | fixed 256 / 64 | 0.739 | 0.347 | 0.431 |
<!-- /table:summary -->

Published references for the whole-document rows, from the
[Pyserini BEIR reproductions](https://castorini.github.io/pyserini/2cr/beir.html): BM25 "flat"
0.679 / 0.322 / 0.236 and BGE-base 0.741 / 0.373 / 0.406 (SciFact / NFCorpus / FiQA).

- **Both baselines reproduce the published numbers.** Whole-document BM25 is within 0.003 of
  Pyserini's nDCG@10 and Recall@100 on every subset, and whole-document dense retrieval within
  0.002, which validates qrels handling, document-id mapping and metrics end to end. The metric
  code also agrees with pytrec_eval, the implementation BEIR reports with, to 1e-15
  (`scripts/crosscheck_metrics.py`).
- **BM25 does not win anywhere here.** bge-base beats it on all three subsets, by 0.06 nDCG@10 on
  SciFact and NFCorpus and by 0.17 on FiQA, where financial forum posts reward semantic matching.
  Older or smaller encoders often lose to BM25 on SciFact and NFCorpus; this one does not.
- **Hybrid fusion does not pay off with this encoder.** Reciprocal rank fusion of BM25 and dense
  ties dense on SciFact and NFCorpus (within 0.003 nDCG@10) and is clearly worse on FiQA (0.356 vs
  0.407): RRF weights both rankings equally, and BM25 is too weak a partner there. It does lift
  Recall@100 on SciFact (0.977 vs 0.966), which matters for a reranker downstream.
- **Chunking barely moves first-stage retrieval** at the default 256-token windows: under 0.01
  nDCG@10 for BM25 and dense on every subset; only hybrid on FiQA moves more (0.370 to 0.356). The
  chunking ablation is reported separately.

Default chunking (fixed 256-token windows, 64 overlap), with corpus size and per-query latency.
Latency answers each query on its own on one RTX 3090: query analysis or encoding, scoring every
chunk, top-1000 selection and max-pooling to documents; hybrid runs both retrievers in sequence.
Full tables: [`reports/results.csv`](reports/results.csv).

<!-- table:retrieval -->
| Retriever | Corpus | Documents | Chunks | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| BM25 | SciFact | 5,183 | 9,245 | 0.684 | 0.924 | 0.649 | 0.9 | 1.8 |
| BM25 | NFCorpus | 3,633 | 6,944 | 0.316 | 0.243 | 0.514 | 0.6 | 0.8 |
| BM25 | FiQA | 57,638 | 74,688 | 0.232 | 0.530 | 0.293 | 1.3 | 3.2 |
| Dense (bge-base) | SciFact | 5,183 | 9,245 | 0.744 | 0.966 | 0.709 | 16.3 | 21.4 |
| Dense (bge-base) | NFCorpus | 3,633 | 6,944 | 0.376 | 0.346 | 0.567 | 15.8 | 23.0 |
| Dense (bge-base) | FiQA | 57,638 | 74,688 | 0.407 | 0.739 | 0.487 | 17.9 | 26.0 |
| Hybrid (RRF) | SciFact | 5,183 | 9,245 | 0.741 | 0.977 | 0.703 | 19.1 | 23.1 |
| Hybrid (RRF) | NFCorpus | 3,633 | 6,944 | 0.374 | 0.344 | 0.569 | 18.2 | 24.6 |
| Hybrid (RRF) | FiQA | 57,638 | 74,688 | 0.356 | 0.709 | 0.430 | 20.2 | 27.7 |
| Dense + rerank (bge-v2-m3) | SciFact | 5,183 | 9,245 | 0.739 | 0.966 | 0.712 | 948.8 | 1211.3 |
| Dense + rerank (bge-v2-m3) | NFCorpus | 3,633 | 6,944 | 0.347 | 0.346 | 0.557 | 914.9 | 1004.7 |
| Dense + rerank (bge-v2-m3) | FiQA | 57,638 | 74,688 | 0.431 | 0.739 | 0.517 | 710.7 | 856.0 |
<!-- /table:retrieval -->

<details>
<summary>Whole-document rows</summary>

<!-- table:retrieval-whole -->
| Retriever | Corpus | Documents | Chunks | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| BM25 | SciFact | 5,183 | 5,183 | 0.678 | 0.925 | 0.643 | 0.9 | 1.3 |
| BM25 | NFCorpus | 3,633 | 3,633 | 0.320 | 0.246 | 0.521 | 0.6 | 1.0 |
| BM25 | FiQA | 57,638 | 57,638 | 0.235 | 0.541 | 0.295 | 1.2 | 3.4 |
| Dense (bge-base) | SciFact | 5,183 | 5,183 | 0.742 | 0.967 | 0.705 | 16.0 | 20.0 |
| Dense (bge-base) | NFCorpus | 3,633 | 3,633 | 0.374 | 0.337 | 0.563 | 15.6 | 21.2 |
| Dense (bge-base) | FiQA | 57,638 | 57,638 | 0.406 | 0.742 | 0.486 | 17.2 | 23.6 |
| Hybrid (RRF) | SciFact | 5,183 | 5,183 | 0.740 | 0.977 | 0.704 | 18.9 | 22.9 |
| Hybrid (RRF) | NFCorpus | 3,633 | 3,633 | 0.371 | 0.341 | 0.568 | 18.0 | 21.6 |
| Hybrid (RRF) | FiQA | 57,638 | 57,638 | 0.370 | 0.712 | 0.445 | 22.6 | 33.6 |
<!-- /table:retrieval-whole -->

</details>

### Reranking: what the cross-encoder buys and costs

bge-reranker-v2-m3 (568M parameters) re-scores every chunk of the dense retriever's top 100
documents (`make eval`, [`reports/retrieval_rerank.csv`](reports/retrieval_rerank.csv)). The
first stage and the reranker were both chosen on the tuning splits, never on test
([`reports/selection.csv`](reports/selection.csv)); latency is measured on a fixed sample of 100
queries per subset, each answered on its own.

<!-- table:rerank -->
| Corpus | Pairs per query | nDCG@10 before | nDCG@10 after | Change | MRR@10 before | MRR@10 after | Retrieval p50 / p95 ms | Rerank p50 / p95 ms | End to end p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| SciFact | 197 | 0.744 | 0.739 | -0.005 | 0.709 | 0.712 | 15 / 29 | 923 / 1227 | 949 / 1211 |
| NFCorpus | 208 | 0.376 | 0.347 | -0.030 | 0.567 | 0.557 | 15 / 28 | 900 / 985 | 915 / 1005 |
| FiQA | 181 | 0.407 | 0.431 | +0.024 | 0.487 | 0.517 | 16 / 17 | 693 / 840 | 711 / 856 |
<!-- /table:rerank -->

- **Reranking is not a free win here.** It helps FiQA (+0.024 nDCG@10, +0.030 MRR@10) and hurts
  NFCorpus (-0.030) and, slightly, SciFact (-0.005). The tuning splits predicted exactly this
  pattern, and on them no reranker beat dense retrieval alone on average: bge-reranker-base lost
  0.08 nDCG@10 and MS MARCO MiniLM-L6 0.03, so a strong bi-encoder leaves a cross-encoder little
  room, especially on medical (NFCorpus) and scientific (SciFact) text far from its training data.
- **It costs 43 to 60 times the first stage:** 0.7 to 0.9 s per query against 15 ms, because
  every query sends roughly 200 (query, chunk) pairs through a 568M-parameter model. Even on FiQA,
  where it helps, that is a poor trade for an interactive system unless the candidate list is cut
  or the reranker distilled.

### Chunking ablation: tuned on SciFact, transferred untouched

Dense retrieval (the best pipeline on the tuning splits) over 13 chunkings: fixed windows and
sentence-packed chunks of 128, 256 and 512 tokens with 0% and 25% overlap, plus whole documents
(`make ablation`, [`reports/ablations.csv`](reports/ablations.csv)). **The configuration was chosen
on SciFact's tuning split (train queries) alone; the NFCorpus and FiQA columns are untouched
transfer**, never used for any choice.

![nDCG@10 against chunk size for each strategy and overlap, per corpus](reports/ablations.png)

<!-- table:ablation -->
| Chunking | SciFact train (tuning) | SciFact test | NFCorpus test | FiQA test |
|---|---|---|---|---|
| whole documents | 0.771 | 0.742 | 0.374 | 0.406 |
| fixed 128 / 0 | 0.772 | 0.749 | 0.381 | 0.389 |
| fixed 128 / 32 | 0.765 | 0.755 | 0.377 | 0.390 |
| fixed 256 / 0 | 0.769 | 0.743 | 0.377 | 0.401 |
| fixed 256 / 64 (default) | 0.766 | 0.744 | 0.376 | 0.407 |
| fixed 512 / 0 | 0.773 | 0.743 | 0.373 | 0.409 |
| fixed 512 / 128 | 0.773 | 0.744 | 0.374 | 0.407 |
| sentence 128 / 0 | 0.775 | 0.759 | 0.379 | 0.384 |
| sentence 128 / 32 (selected) | 0.778 | 0.755 | 0.379 | 0.388 |
| sentence 256 / 0 | 0.773 | 0.746 | 0.375 | 0.407 |
| sentence 256 / 64 | 0.776 | 0.747 | 0.377 | 0.412 |
| sentence 512 / 0 | 0.771 | 0.745 | 0.374 | 0.408 |
| sentence 512 / 128 | 0.771 | 0.745 | 0.374 | 0.406 |
<!-- /table:ablation -->

- **The tuned choice transfers to SciFact's test split and to NFCorpus, but not to FiQA.**
  Sentence chunks of 128 tokens with 32 overlap win on the tuning split and add 0.011 nDCG@10 on
  SciFact test and 0.002 on NFCorpus over the a-priori default, but cost 0.019 on FiQA, where the
  best setting is the opposite, larger sentence chunks (256 / 64, 0.412). Abstracts reward small,
  focused chunks; forum answers need their context. One tuned chunking is not a property of
  retrieval in general.
- **Sentence boundaries help more often than not:** at equal size and overlap, sentence packing
  matches or beats fixed windows in 12 of 18 test cells, most clearly on SciFact at 128 tokens.
- **The effect is real but bounded:** chunking moves nDCG@10 by 0.007 to 0.028 within a corpus,
  smaller than the gap between BM25 and dense retrieval but comparable to the reranker's effect.

### Index study: exact vs approximate search

Dense search over FiQA's 74,650 non-empty chunk embeddings (768-d), one query at a time on two CPU
threads (`scripts/index_study.py`). Recall is the overlap of each index's top 100 with exact search;
nDCG@10 shows what that loss costs downstream. The retrieval rows above use exact search on the GPU.

![Recall relative to exact search against query latency for flat, HNSW and IVF-PQ indexes](reports/index_study.png)

<!-- table:index -->
| Index | Setting | Recall vs exact | nDCG@10 | p50 ms | p95 ms | Build s | Memory MB |
|---|---|---|---|---|---|---|---|
| flat | exact | 1.000 | 0.407 | 14.36 | 17.51 | 0.2 | 229.3 |
| hnsw | efSearch=16 | 0.597 | 0.384 | 0.22 | 0.34 | 65.9 | 249.6 |
| hnsw | efSearch=32 | 0.749 | 0.397 | 0.34 | 0.54 | 65.9 | 249.6 |
| hnsw | efSearch=64 | 0.871 | 0.402 | 0.56 | 0.77 | 65.9 | 249.6 |
| hnsw | efSearch=128 | 0.945 | 0.405 | 0.98 | 1.31 | 65.9 | 249.6 |
| hnsw | efSearch=256 | 0.981 | 0.407 | 1.82 | 2.74 | 65.9 | 249.6 |
| hnsw | efSearch=512 | 0.995 | 0.407 | 3.17 | 4.46 | 65.9 | 249.6 |
| ivfpq | nprobe=1 | 0.214 | 0.221 | 0.27 | 0.36 | 11.6 | 11.7 |
| ivfpq | nprobe=4 | 0.457 | 0.297 | 0.30 | 0.40 | 11.6 | 11.7 |
| ivfpq | nprobe=16 | 0.640 | 0.353 | 0.38 | 0.52 | 11.6 | 11.7 |
| ivfpq | nprobe=64 | 0.701 | 0.369 | 0.68 | 0.82 | 11.6 | 11.7 |
| ivfpq | nprobe=256 | 0.710 | 0.372 | 1.86 | 2.25 | 11.6 | 11.7 |
<!-- /table:index -->

- **Exact search is affordable at this scale.** 14 ms per query on two CPU threads, and faster on
  the GPU; below about a million vectors an approximate index is an optimisation, not a necessity.
- **HNSW is the practical approximate choice:** at efSearch 128 it keeps 94.5% of the exact top
  100 and loses 0.002 nDCG@10 at under 1 ms, for 9% more memory and a 66 s build.
- **IVF-PQ trades accuracy for memory:** 20x smaller (11.7 MB vs 229 MB), but product quantisation
  caps recall near 0.71 no matter how many lists are probed, costing 0.036 nDCG@10. It needs a
  re-scoring step with full vectors before it is usable here.

### End to end: retrieve, read, abstain (SQuAD 2.0)

The reader (`deepset/roberta-base-squad2`) answers from the top 5 paragraphs of a corpus of all
20,233 SQuAD 2.0 paragraphs, the 1,204 dev paragraphs among 19,029 train distractors, or from the
gold paragraph alone (oracle). The abstention threshold is tuned per row on 10 dev articles held
out whole; every number below is on the other 25 articles: 4,260 answerable and 4,189 unanswerable
questions (`make eval`, [`reports/e2e_results.csv`](reports/e2e_results.csv)). Answerable EM and F1
and abstention accuracy are separate columns on purpose: a blended score hides the trade-off.

<!-- table:e2e -->
| Retrieval | Passages read | Gold passage retrieved | Answerable EM | Answerable F1 | Abstention accuracy | Retrieval p50 / p95 ms | Reader p50 / p95 ms | End to end p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| Oracle passage | 1 | - | 0.741 | 0.790 | 0.874 | 0 / 0 | 17 / 34 | 17 / 34 |
| BM25 | 5 | 0.866 | 0.585 | 0.625 | 0.885 | 1 / 4 | 23 / 38 | 24 / 41 |
| Dense (bge-base) | 5 | 0.773 | 0.516 | 0.553 | 0.901 | 17 / 21 | 22 / 27 | 39 / 47 |
<!-- /table:e2e -->

![F1 on answerable questions against abstention accuracy as the threshold varies](reports/e2e_curve.png)

- **The oracle gap is the headline number: 0.790 F1 with the gold paragraph, 0.625 with the best
  retriever, so retrieval costs 0.165 F1 on answerable questions.** 11.9% of answerable questions
  (506 of 4,260) never get their gold paragraph into the top 5 and score 0.03 F1; on the rest the
  reader reaches 0.705, below the oracle's 0.790 because it must pick among five passages and its
  tuned threshold is stricter.
- **BM25 beats dense retrieval on SQuAD** (gold paragraph in the top 5 for 86.6% vs 77.3% of
  questions, 0.625 vs 0.553 F1), the reverse of BEIR. SQuAD questions were written while reading the
  paragraph and reuse its words, which favours exact term matching; the dense model's advantage on
  BEIR does not carry over.
- **Abstention is a dial, not a constant.** Along the curve the oracle reader trades answerable F1
  for abstention accuracy smoothly; the tuned thresholds sit near 0.87 to 0.90 abstention accuracy,
  and an application that cares more about wrong answers than missed ones can move along it.
- **The reader is validated in isolation:** with gold paragraphs on the full dev set and threshold 0
  it scores 79.81 EM / 82.80 F1 on the official blended metric, against 79.93 / 82.95 on its model card
  ([`reports/reader_check.csv`](reports/reader_check.csv)).
- Latency is per question on one RTX 3090, over a fixed sample of 500: reading 5 paragraphs takes
  22 to 23 ms at p50, so retrieval plus reading stays under 50 ms at p95.

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
make eval    # BM25, dense and hybrid on the BEIR test splits -> reports/retrieval_*.csv
make study   # exact vs HNSW vs IVF-PQ on FiQA -> reports/index_study.csv and .png
make report  # reports/results.csv, and refresh every table in this README
```

Each `make eval` writes the exact config and the TREC run files to a timestamped `runs/`
directory and logs to a local MLflow store (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).
To recompute the metrics of a run with pytrec_eval:

```bash
uv pip install --python .venv/bin/python -e ".[crosscheck]"
.venv/bin/python scripts/crosscheck_metrics.py runs/<timestamp>-retrieval
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
