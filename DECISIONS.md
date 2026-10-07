# Decisions

A short log of non-obvious choices and the reasoning behind them, oldest first.

## Reader scope: extractive only

SQuAD 2.0 scores answers by EM and F1 against spans of the given context, so an extractive reader
is measured by exactly the metric the benchmark was built for. Its abstention is a score threshold
(best span vs. the null answer) that can be tuned on held-out data, and published SQuAD 2.0 reader
numbers exist to sanity-check the evaluation harness, the same way published BEIR numbers check the
retrieval side. A generative path would be scored by lexical overlap that penalises correct
paraphrases, its abstention would hinge on prompt wording, and an instruction-tuned LLM may have
seen SQuAD's Wikipedia passages in pre-training. It would make the evaluation weaker, so it is out
of scope.

## Data: raw upstream files at pinned Hub commits

`scripts/download_data.py` fetches the original files (BEIR parquet corpora and queries, TSV qrels,
SQuAD 2.0 parquet) from the Hugging Face Hub at commit SHAs pinned in `configs/default.yaml`, and
verifies each file's size and hash (sha256 for LFS files, the git blob hash otherwise). The config
loader rejects branch names, so upstream edits cannot silently change the data behind an unchanged
config. `datasets.load_dataset` would also work, but it resolves `main` unless told otherwise and
keeps a second, converted Arrow copy of everything; since nothing else needs it, `datasets` is not a
dependency.

## Evaluation vs. tuning splits

BEIR test splits are evaluation-only. Anything tuned is tuned on `dev` for NFCorpus and FiQA, and on
`train` for SciFact, which ships no dev split. The config loader refuses a tuning split equal to the
evaluation split.

## Environment

Python 3.11 managed by uv, dependencies pinned to minor versions in `pyproject.toml`. PyTorch is
installed from the CUDA 12.6 wheel index instead of PyPI's CUDA 13 default: the same features, and
it runs on a wider range of installed drivers.

## Chunk lengths are WordPiece tokens

Chunk sizes and overlaps are counted with the bert-base-uncased WordPiece tokenizer (pinned
commit), the vocabulary shared by the BERT-family encoders used downstream, so a 256-token chunk
is 256 tokens to the encoder rather than an estimate from whitespace words. The tokenizer's
character offsets map every chunk back to an exact span of its document. The budget counts body
tokens; the title is prepended to every chunk (BEIR's `title + " " + text` convention), adding up
to ~90 tokens on SciFact.

## Two chunkers

`fixed` slides a window of exactly `chunk_size` tokens whose neighbours share exactly `overlap`
tokens: the textbook baseline, which cuts mid-sentence and occasionally mid-word. `sentence` packs
whole sentences up to the budget and repeats trailing sentences worth at most `overlap` tokens in
the next chunk; a single sentence longer than the budget falls back to fixed windows. Sentences
are split by a regular expression with an abbreviation guard (`et al.`, `Fig.`, initials), which is
enough for abstracts and forum posts and adds no model or dependency.

## Chunk-to-document mapping

Every chunk stores its parent's `doc_id` and position explicitly; nothing is ever parsed back out
of a chunk id, so ids like `MED-10`, `0123` or ids containing the `#` separator are safe (the
test exercises exactly these). Every document yields at least one chunk, empty ones included, so
no judged document can silently fall out of the evaluation.

## Empty FiQA documents are kept

FiQA's corpus has 38 documents with neither title nor text, and the official qrels judge 38 rows
on them relevant (35 train, 2 dev, 1 test). They stay in, so the corpus (57,638 documents) and the
qrels match the official release and published numbers remain comparable. No content-based
retriever can find them, which caps attainable recall for one FiQA test query; retrievers must
never score an empty text.

## Default chunking chosen a priori

`fixed`, 256 tokens, 64 overlap (25%), chosen before any retrieval result was seen. The ablation
grid (fixed and sentence, 128/256/512 tokens, 0% and 25% overlap, plus whole documents) is
evaluated later, and any choice among its configurations is made on SciFact's tuning split.

## Tokenisation is cached by content

Tokenising FiQA takes ~17 s per run. Offsets are memoised in `data/cache` under a SHA-256 of the
exact document texts and the serialised tokenizer, so a stale cache can never be served.

## BM25: our own Lucene-formula implementation

`rank_bm25` uses a different IDF (negative values floored at a fraction of the mean) and loops over
every document in Python for each query term; Pyserini reproduces Lucene exactly but needs a JVM
and a Lucene index per chunking configuration. `sparse.py` instead implements Lucene's BM25
(`idf = ln(1 + (N - df + 0.5) / (df + 0.5))`, `tf / (tf + k1 (1 - b + b dl / avgdl))`) with an
approximation of Anserini's analyzer: lowercase, word tokens kept whole across `.` and `'`
(close to UAX#29), possessive `'s` stripped, Lucene's 33 English stopwords, Porter stemming. The
index stores each term's postings with their precomputed BM25 contributions, and a query is scored
term at a time: every posting list is added into the query's score vector over all chunks, exactly,
with no pruning. It runs on the configured device, the GPU by default, the same device the dense
encoders will use. `k1 = 0.9, b = 0.4` are Anserini's defaults and are not tuned.

Scores are bitwise reproducible across processes: terms are added in a fixed order and no chunk
repeats within a posting list, so no two additions ever race. A first version scored all queries
with one cuSPARSE sparse-dense product instead; its results drifted by up to 4e-6 between
processes, even with PyTorch's deterministic mode on. Rankings and metrics never changed, but
identical runs should give identical files, and now do.

## BM25 reproduces the published numbers

Whole-document BM25 (title and text concatenated, as in Pyserini's "flat" BEIR baseline) against
the [published reproduction](https://castorini.github.io/pyserini/2cr/beir.html):

| Subset   | nDCG@10 ours / published | Recall@100 ours / published |
|----------|--------------------------|-----------------------------|
| SciFact  | 0.678 / 0.679            | 0.925 / 0.925               |
| NFCorpus | 0.320 / 0.322            | 0.246 / 0.246               |
| FiQA     | 0.235 / 0.236            | 0.541 / 0.539               |

Every difference is at most 0.003, the size expected from Lucene's lossy one-byte document-length
encoding and its UAX#29 tokenizer versus our regex (Kamphuis et al., "Which BM25 Do You Mean?",
ECIR 2020). The qrels handling is therefore right, and the BEIR paper's Elasticsearch numbers
(0.665, 0.325, 0.236) are in the same range. Independently, `scripts/crosscheck_metrics.py`
re-scores the saved runs with pytrec_eval and agrees to 1e-15 on all six.

## Metric definitions follow trec_eval

nDCG uses linear gains (NFCorpus has grades 1 and 2) and the ideal ranking of all judged relevant
documents; ties are broken by document id descending, as trec_eval does; every judged query counts,
even one whose ranking is empty. MRR@10 is not reported by BEIR and is computed the usual way
(reciprocal rank of the first relevant document in the top 10). BEIR's evaluator by default also
drops any retrieved document whose id equals the query id, a fix for ArguAna and Quora where
queries are drawn from the corpus; it is not applied here, although 55 FiQA test query ids happen
to coincide with unrelated document ids.

## Chunk scores become document scores by max-pooling

Retrieval runs over chunks; each document takes the score of its best chunk (MaxP). Pooling the
top 1,000 chunks is exact for the top 100 documents whenever they cover at least 100 documents,
because a document whose best chunk was not retrieved cannot outscore a retrieved one; when the
candidates do not, `max_pool` raises instead of returning a silently truncated ranking. Chunks with
no query term in common are not retrieved at all, as in Lucene, rather than ranked arbitrarily among
zero scores, which would add chance hits to Recall@100.

## Experiment tracking

MLflow 3.17 refuses the plain `mlruns/` file store, so runs go to SQLite at `mlruns/mlflow.db` with
artifacts beside it; everything stays local and gitignored. Each run directory under `runs/` keeps
the exact config and the TREC run files.
