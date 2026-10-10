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

## Dense retriever: bge-base-en-v1.5

A strong general-purpose base-size encoder with published BEIR numbers to check against, a
512-token window and the same WordPiece vocabulary the chunk sizes are counted in. Queries get the
instruction prefix the model was trained with; embeddings are unit-normalised, so inner product is
cosine similarity. Whole-document results land within 0.002 of the published BGE-base nDCG@10 and
Recall@100 on all three subsets, which validates encoding, pooling and the prefix. Weights are
pinned by commit and fetched by `make data`; nothing is downloaded at inference. Empty chunks
(FiQA's empty documents) are never embedded, since the embedding of an empty string is noise that
could be retrieved.

## Index type: exact search, with HNSW as the next step

The evaluation uses an exact `IndexFlatIP` on the GPU, the device that already holds the encoder:
at 75k vectors exact search is cheap, removes approximation error from every reported number, and
is bitwise reproducible across processes (verified, like BM25). The index study then measures what
approximate indexes would buy on the CPU, where HNSW lives in FAISS: recall relative to exact
search, downstream nDCG@10, single-query latency on two threads, build time and serialised size.
IVF-PQ k-means and codebook training run on the GPU. Settings are a priori (HNSW M 32,
efConstruction 200; IVF 1024 lists, 96 x 8-bit sub-quantisers) and swept only at query time.

The study keeps exact search as the index for every corpus here: at 75k vectors it costs 14 ms on
two CPU threads and less on the GPU, and it adds no approximation to the results. HNSW at efSearch
128 would be the choice for a larger corpus (94.5% of the exact top 100, 0.002 nDCG@10 lost, under
1 ms); IVF-PQ only when memory is the constraint, since its recall stops near 0.71 without a
re-scoring step.

## Hybrid by reciprocal rank fusion, untuned

Fusion uses RRF with k = 60, the constant from Cormack et al. (2009), on the document rankings of
BM25 and dense retrieval. It needs no score normalisation, which matters because BM25 scores are
unbounded and cosine scores are not. A weighted fusion could be tuned on the tuning splits, but RRF
is the standard untuned baseline, and its failure on FiQA, where BM25 is much weaker than dense, is
itself the finding: equal-weight fusion assumes two comparably good rankers.

## Embeddings cached by content

Encoding all corpora takes about four minutes on the GPU. Embeddings are memoised in `data/cache`
under a SHA-256 of the texts and every encoder setting (model revision, length, precision), and the
cache stores the original encode time, so reported index times stay honest on cached reruns.

## README tables are generated

`make report` renders every results table from the CSVs in `reports/` into the README between
marker comments, so a number in the README cannot drift from the run that produced it.

## First stage and reranker are chosen on the tuning splits

The "best first-stage retriever" and the reranker were picked by `scripts/select_models.py` on
SciFact train, NFCorpus dev and FiQA dev only, by mean nDCG@10, from candidates fixed in advance
(`reports/selection.csv`). Dense retrieval wins the first stage (0.515 vs hybrid 0.489 and BM25
0.413). Of three cross-encoders spanning model size, MS MARCO MiniLM-L6 (22M), bge-reranker-base
(278M) and bge-reranker-v2-m3 (568M), none beats dense retrieval alone on average: v2-m3 comes
closest (0.511) and is the one reported, helping FiQA but hurting SciFact and NFCorpus. Choosing
on the test splits would have been easier and would have overstated every reranking result.

## Cross-encoder and reader logits are kept in fp32

Running the models in fp16 doubles throughput, but fp16 logits are coarse: the first reranking run
produced only 3,864 distinct scores for 59,026 candidates, so ranks were decided by document-id
tie-breaking. The classification head now runs in fp32 on the fp16 encoder output (57,756 distinct
scores, same speed), and the reader casts start and end logits to fp32 before any span arithmetic.
On this data the tie fix happened not to change nDCG@10, but the ranking is no longer an artefact.

## Reader: an off-the-shelf SQuAD 2.0 model, validated before use

`deepset/roberta-base-squad2` is read with its training window (384 tokens, stride 128). Spans are
searched exhaustively within each window (up to 30 tokens) on the GPU, and each question keeps the
window whose best span beats its own no-answer score by the largest margin; the reader abstains
when that margin is below a threshold. On the full dev set with gold passages and threshold 0 it
reproduces the model card within 0.2 points on the official blended score
(`reports/reader_check.csv`), which validates the reading and scoring code before any retrieval is
involved.

## SQuAD end-to-end setup

The corpus is every distinct paragraph of SQuAD 2.0 train and dev (20,233), so the 1,204 dev
paragraphs sit among 19,029 distractors from other articles; paragraphs are indexed whole, since
they are already passage-sized and the gold answer offsets are defined on them. The abstention
threshold is the only fitted parameter. It is tuned per pipeline on 10 dev articles held out as a
whole, never on the evaluation articles: unanswerable questions are written against a specific
paragraph, so splitting by question would leak paragraph-level cues between tuning and evaluation.
The threshold maximises the official overall F1 on the tuning articles; the results report answerable
EM and F1 and abstention accuracy separately, and the full curve shows every other operating point.
The oracle row reads the gold paragraph, so the gap between it and a retrieval row is error caused by
retrieval rather than by the reader.

## The ablation's winner does not become the default

The chunking selected on SciFact's tuning split (sentence-packed, 128 tokens, 32 overlap) improves
SciFact test by 0.011 nDCG@10 and NFCorpus by 0.002, but loses 0.019 on FiQA. Adopting it globally
would trade one corpus against another on evidence from a single domain, so the a-priori default
(fixed 256 / 64) stays the pipeline's chunking and every other table uses it. A deployment on one
corpus should tune chunking on that corpus's own held-out queries; the ablation shows why.

## Reranking is reported although it does not win on average

The reranked BEIR pipeline is in the results with both sides of the trade: on
test it moves nDCG@10 by -0.005, -0.030 and +0.024 (SciFact, NFCorpus, FiQA) for 0.7 to 0.9 s of
added latency per query. Candidates are all chunks of the top 100 documents, scored in a single
fp16 batch per query; a smaller candidate list or a distilled reranker would cut the cost, but
neither was tuned, to keep the comparison with the selection run exact.

## Where BM25 beat dense retrieval, and what follows

On BEIR it never did: bge-base beats BM25 on all three subsets, at the default chunking by 0.06
nDCG@10 on SciFact and NFCorpus and 0.18 on FiQA, although older encoders often lose to BM25 on the
first two. On SQuAD it does, clearly: BM25 puts the gold paragraph in the top 5 for 86.6% of
evaluation questions and dense retrieval for 77.3%, and answerable F1 follows (0.625 vs 0.553).
SQuAD questions were written by annotators looking at the paragraph and reuse its words, which is
exactly what exact term matching rewards; BEIR's queries were written independently of the
documents.

The conclusion is that "the best retriever" is a property of a corpus and its queries, not of the
retriever. The BEIR selection (dense) is therefore not carried over to SQuAD; the service retrieves
with BM25 because SQuAD's own tuning articles say so, and both rows stay in every table.

## Reranking on SQuAD: better retrieval, worse answers

bge-reranker-v2-m3 re-scores the top 20 paragraphs of each first stage (20 rather than BEIR's 100:
the reader reads 5, and 100 would cost about 0.5 s per question). It does what it should for
retrieval: the gold paragraph reaches BM25's top 5 for 91.8% of questions instead of 86.6%, and
dense retrieval's for 86.7% instead of 77.3%. End to end the picture splits. After dense retrieval
it lifts answerable F1 from 0.553 to 0.601; after BM25 it lowers it from 0.625 to 0.613, and on the
tuning articles too (official F1 0.766 vs 0.771). Where the gold paragraph is among the five, the
reader scores 0.654 F1 after reranking against 0.705 without, and the tuned threshold rises from 4.3
to 5.2: a likely reading is that the reranked five are closer to the question as a group, so the
reader is distracted more often and needs a stricter threshold. Either way, a better retrieval
metric did not become a better answer, and at about 130 ms per question (five times the rest of the
pipeline) the reranker is not served.

## The served pipeline is chosen on the tuning articles

The service answers with one of the evaluated end-to-end rows, picked by its official F1 on the 10
held-out tuning articles (`tune_blended_f1` in `reports/e2e_results.csv`): BM25, top 5 paragraphs,
no reranker. The evaluation articles played no part. Its abstention threshold is read from the same
file at start-up, so the service and the reported numbers cannot disagree about it.

## Abstention threshold

The reader abstains when its best span beats the no-answer score by less than a threshold. That
threshold is fitted per pipeline on the 10 tuning articles (3,424 questions, 1,668 answerable),
over 101 quantiles of the tuning margins, maximising SQuAD's official overall F1, which counts a
correct abstention as an exact match and so weighs answering and abstaining by how often each is
right. The fitted values (1.9 for the oracle passage, 4.3 to 5.3 after retrieval) sit well above
the default of 0: with five retrieved paragraphs there are five chances for a spurious span, so the
reader must be more confident before it answers. Answerable F1 and abstention accuracy are always
reported separately, with the full curve, because a different application would pick a different
point on it.

## Service: one process, one GPU pipeline, one request at a time

`serve.py` loads the corpus, index and models once at start-up and answers `POST /query` under a
lock, since the models share one GPU and batching across requests would complicate latency without
being needed at this scale. The response carries the answer (or `null` with `abstained: true`), the
reader's margin, the five paragraphs read with their retrieval (and, if enabled, rerank) scores,
the index of the paragraph the answer came from, and per-stage timings. No network access happens
after start-up; the Docker image sets `HF_HUB_OFFLINE` to make that explicit. Corpora and weights
(about 4 GB) are mounted into the container rather than baked into the image.

## Latency is measured through the service

`scripts/benchmark_latency.py` sends 20 warm-up and then 300 timed questions, one at a time,
through FastAPI's in-process test client: end to end therefore includes request validation and JSON
encoding but no network hop. Stage times come from the pipeline itself; every stage ends by copying
its result to the host, so GPU work is finished when its timer stops. The reranked variant is timed
too, so its cost is on record even though it is not served.
