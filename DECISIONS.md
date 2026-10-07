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
