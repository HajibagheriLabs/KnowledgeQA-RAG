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
