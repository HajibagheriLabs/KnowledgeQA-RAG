import argparse

import numpy as np
import pandas as pd

from knowledgeqa.chunking import (
    chunk_corpus,
    corpus_offsets,
    load_tokenizer,
    token_offsets,
)
from knowledgeqa.config import load_config
from knowledgeqa.ingest import load_corpus
from knowledgeqa.runtime import setup


def distribution(prefix: str, values: np.ndarray) -> dict[str, float]:
    stats = {"mean": values.mean(), "p50": np.percentile(values, 50)}
    stats |= {"p90": np.percentile(values, 90), "p99": np.percentile(values, 99)}
    stats |= {"max": values.max()}
    return {f"{prefix}_{name}": float(value) for name, value in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Document counts, token lengths and chunk counts per chunking configuration."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    setup(config)
    tokenizer = load_tokenizer(config)
    batch_size = config.chunking.batch_size

    rows = []
    for name in config.data.beir:
        docs = load_corpus(config, name)
        offsets = corpus_offsets(config, name, docs, tokenizer)
        titles = token_offsets([doc.title for doc in docs], tokenizer, batch_size)
        title_tokens = np.array([len(o) for o in titles])
        doc_tokens = title_tokens + np.array([len(o) for o in offsets])
        corpus = {
            "corpus": name,
            "documents": len(docs),
            "empty_documents": sum(not doc.title and not doc.text for doc in docs),
            **distribution("doc_tokens", doc_tokens),
        }
        for chunking in config.chunking.ablation_configs():
            chunks = chunk_corpus(docs, offsets, chunking)
            parents = np.array([chunk.doc_index for chunk in chunks])
            per_doc = np.bincount(parents, minlength=len(docs))
            chunk_tokens = np.array([chunk.n_tokens for chunk in chunks])
            if chunking.prepend_title:
                chunk_tokens = chunk_tokens + title_tokens[parents]
            rows.append(
                corpus
                | {
                    "chunking": chunking.name,
                    "strategy": chunking.strategy,
                    "chunk_size": chunking.chunk_size,
                    "overlap": chunking.overlap,
                    "chunks": len(chunks),
                    "chunks_per_doc_mean": float(per_doc.mean()),
                    "chunks_per_doc_max": int(per_doc.max()),
                    "docs_split": float((per_doc > 1).mean()),
                    **distribution("chunk_tokens", chunk_tokens),
                }
            )

    frame = pd.DataFrame(rows).round(3)
    config.paths.reports.mkdir(parents=True, exist_ok=True)
    out = config.paths.reports / "corpus_stats.csv"
    frame.to_csv(out, index=False)
    print(
        frame.drop_duplicates("corpus").filter(regex="^(corpus|documents|empty|doc_)")
    )
    print(frame[["corpus", "chunking", "chunks", "chunks_per_doc_mean", "docs_split"]])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
