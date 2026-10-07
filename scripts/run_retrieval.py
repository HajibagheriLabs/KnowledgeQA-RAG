import argparse

import pandas as pd

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import ChunkingConfig, load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import evaluate_run, latency_ms, write_trec_run
from knowledgeqa.ingest import check_qrels, eval_qrels, load_corpus, load_queries
from knowledgeqa.pipeline import bm25_stage, dense_stage, hybrid_stage
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking

RETRIEVERS = ("bm25", "dense", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25, dense and hybrid retrieval on the BEIR test splits."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "retrieval")
    start_tracking(config)
    tokenizer = load_tokenizer(config)
    encoder = load_encoder(config, device)
    evaluation = config.evaluation
    # whole documents are what published BEIR numbers index; the default chunking is ours
    whole = ChunkingConfig("none", None, 0, config.chunking.default.prepend_title)

    rows: dict[str, list[dict]] = {retriever: [] for retriever in RETRIEVERS}
    for name in config.data.beir:
        docs = load_corpus(config, name)
        queries = load_queries(config, name)
        qrels = eval_qrels(config, name)
        check_qrels(qrels, [doc.doc_id for doc in docs], queries)
        query_ids = sorted(qrels)
        texts = [queries[query_id] for query_id in query_ids]
        offsets = corpus_offsets(config, name, docs, tokenizer)
        for chunking in (whole, config.chunking.default):
            chunks = chunk_corpus(docs, offsets, chunking)
            cache = config.paths.cache / f"{name}-{chunking.name}-dense.npz"
            bm25 = bm25_stage(config, device, chunks, query_ids, texts)
            semantic = dense_stage(config, encoder, chunks, query_ids, texts, cache)
            stages = {
                "bm25": bm25,
                "dense": semantic,
                "hybrid": hybrid_stage(config, bm25, semantic),
            }
            for retriever, stage in stages.items():
                tag = f"{retriever}-{chunking.name}"
                write_trec_run(
                    run_dir / f"{name}.{tag}.trec", stage.run, evaluation.recall_k, tag
                )
                metrics = evaluate_run(
                    stage.run,
                    qrels,
                    evaluation.ndcg_k,
                    evaluation.recall_k,
                    evaluation.mrr_k,
                )
                metrics |= latency_ms(
                    stage.answer, texts, config.retrieval.latency_warmup
                )
                metrics["index_seconds"] = stage.index_seconds
                params = {
                    "retriever": retriever,
                    "corpus": name,
                    "split": config.data.eval_split,
                    "chunking": chunking.name,
                    "queries": len(query_ids),
                    "documents": len(docs),
                    "chunks": len(chunks),
                    "device": device.type,
                    "seed": config.seed,
                }
                rows[retriever].append(params | metrics)
                log_result(f"{tag}-{name}", params, metrics, run_dir)
                scores = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"{name:9s} {tag:20s} {scores}")

    config.paths.reports.mkdir(parents=True, exist_ok=True)
    for retriever, table in rows.items():
        frame = pd.DataFrame(table).round(4)
        frame.to_csv(config.paths.reports / f"retrieval_{retriever}.csv", index=False)
        frame.to_csv(run_dir / f"retrieval_{retriever}.csv", index=False)
    print(f"wrote reports/retrieval_*.csv and {run_dir}")


if __name__ == "__main__":
    main()
