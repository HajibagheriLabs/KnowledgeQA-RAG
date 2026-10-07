import argparse
import time

import mlflow
import pandas as pd

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import ChunkingConfig, Config, load_config
from knowledgeqa.evaluate import evaluate_run, latency_ms, max_pool, write_trec_run
from knowledgeqa.ingest import check_qrels, eval_qrels, load_corpus, load_queries
from knowledgeqa.runtime import setup, start_run
from knowledgeqa.sparse import build_index, search


def start_tracking(config: Config) -> None:
    config.paths.mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.paths.mlruns / 'mlflow.db'}")
    mlflow.set_experiment(config.tracking.experiment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 baseline on the BEIR test splits."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "bm25")
    start_tracking(config)
    tokenizer = load_tokenizer(config)
    retrieval, evaluation = config.retrieval, config.evaluation
    # whole documents are what published BEIR numbers index; the default chunking is ours
    whole = ChunkingConfig("none", None, 0, config.chunking.default.prepend_title)
    chunkings = [whole, config.chunking.default]

    rows = []
    for name in config.data.beir:
        docs = load_corpus(config, name)
        queries = load_queries(config, name)
        qrels = eval_qrels(config, name)
        check_qrels(qrels, [doc.doc_id for doc in docs], queries)
        query_ids = sorted(qrels)
        texts = [queries[query_id] for query_id in query_ids]
        offsets = corpus_offsets(config, name, docs, tokenizer)
        for chunking in chunkings:
            chunks = chunk_corpus(docs, offsets, chunking)
            chunk_doc_ids = [chunk.doc_id for chunk in chunks]
            started = time.perf_counter()
            index = build_index([chunk.text for chunk in chunks], config.bm25, device)
            index_seconds = time.perf_counter() - started

            hits, scores = search(
                index, texts, retrieval.candidate_chunks, retrieval.query_batch_size
            )
            run = max_pool(query_ids, hits, scores, chunk_doc_ids, evaluation.recall_k)
            run_file = run_dir / f"{name}.{chunking.name}.trec"
            write_trec_run(run_file, run, evaluation.recall_k, f"bm25-{chunking.name}")
            metrics = evaluate_run(
                run, qrels, evaluation.ndcg_k, evaluation.recall_k, evaluation.mrr_k
            )

            def answer(query: str, index=index, chunk_doc_ids=chunk_doc_ids) -> None:
                one = search(index, [query], retrieval.candidate_chunks, 1)
                max_pool([query], *one, chunk_doc_ids, evaluation.recall_k)

            metrics |= latency_ms(answer, texts, retrieval.latency_warmup)
            metrics["index_seconds"] = index_seconds
            params = {
                "retriever": "bm25",
                "corpus": name,
                "split": config.data.eval_split,
                "chunking": chunking.name,
                "queries": len(query_ids),
                "documents": len(docs),
                "chunks": len(chunks),
                "k1": config.bm25.k1,
                "b": config.bm25.b,
                "device": device.type,
                "seed": config.seed,
            }
            rows.append(params | metrics)
            with mlflow.start_run(run_name=f"bm25-{name}-{chunking.name}"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(run_dir / "config.yaml"))
            print(
                f"{name:9s} {chunking.name:13s} "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

    frame = pd.DataFrame(rows).round(4)
    config.paths.reports.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.paths.reports / "retrieval_bm25.csv", index=False)
    frame.to_csv(run_dir / "retrieval_bm25.csv", index=False)
    print(f"wrote {config.paths.reports / 'retrieval_bm25.csv'} and {run_dir}")


if __name__ == "__main__":
    main()
