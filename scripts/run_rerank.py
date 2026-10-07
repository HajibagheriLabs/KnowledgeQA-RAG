import argparse

import numpy as np
import pandas as pd

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import evaluate_run, latency_ms, write_trec_run
from knowledgeqa.ingest import check_qrels, eval_qrels, load_corpus, load_queries
from knowledgeqa.pipeline import bm25_stage, dense_stage, hybrid_stage
from knowledgeqa.rerank import candidate_pairs, chunks_by_doc, load_reranker, rerank_run
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-encoder reranking of the first stage's top documents."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "rerank")
    start_tracking(config)
    settings, evaluation, chunking = (
        config.rerank,
        config.evaluation,
        config.chunking.default,
    )
    tokenizer, encoder = load_tokenizer(config), load_encoder(config, device)
    reranker = load_reranker(config, settings.model, device)
    retriever = f"{settings.first_stage}+rerank"

    rows = []
    for name in config.data.beir:
        docs, queries = load_corpus(config, name), load_queries(config, name)
        qrels = eval_qrels(config, name)
        check_qrels(qrels, [doc.doc_id for doc in docs], queries)
        query_ids = sorted(qrels)
        texts = [queries[query_id] for query_id in query_ids]
        chunks = chunk_corpus(
            docs, corpus_offsets(config, name, docs, tokenizer), chunking
        )
        cache = config.paths.cache / f"{name}-{chunking.name}-dense.npz"
        if settings.first_stage == "bm25":
            first = bm25_stage(config, device, chunks, query_ids, texts)
        else:
            first = dense_stage(config, encoder, chunks, query_ids, texts, cache)
            if settings.first_stage == "hybrid":
                bm25 = bm25_stage(config, device, chunks, query_ids, texts)
                first = hybrid_stage(config, bm25, first)
        grouped = chunks_by_doc(chunks)
        pairs = candidate_pairs(first.run, query_ids, grouped, settings.top_k)
        run = rerank_run(
            reranker, query_ids, pairs, queries, chunks, settings.batch_size
        )
        tag = f"{retriever}-{chunking.name}"
        write_trec_run(run_dir / f"{name}.{tag}.trec", run, evaluation.recall_k, tag)
        cut_offs = (evaluation.ndcg_k, evaluation.recall_k, evaluation.mrr_k)
        metrics = evaluate_run(run, qrels, *cut_offs)
        before = evaluate_run(first.run, qrels, *cut_offs)

        def rerank_only(
            query_id: str, first=first, grouped=grouped, chunks=chunks, queries=queries
        ) -> None:
            one = candidate_pairs(first.run, [query_id], grouped, settings.top_k)
            rerank_run(reranker, [query_id], one, queries, chunks, settings.batch_size)

        def end_to_end(
            query_id: str, first=first, grouped=grouped, chunks=chunks, queries=queries
        ) -> None:
            ranked = {query_id: first.answer(queries[query_id])}
            one = candidate_pairs(ranked, [query_id], grouped, settings.top_k)
            rerank_run(reranker, [query_id], one, queries, chunks, settings.batch_size)

        warmup = config.retrieval.latency_warmup
        rng = np.random.default_rng(config.seed)
        count = min(settings.latency_queries, len(query_ids))
        sample = sorted(rng.choice(query_ids, count, replace=False).tolist())
        metrics |= latency_ms(end_to_end, sample, warmup)
        added = latency_ms(rerank_only, sample, warmup)
        stage = latency_ms(first.answer, [queries[q] for q in sample], warmup)
        metrics |= {f"rerank_{key}": value for key, value in added.items()}
        metrics |= {f"first_stage_{key}": value for key, value in stage.items()}
        metrics |= {f"first_stage_{key}": value for key, value in before.items()}
        metrics["pairs_per_query"] = len(pairs) / len(query_ids)
        params = {
            "retriever": retriever,
            "corpus": name,
            "split": config.data.eval_split,
            "chunking": chunking.name,
            "queries": len(query_ids),
            "documents": len(docs),
            "chunks": len(chunks),
            "reranker": settings.candidates[settings.model].repo,
            "top_k": settings.top_k,
            "device": device.type,
            "seed": config.seed,
        }
        rows.append(params | metrics)
        log_result(f"{tag}-{name}", params, metrics, run_dir)
        print(
            f"{name:9s} {tag} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    frame = pd.DataFrame(rows).round(4)
    frame.to_csv(config.paths.reports / "retrieval_rerank.csv", index=False)
    frame.to_csv(run_dir / "retrieval_rerank.csv", index=False)
    print(f"wrote reports/retrieval_rerank.csv and {run_dir}")


if __name__ == "__main__":
    main()
