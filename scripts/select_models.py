import argparse
import gc

import pandas as pd
import torch

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import evaluate_run
from knowledgeqa.ingest import load_corpus, load_queries, tune_qrels
from knowledgeqa.pipeline import bm25_stage, dense_stage, hybrid_stage
from knowledgeqa.rerank import candidate_pairs, chunks_by_doc, load_reranker, rerank_run
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the first stage and the reranker on the tuning splits only."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "selection")
    start_tracking(config)
    evaluation, chunking = config.evaluation, config.chunking.default
    ndcg = f"ndcg_at_{evaluation.ndcg_k}"
    tokenizer, encoder = load_tokenizer(config), load_encoder(config, device)

    rows, first_stage_runs, corpora = [], {}, {}
    for name in config.data.beir:
        split = config.data.beir[name].tune_split
        docs, queries = load_corpus(config, name), load_queries(config, name)
        qrels = tune_qrels(config, name)
        query_ids = sorted(qrels)
        texts = [queries[query_id] for query_id in query_ids]
        chunks = chunk_corpus(
            docs, corpus_offsets(config, name, docs, tokenizer), chunking
        )
        cache = config.paths.cache / f"{name}-{chunking.name}-dense.npz"
        bm25 = bm25_stage(config, device, chunks, query_ids, texts)
        semantic = dense_stage(config, encoder, chunks, query_ids, texts, cache)
        stages = {
            "bm25": bm25,
            "dense": semantic,
            "hybrid": hybrid_stage(config, bm25, semantic),
        }
        for stage, result in stages.items():
            metrics = evaluate_run(
                result.run,
                qrels,
                evaluation.ndcg_k,
                evaluation.recall_k,
                evaluation.mrr_k,
            )
            params = {
                "corpus": name,
                "split": split,
                "first_stage": stage,
                "reranker": "none",
            }
            rows.append(params | metrics)
            log_result(f"selection-{stage}-{name}", params, metrics, run_dir)
        first_stage_runs[name] = {stage: result.run for stage, result in stages.items()}
        corpora[name] = (queries, qrels, query_ids, chunks)
    del encoder

    frame = pd.DataFrame(rows)
    best_stage = frame.groupby("first_stage")[ndcg].mean().idxmax()
    for reranker_name in config.rerank.candidates:
        reranker = load_reranker(config, reranker_name, device)
        for name, (queries, qrels, query_ids, chunks) in corpora.items():
            pairs = candidate_pairs(
                first_stage_runs[name][best_stage],
                query_ids,
                chunks_by_doc(chunks),
                config.rerank.top_k,
            )
            run = rerank_run(
                reranker, query_ids, pairs, queries, chunks, config.rerank.batch_size
            )
            metrics = evaluate_run(
                run, qrels, evaluation.ndcg_k, evaluation.recall_k, evaluation.mrr_k
            )
            split = config.data.beir[name].tune_split
            params = {
                "corpus": name,
                "split": split,
                "first_stage": best_stage,
                "reranker": reranker_name,
            }
            rows.append(params | metrics)
            tag = f"{best_stage}+{reranker_name}"
            log_result(f"selection-{tag}-{name}", params, metrics, run_dir)
            print(
                f"{name:9s} {split:6s} {best_stage} + {reranker_name}: {ndcg}={metrics[ndcg]:.4f}"
            )
        # the caching allocator would otherwise keep each model's memory for the next one
        del reranker
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).round(4)
    means = frame.groupby(["first_stage", "reranker"], sort=False)[ndcg].mean().round(4)
    print(frame.to_string(index=False))
    print(f"\nmean {ndcg} over the tuning splits:\n{means.to_string()}")
    frame.to_csv(config.paths.reports / "selection.csv", index=False)
    frame.to_csv(run_dir / "selection.csv", index=False)
    print(f"wrote reports/selection.csv and {run_dir}")


if __name__ == "__main__":
    main()
