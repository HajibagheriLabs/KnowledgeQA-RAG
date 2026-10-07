import argparse

import pandas as pd
from matplotlib.figure import Figure

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import ChunkingConfig, Config, load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import evaluate_run
from knowledgeqa.ingest import eval_qrels, load_corpus, load_queries, tune_qrels
from knowledgeqa.pipeline import bm25_stage, dense_stage, hybrid_stage
from knowledgeqa.rerank import candidate_pairs, chunks_by_doc, load_reranker, rerank_run
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking

# reference categorical slots 1-2 for the strategy; overlap is carried by the line style
COLORS = {"fixed": "#2a78d6", "sentence": "#eb6834"}
STYLES = {0: "-", 1: "--"}


def plot(frame: pd.DataFrame, ndcg: str, path) -> None:
    panels = [("tune", "scifact")] + [("test", c) for c in frame["corpus"].unique()]
    fig = Figure(figsize=(12, 3.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    axes = fig.subplots(1, len(panels), sharex=True)
    for ax, (split, corpus) in zip(axes, panels, strict=True):
        data = frame[(frame["split"] == split) & (frame["corpus"] == corpus)]
        ax.set_facecolor("#fcfcfb")
        whole = data[data["strategy"] == "none"][ndcg]
        if len(whole):
            ax.axhline(whole.iloc[0], color="#8a8984", linewidth=1, linestyle=":")
        chunked = data[data["strategy"] != "none"]
        for (strategy, overlap_rank), group in chunked.groupby(
            ["strategy", "overlap_rank"]
        ):
            group = group.sort_values("chunk_size")
            share = "0%" if overlap_rank == 0 else "25%"
            ax.plot(
                group["chunk_size"],
                group[ndcg],
                color=COLORS[strategy],
                linestyle=STYLES[overlap_rank],
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"{strategy}, {share} overlap",
            )
        chosen = data[data["selected"]]
        ax.scatter(
            chosen["chunk_size"],
            chosen[ndcg],
            s=140,
            facecolors="none",
            edgecolors="#0b0b0b",
            linewidths=1.2,
            zorder=5,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512], labels=["128", "256", "512"])
        ax.set_title(f"{corpus} ({split})", loc="left", fontsize=9)
        ax.set_xlabel("chunk size, tokens", fontsize=8, color="#52514e")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.tick_params(labelsize=7, colors="#52514e")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(ndcg.replace("ndcg_at_", "nDCG@"), fontsize=8, color="#52514e")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        "Chunking ablation: chosen on SciFact's tuning split (ringed), others untouched;"
        " dotted line = whole documents; each panel has its own y-scale",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, facecolor=fig.get_facecolor())


def first_stage(config: Config, device, encoder, chunks, query_ids, texts, cache):
    stage = config.rerank.first_stage
    if stage == "bm25":
        return bm25_stage(config, device, chunks, query_ids, texts)
    semantic = dense_stage(config, encoder, chunks, query_ids, texts, cache)
    if stage == "dense":
        return semantic
    return hybrid_stage(
        config, bm25_stage(config, device, chunks, query_ids, texts), semantic
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk size, overlap and strategy ablation on the best pipeline."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "ablation")
    start_tracking(config)
    ablation, evaluation = config.chunking.ablation, config.evaluation
    ndcg = f"ndcg_at_{evaluation.ndcg_k}"
    cut_offs = (evaluation.ndcg_k, evaluation.recall_k, evaluation.mrr_k)
    tokenizer, encoder = load_tokenizer(config), load_encoder(config, device)
    reranker = (
        load_reranker(config, config.rerank.model, device) if ablation.rerank else None
    )
    pipeline = config.rerank.first_stage + ("+rerank" if ablation.rerank else "")
    grid: list[ChunkingConfig] = config.chunking.ablation_configs()

    rows = []
    for name in config.data.beir:
        docs, queries = load_corpus(config, name), load_queries(config, name)
        offsets = corpus_offsets(config, name, docs, tokenizer)
        splits = {"test": eval_qrels(config, name)}
        if name == ablation.tune_corpus:
            splits["tune"] = tune_qrels(config, name)
        for chunking in grid:
            chunks = chunk_corpus(docs, offsets, chunking)
            cache = config.paths.cache / f"{name}-{chunking.name}-dense.npz"
            for split, qrels in splits.items():
                query_ids = sorted(qrels)
                texts = [queries[query_id] for query_id in query_ids]
                run = first_stage(
                    config, device, encoder, chunks, query_ids, texts, cache
                ).run
                if reranker is not None:
                    pairs = candidate_pairs(
                        run, query_ids, chunks_by_doc(chunks), config.rerank.top_k
                    )
                    run = rerank_run(
                        reranker,
                        query_ids,
                        pairs,
                        queries,
                        chunks,
                        config.rerank.batch_size,
                    )
                metrics = evaluate_run(run, qrels, *cut_offs)
                params = {
                    "pipeline": pipeline,
                    "corpus": name,
                    "split": split,
                    "chunking": chunking.name,
                    "strategy": chunking.strategy,
                    "chunk_size": chunking.chunk_size,
                    "overlap": chunking.overlap,
                    "chunks": len(chunks),
                    "queries": len(query_ids),
                }
                rows.append(params | metrics)
                log_result(
                    f"ablation-{name}-{split}-{chunking.name}", params, metrics, run_dir
                )
                print(
                    f"{name:9s} {split:5s} {chunking.name:17s} {ndcg}={metrics[ndcg]:.4f}"
                )

    frame = pd.DataFrame(rows)
    tune = frame[(frame["corpus"] == ablation.tune_corpus) & (frame["split"] == "tune")]
    chosen = tune.loc[tune[ndcg].idxmax(), "chunking"]
    frame["selected"] = frame["chunking"] == chosen
    frame["default"] = frame["chunking"] == config.chunking.default.name
    frame["overlap_rank"] = (frame["overlap"] > 0).astype(int)
    frame.round(4).drop(columns="overlap_rank").to_csv(
        config.paths.reports / "ablations.csv", index=False
    )
    frame.round(4).to_csv(run_dir / "ablations.csv", index=False)
    plot(frame, ndcg, config.paths.reports / "ablations.png")
    print(f"selected on {ablation.tune_corpus} tuning split: {chosen}")
    print(f"wrote reports/ablations.csv, reports/ablations.png and {run_dir}")


if __name__ == "__main__":
    main()
