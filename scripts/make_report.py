import argparse
import re
from pathlib import Path

import pandas as pd

from knowledgeqa.config import load_config

CORPORA = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}
RETRIEVERS = {
    "bm25": "BM25",
    "dense": "Dense (bge-base)",
    "hybrid": "Hybrid (RRF)",
    "dense+rerank": "Dense + rerank (bge-v2-m3)",
}


def chunking_label(name: str) -> str:
    if name == "none":
        return "whole documents"
    strategy, size, overlap = name.split("-")
    return f"{strategy} {size} / {overlap}"


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def summary_table(results: pd.DataFrame, ndcg: str) -> pd.DataFrame:
    table = results.pivot_table(
        index=["retriever", "chunking"], columns="corpus", values=ndcg, sort=False
    ).reset_index()
    return pd.DataFrame(
        {
            "Retriever": table["retriever"].map(RETRIEVERS),
            "Chunking": table["chunking"].map(chunking_label),
            **{
                f"{label} nDCG@10": table[c].map("{:.3f}".format)
                for c, label in CORPORA.items()
            },
        }
    )


def retrieval_table(
    results: pd.DataFrame, chunking: str, metrics: list[str]
) -> pd.DataFrame:
    rows = results[results["chunking"] == chunking]
    return pd.DataFrame(
        {
            "Retriever": rows["retriever"].map(RETRIEVERS),
            "Corpus": rows["corpus"].map(CORPORA),
            "Documents": rows["documents"].map("{:,}".format),
            "Chunks": rows["chunks"].map("{:,}".format),
            "nDCG@10": rows[metrics[0]].map("{:.3f}".format),
            "Recall@100": rows[metrics[1]].map("{:.3f}".format),
            "MRR@10": rows[metrics[2]].map("{:.3f}".format),
            "p50 ms": rows["latency_p50_ms"].map("{:.1f}".format),
            "p95 ms": rows["latency_p95_ms"].map("{:.1f}".format),
        }
    )


def index_table(study: pd.DataFrame, ndcg: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Index": study["family"],
            "Setting": study["setting"],
            "Recall vs exact": study["recall_vs_exact"].map("{:.3f}".format),
            "nDCG@10": study[ndcg].map("{:.3f}".format),
            "p50 ms": study["latency_p50_ms"].map("{:.2f}".format),
            "p95 ms": study["latency_p95_ms"].map("{:.2f}".format),
            "Build s": study["build_seconds"].map("{:.1f}".format),
            "Memory MB": study["memory_mb"].map("{:.1f}".format),
        }
    )


def rerank_table(rerank: pd.DataFrame, ndcg: str, mrr: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Corpus": rerank["corpus"].map(CORPORA),
            "Pairs per query": rerank["pairs_per_query"].map("{:.0f}".format),
            "nDCG@10 before": rerank[f"first_stage_{ndcg}"].map("{:.3f}".format),
            "nDCG@10 after": rerank[ndcg].map("{:.3f}".format),
            "Change": (rerank[ndcg] - rerank[f"first_stage_{ndcg}"]).map(
                "{:+.3f}".format
            ),
            "MRR@10 before": rerank[f"first_stage_{mrr}"].map("{:.3f}".format),
            "MRR@10 after": rerank[mrr].map("{:.3f}".format),
            "Retrieval p50 / p95 ms": rerank["first_stage_latency_p50_ms"].map(
                "{:.0f}".format
            )
            + " / "
            + rerank["first_stage_latency_p95_ms"].map("{:.0f}".format),
            "Rerank p50 / p95 ms": rerank["rerank_latency_p50_ms"].map("{:.0f}".format)
            + " / "
            + rerank["rerank_latency_p95_ms"].map("{:.0f}".format),
            "End to end p50 / p95 ms": rerank["latency_p50_ms"].map("{:.0f}".format)
            + " / "
            + rerank["latency_p95_ms"].map("{:.0f}".format),
        }
    )


def ablation_table(ablations: pd.DataFrame, ndcg: str) -> pd.DataFrame:
    columns = {("scifact", "tune"): "SciFact train (tuning)"}
    columns |= {(c, "test"): f"{label} test" for c, label in CORPORA.items()}
    table = ablations.pivot_table(
        index="chunking", columns=["corpus", "split"], values=ndcg, sort=False
    )
    flags = ablations.drop_duplicates("chunking").set_index("chunking")
    names = [
        chunking_label(c)
        + (" (selected)" if flags.loc[c, "selected"] else "")
        + (" (default)" if flags.loc[c, "default"] else "")
        for c in table.index
    ]
    out = pd.DataFrame({"Chunking": names})
    for key, label in columns.items():
        out[label] = table[key].map("{:.3f}".format).to_list()
    return out


def e2e_table(e2e: pd.DataFrame, recall: str) -> pd.DataFrame:
    def pair(prefix: str) -> pd.Series:
        return [
            "-"
            if pd.isna(row[f"{prefix}_p50_ms"])
            else f"{row[f'{prefix}_p50_ms']:.0f} / {row[f'{prefix}_p95_ms']:.0f}"
            for _, row in e2e.iterrows()
        ]

    labels = {
        "oracle": "Oracle passage",
        "bm25": "BM25",
        "dense": "Dense (bge-base)",
        "bm25+rerank": "BM25 + rerank (bge-v2-m3)",
        "dense+rerank": "Dense + rerank (bge-v2-m3)",
    }
    return pd.DataFrame(
        {
            "Retrieval": e2e["row"].map(labels),
            f"Gold passage in top {recall.rsplit('_', 1)[1]}": e2e[recall].map(
                lambda v: "-" if pd.isna(v) else f"{v:.3f}"
            ),
            "Answerable EM": e2e["answerable_em"].map("{:.3f}".format),
            "Answerable F1": e2e["answerable_f1"].map("{:.3f}".format),
            "Abstention accuracy": e2e["abstention_accuracy"].map("{:.3f}".format),
            "Tuning-article F1": e2e["tune_blended_f1"].map("{:.3f}".format),
            "Retrieval p50 / p95 ms": pair("retrieval"),
            "Rerank p50 / p95 ms": pair("rerank"),
            "Reader p50 / p95 ms": pair("reader"),
            "End to end p50 / p95 ms": pair("end_to_end"),
        }
    )


def latency_table(latency: pd.DataFrame) -> pd.DataFrame:
    stages = {
        "retrieval": "Retrieval",
        "rerank": "Reranking",
        "reader": "Reading",
        "end_to_end": "End to end (HTTP request)",
    }
    return pd.DataFrame(
        {
            "Stage": latency["stage"].map(stages),
            "p50 ms": latency["p50_ms"].map("{:.1f}".format),
            "p95 ms": latency["p95_ms"].map("{:.1f}".format),
            "Mean ms": latency["mean_ms"].map("{:.1f}".format),
            "Queries": latency["queries"],
        }
    )


def fill(readme: str, name: str, table: pd.DataFrame) -> str:
    """Replace the block between <!-- table:name --> markers with a fresh table."""
    pattern = re.compile(
        rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", re.DOTALL
    )
    if not pattern.search(readme):
        raise ValueError(f"README has no table:{name} block")
    return pattern.sub(
        lambda m: m.group(1) + markdown(table) + "\n" + m.group(2), readme
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect results into reports/results.csv and refresh the README tables."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    config = load_config(args.config)
    reports, evaluation = config.paths.reports, config.evaluation

    parts = sorted(reports.glob("retrieval_*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"no retrieval_*.csv in {reports}; run `make eval` first"
        )
    results = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    order = {retriever: i for i, retriever in enumerate(RETRIEVERS)}
    results = results.sort_values(
        "retriever", key=lambda column: column.map(order), kind="stable"
    )
    results.to_csv(reports / "results.csv", index=False)

    metrics = [
        f"ndcg_at_{evaluation.ndcg_k}",
        f"recall_at_{evaluation.recall_k}",
        f"mrr_at_{evaluation.mrr_k}",
    ]
    readme = args.readme.read_text()
    readme = fill(readme, "summary", summary_table(results, metrics[0]))
    default = config.chunking.default.name
    readme = fill(readme, "retrieval", retrieval_table(results, default, metrics))
    readme = fill(readme, "retrieval-whole", retrieval_table(results, "none", metrics))
    study = reports / "index_study.csv"
    if study.is_file():
        readme = fill(readme, "index", index_table(pd.read_csv(study), metrics[0]))
    rerank = reports / "retrieval_rerank.csv"
    if rerank.is_file():
        frame = pd.read_csv(rerank)
        readme = fill(readme, "rerank", rerank_table(frame, metrics[0], metrics[2]))
    ablations = reports / "ablations.csv"
    if ablations.is_file():
        readme = fill(
            readme, "ablation", ablation_table(pd.read_csv(ablations), metrics[0])
        )
    e2e = reports / "e2e_results.csv"
    if e2e.is_file():
        recall = f"retrieval_recall_at_{config.e2e.top_k}"
        readme = fill(readme, "e2e", e2e_table(pd.read_csv(e2e), recall))
    latency = reports / "latency.csv"
    if latency.is_file():
        readme = fill(readme, "latency", latency_table(pd.read_csv(latency)))
    args.readme.write_text(readme)
    print(
        f"wrote {reports / 'results.csv'} from {len(parts)} file(s) and refreshed {args.readme}"
    )


if __name__ == "__main__":
    main()
