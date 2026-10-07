import argparse
import re
from pathlib import Path

import pandas as pd

from knowledgeqa.config import load_config

CORPORA = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}
RETRIEVERS = {"bm25": "BM25", "dense": "Dense (bge-base)", "hybrid": "Hybrid (RRF)"}


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
    args.readme.write_text(readme)
    print(
        f"wrote {reports / 'results.csv'} from {len(parts)} file(s) and refreshed {args.readme}"
    )


if __name__ == "__main__":
    main()
