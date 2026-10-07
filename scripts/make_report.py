import argparse

import pandas as pd

from knowledgeqa.config import load_config


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect every reports/retrieval_*.csv into reports/results.csv."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    reports = config.paths.reports

    parts = sorted(reports.glob("retrieval_*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"no retrieval_*.csv in {reports}; run `make eval` first"
        )
    results = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    results.to_csv(reports / "results.csv", index=False)

    evaluation = config.evaluation
    columns = ["retriever", "chunking", "corpus", "documents", "chunks"]
    columns += [
        f"ndcg_at_{evaluation.ndcg_k}",
        f"recall_at_{evaluation.recall_k}",
        f"mrr_at_{evaluation.mrr_k}",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    table = results[columns].round(3)
    print(markdown(table))
    print(f"\nwrote {reports / 'results.csv'} from {len(parts)} file(s)")


if __name__ == "__main__":
    main()
