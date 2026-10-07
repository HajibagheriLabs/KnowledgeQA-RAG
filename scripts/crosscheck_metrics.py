import argparse
from pathlib import Path

import pytrec_eval

from knowledgeqa.config import load_config
from knowledgeqa.evaluate import evaluate_run, read_trec_run
from knowledgeqa.ingest import eval_qrels

TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute nDCG and recall of saved runs with pytrec_eval, "
        "the trec_eval binding BEIR reports with, and compare with ours."
    )
    parser.add_argument(
        "run_dir", type=Path, help="a runs/<timestamp>-<name> directory"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ndcg_k, recall_k = config.evaluation.ndcg_k, config.evaluation.recall_k

    run_files = sorted(args.run_dir.glob("*.trec"))
    if not run_files:
        raise FileNotFoundError(f"no .trec run files in {args.run_dir}")
    worst = 0.0
    for path in run_files:
        qrels = eval_qrels(config, path.name.split(".")[0])
        run = read_trec_run(path)
        # a query that matched nothing has no lines in a trec file but still counts as zero
        run |= {query_id: {} for query_id in qrels.keys() - run.keys()}
        ours = evaluate_run(run, qrels, ndcg_k, recall_k, config.evaluation.mrr_k)
        measures = {f"ndcg_cut.{ndcg_k}", f"recall.{recall_k}"}
        per_query = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
        reference = {
            f"ndcg_at_{ndcg_k}": sum(
                q[f"ndcg_cut_{ndcg_k}"] for q in per_query.values()
            ),
            f"recall_at_{recall_k}": sum(
                q[f"recall_{recall_k}"] for q in per_query.values()
            ),
        }
        for metric, total in reference.items():
            difference = abs(total / len(qrels) - ours[metric])
            worst = max(worst, difference)
            print(
                f"{path.name:32s} {metric:14s} ours {ours[metric]:.6f}  diff {difference:.1e}"
            )
    if worst > TOLERANCE:
        raise SystemExit(f"metrics disagree with pytrec_eval by up to {worst:.1e}")
    print(f"all metrics agree with pytrec_eval (max difference {worst:.1e})")


if __name__ == "__main__":
    main()
