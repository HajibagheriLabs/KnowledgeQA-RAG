import argparse
import time

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from knowledgeqa.config import load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import rank
from knowledgeqa.ingest import load_squad, paragraph_id, split_by_article
from knowledgeqa.pipeline import QAPipeline, bm25_search, dense_search, squad_corpus
from knowledgeqa.reader import Answer, load_reader, read, score_answers, token_f1
from knowledgeqa.rerank import candidate_pairs, chunks_by_doc, load_reranker, rerank_run
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking

# reference categorical slots 1-2, validated for colour-vision deficiency on a light surface
COLORS = {"oracle": "#2a78d6", "pipeline": "#eb6834"}


def blended(scores: dict[str, float], key: str) -> float:
    # squad's official overall score: an abstention on an unanswerable question counts as exact
    total = scores["answerable"] + scores["unanswerable"]
    unanswered = scores["abstention_accuracy"] * scores["unanswerable"]
    return (scores[key] * scores["answerable"] + unanswered) / total


def answerable_f1(answers, questions, threshold: float) -> float:
    if not questions:
        return float("nan")
    scores = []
    for q in questions:
        predicted = answers[q.qid].text if answers[q.qid].margin > threshold else ""
        scores.append(max(token_f1(predicted, gold) for gold in q.answers))
    return float(np.mean(scores))


def tune_threshold(answers, golds, points: int) -> float:
    """Threshold maximising official overall F1 on the tuning articles."""
    margins = np.array([answers[qid].margin for qid in golds])
    grid = np.quantile(margins[np.isfinite(margins)], np.linspace(0, 1, points))
    return float(
        max(
            grid,
            key=lambda t: blended(score_answers(answers, golds, t), "answerable_f1"),
        )
    )


def plot(curves: pd.DataFrame, chosen: pd.DataFrame, path) -> None:
    fig = Figure(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = fig.subplots()
    ax.set_facecolor("#fcfcfb")
    for (row, label), group in curves.groupby(["row", "label"], sort=False):
        group = group.sort_values("threshold")
        color = COLORS["oracle" if row == "oracle" else "pipeline"]
        ax.plot(
            group["abstention_accuracy"],
            group["answerable_f1"],
            color=color,
            linewidth=2,
            label=label,
        )
        point = chosen[chosen["row"] == row].iloc[0]
        ax.scatter(
            point["abstention_accuracy"],
            point["answerable_f1"],
            s=40,
            color=color,
            edgecolors="#fcfcfb",
            linewidths=1.5,
            zorder=5,
        )
    ax.set_xlabel(
        "abstention accuracy on unanswerable questions", fontsize=8, color="#52514e"
    )
    ax.set_ylabel("F1 on answerable questions", fontsize=8, color="#52514e")
    ax.set_title(
        "Answering vs abstaining as the threshold varies (dot = tuned threshold)",
        loc="left",
        fontsize=9,
    )
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.tick_params(labelsize=7, colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve, rerank, read and abstain on SQuAD 2.0, against an oracle passage."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    run_dir = start_run(config, "e2e")
    start_tracking(config)
    e2e = config.e2e

    questions = load_squad(config, e2e.eval_split)
    tune, evaluation = split_by_article(questions, e2e.tune_articles, config.seed)
    docs, chunks = squad_corpus(config)
    by_id = {doc.doc_id: doc for doc in docs}
    by_doc = chunks_by_doc(chunks)
    reader = load_reader(config, device)
    encoder = load_encoder(config, device)
    reranker = load_reranker(config, config.rerank.model, device)
    rng = np.random.default_rng(config.seed)
    timed = list(
        rng.choice(
            len(evaluation), min(e2e.latency_questions, len(evaluation)), replace=False
        )
    )

    every = tune + evaluation
    ids = [q.qid for q in every]
    texts = [q.question for q in every]
    searches = {
        "bm25": bm25_search(config, device, chunks)[0],
        "dense": dense_search(
            config, encoder, chunks, config.paths.cache / "squad-none-dense.npz"
        )[0],
    }
    batch_sizes = {
        "bm25": config.retrieval.query_batch_size,
        "dense": config.dense.batch_size,
    }
    runs = {
        name: search(ids, texts, batch_sizes[name]) for name, search in searches.items()
    }
    queries = dict(zip(ids, texts, strict=True))
    for name in list(runs):
        pairs = candidate_pairs(runs[name], ids, by_doc, e2e.rerank_depth)
        runs[f"{name}+rerank"] = rerank_run(
            reranker, ids, pairs, queries, chunks, config.rerank.batch_size
        )
    retrieved = {
        name: {qid: rank(run[qid])[: e2e.top_k] for qid in ids}
        for name, run in runs.items()
    }
    rows, curves, chosen = [], [], []
    for row in ["oracle", *retrieved]:

        def passages(group, row=row):
            if row == "oracle":
                return [[q.context] for q in group]
            return [[by_id[d].text for d in retrieved[row][q.qid]] for q in group]

        answers: dict[str, Answer] = {}
        for group in (tune, evaluation):
            found = read(reader, [q.question for q in group], passages(group))
            answers |= dict(zip([q.qid for q in group], found, strict=True))
        tune_golds = {q.qid: q.answers for q in tune}
        golds = {q.qid: q.answers for q in evaluation}
        if row == "oracle":
            # the published reader numbers are blended over the full dev set at threshold 0
            check = score_answers(answers, tune_golds | golds, 0.0)
            check["blended_em"] = blended(check, "answerable_em")
            check["blended_f1"] = blended(check, "answerable_f1")
            check_frame = pd.DataFrame(
                [{"split": e2e.eval_split, "threshold": 0.0} | check]
            )
            check_frame.round(4).to_csv(
                config.paths.reports / "reader_check.csv", index=False
            )
        threshold = tune_threshold(answers, tune_golds, e2e.threshold_points)
        metrics = score_answers(answers, golds, threshold)
        tuned = score_answers(answers, tune_golds, threshold)
        metrics["tune_blended_f1"] = blended(tuned, "answerable_f1")
        if row != "oracle":
            hit = [paragraph_id(q.context) in retrieved[row][q.qid] for q in evaluation]
            metrics[f"retrieval_recall_at_{e2e.top_k}"] = float(np.mean(hit))
            # the reranker can only promote what the first stage put in its pool
            pool = runs[row.removesuffix("+rerank")]
            metrics[f"first_stage_recall_at_{e2e.rerank_depth}"] = float(
                np.mean(
                    [
                        paragraph_id(q.context) in rank(pool[q.qid])[: e2e.rerank_depth]
                        for q in evaluation
                    ]
                )
            )
            # splits the answerable error into questions retrieval failed and ones the reader failed
            for found, label in ((True, "gold_retrieved"), (False, "gold_missed")):
                subset = [
                    q
                    for q, h in zip(evaluation, hit, strict=True)
                    if h == found and q.answers
                ]
                metrics[f"answerable_{label}"] = len(subset)
                metrics[f"answerable_f1_{label}"] = answerable_f1(
                    answers, subset, threshold
                )

        timings: dict[str, list[float]] = {"retrieval": [], "rerank": [], "reader": []}
        if row != "oracle":
            first = row.removesuffix("+rerank")
            pipeline = QAPipeline(
                search=searches[first],
                chunks=chunks,
                by_doc=by_doc,
                docs=by_id,
                reranker=reranker if row.endswith("+rerank") else None,
                reader=reader,
                threshold=threshold,
                rerank_depth=e2e.rerank_depth,
                rerank_batch_size=config.rerank.batch_size,
                top_k=e2e.top_k,
            )
        for index in timed:
            q = evaluation[index]
            if row == "oracle":
                started = time.perf_counter()
                read(reader, [q.question], passages([q]))
                spent = {"reader": (time.perf_counter() - started) * 1000}
            else:
                spent = pipeline.answer(q.question).timings_ms
            for stage, values in timings.items():
                values.append(spent.get(stage, np.nan))
        timings["end_to_end"] = list(np.nansum(list(timings.values()), axis=0))
        for stage, values in timings.items():
            for p in (50, 95):
                metrics[f"{stage}_p{p}_ms"] = float(np.percentile(values, p))
        metrics["threshold"] = threshold
        params = {
            "row": row,
            "split": f"{e2e.eval_split} minus {e2e.tune_articles} tuning articles",
            "paragraphs": len(docs) if row != "oracle" else 1,
            "passages_read": e2e.top_k if row != "oracle" else 1,
            "reranked": e2e.rerank_depth if row.endswith("+rerank") else 0,
            "reader": config.reader.model.repo,
            "seed": config.seed,
        }
        rows.append(params | metrics)
        log_result(f"e2e-{row}", params, metrics, run_dir)
        print(f"{row:14s} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        margins = np.array([answers[q.qid].margin for q in evaluation])
        for t in np.quantile(
            margins[np.isfinite(margins)], np.linspace(0, 1, e2e.threshold_points)
        ):
            point = score_answers(answers, golds, float(t))
            label = "oracle passage" if row == "oracle" else f"{row}, top {e2e.top_k}"
            curves.append({"row": row, "label": label, "threshold": float(t)} | point)
        chosen.append({"row": row} | metrics)

    frame = pd.DataFrame(rows).round(4)
    frame.to_csv(config.paths.reports / "e2e_results.csv", index=False)
    frame.to_csv(run_dir / "e2e_results.csv", index=False)
    curve = pd.DataFrame(curves).round(4)
    curve.to_csv(config.paths.reports / "e2e_curve.csv", index=False)
    # the plotted pipeline is the one the tuning articles favour, as for serving
    best = frame.drop(index=0)["tune_blended_f1"].idxmax()
    shown = {"oracle", frame.loc[best, "row"]}
    plot(
        curve[curve["row"].isin(shown)],
        pd.DataFrame(chosen),
        config.paths.reports / "e2e_curve.png",
    )
    print(f"best pipeline on the tuning articles: {frame.loc[best, 'row']}")
    print(f"wrote reports/e2e_results.csv, e2e_curve.csv, e2e_curve.png and {run_dir}")


if __name__ == "__main__":
    main()
