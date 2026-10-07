import argparse
import hashlib
import time

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import ChunkingConfig, load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.evaluate import rank
from knowledgeqa.ingest import Document, load_squad, split_by_article
from knowledgeqa.pipeline import Stage, bm25_stage, dense_stage
from knowledgeqa.reader import Answer, load_reader, read, score_answers, token_f1
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking

# reference categorical slots 1-2, validated for colour-vision deficiency on a light surface
COLORS = {"oracle": "#2a78d6", "pipeline": "#eb6834"}


def paragraph_id(context: str) -> str:
    return hashlib.sha1(context.encode()).hexdigest()[:16]


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
        description="Retrieve, read and abstain on SQuAD 2.0, against an oracle passage."
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
    paragraphs: dict[str, Document] = {}
    for split in e2e.corpus_splits:
        for q in load_squad(config, split):
            paragraphs.setdefault(
                paragraph_id(q.context),
                Document(paragraph_id(q.context), q.title, q.context, "squad"),
            )
    docs = list(paragraphs.values())
    whole = ChunkingConfig("none", None, 0, config.chunking.default.prepend_title)
    chunks = chunk_corpus(
        docs, corpus_offsets(config, "squad", docs, load_tokenizer(config)), whole
    )
    text_of = {doc.doc_id: doc.text for doc in docs}
    reader = load_reader(config, device)
    encoder = load_encoder(config, device)
    rng = np.random.default_rng(config.seed)
    timed = list(
        rng.choice(
            len(evaluation), min(e2e.latency_questions, len(evaluation)), replace=False
        )
    )

    every = tune + evaluation
    ids = [q.qid for q in every]
    texts = [q.question for q in every]
    stages: dict[str, Stage] = {
        "bm25": bm25_stage(config, device, chunks, ids, texts),
        "dense": dense_stage(
            config,
            encoder,
            chunks,
            ids,
            texts,
            config.paths.cache / "squad-none-dense.npz",
        ),
    }
    retrieved: dict[str, dict[str, list[str]]] = {
        name: {qid: rank(stage.run[qid])[: e2e.top_k] for qid in ids}
        for name, stage in stages.items()
    }
    rows, curves, chosen = [], [], []
    for row in ["oracle", *retrieved]:

        def passages(group, row=row):
            if row == "oracle":
                return [[q.context] for q in group]
            return [[text_of[d] for d in retrieved[row][q.qid]] for q in group]

        answers: dict[str, Answer] = {}
        for group in (tune, evaluation):
            found = read(reader, [q.question for q in group], passages(group))
            answers |= dict(zip([q.qid for q in group], found, strict=True))
        tune_golds = {q.qid: q.answers for q in tune}
        golds = {q.qid: q.answers for q in evaluation}
        threshold = tune_threshold(answers, tune_golds, e2e.threshold_points)
        metrics = score_answers(answers, golds, threshold)
        if row != "oracle":
            hit = [paragraph_id(q.context) in retrieved[row][q.qid] for q in evaluation]
            metrics[f"retrieval_recall_at_{e2e.top_k}"] = float(np.mean(hit))
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

        # latency on a fixed sample of evaluation questions, each answered on its own
        retrieval_ms, reader_ms = [], []
        for index in timed:
            q = evaluation[index]
            started = time.perf_counter()
            if row != "oracle":
                stages[row].answer(q.question)
            middle = time.perf_counter()
            read(reader, [q.question], passages([q]))
            reader_ms.append((time.perf_counter() - middle) * 1000)
            retrieval_ms.append((middle - started) * 1000)
        total = np.array(retrieval_ms) + np.array(reader_ms)
        for label, values in (
            ("retrieval", retrieval_ms),
            ("reader", reader_ms),
            ("end_to_end", total),
        ):
            metrics[f"{label}_p50_ms"] = float(np.percentile(values, 50))
            metrics[f"{label}_p95_ms"] = float(np.percentile(values, 95))
        metrics["threshold"] = threshold
        params = {
            "row": row,
            "split": f"{e2e.eval_split} minus {e2e.tune_articles} tuning articles",
            "paragraphs": len(docs) if row != "oracle" else 1,
            "passages_read": e2e.top_k if row != "oracle" else 1,
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
    best = frame.drop(index=0)["answerable_f1"].idxmax()
    shown = {"oracle", frame.loc[best, "row"]}
    plot(
        curve[curve["row"].isin(shown)],
        pd.DataFrame(chosen),
        config.paths.reports / "e2e_curve.png",
    )
    print(f"wrote reports/e2e_results.csv, e2e_curve.csv, e2e_curve.png and {run_dir}")


if __name__ == "__main__":
    main()
