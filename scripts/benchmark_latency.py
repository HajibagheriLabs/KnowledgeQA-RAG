import argparse
import gc
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from knowledgeqa.config import load_config
from knowledgeqa.ingest import load_squad, split_by_article
from knowledgeqa.runtime import log_result, start_run, start_tracking
from knowledgeqa.serve import create_app

STAGES = ("retrieval", "rerank", "reader", "end_to_end")


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-stage latency of the query service, one request at a time."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run(config, "latency")
    start_tracking(config)
    serve = config.serve

    questions = load_squad(config, config.e2e.eval_split)
    _, evaluation = split_by_article(questions, config.e2e.tune_articles, config.seed)
    rng = np.random.default_rng(config.seed)
    picked = rng.choice(
        len(evaluation), serve.warmup_queries + serve.benchmark_queries, replace=False
    )
    texts = [evaluation[i].question for i in picked]

    rows = []
    for pipeline in serve.benchmark_pipelines:
        settings = replace(config, serve=replace(serve, pipeline=pipeline))
        timings: dict[str, list[float]] = {stage: [] for stage in STAGES}
        abstained = 0
        with TestClient(create_app(settings)) as client:
            health = client.get("/health").json()
            for position, text in enumerate(texts):
                started = time.perf_counter()
                response = client.post("/query", json={"question": text})
                elapsed = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                if position < serve.warmup_queries:
                    continue
                body = response.json()
                abstained += body["abstained"]
                for stage in STAGES[:-1]:
                    timings[stage].append(body["timings_ms"].get(stage, np.nan))
                timings["end_to_end"].append(elapsed)

        gpu = torch.cuda.get_device_name() if health["device"] == "cuda" else "none"
        stats = {
            stage: {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values)),
                "queries": len(values),
            }
            for stage, values in timings.items()
            if not np.isnan(values).all()
        }
        params = {
            "pipeline": pipeline,
            "served": pipeline == serve.pipeline,
            "warmup_discarded": serve.warmup_queries,
            "paragraphs": health["paragraphs"],
            "reranked_paragraphs": settings.e2e.rerank_depth
            if settings.serve.rerank
            else 0,
            "passages_read": config.e2e.top_k,
            "abstained": abstained,
            "device": health["device"],
            "gpu": gpu,
            "cpu": cpu_name(),
            "cpu_threads": config.cpu_threads,
            "seed": config.seed,
        }
        rows += [
            {"pipeline": pipeline, "served": params["served"], "stage": stage}
            | values
            | params
            for stage, values in stats.items()
        ]
        metrics = {
            f"{stage}_{name}": value
            for stage, values in stats.items()
            for name, value in values.items()
            if name != "queries"
        }
        log_result(f"latency-{pipeline}", params, metrics, run_dir)
        # the next pipeline should not run beside this one's weights
        del client
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).round(2)
    frame.to_csv(config.paths.reports / "latency.csv", index=False)
    frame.to_csv(run_dir / "latency.csv", index=False)
    print(frame[["pipeline", "stage", "p50_ms", "p95_ms", "mean_ms"]].to_string())
    print(f"wrote {config.paths.reports / 'latency.csv'} and {run_dir}")


if __name__ == "__main__":
    main()
