import argparse
import time

import faiss
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from knowledgeqa.chunking import chunk_corpus, corpus_offsets, load_tokenizer
from knowledgeqa.config import load_config
from knowledgeqa.dense import cached_encode, encode, load_encoder
from knowledgeqa.evaluate import evaluate_run, latency_ms, max_pool
from knowledgeqa.ingest import eval_qrels, load_corpus, load_queries
from knowledgeqa.runtime import log_result, setup, start_run, start_tracking

# reference categorical slots 1-3, validated for colour-vision deficiency on a light surface
COLORS = {"flat": "#2a78d6", "hnsw": "#eb6834", "ivfpq": "#1baf7a"}
MARKERS = {"flat": "s", "hnsw": "o", "ivfpq": "^"}


def memory_mb(index: faiss.Index) -> float:
    return faiss.serialize_index(index).nbytes / 1e6


def build_hnsw(vectors: np.ndarray, m: int, ef_construction: int) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivfpq(vectors: np.ndarray, lists: int, m: int, bits: int) -> faiss.Index:
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer, vectors.shape[1], lists, m, bits, faiss.METRIC_INNER_PRODUCT
    )
    # k-means and codebook training are the expensive part, so they run on the gpu
    options = faiss.GpuClonerOptions()
    options.useFloat16 = True
    resources = faiss.StandardGpuResources()
    gpu = faiss.index_cpu_to_gpu(resources, 0, index, options)
    gpu.train(vectors)
    gpu.add(vectors)
    return faiss.index_gpu_to_cpu(gpu)


def plot(frame: pd.DataFrame, path) -> None:
    fig = Figure(figsize=(7, 4.2), dpi=150)
    ax = fig.subplots()
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for family, group in frame.groupby("family", sort=False):
        group = group.sort_values("latency_p50_ms")
        ax.plot(
            group["latency_p50_ms"],
            group["recall_vs_exact"],
            color=COLORS[family],
            marker=MARKERS[family],
            markersize=6,
            linewidth=2,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.5,
            label=family,
        )
        last = group.iloc[-1]
        ax.annotate(
            f"{family}",
            (last["latency_p50_ms"], last["recall_vs_exact"]),
            xytext=(6, -3),
            textcoords="offset points",
            color="#0b0b0b",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel(
        "query latency p50, ms (log scale, one query at a time)", color="#52514e"
    )
    ax.set_ylabel("recall@100 relative to exact search", color="#52514e")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("FAISS index trade-off on FiQA chunks", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact vs HNSW vs IVF-PQ search on the largest corpus."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    faiss.omp_set_num_threads(config.cpu_threads)
    run_dir = start_run(config, "index-study")
    start_tracking(config)
    study, evaluation = config.index_study, config.evaluation
    name, depth = study.corpus, evaluation.recall_k

    docs = load_corpus(config, name)
    chunking = config.chunking.default
    chunks = chunk_corpus(
        docs, corpus_offsets(config, name, docs, load_tokenizer(config)), chunking
    )
    doc_ids = [chunk.doc_id for chunk in chunks]
    rows = np.array([i for i, chunk in enumerate(chunks) if chunk.text], dtype=np.int64)
    encoder = load_encoder(config, device)
    cache = config.paths.cache / f"{name}-{chunking.name}-dense.npz"
    vectors, _ = cached_encode(
        encoder, [chunks[i].text for i in rows], config.dense, cache
    )
    queries, qrels = load_queries(config, name), eval_qrels(config, name)
    query_ids = sorted(qrels)
    texts = [config.dense.query_prefix + queries[q] for q in query_ids]
    query_vectors = encode(encoder, texts, config.dense.batch_size)

    def timed(build) -> tuple[faiss.Index, float]:
        started = time.perf_counter()
        index = build()
        return index, time.perf_counter() - started

    def flat() -> faiss.Index:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    exact, exact_seconds = timed(flat)
    _, truth = exact.search(query_vectors, depth)
    hnsw, hnsw_seconds = timed(
        lambda: build_hnsw(vectors, study.hnsw_m, study.hnsw_ef_construction)
    )
    ivfpq, ivfpq_seconds = timed(
        lambda: build_ivfpq(
            vectors, study.ivf_lists, study.pq_subquantizers, study.pq_bits
        )
    )
    position = {query_id: i for i, query_id in enumerate(query_ids)}
    variants = [("flat", "exact", exact, None, exact_seconds)]
    variants += [
        ("hnsw", f"efSearch={ef}", hnsw, ef, hnsw_seconds)
        for ef in study.hnsw_ef_search
    ]
    variants += [
        ("ivfpq", f"nprobe={n}", ivfpq, n, ivfpq_seconds) for n in study.ivf_nprobe
    ]

    results = []
    for family, setting, index, knob, build_seconds in variants:
        if family == "hnsw":
            index.hnsw.efSearch = knob
        if family == "ivfpq":
            index.nprobe = knob
        scores, ids = index.search(query_vectors, depth)
        overlap = [
            len(set(a) & set(b)) / depth for a, b in zip(ids, truth, strict=True)
        ]
        hits = np.where(ids >= 0, rows[np.maximum(ids, 0)], 0)
        padded = np.where(ids >= 0, scores, -np.inf)
        # only the top documents are needed here, so pool at the ndcg cut-off
        run = max_pool(query_ids, hits, padded, doc_ids, evaluation.ndcg_k)
        quality = evaluate_run(
            run, qrels, evaluation.ndcg_k, evaluation.recall_k, evaluation.mrr_k
        )

        def answer(query_id: str, index=index) -> object:
            row = position[query_id]
            return index.search(query_vectors[row : row + 1], depth)

        metrics = {
            "recall_vs_exact": float(np.mean(overlap)),
            f"ndcg_at_{evaluation.ndcg_k}": quality[f"ndcg_at_{evaluation.ndcg_k}"],
            **latency_ms(answer, query_ids, config.retrieval.latency_warmup),
            "build_seconds": build_seconds,
            "memory_mb": memory_mb(index),
        }
        params = {
            "corpus": name,
            "chunking": chunking.name,
            "vectors": len(rows),
            "dimension": vectors.shape[1],
            "family": family,
            "setting": setting,
            "cpu_threads": config.cpu_threads,
        }
        results.append(params | metrics)
        log_result(f"index-{family}-{setting}", params, metrics, run_dir)
        print(
            f"{family:6s} {setting:13s} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    frame = pd.DataFrame(results).round(4)
    frame.to_csv(config.paths.reports / "index_study.csv", index=False)
    frame.to_csv(run_dir / "index_study.csv", index=False)
    plot(frame, config.paths.reports / "index_study.png")
    print(f"wrote reports/index_study.csv, reports/index_study.png and {run_dir}")


if __name__ == "__main__":
    main()
