import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledgeqa.config import Config, load_config
from knowledgeqa.dense import load_encoder
from knowledgeqa.pipeline import (
    QAPipeline,
    Response,
    bm25_search,
    dense_search,
    squad_corpus,
)
from knowledgeqa.reader import load_reader
from knowledgeqa.rerank import chunks_by_doc, load_reranker
from knowledgeqa.runtime import setup

DEFAULT_CONFIG = "configs/default.yaml"


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def served_threshold(config: Config) -> float:
    """Abstention threshold tuned for the served pipeline by scripts/run_e2e.py."""
    path = config.paths.reports / "e2e_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/run_e2e.py first")
    frame = pd.read_csv(path)
    rows = frame[frame["row"] == config.serve.pipeline]
    if len(rows) != 1:
        raise ValueError(f"{path} has no row for pipeline {config.serve.pipeline!r}")
    return float(rows["threshold"].iloc[0])


def load_pipeline(config: Config, device: torch.device) -> QAPipeline:
    docs, chunks = squad_corpus(config)
    if config.serve.first_stage == "bm25":
        search, _ = bm25_search(config, device, chunks)
    else:
        encoder = load_encoder(config, device)
        cache = config.paths.cache / "squad-none-dense.npz"
        search, _ = dense_search(config, encoder, chunks, cache)
    reranker = (
        load_reranker(config, config.rerank.model, device)
        if config.serve.rerank
        else None
    )
    return QAPipeline(
        search=search,
        chunks=chunks,
        by_doc=chunks_by_doc(chunks),
        docs={doc.doc_id: doc for doc in docs},
        reranker=reranker,
        reader=load_reader(config, device),
        threshold=served_threshold(config),
        rerank_depth=config.e2e.rerank_depth,
        rerank_batch_size=config.rerank.batch_size,
        top_k=config.e2e.top_k,
    )


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """The service; the config path falls back to $KNOWLEDGEQA_CONFIG, then the default."""
    path = config_path or os.environ.get("KNOWLEDGEQA_CONFIG", DEFAULT_CONFIG)
    config = load_config(path)
    # one gpu pipeline, so requests are answered one at a time
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        device = setup(config)
        app.state.pipeline = load_pipeline(config, device)
        app.state.device = str(device)
        yield

    app = FastAPI(title="KnowledgeQA", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        pipeline: QAPipeline = app.state.pipeline
        return {
            "status": "ok",
            "pipeline": config.serve.pipeline,
            "paragraphs": len(pipeline.docs),
            "device": app.state.device,
        }

    @app.post("/query")
    def query(body: Query) -> Response:
        question = " ".join(body.question.split())
        if not question:
            raise HTTPException(422, "question is blank")
        with lock:
            return app.state.pipeline.answer(question)

    return app
