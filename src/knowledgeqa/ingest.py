from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from knowledgeqa.config import Config

Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str


def clean_text(value: Any) -> str:
    """Collapse whitespace runs to single spaces; a missing value becomes ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {value!r}")
    return " ".join(value.split())


def clean_id(value: Any) -> str:
    # ids stay strings end to end: an int cast would turn "0123" into 123 and detach it
    # from the qrels without any error
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"expected a string or integer id, got {value!r}")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("empty id")
    return cleaned


def normalize_document(record: Mapping[str, Any], source: str) -> Document:
    """Map one raw BEIR corpus record onto the internal schema.

    Empty documents are kept: FiQA ships some, and its qrels mark several of them relevant.
    """
    return Document(
        doc_id=clean_id(record["_id"]),
        title=clean_text(record.get("title")),
        text=clean_text(record.get("text")),
        source=source,
    )


def beir_dir(config: Config, name: str) -> Path:
    if name not in config.data.beir:
        raise KeyError(
            f"unknown BEIR dataset {name!r}, expected one of {sorted(config.data.beir)}"
        )
    return config.paths.raw / "beir" / name


def load_corpus(config: Config, name: str) -> list[Document]:
    frame = pd.read_parquet(beir_dir(config, name) / "corpus")
    docs = [normalize_document(record, name) for record in frame.to_dict("records")]
    if len({doc.doc_id for doc in docs}) != len(docs):
        raise ValueError(f"{name}: duplicate document ids in corpus")
    return docs


def load_queries(config: Config, name: str) -> dict[str, str]:
    frame = pd.read_parquet(beir_dir(config, name) / "queries")
    queries = {
        clean_id(r["_id"]): clean_text(r["text"]) for r in frame.to_dict("records")
    }
    if len(queries) != len(frame):
        raise ValueError(f"{name}: duplicate query ids")
    return queries


def load_qrels(config: Config, name: str, split: str) -> Qrels:
    path = beir_dir(config, name) / "qrels" / f"{split}.tsv"
    frame = pd.read_csv(
        path, sep="\t", dtype={"query-id": str, "corpus-id": str, "score": int}
    )
    if list(frame.columns) != ["query-id", "corpus-id", "score"]:
        raise ValueError(f"{path}: unexpected columns {list(frame.columns)}")
    if frame.duplicated(["query-id", "corpus-id"]).any():
        raise ValueError(f"{path}: a (query, document) pair is judged twice")
    qrels: Qrels = {}
    for query_id, doc_id, score in frame.itertuples(index=False):
        qrels.setdefault(clean_id(query_id), {})[clean_id(doc_id)] = int(score)
    return qrels


def eval_qrels(config: Config, name: str) -> Qrels:
    return load_qrels(config, name, config.data.eval_split)


def tune_qrels(config: Config, name: str) -> Qrels:
    """Qrels of the tuning split, refusing any query that also belongs to the evaluation split."""
    tune = load_qrels(config, name, config.data.beir[name].tune_split)
    shared = set(tune) & set(eval_qrels(config, name))
    if shared:
        raise ValueError(
            f"{name}: {len(shared)} queries are in both the tuning and the evaluation split, "
            f"e.g. {sorted(shared)[:3]}"
        )
    return tune


def check_qrels(
    qrels: Qrels, doc_ids: Collection[str], query_ids: Collection[str]
) -> None:
    """Raise if judgements point at documents or queries that were not loaded."""
    unknown_queries = set(qrels) - set(query_ids)
    unknown_docs = {d for judged in qrels.values() for d in judged} - set(doc_ids)
    if unknown_queries or unknown_docs:
        raise ValueError(
            f"qrels reference {len(unknown_queries)} unknown queries and "
            f"{len(unknown_docs)} unknown documents"
        )
