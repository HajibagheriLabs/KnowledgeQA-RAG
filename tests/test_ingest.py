from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from knowledgeqa.config import Config, load_config
from knowledgeqa.ingest import (
    Document,
    eval_qrels,
    load_corpus,
    normalize_document,
    tune_qrels,
)

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def config_with_raw_dir(raw: Path) -> Config:
    config = load_config(DEFAULT)
    return replace(config, paths=replace(config.paths, raw=raw))


def write_parquet(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path)


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["query-id", "corpus-id", "score"]).to_csv(
        path, sep="\t", index=False
    )


def test_normaliser_handles_all_three_corpora(tmp_path):
    config = config_with_raw_dir(tmp_path)
    raw = {
        "scifact": [
            {
                "_id": "4983",
                "title": "White  matter",
                "text": "Alterations of\nthe brain.",
            }
        ],
        "nfcorpus": [
            {"_id": "MED-10", "title": "Statin Use", "text": "Recent  studies."}
        ],
        # fiqa has no titles, and ships documents that are empty yet judged relevant
        "fiqa": [
            {"_id": "0123", "title": "", "text": "  I'm not saying  that. "},
            {"_id": "599762", "title": None, "text": ""},
        ],
    }
    for name, records in raw.items():
        write_parquet(
            tmp_path / "beir" / name / "corpus" / "corpus-00000-of-00001.parquet",
            records,
        )

    assert load_corpus(config, "scifact") == [
        Document("4983", "White matter", "Alterations of the brain.", "scifact")
    ]
    assert load_corpus(config, "nfcorpus") == [
        Document("MED-10", "Statin Use", "Recent studies.", "nfcorpus")
    ]
    assert load_corpus(config, "fiqa") == [
        Document("0123", "", "I'm not saying that.", "fiqa"),
        Document("599762", "", "", "fiqa"),
    ]

    assert normalize_document({"_id": 4983, "text": "x"}, "scifact").doc_id == "4983"
    with pytest.raises(TypeError):
        normalize_document({"_id": 4983.0, "text": "x"}, "scifact")
    with pytest.raises(ValueError):
        normalize_document({"_id": " ", "text": "x"}, "scifact")
    with pytest.raises(KeyError):
        normalize_document({"title": "no id", "text": "x"}, "scifact")


def test_tuning_split_never_overlaps_evaluation_split(tmp_path):
    config = config_with_raw_dir(tmp_path)
    qrels = tmp_path / "beir" / "scifact" / "qrels"
    write_qrels(qrels / "train.tsv", [("1", "d1", 1), ("2", "d2", 1)])
    write_qrels(qrels / "test.tsv", [("3", "d3", 1), ("3", "d1", 1)])

    assert config.data.beir["scifact"].tune_split == "train"
    assert set(tune_qrels(config, "scifact")) == {"1", "2"}
    assert eval_qrels(config, "scifact") == {"3": {"d3": 1, "d1": 1}}

    write_qrels(qrels / "train.tsv", [("1", "d1", 1), ("3", "d3", 1)])
    with pytest.raises(ValueError, match="both the tuning and the evaluation split"):
        tune_qrels(config, "scifact")

    leaky = replace(config.data.beir["scifact"], tune_split="test")
    with pytest.raises(ValueError, match="leaks"):
        replace(config.data, beir={"scifact": leaky})
