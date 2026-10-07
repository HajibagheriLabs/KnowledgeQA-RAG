from pathlib import Path

import pytest
import yaml

from knowledgeqa.config import Config, config_to_dict, load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads_and_round_trips(tmp_path):
    config = load_config(DEFAULT)
    assert isinstance(config, Config)
    assert set(config.data.beir) == {"scifact", "nfcorpus", "fiqa"}
    assert config.data.eval_split == "test"

    dumped = tmp_path / "dumped.yaml"
    dumped.write_text(yaml.safe_dump(config_to_dict(config)))
    assert load_config(dumped) == config


def test_config_rejects_bad_input(tmp_path):
    def load_modified(edit) -> Config:
        raw = yaml.safe_load(DEFAULT.read_text())
        edit(raw)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw))
        return load_config(path)

    with pytest.raises(ValueError, match="unknown keys"):
        load_modified(lambda raw: raw.update(sed=1))
    with pytest.raises(TypeError, match="cpu_threads"):
        load_modified(lambda raw: raw.update(cpu_threads="2"))
    with pytest.raises(ValueError, match="commit sha"):
        load_modified(lambda raw: raw["data"]["squad"].update(revision="main"))
    with pytest.raises(ValueError, match="device"):
        load_modified(lambda raw: raw.update(device="tpu"))
