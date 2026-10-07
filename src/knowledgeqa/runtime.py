import os
import random
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from knowledgeqa.config import Config, config_to_dict


def setup(config: Config) -> torch.device:
    """Seed every RNG, cap CPU threads and return the configured device."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # the tokenizers thread pool reads this lazily on first use, so setting it here still applies
    os.environ["RAYON_NUM_THREADS"] = str(config.cpu_threads)
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "config asks for device 'cuda' but torch sees no GPU; set device: cpu"
        )
    return torch.device(config.device)


def start_run(config: Config, name: str) -> Path:
    """Create a timestamped run directory holding the exact config used."""
    run_dir = config.paths.runs / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{name}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False)
    )
    return run_dir


def start_tracking(config: Config) -> None:
    config.paths.mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.paths.mlruns / 'mlflow.db'}")
    mlflow.set_experiment(config.tracking.experiment)


def log_result(
    name: str, params: dict, metrics: dict[str, float], run_dir: Path
) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(run_dir / "config.yaml"))
