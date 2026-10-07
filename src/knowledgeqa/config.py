import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, get_args, get_origin, get_type_hints

import yaml

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class HubSource:
    repo: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        # a branch name would let upstream edits change the data behind an unchanged config
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError(
                f"{self.repo}: revision must be a full commit sha, got {self.revision!r}"
            )
        if not self.files:
            raise ValueError(f"{self.repo}: files must list at least one pattern")


@dataclass(frozen=True)
class BeirDataset:
    corpus: HubSource
    qrels: HubSource
    tune_split: str


@dataclass(frozen=True)
class DataConfig:
    disk_headroom_gb: float
    eval_split: str
    beir: dict[str, BeirDataset]
    squad: HubSource

    def __post_init__(self) -> None:
        if self.disk_headroom_gb < 0:
            raise ValueError("data.disk_headroom_gb must be >= 0")
        if not self.beir:
            raise ValueError("data.beir must list at least one dataset")
        for name, dataset in self.beir.items():
            if dataset.tune_split == self.eval_split:
                raise ValueError(
                    f"data.beir.{name}: tune_split equals eval_split ({self.eval_split!r}), "
                    "tuning on the evaluation split leaks"
                )


@dataclass(frozen=True)
class Paths:
    raw: Path
    reports: Path
    runs: Path


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    cpu_threads: int
    paths: Paths
    data: DataConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be >= 1")


def load_config(path: str | Path) -> Config:
    """Parse a YAML config, rejecting unknown keys, missing keys and wrongly typed values."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return _build(Config, raw, "config")


def config_to_dict(config: Config) -> dict[str, Any]:
    return _plain(asdict(config))


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown, missing = sorted(set(data) - names), sorted(names - set(data))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    hints = get_type_hints(cls)
    return cls(
        **{
            key: _coerce(hints[key], value, f"{where}.{key}")
            for key, value in data.items()
        }
    )


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if is_dataclass(tp):
        return _build(tp, value, where)
    origin, args = get_origin(tp), get_args(tp)
    if origin is UnionType:
        if value is None and NoneType in args:
            return None
        (inner,) = [arg for arg in args if arg is not NoneType]
        return _coerce(inner, value, where)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        return {
            str(key): _coerce(args[1], item, f"{where}.{key}")
            for key, item in value.items()
        }
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        return tuple(
            _coerce(args[0], item, f"{where}[{i}]") for i, item in enumerate(value)
        )
    if tp is Path and isinstance(value, str):
        return Path(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    # bool is a subclass of int, so a stray `true` must not pass as a number
    if isinstance(value, tp) and (tp is bool or not isinstance(value, bool)):
        return value
    raise TypeError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
