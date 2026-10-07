import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, get_args, get_origin, get_type_hints

import yaml

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
DEVICES = ("cpu", "cuda")
STRATEGIES = ("none", "fixed", "sentence")


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
class ChunkingConfig:
    strategy: str
    chunk_size: int | None
    overlap: int
    prepend_title: bool

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"chunking strategy must be one of {STRATEGIES}, got {self.strategy!r}"
            )
        if self.strategy == "none":
            if self.chunk_size is not None or self.overlap != 0:
                raise ValueError(
                    "strategy 'none' keeps whole documents: chunk_size null, overlap 0"
                )
        elif (
            self.chunk_size is None
            or self.chunk_size < 1
            or not 0 <= self.overlap < self.chunk_size
        ):
            raise ValueError(
                f"chunking needs chunk_size >= 1 and 0 <= overlap < chunk_size, "
                f"got {self.chunk_size} and {self.overlap}"
            )

    @property
    def name(self) -> str:
        if self.strategy == "none":
            return "none"
        return f"{self.strategy}-{self.chunk_size}-{self.overlap}"


@dataclass(frozen=True)
class ChunkingAblation:
    strategies: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    overlap_fractions: tuple[float, ...]
    include_unchunked: bool

    def __post_init__(self) -> None:
        if "none" in self.strategies:
            raise ValueError(
                "chunking.ablation: use include_unchunked instead of strategy 'none'"
            )
        if not all(0 <= fraction < 1 for fraction in self.overlap_fractions):
            raise ValueError("chunking.ablation: overlap fractions must lie in [0, 1)")


@dataclass(frozen=True)
class ChunkingSection:
    tokenizer: HubSource
    batch_size: int
    default: ChunkingConfig
    ablation: ChunkingAblation

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("chunking.batch_size must be >= 1")
        self.ablation_configs()

    def ablation_configs(self) -> list[ChunkingConfig]:
        """Every configuration of the ablation grid, sharing the default's title handling."""
        title = self.default.prepend_title
        grid = (
            [ChunkingConfig("none", None, 0, title)]
            if self.ablation.include_unchunked
            else []
        )
        grid += [
            ChunkingConfig(strategy, size, round(size * fraction), title)
            for strategy in self.ablation.strategies
            for size in self.ablation.chunk_sizes
            for fraction in self.ablation.overlap_fractions
        ]
        return grid


@dataclass(frozen=True)
class BM25Config:
    k1: float
    b: float
    stemmer: str

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("bm25.k1 must be > 0")
        if not 0 <= self.b <= 1:
            raise ValueError("bm25.b must lie in [0, 1]")


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_chunks: int
    query_batch_size: int
    latency_warmup: int

    def __post_init__(self) -> None:
        if (
            self.candidate_chunks < 1
            or self.query_batch_size < 1
            or self.latency_warmup < 0
        ):
            raise ValueError(
                "retrieval: candidate_chunks, query_batch_size >= 1 and latency_warmup >= 0"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    ndcg_k: int
    recall_k: int
    mrr_k: int

    def __post_init__(self) -> None:
        if min(self.ndcg_k, self.recall_k, self.mrr_k) < 1:
            raise ValueError("evaluation cut-offs must be >= 1")


@dataclass(frozen=True)
class TrackingConfig:
    experiment: str


@dataclass(frozen=True)
class Paths:
    raw: Path
    models: Path
    cache: Path
    reports: Path
    runs: Path
    mlruns: Path

    def model_dir(self, source: HubSource) -> Path:
        return self.models / source.repo


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    cpu_threads: int
    paths: Paths
    data: DataConfig
    chunking: ChunkingSection
    bm25: BM25Config
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.retrieval.candidate_chunks < self.evaluation.recall_k:
            raise ValueError(
                "retrieval.candidate_chunks must be >= evaluation.recall_k"
            )
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
