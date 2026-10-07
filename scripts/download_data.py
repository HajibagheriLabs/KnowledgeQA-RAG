import argparse
import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from knowledgeqa.config import HubSource, load_config


@dataclass(frozen=True)
class RemoteFile:
    source: HubSource
    repo_type: str
    filename: str
    size: int
    sha256: str | None
    blob_id: str
    local_dir: Path

    @property
    def path(self) -> Path:
        return self.local_dir / self.filename


def list_remote(
    api: HfApi, source: HubSource, repo_type: str, local_dir: Path
) -> list[RemoteFile]:
    info = api.repo_info(
        source.repo, repo_type=repo_type, revision=source.revision, files_metadata=True
    )
    matched = [
        sibling
        for sibling in info.siblings
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in source.files)
    ]
    if not matched:
        raise RuntimeError(
            f"{source.repo}@{source.revision}: nothing matches {list(source.files)}"
        )
    return [
        RemoteFile(
            source=source,
            repo_type=repo_type,
            filename=sibling.rfilename,
            size=sibling.size,
            sha256=sibling.lfs.sha256 if sibling.lfs else None,
            blob_id=sibling.blob_id,
            local_dir=local_dir,
        )
        for sibling in matched
    ]


def digest(path: Path, algorithm: str, git_blob: bool = False) -> str:
    hasher = hashlib.new(algorithm)
    if git_blob:
        hasher.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def is_intact(remote: RemoteFile) -> bool:
    if not remote.path.is_file() or remote.path.stat().st_size != remote.size:
        return False
    if remote.sha256 is not None:
        return digest(remote.path, "sha256") == remote.sha256
    # non-lfs files are plain git blobs, identified by sha1 over a length header plus content
    return digest(remote.path, "sha1", git_blob=True) == remote.blob_id


def check_disk(directory: Path, needed_bytes: int, headroom_gb: float) -> None:
    free = shutil.disk_usage(directory).free
    if needed_bytes + headroom_gb * 1e9 > free:
        raise RuntimeError(
            f"{directory}: need {needed_bytes / 1e6:.1f} MB plus {headroom_gb:.1f} GB headroom, "
            f"only {free / 1e9:.1f} GB free"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the BEIR subsets, SQuAD 2.0 and the chunking tokenizer."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = config.paths.raw
    raw.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    remote: list[RemoteFile] = []
    for name, dataset in config.data.beir.items():
        remote += list_remote(api, dataset.corpus, "dataset", raw / "beir" / name)
        remote += list_remote(
            api, dataset.qrels, "dataset", raw / "beir" / name / "qrels"
        )
    remote += list_remote(api, config.data.squad, "dataset", raw)
    tokenizer = config.chunking.tokenizer
    remote += list_remote(api, tokenizer, "model", config.paths.model_dir(tokenizer))

    pending = [item for item in remote if not is_intact(item)]
    check_disk(raw, sum(item.size for item in pending), config.data.disk_headroom_gb)
    for item in pending:
        # a complete-looking but corrupt file would otherwise be trusted by the hub client
        item.path.unlink(missing_ok=True)
        hf_hub_download(
            repo_id=item.source.repo,
            filename=item.filename,
            repo_type=item.repo_type,
            revision=item.source.revision,
            local_dir=item.local_dir,
        )
        if not is_intact(item):
            raise RuntimeError(
                f"{item.path}: size or hash does not match the pinned revision"
            )

    manifest = [
        {
            "repo": item.source.repo,
            "revision": item.source.revision,
            "file": item.filename,
            "path": str(item.path),
            "bytes": item.size,
            "sha256": item.sha256,
            "git_blob": item.blob_id,
        }
        for item in remote
    ]
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total_mb = sum(item.size for item in remote) / 1e6
    print(
        f"{len(remote)} files verified ({total_mb:.1f} MB), {len(pending)} downloaded, "
        f"{len(remote) - len(pending)} already present"
    )


if __name__ == "__main__":
    main()
