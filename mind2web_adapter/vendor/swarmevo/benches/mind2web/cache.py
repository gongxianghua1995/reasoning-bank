"""Manage reproducible conversions of explicitly trusted official Mind2Web data."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import pickle
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from swarmevo.benches.mind2web.dataset import (
    DOM_IDENTITY,
    OFFICIAL_COUNTS,
    file_hash,
    load_inventory,
    partition_inventory,
    prepare,
    read_case,
    write_json,
)


def official_sources(data_root: Path, *, diagnostic: bool = False) -> dict[str, str]:
    source = data_root / "data" if (data_root / "data").is_dir() else data_root
    files = {}
    for split in OFFICIAL_COUNTS:
        paths = sorted((source / split).glob("*.json"))
        if not paths and not diagnostic:
            raise FileNotFoundError(f"missing official split {split}; provide --data-root")
        files.update({str(path.relative_to(data_root)): file_hash(path) for path in paths})
    if not files:
        raise FileNotFoundError(f"no official Mind2Web JSON files in {data_root}")
    return files


def export_partition(data_root: Path, seed: int, *, diagnostic: bool = False) -> dict:
    """Read official metadata without converting webpages or loading candidate ranks."""
    import ijson

    files = official_sources(data_root, diagnostic=diagnostic)
    tasks = {}
    for name in files:
        split = Path(name).parent.name
        with (data_root / name).open("rb") as stream:
            for task in ijson.items(stream, "item"):
                task_id = task["annotation_id"]
                if task_id in tasks:
                    raise ValueError(f"duplicate annotation_id: {task_id}")
                tasks[task_id] = {
                    "annotation_id": task_id,
                    "website": task["website"],
                    "official_split": split,
                }
    if not any(item["official_split"] == "train" for item in tasks.values()):
        raise ValueError("train is empty")
    if not diagnostic:
        for split, count in OFFICIAL_COUNTS.items():
            actual = sum(item["official_split"] == split for item in tasks.values())
            if actual != count:
                raise ValueError(f"{split}: expected {count} tasks, got {actual}")
    manifest = partition_inventory({"tasks": tasks, "source_files": files}, seed)
    manifest.pop("tasks")
    return manifest


def ensure_cache(
    data_root: Path, trusted_official_pickle: Path, cache_root: Path, *, diagnostic: bool = False
) -> Path:
    """Publish a complete cache atomically; serialize builders of the same identity."""
    data_root = data_root.resolve()
    ranks_file = trusted_official_pickle.resolve()
    source_files = official_sources(data_root, diagnostic=diagnostic)
    metadata_path = data_root / "download-manifest.json"
    if not metadata_path.exists():
        metadata_path = data_root / "prefetch-revision.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    identity = {
        "source_files": source_files,
        "source_repository": metadata.get("repository"),
        "source_revision": metadata.get("revision"),
        "ranks_sha256": file_hash(ranks_file),
        "conversion": {"version": 4, "observation": DOM_IDENTITY, "top_k": 50},
        "diagnostic": diagnostic,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / key
    logger = logging.getLogger(__name__)
    with (cache_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            try:
                manifest, parts = load_inventory(target)
                if manifest.get("cache_identity") != identity:
                    raise ValueError("cache identity changed")
                for cases in parts.values():
                    for case in cases:
                        read_case(case)
            except (ValueError, KeyError, OSError, TypeError, AttributeError) as error:
                logger.warning("mind2web_cache rebuild=%s reason=%s", key, error)
            else:
                logger.info("mind2web_cache hit=%s", key)
                return target
        logger.info("mind2web_cache converting=%s", key)
        with TemporaryDirectory(prefix=f".{key}-", dir=cache_root) as temporary:
            staging = Path(temporary)
            with ranks_file.open("rb") as stream:
                scores = pickle.load(stream)  # noqa: S301 -- explicitly trusted official input
            ranks = {
                str(sample): {str(element): int(rank) for element, rank in items.items()}
                for sample, items in scores["ranks"].items()
            }
            del scores
            rank_json = staging / "ranks.json"
            write_json(rank_json, {"ranks": ranks})
            del ranks
            manifest = prepare(data_root, rank_json, staging / "inventory", diagnostic=diagnostic)
            if (official_sources(data_root, diagnostic=diagnostic) != source_files
                    or file_hash(ranks_file) != identity["ranks_sha256"]
                    or manifest["source_repository"] != identity["source_repository"]
                    or manifest["source_revision"] != identity["source_revision"]):
                raise ValueError("official source changed during conversion")
            manifest["ranks_sha256"] = identity["ranks_sha256"]
            manifest["cache_identity"] = identity
            write_json(staging / "inventory/manifest.json", manifest)
            load_inventory(staging / "inventory")
            if target.exists():
                shutil.rmtree(target)
            (staging / "inventory").rename(target)
        logger.info("mind2web_cache ready=%s", key)
    return target
