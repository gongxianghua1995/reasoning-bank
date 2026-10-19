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
    INVENTORY_VERSION,
    OFFICIAL_COUNTS,
    OFFICIAL_STEPS,
    file_hash,
    load_inventory,
    partition_inventory,
    prepare,
    read_case,
    write_json,
)
from swarmevo.benches.mind2web.source import (
    DATASET_REVISION,
    DATASET_SOURCE,
    action_rows,
    official_sources,
    row_metadata,
    validate_step_identity,
)


def export_partition(data_root: Path, seed: int, *, diagnostic: bool = False) -> dict:
    """Read published metadata columns without HTML, screenshots or candidate ranks."""
    files = official_sources(data_root, diagnostic=diagnostic)
    tasks = {}
    for split, _name, row in action_rows(data_root, files, metadata_only=True):
        validate_step_identity(row)
        task_id = row["annotation_id"]
        entry = tasks.setdefault(task_id, {**row_metadata(row), "official_split": split,
                                          "step_ids": []})
        if (row_metadata(row) != {key: entry[key] for key in row_metadata(row)}
                or entry["official_split"] != split):
            raise ValueError(f"{task_id}: inconsistent metadata or split")
        entry["step_ids"].append([row["target_action_index"], row["action_uid"]])
    for task_id, entry in tasks.items():
        steps = entry["step_ids"]
        if len({s[0] for s in steps}) != len(steps) or len({s[1] for s in steps}) != len(steps):
            raise ValueError(f"{task_id}: duplicate published step")
        steps.sort()
    if not any(item["official_split"] == "train" for item in tasks.values()):
        raise ValueError("train is empty")
    if not diagnostic:
        for split, count in OFFICIAL_COUNTS.items():
            selected = [t for t in tasks.values() if t["official_split"] == split]
            if len(selected) != count or sum(len(t["step_ids"]) for t in selected) != OFFICIAL_STEPS[split]:
                raise ValueError(f"{split}: expected {count} tasks and published step inventory")
    manifest = partition_inventory({"tasks": tasks, "source_files": files,
        "source_repository": DATASET_SOURCE, "source_revision": DATASET_REVISION,
        "step_ids": {key: value["step_ids"] for key, value in sorted(tasks.items())}}, seed)
    manifest.pop("tasks")
    return manifest


def ensure_cache(
    data_root: Path, trusted_official_pickle: Path, cache_root: Path, *, diagnostic: bool = False
) -> Path:
    """Publish a complete cache atomically; serialize builders of the same identity."""
    data_root = data_root.resolve()
    ranks_file = trusted_official_pickle.resolve()
    source_files = official_sources(data_root, diagnostic=diagnostic)
    identity = {
        "source_files": source_files,
        "source_repository": DATASET_SOURCE,
        "source_revision": DATASET_REVISION,
        "ranks_sha256": file_hash(ranks_file),
        "conversion": {"version": INVENTORY_VERSION, "observation": DOM_IDENTITY, "top_k": 50},
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
                checked_images = set()
                for cases in parts.values():
                    for case in cases:
                        task = read_case(case)
                        for step in task["steps"]:
                            image = step["image"]
                            if image is None:
                                continue
                            if image["path"] not in checked_images:
                                if file_hash(Path(image["path"])) != image["sha256"]:
                                    raise ValueError("cached screenshot changed")
                                checked_images.add(image["path"])
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
