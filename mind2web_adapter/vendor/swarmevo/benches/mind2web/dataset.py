"""Prepare immutable, compact cases and deterministic website/task holdouts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from swarmevo.benches.protocol import Case

OFFICIAL_COUNTS = {"train": 1009, "test_task": 252, "test_website": 177, "test_domain": 912}
LABEL_ATTRIBUTES = {"is_original_target", "is_top_level_target", "is_target", "data-positive"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def make_holdout(tasks: list[dict], seed: int = 42) -> dict:
    """Partition complete official-training cases with one seeded shuffle."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    ids = sorted(task["annotation_id"] for task in tasks)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training case IDs")
    random.Random(seed).shuffle(ids)
    train_count = math.ceil(len(ids) * 0.3)
    validation_count = math.ceil(len(ids) * 0.1)
    if train_count + validation_count > len(ids):
        raise ValueError("training inventory is too small for disjoint train/validation")
    parts = {
        "train": ids[:train_count],
        "validation": ids[train_count:train_count + validation_count],
        "unused": ids[train_count + validation_count:],
    }
    identity = json.dumps({"seed": seed, "parts": parts}, sort_keys=True).encode()
    return {
        "parts": parts,
        "split_sha256": hashlib.sha256(identity).hexdigest(),
        "seed": seed,
        "train_fraction": 0.3,
        "validation_fraction": 0.1,
    }


def partition_inventory(manifest: dict, seed: int = 42) -> dict:
    """Project the same case files onto a reproducible train/validation split."""
    tasks = manifest["tasks"]
    split = make_holdout(
        [item for item in tasks.values() if item["official_split"] == "train"], seed
    )
    for name in OFFICIAL_COUNTS:
        if name != "train":
            split["parts"][name] = sorted(
                key for key, item in tasks.items() if item["official_split"] == name
            )
    return {
        **manifest,
        **split,
        "counts": {name: len(ids) for name, ids in split["parts"].items()},
        "website_counts": {
            name: len({tasks[key]["website"] for key in ids})
            for name, ids in split["parts"].items()
        },
    }


DOM_IDENTITY = {
    "format": "mind2web-compact-html-v1",
    "revision": "33bd95caeee7bba22dd08ecc935845e15c5e5dc7",
    "max_depth": 5,
    "max_children": 50,
    "max_sibling": 3,
    "max_value_length": 5,
    "max_length": 20,
}


def compact_observation(text: str, candidate_ids: list[str]) -> tuple[dict, dict]:
    """Apply the pinned official representation to the original cleaned snapshot."""
    from lxml import etree

    from swarmevo.benches.mind2web._vendor.dom_utils import get_tree_repr, prune_tree

    tree = etree.fromstring((text or "<html/>").encode())
    present = {node.get("backend_node_id") for node in tree.iter()}
    located = [eid for eid in candidate_ids if eid in present]
    compact, mapping = get_tree_repr(
        prune_tree(
            tree, located, max_depth=DOM_IDENTITY["max_depth"],
            max_children=DOM_IDENTITY["max_children"], max_sibling=DOM_IDENTITY["max_sibling"],
        ),
        keep_html_brackets=True, id_mapping={},
        max_value_length=DOM_IDENTITY["max_value_length"], max_length=DOM_IDENTITY["max_length"],
    )
    original_to_local = {eid: str(local) for eid, local in mapping.items()}
    local_to_original = {local: eid for eid, local in original_to_local.items()}
    return {
        "format": DOM_IDENTITY["format"],
        "html": compact,
        "candidate_ids": [original_to_local[eid] for eid in located],
        "missing_candidates": [
            {"candidate_index": index, "reason": "absent from recorded DOM"}
            for index, eid in enumerate(candidate_ids)
            if eid not in present
        ],
    }, {
        "original_to_local": original_to_local,
        "local_to_original": local_to_original,
    }


def normalize_task(task: dict, ranks: dict, top_k: int = 50) -> dict:
    history = task["action_reprs"]
    if not task["actions"] or len(history) != len(task["actions"]):
        raise ValueError(f"{task['annotation_id']}: invalid annotated action sequence")
    steps = []
    for index, action in enumerate(task["actions"]):
        sample_id = f"{task['annotation_id']}_{action['action_uid']}"
        order = ranks[sample_id]
        candidates = {}
        for item in action["pos_candidates"] + action["neg_candidates"]:
            element_id = str(item["backend_node_id"])
            rank = order.get(element_id)
            if type(rank) is not int or rank < 0:
                raise ValueError(f"{sample_id}: missing/invalid rank for {element_id}")
            if rank >= top_k:
                continue
            attributes = item.get("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes)
            safe = {
                key: value
                for key, value in attributes.items()
                if key.lower() not in LABEL_ATTRIBUTES
            }
            candidates[element_id] = {
                "element_id": element_id,
                "tag": item["tag"],
                "attributes": safe,
                "rank": rank,
            }
        ordered = sorted(candidates.values(), key=lambda item: (item["rank"], item["element_id"]))
        # Positive/negative membership never influences candidate selection.
        view, mapping = compact_observation(
            action["cleaned_html"], [item["element_id"] for item in ordered]
        )
        operation = action["operation"]
        if operation["op"] not in {"CLICK", "TYPE", "SELECT"}:
            raise ValueError(f"{sample_id}: unsupported operation")
        steps.append(
            {
                "action_uid": action["action_uid"],
                "observation": {
                    "task": task["confirmed_task"],
                    "previous_actions": history[:index],
                    **view,
                },
                "id_mapping": mapping,
                "raw": {"cleaned_html": action["cleaned_html"], "candidates": ordered},
                "gold": {
                    "element_ids": sorted(
                        {str(item["backend_node_id"]) for item in action["pos_candidates"]}
                    ),
                    "operation": operation["op"],
                    "value": str(operation.get("value", "")),
                },
            }
        )
    if len({step["action_uid"] for step in steps}) != len(steps):
        raise ValueError(f"{task['annotation_id']}: duplicate action_uid")
    return {
        key: task[key]
        for key in ("annotation_id", "website", "domain", "subdomain", "confirmed_task")
    } | {"steps": steps}


def prepare(
    data_root: Path, ranks_file: Path, output: Path, seed: int = 42, *, diagnostic: bool = False
) -> dict:
    import ijson

    if (output / "manifest.json").exists():
        raise FileExistsError("prepared inventory already exists; use a new directory")
    ranks = json.loads(ranks_file.read_text(encoding="utf-8"))["ranks"]
    source_root = data_root / "data" if (data_root / "data").exists() else data_root
    tasks, source_files, split_ids = {}, {}, {}
    for split in OFFICIAL_COUNTS:
        paths = sorted((source_root / split).glob("*.json"))
        if not paths and not diagnostic:
            raise FileNotFoundError(f"missing official split {split}")
        split_ids[split] = []
        for path in paths:
            source_files[str(path.relative_to(data_root))] = file_hash(path)
            with path.open("rb") as source:
                for task in ijson.items(source, "item"):
                    normalized = normalize_task(task, ranks)
                    task_id = normalized["annotation_id"]
                    if task_id in tasks:
                        raise ValueError(f"duplicate annotation_id: {task_id}")
                    # File identity never uses untrusted dataset text as a path.
                    filename = hashlib.sha256(task_id.encode()).hexdigest() + ".json"
                    target = output / "cases" / filename
                    write_json(target, normalized)
                    tasks[task_id] = {
                        "annotation_id": task_id,
                        "website": normalized["website"],
                        "domain": normalized["domain"],
                        "subdomain": normalized["subdomain"],
                        "spec": normalized["confirmed_task"],
                        "file": f"cases/{filename}",
                        "sha256": file_hash(target),
                        "official_split": split,
                        "n_steps": len(normalized["steps"]),
                    }
                    split_ids[split].append(task_id)
        if not diagnostic and len(split_ids[split]) != OFFICIAL_COUNTS[split]:
            raise ValueError(
                f"{split}: expected {OFFICIAL_COUNTS[split]} tasks, got {len(split_ids[split])}"
            )
    if not split_ids["train"]:
        raise ValueError("train is empty")
    holdout = make_holdout([tasks[item] for item in split_ids["train"]], seed)
    for split in OFFICIAL_COUNTS:
        if split != "train":
            holdout["parts"][split] = sorted(split_ids[split])
    download_manifest = data_root / "download-manifest.json"
    prefetch_manifest = data_root / "prefetch-revision.json"
    identity_path = download_manifest if download_manifest.exists() else prefetch_manifest
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    manifest = {
        "version": 4,
        "observation": DOM_IDENTITY,
        "diagnostic": diagnostic,
        "top_k": 50,
        "source_repository": identity.get("repository"),
        "source_revision": identity.get("revision"),
        "source_files": source_files,
        "ranks_sha256": file_hash(ranks_file),
        "tasks": tasks,
        **holdout,
        "counts": {key: len(value) for key, value in holdout["parts"].items()},
        "website_counts": {
            key: len({tasks[item]["website"] for item in ids})
            for key, ids in holdout["parts"].items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_inventory(root: Path, *, seed: int | None = None) -> tuple[dict, dict[str, list[Case]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        type(manifest.get("version")) is not int
        or manifest["version"] != 4
        or manifest.get("observation") != DOM_IDENTITY
        or manifest.get("top_k") != 50
        or type(manifest.get("diagnostic")) is not bool
        or type(manifest.get("seed")) is not int
        or manifest.get("train_fraction") != 0.3
        or manifest.get("validation_fraction") != 0.1
    ):
        raise ValueError("unsupported Mind2Web inventory")
    expected_parts = {
        "train",
        "validation",
        "unused",
        "test_task",
        "test_website",
        "test_domain",
    }
    if set(manifest["parts"]) != expected_parts:
        raise ValueError("inventory is missing an official/internal split")
    expected = partition_inventory(manifest, manifest["seed"])
    if (manifest["parts"] != expected["parts"]
            or manifest.get("split_sha256") != expected["split_sha256"]):
        raise ValueError("inventory differs from deterministic random case split")
    if manifest["counts"] != expected["counts"] or manifest["website_counts"] != expected["website_counts"]:
        raise ValueError("inventory counts differ from assigned tasks")
    if seed is not None:
        manifest = partition_inventory(manifest, seed)
    parts = {}
    used_ids = set()
    for split, ids in manifest["parts"].items():
        cases = []
        for task_id in ids:
            if task_id in used_ids:
                raise ValueError("inventory partitions overlap")
            used_ids.add(task_id)
            item = manifest["tasks"][task_id]
            expected_split = (
                "train" if split in {"train", "validation", "unused"} else split
            )
            if item["official_split"] != expected_split or item["annotation_id"] != task_id:
                raise ValueError("official split/case identity changed")
            path = (root / item["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("case path escapes inventory")
            payload = {
                "spec": item["spec"],
                "file": str(path),
                "sha256": item["sha256"],
                "official_split": item["official_split"],
                "n_steps": item["n_steps"],
            }
            cases.append(Case(task_id, item["website"], payload))
        parts[split] = cases
    if used_ids != set(manifest["tasks"]):
        raise ValueError("inventory has unassigned tasks")
    counts = {split: len(cases) for split, cases in parts.items()}
    if counts != manifest["counts"]:
        raise ValueError("inventory counts differ from assigned tasks")
    if not manifest["diagnostic"]:
        for split, count in OFFICIAL_COUNTS.items():
            if sum(item["official_split"] == split for item in manifest["tasks"].values()) != count:
                raise ValueError("incomplete official inventory")
    return manifest, parts


def read_case(case: Case) -> dict:
    path = Path(case.payload["file"])
    if file_hash(path) != case.payload["sha256"]:
        raise ValueError(f"case content changed: {case.id}")
    task = json.loads(path.read_text(encoding="utf-8"))
    if (
        task["annotation_id"] != case.id
        or task["website"] != case.family_id
        or len(task["steps"]) != case.payload["n_steps"]
        or task["confirmed_task"] != case.payload["spec"]
    ):
        raise ValueError("case identity mismatch")
    return task
