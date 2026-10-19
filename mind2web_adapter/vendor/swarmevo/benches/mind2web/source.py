"""Read and verify the pinned official action-row snapshot before conversion."""
from __future__ import annotations

import json
from pathlib import Path

from swarmevo.benches.mind2web.dataset import file_hash

DATASET_SOURCE = "osunlp/Multimodal-Mind2Web"
DATASET_REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"
SNAPSHOT = Path(__file__).with_name("snapshot.json")
METADATA_COLUMNS = ["annotation_id", "website", "domain", "subdomain", "confirmed_task",
                    "action_uid", "target_action_index", "action_reprs"]


def dataset_options(options: dict) -> dict:
    expected = {"dataset_source": DATASET_SOURCE, "dataset_revision": DATASET_REVISION}
    for name, value in expected.items():
        if options.get(name) != value:
            raise ValueError(f"bench_cfg.{name} must explicitly be {value}")
    if options.get("input_mode") not in {"text", "multimodal"}:
        raise ValueError("bench_cfg.input_mode must explicitly be text or multimodal")
    return {**expected, "input_mode": options["input_mode"]}


def official_sources(data_root: Path, *, diagnostic: bool = False) -> dict[str, str]:
    """Use shipped upstream LFS hashes, never local declarations as proof of origin."""
    pinned = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    metadata = data_root / "download-manifest.json"
    if not metadata.is_file():
        raise ValueError("Multimodal-Mind2Web requires download-manifest.json and official Parquet files")
    local = json.loads(metadata.read_text(encoding="utf-8"))
    if (local.get("repository") != DATASET_SOURCE
            or local.get("revision") != DATASET_REVISION):
        raise ValueError("dataset source/revision differs from pinned Multimodal-Mind2Web")
    paths = {str(p.relative_to(data_root)) for p in data_root.rglob("*.parquet")}
    expected = pinned["files"]
    if not paths or not paths <= expected.keys() or (not diagnostic and paths != expected.keys()):
        raise ValueError("official snapshot file inventory differs from pinned Multimodal-Mind2Web")
    hashes = {}
    for name in sorted(paths):
        path = data_root / name
        record = expected[name]
        actual = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        if actual != record or local.get("files", {}).get(name) != record:
            raise ValueError(f"official snapshot content mismatch: {name}")
        hashes[name] = actual["sha256"]
    return hashes


def split_for_file(name: str) -> str:
    return Path(name).name.split("-", 1)[0]


def action_rows(root: Path, files: dict, *, metadata_only: bool = False):
    import pyarrow.parquet as pq

    for name in files:
        source = pq.ParquetFile(root / name)
        for batch in source.iter_batches(batch_size=16,
                                         columns=METADATA_COLUMNS if metadata_only else None):
            for row in batch.to_pylist():
                # The pinned Arrow schema stores indices and each candidate as strings.
                if isinstance(row["target_action_index"], str):
                    row["target_action_index"] = int(row["target_action_index"])
                if not metadata_only:
                    row["operation"] = json.loads(row["operation"]) if isinstance(row["operation"], str) else row["operation"]
                    for field in ("pos_candidates", "neg_candidates"):
                        candidates = json.loads(row[field]) if isinstance(row[field], str) else row[field]
                        row[field] = [json.loads(item) if isinstance(item, str) else item for item in candidates]
                yield split_for_file(name), name, row


def row_metadata(row: dict) -> dict:
    return {key: row[key] for key in ("annotation_id", "website", "domain", "subdomain",
                                    "confirmed_task", "action_reprs")}


def validate_step_identity(row: dict) -> None:
    index = row["target_action_index"]
    if (type(index) is not int or index < 0 or index >= len(row["action_reprs"])
            or not isinstance(row["action_uid"], str) or not row["action_uid"]):
        raise ValueError(f"{row['annotation_id']}: invalid published action identity")
