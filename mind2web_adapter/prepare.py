"""Build the shared official conversion and export exactly the same split."""
import argparse
import logging
from pathlib import Path

from .common import write_json
from swarmevo.benches.mind2web.cache import ensure_cache, export_partition
from swarmevo.benches.mind2web.dataset import load_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--ranks', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    logging.basicConfig(level=logging.INFO)
    split = export_partition(a.data_root, a.seed)
    print('Official split counts:', split['counts'], flush=True)
    inventory = ensure_cache(a.data_root, a.ranks, a.cache_root)
    manifest, _ = load_inventory(inventory, seed=a.seed)
    assert manifest['parts'] == split['parts']
    assert manifest['split_sha256'] == split['split_sha256']
    write_json(a.output, {'inventory': str(inventory.resolve()), 'split': split,
                         'cache_identity': manifest['cache_identity']})
    print('Verified shared inventory:', inventory, flush=True)


if __name__ == '__main__':
    main()
