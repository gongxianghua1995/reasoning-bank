"""Evaluate a completed first-epoch bank and its matched no-memory control."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time

from .common import load, write_json, file_hash
from .campaign import ROOT, TEST_SPLITS, execute


def prepare(args):
    from swarmevo.benches.mind2web.dataset import load_inventory
    inventory = args.inventory.resolve()
    manifest, parts = load_inventory(inventory, seed=42)
    counts = {'train':303, 'validation':101, 'test_task':177,
              'test_website':142, 'test_domain':694}
    if manifest['diagnostic'] or any(len(parts[k]) != n for k,n in counts.items()):
        raise ValueError('Complete unified seed42 inventory required')
    source = args.memory_from.resolve()
    bank, checkpoint = load(source), load(source.with_name('summary.json'))
    entries = bank['entries']
    if (checkpoint['epoch'] != 1 or checkpoint['memory_sha256'] != file_hash(source)
            or bank['identity']['split_sha256'] != manifest['split_sha256']
            or len(entries) != 303
            or {e['case_id'] for e in entries} != set(manifest['parts']['train'])
            or any(e['episode_id'] != 'epoch-1:' + e['case_id'] for e in entries)):
        raise ValueError('Expected complete first-epoch bank from the same training split')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'source_at_launch'
    shutil.copytree(ROOT/'mind2web_adapter', frozen/'mind2web_adapter',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(out/'source_hashes.json', {str(p.relative_to(frozen)):file_hash(p)
                                        for p in frozen.rglob('*') if p.is_file()})
    for name in ('aligned', 'stream'):
        write_json(out/f'config_{name}.json', load(ROOT/f'mind2web_adapter/config_{name}.json'))
    write_json(out/'splits.json', {k:manifest[k] for k in ('parts','counts','split_sha256','source_files')})
    shutil.copy2(source, out/'selected_memory.json')
    write_json(out/'selection.json', {'selected_epoch':1, 'memory_sha256':file_hash(source),
        'source_memory':str(source), 'source_checkpoint':checkpoint,
        'rule':'Fixed completed epoch 1 by user instruction; no validation or test selection; epoch 2 excluded.'})
    base = [sys.executable, '-u', '-m', 'mind2web_adapter.run', '--inventory', str(inventory),
            '--seed', '42', '--embedding-cache', str(args.embedding_cache.resolve()),
            '--continue-case-errors', '--mode', 'eval']
    jobs = []
    def shards(group, split, depends=()):
        names = []
        for i in range(8):
            ids = manifest['parts'][split][i::8]
            name = f'{group}_{split}_{i:02d}'
            names.append(name)
            ids_file = out/'case_lists'/f'{name}.json'
            write_json(ids_file, ids)
            cmd = base + ['--scope', 'global' if group=='aligned' else 'none', '--split', split,
                '--case-ids-file', str(ids_file), '--config', str(out/'config_aligned.json'),
                '--output', str(out/name)]
            if group == 'aligned':
                cmd += ['--memory-from', str(out/'selected_memory.json')]
            jobs.append({'name':name, 'group':group, 'split':split, 'command':cmd,
                'depends':list(depends), 'slots':1, 'expected':len(ids),
                'case_ids':ids, 'output':str(out/name)})
        return names
    validation = shards('aligned', 'validation')
    for split in TEST_SPLITS:
        shards('aligned', split, depends=validation)
    for split in ('validation', *TEST_SPLITS):
        shards('none', split)
    plan = {'created':time.time(), 'source_root':str(frozen), 'inventory':str(inventory),
        'inventory_sha256':file_hash(inventory/'manifest.json'), 'split_sha256':manifest['split_sha256'],
        'workers':args.workers, 'training_workers':0, 'jobs':jobs,
        'expected_episodes':sum(j['expected'] for j in jobs),
        'protocol':'Unified substrate; fixed epoch-1 ReasoningBank official-feedback bank; frozen validation/tests and paired no-memory control.',
        'model_note':'Text, thinking enabled, 64000 actor output cap; both groups use identical actor configuration.'}
    write_json(out/'plan.json', plan)
    inputs = [out/'plan.json', out/'config_aligned.json', out/'config_stream.json', out/'splits.json',
              out/'selected_memory.json', out/'selection.json', *sorted((out/'case_lists').glob('*.json'))]
    write_json(out/'input_hashes.json', {str(p.relative_to(out)):file_hash(p) for p in inputs})
    write_json(out/'status.json', {'status':'prepared', 'updated':time.time()})
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--memory-from', type=Path)
    p.add_argument('--embedding-cache', type=Path, default=ROOT/'.models')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if not a.execute:
        if not a.inventory or not a.memory_from or not 1 <= a.workers <= 16:
            p.error('Inventory, first-epoch checkpoint and 1..16 workers required')
        prepare(a)
    from dotenv import load_dotenv
    if a.env_file:
        load_dotenv(a.env_file, override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing')
    os.environ['REASONINGBANK_RATE_FILE'] = str(out/'request_admission.txt')
    frozen = out/'source_at_launch'
    if ROOT != frozen:
        os.environ['PYTHONPATH'] = str(frozen)
        os.chdir(frozen)
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'mind2web_adapter.frozen_campaign',
                                  '--execute', '--output', str(out)])
    raise SystemExit(execute(out))


if __name__ == '__main__':
    main()
