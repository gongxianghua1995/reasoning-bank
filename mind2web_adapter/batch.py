"""Run the complete fixed Mind2Web campaign from a frozen source snapshot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import file_hash, load, write_json

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('validation', 'test_task', 'test_website', 'test_domain')


def prepare(args):
    from swarmevo.benches.mind2web.dataset import load_inventory

    inventory = args.inventory.resolve()
    manifest, parts = load_inventory(inventory, seed=42)
    if manifest['diagnostic'] or manifest['counts']['train'] != 303:
        raise ValueError('Complete official seed-42 inventory required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'source_at_launch'
    shutil.copytree(ROOT / 'mind2web_adapter', frozen / 'mind2web_adapter',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(out / 'source_hashes.json', {
        str(p.relative_to(frozen)): file_hash(p) for p in frozen.rglob('*') if p.is_file()})
    write_json(out / 'config.json', load(args.config))
    base = [sys.executable, '-u', '-m', 'mind2web_adapter.run',
            '--inventory', str(inventory), '--seed', '42',
            '--continue-case-errors',
            '--config', str(out / 'config.json'),
            '--embedding-cache', str(args.embedding_cache.resolve())]
    stages = []

    def stage(name, split, scope, options, repeats=1):
        stages.append({'name': name, 'split': split, 'scope': scope,
                       'expected_episodes': len(parts[split]) * repeats,
                       'expected_steps': sum(manifest['tasks'][c.id]['n_steps']
                                             for c in parts[split]) * repeats,
                       'output': str(out / name),
                       'command': base + options + ['--scope', scope, '--output', str(out / name)]})

    train_options = ['--mode', 'train', '--epochs', str(args.epochs)]
    if args.resume_train_from:
        train_options += ['--resume-from', str(args.resume_train_from.resolve())]
    stage('train', 'train', 'global', train_options, args.epochs)
    for scope in ('global', 'none'):
        for split in SPLITS:
            options = ['--mode', 'eval', '--split', split]
            if scope == 'global':
                options += ['--memory-from', str(out / 'train/memory.json')]
            stage(f'{"frozen" if scope == "global" else "none"}_{split}', split, scope, options)
    plan = {'created': time.time(), 'seed': 42, 'epochs': args.epochs,
            'python': sys.executable, 'source_root': str(frozen),
            'inventory': str(inventory), 'inventory_sha256': file_hash(inventory / 'manifest.json'),
            'split_sha256': manifest['split_sha256'], 'counts': manifest['counts'],
            'resume_train_from': str(args.resume_train_from.resolve()) if args.resume_train_from else None,
            'config_sha256': file_hash(out / 'config.json'), 'stages': stages,
            'expected_episodes': sum(s['expected_episodes'] for s in stages),
            'expected_steps': sum(s['expected_steps'] for s in stages),
            'failure_policy': 'Continue recoverable case failures, report coverage and incomplete cases; stop on quota/auth/integrity failures.'}
    write_json(out / 'plan.json', plan)
    write_json(out / 'batch_status.json', {'status': 'prepared', 'updated': time.time(),
               'expected_episodes': plan['expected_episodes'], 'completed_episodes': 0})
    return plan


def execute(out):
    plan = load(out / 'plan.json')
    lock = (out / '.driver.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if load(out / 'batch_status.json')['status'] != 'prepared':
            raise ValueError('Batch already started; existing runs cannot be overwritten')
    except BaseException:
        lock.close()
        raise
    started = time.time()
    process = None
    finished = []
    active = None

    def status(state, **extra):
        value = {'status': state, 'updated': time.time(), 'started': started,
                 'elapsed_seconds': round(time.time() - started, 2), 'pid': os.getpid(),
                 'active_stage': active['name'] if active else None,
                 'finished_stages': [s['name'] for s in finished],
                 'expected_episodes': plan['expected_episodes'],
                 'completed_episodes': sum(load(Path(s['output']) / 'summary.json')['completed'] for s in finished),
                 'processed_episodes': sum(s['expected_episodes'] for s in finished),
                 'failed_episodes': sum(load(Path(s['output']) / 'summary.json').get('failed', 0) for s in finished), **extra}
        if active and (Path(active['output']) / 'status.json').exists():
            value['active_progress'] = load(Path(active['output']) / 'status.json')
            value['completed_episodes'] += value['active_progress'].get('completed', 0)
            value['processed_episodes'] += value['active_progress'].get('processed', value['active_progress'].get('completed', 0))
            value['failed_episodes'] += value['active_progress'].get('failed', 0)
        write_json(out / 'batch_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for active in plan['stages']:
            for name, expected in load(out / 'source_hashes.json').items():
                if file_hash(Path(plan['source_root']) / name) != expected:
                    raise ValueError('Launch source snapshot changed')
            if file_hash(out / 'config.json') != plan['config_sha256']:
                raise ValueError('Batch configuration changed')
            if file_hash(Path(plan['inventory']) / 'manifest.json') != plan['inventory_sha256']:
                raise ValueError('Inventory manifest changed')
            status('running')
            with (out / f'{active["name"]}.console.log').open('x') as log:
                process = subprocess.Popen(active['command'], cwd=plan['source_root'],
                    stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
                while process.poll() is None:
                    status('running', child_pid=process.pid)
                    time.sleep(15)
            code = process.returncode
            process = None
            if code:
                status('error', reason='Stage exited before completion', exit_code=code)
                return code if code > 0 else 1
            summary = load(Path(active['output']) / 'summary.json')
            if (summary['status'] not in {'finished', 'finished_with_failures'} or
                    summary.get('processed', summary['completed']) != active['expected_episodes']):
                raise ValueError('Stage coverage does not match the plan')
            with (out / f'{active["name"]}.audit.log').open('x') as log:
                subprocess.run([plan['python'], '-m', 'mind2web_adapter.summarize', active['output']],
                    cwd=plan['source_root'], stdout=log, stderr=log, check=True)
            finished.append(active)
            write_json(out / 'results.json', [
                {'stage': s['name'], 'summary': load(Path(s['output']) / 'summary.json')}
                for s in finished])
        active = None
        status('finished_with_failures' if any(load(Path(s['output']) / 'summary.json').get('failed', 0)
                                             for s in finished) else 'finished')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        status('error', error=type(exc).__name__)
        return 1
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'mind2web_adapter/config.json')
    parser.add_argument('--embedding-cache', type=Path, default=ROOT / '.models')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--resume-train-from', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        if not args.inventory or args.epochs < 1:
            parser.error('inventory and positive epochs required')
        plan = prepare(args)
        if args.prepare_only:
            print(json.dumps({'output': str(args.output.resolve()),
                  'episodes': plan['expected_episodes'], 'steps': plan['expected_steps']}))
            return
    from dotenv import load_dotenv
    if args.env_file:
        load_dotenv(args.env_file.resolve(), override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('OPENAI_API_KEY is missing')
    out = args.output.resolve()
    frozen = out / 'source_at_launch'
    if ROOT != frozen:
        os.environ['PYTHONPATH'] = str(frozen)
        os.chdir(frozen)
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'mind2web_adapter.batch',
                                 '--execute', '--output', str(out)])
    raise SystemExit(execute(out))


if __name__ == '__main__':
    main()
