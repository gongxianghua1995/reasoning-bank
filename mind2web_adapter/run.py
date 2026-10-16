"""Run sequential learning or train/frozen evaluation on offline Mind2Web."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

from .common import Progress, digest, event, file_hash, load, write_json

ROOT = Path(__file__).resolve().parents[1]


def validate_protocol(options):
    if options['mode'] == 'stream':
        if options.get('epochs', 1) != 1:
            raise ValueError('Streaming visits each task once; epochs must be one')
        if options.get('memory_from') or options.get('case_ids_file'):
            raise ValueError('Streaming starts empty and cannot import memory or use evaluation shards')


def select_cases(cases, options):
    ids_file = options.get('case_ids_file')
    if ids_file:
        if options['mode'] != 'eval' or options.get('max_cases') is not None:
            raise ValueError('Explicit case IDs require evaluation without max_cases')
        ids = load(ids_file)
        allowed = {case.id for case in cases}
        if (not isinstance(ids, list) or not ids or
                not all(isinstance(cid, str) for cid in ids) or
                len(set(ids)) != len(ids) or not set(ids) <= allowed):
            raise ValueError('Case IDs must be a unique nonempty subset of the selected split')
        selected = set(ids)
        return [case for case in cases if case.id in selected]
    return cases[:options['max_cases']] if options.get('max_cases') is not None else cases


def recoverable_case_error(exc, learning=False):
    from .model import DeadlineExceeded, is_quota_error
    return not is_quota_error(exc) and (
        isinstance(exc, DeadlineExceeded) or
        type(exc).__name__ in {'APITimeoutError', 'APIConnectionError', 'RateLimitError', 'InternalServerError'} or
        (learning and isinstance(exc, ValueError)))


def record_failure(folder, case, episode, phase, exc, before, bank_path, config):
    if file_hash(bank_path) != before:
        raise ValueError('Failed case changed memory; stopping for inspection')
    actions = load(folder / 'predictions_local.json') if (folder / 'predictions_local.json').exists() else []
    checkpoint = load(folder / 'execution_checkpoint.json') if (folder / 'execution_checkpoint.json').exists() else {}
    failure = {'case_id': case.id, 'episode_id': episode, 'website': case.family_id,
               'phase': phase, 'error': type(exc).__name__, 'completed_steps': len(actions),
               'total_steps': len(load(folder / 'public_case.json')['observations']),
               'action_elapsed_seconds': checkpoint.get('elapsed_seconds'),
               'case_budget_exhausted': checkpoint.get('elapsed_seconds', 0) >= config['case_seconds'] - 5,
               'metrics_available': phase == 'learning', 'memory_unchanged': True}
    write_json(folder / 'failure.json', failure)
    event(folder, 'case_failed', **failure)
    return failure


def snapshot(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / 'mind2web_adapter', out / 'source_at_run/mind2web_adapter',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {str(f.relative_to(out / 'source_at_run')): file_hash(f)
              for f in (out / 'source_at_run').rglob('*') if f.is_file()}
    write_json(out / 'source_hashes.json', hashes)
    write_json(out / 'config.json', load(a.config))
    # Copy only the explicitly provided bank; never import other experiments implicitly.
    if a.memory_from:
        shutil.copy2(a.memory_from, out / 'memory.json')
        write_json(out / 'memory_source.json', {'path': str(a.memory_from.resolve()),
                   'sha256': file_hash(a.memory_from)})
    options = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()
               if k not in {'env_file', 'frozen', 'config'}}
    write_json(out / 'options.json', options)
    if a.resume_from:
        from .resume import restore_learning_failure
        restore_learning_failure(a.resume_from, out, load(a.config), options)
    os.environ['PYTHONPATH'] = str(out / 'source_at_run')
    os.chdir(out / 'source_at_run')
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'mind2web_adapter.run', '--frozen',
                             '--output', str(out)])


def execute_run(out):
    from .agent import execute, original_predictions, public_case
    from .memory import Bank, learn
    from swarmevo.benches.mind2web.dataset import load_inventory, read_case
    from swarmevo.benches.mind2web.metrics import aggregate, score_case

    options, config = load(out / 'options.json'), load(out / 'config.json')
    progress = Progress(out, mode=options['mode'], phase='preparing', completed=0)
    completed = []
    failures, outcomes = [], []
    try:
        validate_protocol(options)
        if config.get('feedback_source', 'self') != 'self' and options['mode'] != 'eval':
            raise ValueError('Official-feedback learning must use the aligned runner')
        streaming = options['mode'] == 'stream'
        learning_enabled = options['mode'] in {'train', 'stream'} and options['scope'] != 'none'
        recovery = load(out / 'resume.json') if options.get('resume_from') else None
        if recovery:
            completed = load(out / 'results.json')
            outcomes = load(out / 'case_outcomes.json') if (out / 'case_outcomes.json').exists() else [
                {'episode_id': r['episode_id'], 'status': 'complete'} for r in completed]
            failures = load(out / 'failures.json') if (out / 'failures.json').exists() else []
            progress.update(completed=len(completed), processed=len(outcomes), failed=len(failures),
                            recovered_prefix=len(outcomes))
        for name, value in load(out / 'source_hashes.json').items():
            if file_hash(out / 'source_at_run' / name) != value:
                raise ValueError('Frozen source changed')
        inventory = Path(options['inventory'])
        manifest, parts = load_inventory(inventory, seed=options['seed'])
        if manifest['diagnostic']:
            raise ValueError('Real runs require all four official splits')
        split = 'train' if options['mode'] == 'train' else options['split']
        cases = select_cases(parts[split], options)
        if not cases:
            raise ValueError('No selected cases')
        bank_identity = {'split_sha256': manifest['split_sha256'],
            'data_identity': digest(json.dumps(manifest['cache_identity'], sort_keys=True)),
            'embedding_model': config['embedding_model'], 'model': config['model'],
            'retrieval': 'web-query-cosine-top1-v1'}
        query_format = config.get('embedding_query_format', 'instruction')
        if query_format not in {'plain', 'instruction'}:
            raise ValueError('Unknown embedding query format')
        if query_format == 'plain':
            bank_identity['retrieval'] = 'web-query-cosine-top1-plain-v2'
        if config.get('feedback_source') == 'official':
            bank_identity['feedback_source'] = 'official'
        if streaming:
            bank_identity.update(memory_protocol='stream-empty-v1', stream_split=split,
                stream_case_order_sha256=digest(json.dumps([c.id for c in cases])))
        embedder = None
        embedding_files = {}
        if options['scope'] != 'none':
            from fastembed import TextEmbedding
            cache = Path(options['embedding_cache'])
            embedding_files = {str(f.relative_to(cache)): file_hash(f)
                               for f in cache.rglob('*') if f.is_file()}
            if not embedding_files:
                raise ValueError('No local embedding weights')
            os.environ['HF_HUB_OFFLINE'] = '1'
            embedder = TextEmbedding(config['embedding_model'], cache_dir=str(cache),
                    threads=config['embedding_threads'], local_files_only=True)
        bank = Bank(out / 'memory.json', embedder, bank_identity, options['scope'],
                    frozen=options['mode'] == 'eval', query_format=query_format)
        if options['mode'] in {'train', 'stream'} and bank.state['entries'] and not recovery:
            raise ValueError('Training/streaming must start from empty memory')
        if options['mode'] == 'eval' and options['scope'] != 'none' and not bank.state['entries']:
            raise ValueError('Frozen evaluation requires trained memory')
        allowed_ids = {c.id for c in cases} if streaming else set(manifest['parts']['train'])
        if any(e['case_id'] not in allowed_ids for e in bank.state['entries']):
            raise ValueError('Bank contains cases outside the allowed learning partition')
        if not (out / 'memory.json').exists():
            write_json(out / 'memory.json', bank.state)
        memory_initial = file_hash(out / 'memory.json')
        write_json(out / 'splits.json', {key: manifest[key] for key in
            ['seed', 'split_sha256', 'parts', 'counts', 'website_counts', 'source_files']})
        write_json(out / 'manifest.json', {'config': config, 'options': options,
            'selected_case_ids': [c.id for c in cases], 'split': split,
            'bank_identity': bank_identity, 'inventory_manifest_sha256': file_hash(inventory / 'manifest.json'),
            'cache_identity': manifest['cache_identity'], 'embedding_files': embedding_files,
            'memory_initial_sha256': memory_initial,
            'resume_from': recovery['source'] if recovery else None,
            'protocol': 'single-agent; offline fixed observation; ' +
                ('empty-bank sequential test-time learning; previous tasks only; ' if streaming else 'train-only memory; ') +
                'LLM self-judgment; no MaTTS',
            'learning_protocol': 'stream' if streaming else 'train_then_frozen',
            'case_order': 'selected split manifest order; training epochs >1 shuffle with seed+epoch-1',
            'paper_deviations': config.get('paper_deviations', [])})
        for epoch in range(1, options['epochs'] + 1 if options['mode'] == 'train' else 2):
            ordered = list(cases)
            if epoch > 1:
                random.Random(options['seed'] + epoch - 1).shuffle(ordered)
            for case in ordered:
                episode = f'epoch-{epoch}:{case.id}'
                if episode in {r['episode_id'] for r in outcomes}:
                    continue
                if streaming:
                    previous_ids = {r['episode_id'].split(':', 1)[1] for r in outcomes}
                    if any(e['case_id'] not in previous_ids for e in bank.state['entries']):
                        raise ValueError('Streaming memory contains current/future task information')
                folder = out / 'cases' / f'{epoch:02d}_{digest(case.id)[:16]}'
                folder.mkdir(parents=True, exist_ok=bool(recovery and episode == recovery['failed_episode']))
                progress.update(phase='retrieval', active=case.id, epoch=epoch, step=0)
                task = read_case(case)
                public = public_case(task)
                before = file_hash(out / 'memory.json')
                selected, text, vector, scores = bank.select(case.id, case.family_id, public['task'])
                reusing = recovery and episode == recovery['failed_episode']
                try:
                    if reusing:
                        from .resume import recovered_actions
                        partial = recovery.get('kind') == 'actions'
                        predictions = recovered_actions(folder, public, selected, text, before, partial=partial)
                        progress.update(step=len(predictions), total_steps=len(public['observations']))
                        event(folder, 'recovery_reused_actions', steps=len(predictions), source=recovery['source'])
                        if partial:
                            predictions = execute(public, config, folder, text, progress,
                                resume_steps=len(predictions), previous_elapsed=recovery['previous_elapsed'])
                    else:
                        write_json(folder / 'public_case.json', public)
                        write_json(folder / 'case_identity.json', {'case_id': case.id, 'website': case.family_id,
                            'source_sha256': case.payload['sha256'], 'episode_id': episode, 'split': split})
                        write_json(folder / 'retrieval.json', {'episode_ids': selected, 'selected_memory': text,
                            'scores': scores, 'memory_before_sha256': before, 'scope': options['scope']})
                        predictions = execute(public, config, folder, text, progress)
                except Exception as exc:
                    if not options.get('continue_case_errors') or not recoverable_case_error(exc):
                        raise
                    failures.append(record_failure(folder, case, episode, 'solve', exc, before, out / 'memory.json', config))
                    outcomes.append({'episode_id': episode, 'status': 'action_failed'})
                    write_json(out / 'failures.json', failures)
                    write_json(out / 'case_outcomes.json', outcomes)
                    progress.update(completed=len(completed), processed=len(outcomes), failed=len(failures))
                    continue
                # Learning sees only public observations/agent outputs. Compute scores afterwards.
                learning_error = None
                learning_exception = None
                try:
                    if learning_enabled:
                        progress.update(phase='learning')
                        entry = learn(public, folder, config, vector, episode)
                except Exception as exc:
                    learning_error = type(exc).__name__
                    learning_exception = exc
                else:
                    if learning_enabled:
                        bank.commit(entry)
                progress.update(phase='scoring')
                original = original_predictions(predictions, task)
                metrics = score_case(original, task)
                if reusing and recovery.get('kind', 'learning') == 'learning' and metrics != load(out / 'inherited_results.json')[-1]['metrics']:
                    raise ValueError('Recovered prediction scores changed')
                write_json(folder / 'predictions.json', original)
                write_json(folder / 'metrics.json', metrics)
                after = file_hash(out / 'memory.json')
                if options['mode'] == 'eval' and after != memory_initial:
                    raise ValueError('Frozen memory changed')
                result = {'case_id': case.id, 'episode_id': episode, 'website': case.family_id,
                    'metrics': metrics, 'memory_before_sha256': before, 'memory_after_sha256': after,
                    'learning_error': learning_error, 'selected_memory_ids': selected}
                write_json(folder / 'result.json', result)
                completed.append(result)
                write_json(out / 'results.json', completed)
                if learning_error:
                    if not options.get('continue_case_errors') or not recoverable_case_error(learning_exception, learning=True):
                        raise learning_exception
                    failures.append(record_failure(folder, case, episode, 'learning', learning_exception,
                                                   before, out / 'memory.json', config))
                    write_json(out / 'failures.json', failures)
                outcomes.append({'episode_id': episode, 'status': 'learning_failed' if learning_error else 'complete'})
                write_json(out / 'case_outcomes.json', outcomes)
                progress.update(completed=len(completed), processed=len(outcomes), failed=len(failures))
        if options['memory_from'] and file_hash(Path(options['memory_from'])) != load(out / 'memory_source.json')['sha256']:
            raise ValueError('Source memory changed during frozen evaluation')
        summary = {'status': 'finished_with_failures' if failures else 'finished', 'mode': options['mode'], 'split': split, 'scope': options['scope'],
                   'learning_protocol': 'stream' if streaming else 'train_then_frozen',
                   'metrics': aggregate([r['metrics'] for r in completed]),
                   'metrics_population': 'Fully predicted cases only; incomplete cases are excluded and reported in coverage.',
                   'processed': len(outcomes), 'failed': len(failures),
                   'coverage': {'scheduled': len(cases) * (options['epochs'] if options['mode'] == 'train' else 1),
                       'scored': len(completed), 'unscored': len(outcomes) - len(completed),
                       'learning_failed': sum(f['phase'] == 'learning' for f in failures),
                       'observed_task_success_over_processed': sum(r['metrics']['task_success_rate'] for r in completed) / len(outcomes)},
                   'memory_entries': len(bank.state['entries']),
                   'frozen_memory_unchanged': file_hash(out / 'memory.json') == memory_initial if options['mode'] == 'eval' else None,
                   'source_memory_unchanged': True if options['memory_from'] else None,
                   'completed': len(completed)}
        write_json(out / 'summary.json', summary)
        progress.close(status=summary['status'], phase=None, active=None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        event(out, 'run_error', error=type(exc).__name__)
        progress.close(status='error', error=type(exc).__name__)
        print('Run stopped:', type(exc).__name__, flush=True)
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'eval', 'stream'])
    p.add_argument('--scope', choices=['global', 'local', 'none'], default='global')
    p.add_argument('--inventory', type=Path)
    p.add_argument('--split', choices=['validation', 'test_task', 'test_website', 'test_domain'], default='validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-cases', type=int)
    p.add_argument('--case-ids-file', type=Path, help='Evaluation-only subset from the selected split, for independent frozen shards')
    p.add_argument('--memory-from', type=Path)
    p.add_argument('--resume-from', type=Path, help='Recover a failed train/stream action or learning stage into a new output')
    p.add_argument('--continue-case-errors', action='store_true', help='Record recoverable per-case failures and continue; quota/integrity errors remain fatal')
    p.add_argument('--embedding-cache', type=Path, default=ROOT / '.models')
    p.add_argument('--config', type=Path, help='Defaults to config_stream.json for stream, config.json otherwise')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.frozen:
        raise SystemExit(execute_run(a.output.resolve()))
    if not a.mode or not a.inventory or a.epochs < 1 or (a.max_cases is not None and a.max_cases < 1):
        p.error('mode, inventory and positive epochs/max-cases are required')
    try:
        validate_protocol(vars(a))
    except ValueError as exc:
        p.error(str(exc))
    if a.config is None:
        a.config = ROOT / 'mind2web_adapter' / ('config_stream.json' if a.mode == 'stream' else 'config.json')
    if (a.mode == 'train' or a.scope == 'none') and a.memory_from:
        p.error('Training and no-memory runs cannot import memory')
    if a.mode == 'eval' and a.scope != 'none' and not a.memory_from:
        p.error('Frozen memory evaluation requires --memory-from')
    if a.resume_from and (a.mode not in {'train', 'stream'} or a.scope == 'none' or a.memory_from):
        p.error('Recovery requires training/streaming with memory, without --memory-from')
    from dotenv import load_dotenv
    if a.env_file:
        load_dotenv(a.env_file.resolve(), override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY is missing')
    snapshot(a)


if __name__ == '__main__':
    main()
