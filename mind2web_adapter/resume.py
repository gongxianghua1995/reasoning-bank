"""Recover a failed training step or learning stage into an auditable new run."""
import json
from pathlib import Path
import random
import shutil
import time

from .common import digest, file_hash, load, write_json


def schedule(case_ids, epochs, seed):
    result = []
    for epoch in range(1, epochs + 1):
        ordered = list(case_ids)
        if epoch > 1:
            random.Random(seed + epoch - 1).shuffle(ordered)
        result.extend(f'epoch-{epoch}:{case_id}' for case_id in ordered)
    return result


def folder_name(episode):
    epoch, case_id = episode.split(':', 1)
    return f'{int(epoch.removeprefix("epoch-")):02d}_{digest(case_id)[:16]}'


def restore_learning_failure(source, out, config, options):
    source, out = Path(source).resolve(), Path(out).resolve()
    old = load(source / 'options.json')
    if options['mode'] not in {'train', 'stream'} or options['scope'] == 'none':
        raise ValueError('Learning recovery requires a memory training/streaming run')
    if load(source / 'status.json')['status'] != 'error':
        raise ValueError('Recovery requires a stopped, failed source run')
    for key in ('mode', 'scope', 'inventory', 'seed', 'epochs', 'max_cases', 'embedding_cache'):
        if old[key] != options[key]:
            raise ValueError(f'Recovery changed option: {key}')
    if options['mode'] == 'stream' and old.get('split') != options.get('split'):
        raise ValueError('Streaming recovery cannot change split')
    old_config = load(source / 'config.json')
    migration_path = out / 'source_at_run/mind2web_adapter/runtime_migration.json'
    migrations = load(migration_path) if migration_path.exists() else {}
    if old_config != config:
        compatible = ({k: v for k, v in old_config.items() if k != 'timeout_recoveries'} ==
                      {k: v for k, v in config.items() if k != 'timeout_recoveries'})
        if not (compatible and old_config.get('timeout_recoveries', 0) == 0 and
                config.get('timeout_recoveries') == 2 and migrations):
            raise ValueError('Recovery must use the same model and budgets')
    manifest = load(source / 'manifest.json')
    if file_hash(Path(options['inventory']) / 'manifest.json') != manifest['inventory_manifest_sha256']:
        raise ValueError('Recovery inventory changed')
    weights = {str(p.relative_to(Path(options['embedding_cache']))): file_hash(p)
               for p in Path(options['embedding_cache']).rglob('*') if p.is_file()}
    if weights != manifest['embedding_files']:
        raise ValueError('Recovery embedding weights changed')
    orchestration = {'run.py', 'batch.py', 'resume.py', 'summarize.py'}
    changes = []
    for name, expected in load(source / 'source_hashes.json').items():
        if file_hash(source / 'source_at_run' / name) != expected:
            raise ValueError('Source execution snapshot changed')
        current = out / 'source_at_run' / name
        if not current.exists() or file_hash(current) != expected:
            changes.append(name)
            approved = migrations.get(name, {})
            runtime_upgrade = (approved.get('before') == expected and current.exists() and
                               approved.get('after') == file_hash(current))
            if not (name.startswith('mind2web_adapter/tests/') or runtime_upgrade or
                    name in {f'mind2web_adapter/{n}' for n in orchestration}):
                raise ValueError(f'Recovery changes method code: {name}')
    rows = load(source / 'results.json') if (source / 'results.json').exists() else []
    prior_failures = load(source / 'failures.json') if (source / 'failures.json').exists() else []
    if (source / 'case_outcomes.json').exists():
        outcomes = load(source / 'case_outcomes.json')
    else:
        legacy_prefix = rows[:-1] if rows and rows[-1]['learning_error'] else rows
        outcomes = [{'episode_id': r['episode_id'], 'status': 'complete'} for r in legacy_prefix]
    handled_ids = [o['episode_id'] for o in outcomes]
    unfinished = [r for r in rows if r['episode_id'] not in handled_ids]
    learning = bool(unfinished)
    if learning and (len(unfinished) != 1 or not unfinished[0]['learning_error']):
        raise ValueError('Unaccounted scored cases outside outcome ledger')
    prefix = [r for r in rows if r['episode_id'] in handled_ids]
    ordered = schedule(manifest['selected_case_ids'], options['epochs'], options['seed'])
    if handled_ids != ordered[:len(handled_ids)]:
        raise ValueError('Outcomes are not a consecutive training prefix')
    if len(handled_ids) >= len(ordered):
        raise ValueError('No unfinished episode')
    failed_episode = ordered[len(handled_ids)]
    scored_ids = [o['episode_id'] for o in outcomes if o['status'] != 'action_failed']
    if [r['episode_id'] for r in rows] != scored_ids + ([failed_episode] if learning else []):
        raise ValueError('Scored results differ from outcome ledger')
    if {f['episode_id'] for f in prior_failures} != {o['episode_id'] for o in outcomes if o['status'] != 'complete'}:
        raise ValueError('Failure ledger differs from outcome ledger')
    if load(source / 'status.json').get('active') != failed_episode.split(':', 1)[1]:
        raise ValueError('Failure is not at the expected next episode')
    memory = load(source / 'memory.json')
    learned = [r for r in prefix if not r['learning_error']]
    if [e['episode_id'] for e in memory['entries']] != [r['episode_id'] for r in learned]:
        raise ValueError('Memory does not match the successfully learned prefix')
    memory_by_episode = {e['episode_id']: e for e in memory['entries']}
    memory_hash = file_hash(source / 'memory.json')
    failed_source = source / 'cases' / folder_name(failed_episode)
    if load(failed_source / 'retrieval.json')['memory_before_sha256'] != memory_hash:
        raise ValueError('Failed learning modified memory')
    if learning and rows[-1]['memory_after_sha256'] != memory_hash:
        raise ValueError('Failed learning modified memory')
    expected_folders = {folder_name(eid) for eid in handled_ids} | {folder_name(failed_episode)}
    if {p.name for p in (source / 'cases').iterdir() if p.is_dir()} != expected_folders:
        raise ValueError('Source contains unaccounted case folders')
    for row in rows:
        folder = source / 'cases' / folder_name(row['episode_id'])
        if load(folder / 'result.json') != row:
            raise ValueError('Per-case result differs from result ledger')
        generation = load(folder / 'generation.json')
        if generation['status'] != 'complete' or generation['steps'] != row['metrics']['n_steps']:
            raise ValueError('Recovery requires complete action generation')
        if row['episode_id'] in memory_by_episode and load(folder / 'memory_entry.json') != memory_by_episode[row['episode_id']]:
            raise ValueError('Memory entries differ from case artifacts')
    for failure in prior_failures:
        if load(source / 'cases' / folder_name(failure['episode_id']) / 'failure.json') != failure:
            raise ValueError('Per-case failure differs from ledger')
    actions = load(failed_source / 'predictions_local.json') if (failed_source / 'predictions_local.json').exists() else []
    trajectory = load(failed_source / 'trajectory.json') if (failed_source / 'trajectory.json').exists() else []
    public = load(failed_source / 'public_case.json')
    if ([s.get('final_action') for s in trajectory[:len(actions)]] != actions or
            len(trajectory) < len(actions) or len(actions) > len(public['observations'])):
        raise ValueError('Invalid completed action prefix')
    if not learning and (failed_source / 'generation.json').exists():
        raise ValueError('Uncommitted completed case requires separate inspection')
    previous_elapsed = 0
    if not learning:
        if (failed_source / 'execution_checkpoint.json').exists():
            previous_elapsed = load(failed_source / 'execution_checkpoint.json')['elapsed_seconds']
        else:
            events = [json.loads(line) for line in (failed_source / 'events.jsonl').read_text().splitlines()]
            starts = [e['time'] for e in events if e['event'] == 'step_started']
            previous_elapsed = events[-1]['time'] - starts[0] if starts else 0
        if previous_elapsed >= config['case_seconds'] and not options.get('continue_case_errors'):
            raise ValueError('Original case time budget exhausted; cannot renew it')
    hashes = {str(p.relative_to(source)): file_hash(p) for p in (source / 'cases').rglob('*') if p.is_file()}
    hashes.update({name: file_hash(source / name) for name in
                   ('memory.json', 'results.json', 'status.json', 'manifest.json', 'options.json', 'config.json',
                    'case_outcomes.json', 'failures.json')
                   if (source / name).exists()})
    shutil.copytree(source / 'cases', out / 'cases')
    shutil.copy2(source / 'memory.json', out / 'memory.json')
    for name, expected in hashes.items():
        if file_hash(source / name) != expected:
            raise ValueError('Source artifacts changed during recovery copy')
        if name.startswith('cases/') or name == 'memory.json':
            if file_hash(out / name) != expected:
                raise ValueError('Recovery copy differs from source')
    failed = out / 'cases' / folder_name(failed_episode)
    kind = 'learning' if learning else 'actions'
    archive = failed / f'{kind}_recovery_{len(list(failed.glob(f"{kind}_recovery_*"))) + 1:02d}'
    archive.mkdir()
    if learning:
        for pattern in ('judge_request_*.json', 'extract_request_*.json', 'memory_raw_attempt_*.md',
                        'self_judgment.json', 'memory_entry.json', 'result.json'):
            for p in failed.glob(pattern):
                shutil.move(str(p), archive / p.name)
    else:
        for p in (failed / 'requests').glob('*.json'):
            if int(p.name.split('_')[0]) >= len(actions):
                shutil.move(str(p), archive / p.name)
        if (failed / 'trajectory.json').exists():
            shutil.copy2(failed / 'trajectory.json', archive / 'trajectory.json')
        write_json(failed / 'trajectory.json', trajectory[:len(actions)])
        write_json(failed / 'predictions_local.json', actions)
    write_json(out / 'inherited_results.json', rows)
    write_json(out / 'results.json', prefix)
    write_json(out / 'case_outcomes.json', outcomes)
    write_json(out / 'failures.json', prior_failures)
    recovery = {'source': str(source), 'created': time.time(), 'source_artifact_hashes': hashes,
                'changed_source_files': changes, 'completed_prefix': len(outcomes), 'scored_prefix': len(prefix), 'kind': kind,
                'failed_episode': failed_episode, 'reused_steps': len(actions), 'previous_elapsed': previous_elapsed,
                'config_changes': {k: {'before': old_config.get(k), 'after': v}
                                   for k, v in config.items() if old_config.get(k) != v},
                'source_elapsed_seconds': load(source / 'status.json')['elapsed_seconds'],
                'policy': 'Reuse completed actions; preserve case time consumed and all calls/usage; learning recovery gets fresh phase budgets.'}
    write_json(out / 'resume.json', recovery)
    return recovery


def recovered_actions(folder, public, selected, text, before, partial=False):
    folder = Path(folder)
    if load(folder / 'public_case.json') != public:
        raise ValueError('Recovery public observations changed')
    retrieval = load(folder / 'retrieval.json')
    if (retrieval['episode_ids'] != selected or retrieval['selected_memory'] != text or
            retrieval['memory_before_sha256'] != before):
        raise ValueError('Recovery retrieval differs from the original trajectory')
    actions = load(folder / 'predictions_local.json')
    trajectory = load(folder / 'trajectory.json')
    if ((len(actions) != len(public['observations']) and not partial) or
            len(actions) > len(public['observations']) or [s['final_action'] for s in trajectory] != actions):
        raise ValueError('Recovery trajectory/actions are incomplete or inconsistent')
    for i, step in enumerate(trajectory):
        if step['step'] != i or step['observation'] != public['observations'][i]:
            raise ValueError('Recovery step observations changed')
    return actions
