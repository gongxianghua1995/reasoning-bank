"""Freeze epoch-one memory and evaluate the full validation split in independent shards."""
import argparse
from collections import Counter, defaultdict
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
from swarmevo.benches.mind2web.dataset import load_inventory
from swarmevo.benches.mind2web.metrics import aggregate

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows, scheduled):
    ms = [r['metrics'] for r in rows]
    steps = sum(m['n_steps'] for m in ms)
    correct = sum(sum(s['step_success_rate'] for s in m['steps']) for m in ms)
    successes = sum(m['task_success_rate'] for m in ms)
    return {'macro': aggregate(ms), 'scored': len(ms), 'scheduled': scheduled,
            'unscored': scheduled - len(ms), 'steps': steps, 'correct_steps': correct,
            'micro_step_sr': correct / steps if steps else None,
            'strict_successes': successes,
            'observed_strict_success_over_scheduled': successes / scheduled}


def calls(folders):
    usage, requests, responses, errors = defaultdict(Counter), Counter(), Counter(), Counter()
    for folder in folders:
        path = folder / 'events.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer can have an incomplete final line.
            phase = row.get('phase', 'unknown')
            if row['event'] == 'model_request':
                requests[phase] += 1
            elif row['event'] == 'model_response':
                responses[phase] += 1
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    usage[phase][key] += (row.get('usage') or {}).get(key, 0) or 0
            elif row['event'] == 'model_error':
                errors[row['error']] += 1
    return {'requests': dict(requests), 'responses': dict(responses), 'errors': dict(errors),
            'returned_usage_by_phase': {p: dict(u) for p, u in usage.items()},
            'note': 'Returned usage only; requests without usage may still incur provider cost. No currency estimate.'}


def prepare(source, out, workers):
    opts = load(source / 'options.json')
    manifest, parts = load_inventory(Path(opts['inventory']), seed=42)
    if manifest['diagnostic'] or len(parts['validation']) != 101:
        raise ValueError('Full official validation inventory required')
    rows = [r for r in load(source / 'results.json') if r['episode_id'].startswith('epoch-1:')]
    outcomes = [r for r in load(source / 'case_outcomes.json') if r['episode_id'].startswith('epoch-1:')]
    failures = [r for r in load(source / 'failures.json') if r['episode_id'].startswith('epoch-1:')]
    expected = [f'epoch-1:{c.id}' for c in parts['train']]
    if [r['episode_id'] for r in outcomes] != expected:
        raise ValueError('Epoch one is not completely processed')
    bank = load(source / 'memory.json')
    excluded = [e for e in bank['entries'] if not e['episode_id'].startswith('epoch-1:')]
    bank['entries'] = [e for e in bank['entries'] if e['episode_id'].startswith('epoch-1:')]
    learned = [r['episode_id'] for r in rows if not r.get('learning_error')]
    if [e['episode_id'] for e in bank['entries']] != learned:
        raise ValueError('Memory entries do not match learned epoch-one outcomes')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'epoch1_memory.json', bank)
    boundary_hash = rows[-1]['memory_after_sha256']
    if file_hash(out / 'epoch1_memory.json') != boundary_hash:
        raise ValueError('Reconstructed memory is not byte-identical to epoch-one boundary')
    later = [r for r in load(source / 'results.json') if r['episode_id'].startswith('epoch-2:')]
    if later and later[0]['memory_before_sha256'] != boundary_hash:
        raise ValueError('Epoch-two initial memory differs')
    write_json(out / 'epoch1_results.json', rows)
    write_json(out / 'epoch1_failures.json', failures)
    write_json(out / 'epoch1_outcomes.json', outcomes)
    train = {**metrics(rows, len(expected)), 'processed': len(outcomes), 'failed': len(failures),
             'memory_entries': len(bank['entries']),
             'memory_items': sum(len(e['memory_items']) for e in bank['entries']),
             'usage': calls(sorted((source / 'cases').glob('01_*')))}
    write_json(out / 'epoch1_summary.json', train)
    write_json(out / 'memory_provenance.json', {'source': str(source),
        'source_memory_sha256': file_hash(source / 'memory.json'),
        'epoch1_memory_sha256': boundary_hash, 'epoch1_boundary_hash_matches': True,
        'epoch2_initial_hash_matches': bool(later), 'excluded_epoch2_entries': len(excluded),
        'excluded_epoch2_episode_ids': [e['episode_id'] for e in excluded],
        'source_results_sha256': file_hash(source / 'results.json'),
        'epoch2_cost_excluded_from_epoch1_training_cost': True})
    frozen = out / 'source_at_launch'
    shutil.copytree(ROOT / 'mind2web_adapter', frozen / 'mind2web_adapter',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(out / 'source_hashes.json', {str(p.relative_to(frozen)): file_hash(p)
               for p in frozen.rglob('*') if p.is_file()})
    shutil.copy2(source / 'config.json', out / 'config.json')
    # Assign complete tasks by estimated steps; evaluation cannot write memory.
    bins, weights = [[] for _ in range(workers)], [0] * workers
    for case in sorted(parts['validation'], key=lambda c: -manifest['tasks'][c.id]['n_steps']):
        i = min(range(workers), key=lambda j: weights[j])
        bins[i].append(case.id)
        weights[i] += manifest['tasks'][case.id]['n_steps']
    jobs = []
    for i, ids in enumerate(bins):
        ids_file = out / f'case_ids_{i:02d}.json'
        write_json(ids_file, ids)
        for scope in ('global', 'none'):
            name = f'{scope}_{i:02d}'
            command = [sys.executable, '-u', '-m', 'mind2web_adapter.run', '--mode', 'eval',
                '--scope', scope, '--split', 'validation', '--seed', '42',
                '--inventory', opts['inventory'], '--config', str(out / 'config.json'),
                '--embedding-cache', opts['embedding_cache'], '--continue-case-errors',
                '--case-ids-file', str(ids_file), '--output', str(out / name)]
            if scope == 'global':
                command += ['--memory-from', str(out / 'epoch1_memory.json')]
            jobs.append({'name': name, 'scope': scope, 'case_ids': ids,
                         'steps': weights[i], 'command': command})
    plan = {'created': time.time(), 'source_training': str(source), 'workers': workers,
            'memory_sha256': boundary_hash, 'split_sha256': manifest['split_sha256'],
            'validation_ids': [c.id for c in parts['validation']],
            'validation_steps': sum(weights), 'jobs': jobs,
            'protocol': 'One training epoch; frozen validation and no-memory validation; no official test; unchanged model, prompts, action budgets and scoring.'}
    write_json(out / 'plan.json', plan)
    write_json(out / 'status.json', {'status': 'prepared', 'updated': time.time()})
    return plan


def collect(out, plan, final=False):
    groups = {}
    for scope in ('global', 'none'):
        rows, outcomes, failures, folders, selected = [], [], [], [], []
        for job in plan['jobs']:
            if job['scope'] != scope:
                continue
            folder = out / job['name']
            for name, target in [('results.json', rows), ('case_outcomes.json', outcomes), ('failures.json', failures)]:
                if (folder / name).exists():
                    target.extend(load(folder / name))
            folders.extend((folder / 'cases').glob('*'))
            if (folder / 'manifest.json').exists():
                manifest = load(folder / 'manifest.json')
                if set(manifest['selected_case_ids']) != set(job['case_ids']):
                    raise ValueError('Shard selected wrong cases')
                selected.extend(manifest['selected_case_ids'])
                if file_hash(folder / 'memory.json') != manifest['memory_initial_sha256']:
                    raise ValueError('Evaluation modified memory')
                if scope == 'global' and manifest['memory_initial_sha256'] != plan['memory_sha256']:
                    raise ValueError('Shard imported wrong memory')
            if final:
                summary = load(folder / 'summary.json')
                if summary['status'] not in ('finished', 'finished_with_failures'):
                    raise ValueError('Shard not finished')
                if not summary['frozen_memory_unchanged']:
                    raise ValueError('Frozen memory changed')
        ids = [r['case_id'] for r in rows]
        handled = [r['episode_id'].split(':', 1)[1] for r in outcomes]
        if len(ids) != len(set(ids)) or len(handled) != len(set(handled)):
            raise ValueError('Duplicate case in merged results')
        if not set(ids) <= set(plan['validation_ids']) or not set(handled) <= set(plan['validation_ids']):
            raise ValueError('Case outside validation')
        if final and (set(handled) != set(plan['validation_ids']) or
                      len(selected) != len(set(selected)) or set(selected) != set(plan['validation_ids'])):
            raise ValueError('Incomplete shard coverage')
        order = {cid: i for i, cid in enumerate(plan['validation_ids'])}
        rows.sort(key=lambda r: order[r['case_id']])
        groups[scope] = {**metrics(rows, 101), 'processed': len(outcomes), 'failed': len(failures),
                         'usage': calls(folders)}
        write_json(out / f'{scope}_results.json', rows)
        write_json(out / f'{scope}_failures.json', failures)
    global_rows = {r['case_id']: r['metrics'] for r in load(out / 'global_results.json')}
    none_rows = {r['case_id']: r['metrics'] for r in load(out / 'none_results.json')}
    common = sorted(global_rows.keys() & none_rows.keys())
    paired = {'n_cases': len(common), 'case_ids': common}
    if common:
        for metric in ('step_success_rate', 'element_accuracy', 'operation_f1', 'task_success_rate'):
            paired[metric + '_delta'] = sum(global_rows[c][metric] - none_rows[c][metric] for c in common) / len(common)
    result = {'complete': final, 'updated': time.time(), 'groups': groups, 'paired': paired,
              'training': load(out / 'epoch1_summary.json'),
              'interpretation': 'Macro metrics on fully predicted cases; report coverage and paired common cases. Validation only, not final test performance.'}
    write_json(out / 'comparison.json', result)
    render(out, result)
    return result


def render(out, result):
    def pct(v):
        return '—' if v is None else f'{100*v:.2f}%'
    train = result['training']
    lines = ['# ReasoningBank 一轮训练验证集实验', '',
             '状态：' + ('已完成验证集双组执行。' if result['complete'] else '验证执行中，以下为阶段性结果。'), '',
             '第一轮训练记忆经边界哈希核验恢复，排除第二轮经历。单 Agent、离线固定观察、global cosine top-1、BGE-small、本地记忆；模型 DeepSeek-V4-Flash-0731。', '',
             '## 第一轮训练', '',
             f"303 条已处理，{train['scored']} 条完整评分，{train['failed']} 条失败留档；{train['memory_entries']} 份经历、{train['memory_items']} 条经验。宏平均 Step SR {pct(train['macro'].get('step_success_rate'))}。", '',
             '## 验证结果', '',
             '| 组别 | 已处理 / 计划 | 完整评分 | 失败 | Step SR 宏平均 | Element Acc | Operation F1 | Task SR 完整子集 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope, title in [('global', '第一轮冻结记忆'), ('none', '无记忆')]:
        g = result['groups'][scope]
        m = g['macro']
        lines.append(f"| {title} | {g['processed']}/101 | {g['scored']} | {g['failed']} | {pct(m.get('step_success_rate'))} | {pct(m.get('element_accuracy'))} | {pct(m.get('operation_f1'))} | {pct(m.get('task_success_rate'))} |")
    paired = result['paired']
    lines += ['', f"两组共同完整评分 {paired['n_cases']} 题；该共同集合的记忆组减无记忆组 Step SR 差值为 {pct(paired.get('step_success_rate_delta'))}（百分比点）。", '',
              '未完整预测任务单独报告，不补造完整分数。覆盖不足时表内准确率仅代表完成子集；两组失败集合不同须结合共同集合比较。离线整题成功要求所有标注步骤预测正确，不代表真实浏览器任务完成。', '',
              '## 调用成本', '',
              '| 阶段 | 已返回输入 tokens | 已返回输出 tokens | 模型请求次数 |',
              '|---|---:|---:|---:|']
    for title, usage in [('第一轮训练', train['usage'])] + [(scope, g['usage']) for scope, g in result['groups'].items()]:
        phases = usage['returned_usage_by_phase'].values()
        lines.append(f"| {title} | {sum(p.get('prompt_tokens',0) for p in phases)} | {sum(p.get('completion_tokens',0) for p in usage['returned_usage_by_phase'].values())} | {sum(usage['requests'].values())} |")
    lines += ['', '成本仅统计已返回 usage；超时请求可能仍产生服务端费用。第二轮废弃执行成本另留在原训练目录，不计入一轮方案。', '',
              '本报告使用完整 101 条验证清单，不使用官方测试集；单次运行不足以证明统计稳定的收益。', '',
              '## 产物', '',
              '- plan.json：完整 case ID、分片和执行命令。',
              '- memory_provenance.json：第一轮边界哈希与剔除的第二轮经历。',
              '- comparison.json：汇总、共同完整集合差值与用量。',
              '- global_results.json / none_results.json：逐题结果；对应 failures.json 保存未完成记录。',
              '- 各分片保留原始请求、响应、轨迹、评分、源码和记忆快照。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def execute(out):
    lock = (out / '.validation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load(out / 'plan.json')
    if load(out / 'status.json')['status'] != 'prepared':
        raise ValueError('Campaign already started')
    for name, sha in load(out / 'source_hashes.json').items():
        if file_hash(out / 'source_at_launch' / name) != sha:
            raise ValueError('Frozen source changed')
    if file_hash(out / 'epoch1_memory.json') != plan['memory_sha256']:
        raise ValueError('Source memory changed')
    started, pending, running, done = time.time(), list(plan['jobs']), {}, []
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while pending or running:
            while pending and len(running) < plan['workers']:
                job = pending.pop(0)
                log = (out / f"{job['name']}.console.log").open('w')
                proc = subprocess.Popen(job['command'], cwd=out / 'source_at_launch', stdout=log, stderr=subprocess.STDOUT)
                running[job['name']] = (proc, log)
            for name, (proc, log) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[name]
                if proc.returncode != 0:
                    raise RuntimeError(f'{name} exited {proc.returncode}; see console log')
                done.append(name)
            result = collect(out, plan)
            status = {'status': 'running', 'updated': time.time(), 'started': started,
                      'elapsed_seconds': time.time() - started, 'pid': os.getpid(),
                      'active': {name: proc.pid for name, (proc, _) in running.items()},
                      'finished_shards': done, 'pending_shards': len(pending),
                      'groups': {s: {k: g[k] for k in ('processed', 'scored', 'failed')} for s, g in result['groups'].items()}}
            write_json(out / 'status.json', status)
            print(json.dumps(status), flush=True)
            if pending or running:
                time.sleep(15)
        collect(out, plan, final=True)
        if file_hash(out / 'epoch1_memory.json') != plan['memory_sha256']:
            raise ValueError('Source memory changed')
        status.update(status='finished', updated=time.time(), elapsed_seconds=time.time()-started)
        write_json(out / 'status.json', status)
        return 0
    except BaseException as exc:
        for proc, _ in running.values():
            proc.terminate()
        for proc, log in running.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        write_json(out / 'status.json', {'status': 'failed', 'updated': time.time(),
                   'error': type(exc).__name__, 'message': str(exc), 'finished_shards': done})
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-training', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if not args.execute:
        if not args.source_training or not 1 <= args.workers <= 16:
            p.error('source-training and workers 1..16 required')
        prepare(args.source_training.resolve(), out, args.workers)
    from dotenv import load_dotenv
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing')
    frozen = out / 'source_at_launch'
    if ROOT != frozen:
        os.environ['PYTHONPATH'] = str(frozen)
        os.chdir(frozen)
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'mind2web_adapter.validation',
                                '--execute', '--output', str(out)])
    raise SystemExit(execute(out))


if __name__ == '__main__':
    main()
