"""Continue a fixed-bank campaign, preserving scored cases and partial actions."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import load, write_json, file_hash
from .campaign import collect, validation_rows, audit_job

ROOT = Path(__file__).resolve().parents[1]


def execute(out, recovery):
    lock = (out/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load(out/'plan.json')
    if any(j['group'] not in {'aligned','none'} for j in plan['jobs']):
        raise ValueError('Only fixed-bank evaluation campaigns may continue')
    for name, sha in load(out/'input_hashes.json').items():
        if file_hash(out/name) != sha:
            raise ValueError('Original scientific inputs changed')
    for name, sha in load(recovery/'source_hashes.json').items():
        if file_hash(recovery/'source'/name) != sha:
            raise ValueError('Recovery source changed')
    if file_hash(Path(plan['inventory'])/'manifest.json') != plan['inventory_sha256']:
        raise ValueError('Dataset manifest changed')
    jobs = {j['name']:j for j in plan['jobs']}
    policy = load(recovery/'policy.json') if (recovery/'policy.json').exists() else {}
    workers = policy.get('workers', plan['workers'])
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Invalid runtime concurrency')
    def missing(job):
        rows = validation_rows(out,job)
        observed = {r['case_id'] for r in rows}
        if not observed <= set(job['case_ids']):
            raise ValueError('Unexpected scored cases')
        return [cid for cid in job['case_ids'] if cid not in observed]
    done = [name for name,j in jobs.items() if not missing(j)]
    pending = [name for name in jobs if name not in done]
    active, failed, attempts = {}, {}, {}
    if (recovery/'handoff.json').exists():
        from .campaign_live import AdoptedProcess, process_info
        handoff = load(recovery/'handoff.json')
        old = process_info(handoff['old_pid'])
        if old and old['start_ticks'] == handoff['old_start_ticks'] and old['state'] not in {'Z','X'}:
            raise ValueError('Previous controller still active')
        for name, child in handoff['children'].items():
            info = process_info(child['pid'])
            if info and info['start_ticks'] != child['start_ticks']:
                raise ValueError('Worker PID reused')
            target = Path(child['output'])
            proc = AdoptedProcess(child['pid'], child['start_ticks'], target)
            active[name] = (proc, (recovery/f'{name}.adopted.log').open('a'), target)
            attempts[name] = 1
            if name in pending:
                pending.remove(name)
            if name in done:
                done.remove(name)
    ledger = load(out/'validation_retries.json') if (out/'validation_retries.json').exists() else {}
    costs = {}
    started = time.time()
    def stop(signum, frame):
        raise InterruptedError('Evaluation continuation stopped')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        while pending or active:
            while pending and len(active) < workers:
                # Keep B priority while sharing slots across its independent domains.
                def priority(name):
                    job = jobs[name]
                    same = sum(jobs[n]['group']==job['group'] and jobs[n]['split']==job['split'] for n in active)
                    return (0 if job['group']=='aligned' else 1,
                            0 if job['split']=='validation' else 1, same)
                name = min(pending, key=priority)
                pending.remove(name)
                job = jobs[name]
                ids = missing(job)
                if not ids:
                    done.append(name)
                    continue
                previous = ledger.get(name, [])
                roots = [job['output']] + [r['output'] for r in previous]
                index = len(previous)+1
                target = out/f'{name}_continue{index}'
                ids_path = out/'retry_case_lists'/f'{name}_continue{index}.json'
                write_json(ids_path,ids)
                command = list(job['command'])
                command[0] = sys.executable
                command[command.index('--output')+1] = str(target)
                command[command.index('--case-ids-file')+1] = str(ids_path)
                command += ['--eval-windows','3','--eval-recovery-roots',*roots]
                attempt = {'output':str(target),'case_ids':ids,'expected':len(ids),
                           'recovery':str(recovery),'policy':'Preserve earliest trajectory; 3 bounded action windows'}
                ledger.setdefault(name,[]).append(attempt)
                write_json(out/'validation_retries.json',ledger)
                log = (out/f'{name}_continue{index}.console.log').open('x')
                proc = subprocess.Popen(command,cwd=recovery/'source',
                    env={**os.environ,'PYTHONPATH':str(recovery/'source')},
                    stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
                active[name] = (proc,log,target)
                attempts[name] = attempts.get(name,0)+1
            for name,(proc,log,target) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del active[name]
                costs[target.name] = audit_job(target)
                write_json(recovery/'usage.json',costs)
                if proc.returncode:
                    failed[name] = f'exit_{proc.returncode}'
                    raise RuntimeError(f'Fatal evaluation worker failure: {name}')
                summary = load(target/'summary.json')
                if summary['processed'] != ledger[name][-1]['expected']:
                    raise ValueError('Unexpected evaluation coverage')
                absent = missing(jobs[name])
                if not absent:
                    done.append(name)
                elif attempts[name] < 3:
                    pending.append(name)
                else:
                    # A fixed checkpoint has no validation-dependent selection.
                    # Keep missing cases explicit while other evaluations proceed.
                    failed[name] = f'missing_{len(absent)}_after_continuation'
            report = collect(out,plan,done,failed,active)
            state = {'status':'running','pid':os.getpid(),'started':started,'updated':time.time(),
                'controller_mode':'fixed_bank_continuation','recovery':str(recovery),
                'runtime_workers':workers,
                'processed':sum(g['scored'] for g in report['groups'].values()),
                'expected_episodes':plan['expected_episodes'],'done':done,'failed':failed,
                'pending':len(pending),'active':{}}
            for name,(proc,_,target) in active.items():
                status = target/'status.json'
                state['active'][name] = {'pid':proc.pid,'output':str(target),
                    'progress':load(status) if status.exists() else None}
            write_json(out/'status.json',state)
            if pending or active:
                time.sleep(15)
        state.update(status='finished' if not failed else 'incomplete',updated=time.time())
        write_json(out/'status.json',state)
        return int(bool(failed))
    except BaseException as exc:
        for proc,_,_ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        for proc,log,_ in active.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            log.close()
        write_json(out/'status.json',{'status':'error','updated':time.time(),
            'error':type(exc).__name__,'done':done,'failed':failed,'recovery':str(recovery),
            'interrupted_jobs':list(active)})
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--recovery',type=Path)
    p.add_argument('--workers',type=int)
    p.add_argument('--handoff-file',type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    from dotenv import load_dotenv
    if a.env_file:
        load_dotenv(a.env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('Missing model credentials')
    os.environ['REASONINGBANK_RATE_FILE'] = str(out/'request_admission.txt')
    if a.recovery:
        raise SystemExit(execute(out,a.recovery.resolve()))
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load(out/'status.json')
        if state['status'] not in {'error','incomplete'} and not a.handoff_file:
            raise ValueError('Continuation requires a stopped incomplete campaign')
        recovery = out/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        recovery.mkdir(parents=True,exist_ok=False)
        write_json(recovery/'status_before.json',state)
        if a.handoff_file:
            handoff = load(a.handoff_file)
            if handoff['output'] != str(out) or handoff['plan_sha256'] != file_hash(out/'plan.json'):
                raise ValueError('Handoff plan differs')
            write_json(recovery/'handoff.json',handoff)
        if (out/'validation_retries.json').exists():
            shutil.copy2(out/'validation_retries.json',recovery/'retries_before.json')
        shutil.copytree(ROOT/'mind2web_adapter',recovery/'source/mind2web_adapter',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write_json(recovery/'source_hashes.json',{str(f.relative_to(recovery/'source')):file_hash(f)
            for f in (recovery/'source').rglob('*') if f.is_file()})
        workers = a.workers if a.workers is not None else load(out/'plan.json')['workers']
        if not 1 <= workers <= 16:
            raise ValueError('Concurrency must be 1..16')
        write_json(recovery/'policy.json',{'workers':workers, 'change':'Preserve partial actions and live adopted workers; share slots across independent test domains.',
            'unchanged':'Model, generation budgets, prompts, dataset, scoring and selected epoch-1 memory.',
            'prefix_selection':'Earliest saved trajectory, then exact-prefix extensions; never score-based.'})
    os.environ['PYTHONPATH'] = str(recovery/'source')
    os.chdir(recovery/'source')
    os.execv(sys.executable,[sys.executable,'-u','-m','mind2web_adapter.continue_frozen',
                             '--output',str(out),'--recovery',str(recovery)])


if __name__ == '__main__':
    main()
