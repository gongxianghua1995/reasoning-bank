"""Adopt existing campaign children without restarting scientific execution.

The original immutable plan/source/config remain authoritative. Only controller
capacity changes. A stopped controller is inspected before handoff, and PID
start times protect against adopting or terminating an unrelated process.
"""
import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from .common import file_hash, load, write_json


def process_info(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'state':fields[0],'start_ticks':int(fields[19]),'ppid':int(fields[1])}
    except FileNotFoundError:
        return None


class AdoptedProcess:
    def __init__(self, pid, start_ticks, folder):
        self.pid=pid; self.start_ticks=start_ticks; self.folder=Path(folder)
        self.returncode=None

    def poll(self):
        info=process_info(self.pid)
        if info and info['start_ticks']!=self.start_ticks:
            raise RuntimeError('Adopted PID was reused')
        if info and info['state'] not in {'Z','X'}:
            return None
        summary=self.folder/'summary.json'
        self.returncode=0 if summary.exists() and load(summary).get('status') in {
            'finished','finished_with_failures'} else 1
        return self.returncode

    def wait(self, timeout=None):
        started=time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-started>=timeout:
                raise subprocess.TimeoutExpired('adopted campaign child',timeout)
            time.sleep(.1)
        return self.returncode


def restore_handoff(out, plan, handoff):
    if Path(handoff['output']).resolve()!=out.resolve():
        raise ValueError('Handoff belongs to another campaign')
    if file_hash(out/'plan.json')!=handoff['plan_sha256']:
        raise ValueError('Handoff plan changed')
    previous=process_info(handoff['old_pid'])
    if previous and previous['start_ticks']==handoff['old_start_ticks'] and previous['state'] not in {'Z','X'}:
        raise ValueError('Old controller is still alive')
    jobs={j['name']:j for j in plan['jobs']}
    state=handoff['state'];done=list(state['done']);failed=dict(state['failed']);blocked=list(state['blocked'])
    occupied=set(done)|set(failed)|set(blocked)
    if occupied & set(handoff['children']):raise ValueError('Completed job also marked active')
    pending=[name for name in jobs if name not in occupied and name not in handoff['children']]
    for name in pending:
        # Existing output requires explicit adoption. Never repeat a started job.
        if Path(jobs[name]['output']).exists() or (out/f'{name}.console.log').exists():
            raise ValueError(f'Unreconciled started job: {name}')
    running={}
    for name,child in handoff['children'].items():
        if name not in jobs:raise ValueError('Unknown adopted job')
        running[name]=(AdoptedProcess(child['pid'],child['start_ticks'],child['output']),
                       (out/f'{name}.console.log').open('a'))
    costs=load(out/'usage.json') if (out/'usage.json').exists() else {}
    return pending,running,done,failed,blocked,costs,state['started']


def capacity_reader(path):
    def read():
        value=load(path)['workers']
        if type(value) is not int or not 8<=value<=32:
            raise ValueError('Runtime workers must be an integer from 8 to 32')
        return value
    return read


def capture(out, pid):
    """Called only while the old parent is stopped; children keep running."""
    plan=load(out/'plan.json');state=load(out/'status.json')
    parent=process_info(pid)
    if not parent or parent['state'] not in {'T','t'} or state['pid']!=pid:
        raise ValueError('Expected stopped original controller')
    if state['status']!='running' or state.get('failed') or state.get('blocked'):
        raise ValueError('Only a healthy running campaign can be adopted')
    argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if b'mind2web_adapter.campaign' not in argv or str(out).encode() not in argv:
        raise ValueError('Unexpected controller command')
    if (out/'validation_retries.json').exists() and load(out/'validation_retries.json'):
        raise ValueError('Initial handoff during validation retries is unsupported')
    jobs={j['name']:j for j in plan['jobs']}; by_output={j['output']:j['name'] for j in plan['jobs']}
    child_ids=[int(x) for x in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
    children={};seen=set()
    for child_pid in child_ids:
        info=process_info(child_pid)
        if not info:raise ValueError('Child vanished during capture')
        cmd=(Path('/proc')/str(child_pid)/'cmdline').read_bytes().split(b'\0')
        if b'--output' in cmd:
            output=cmd[cmd.index(b'--output')+1].decode();name=by_output.get(output)
        else:
            matches=[n for n,v in state['active'].items() if v['pid']==child_pid]
            name=matches[0] if len(matches)==1 else None
            output=jobs[name]['output'] if name else None
        if name is None or name in seen or name in state['done']:
            raise ValueError('Unrecognized or duplicate controller child')
        seen.add(name)
        children[name]={'pid':child_pid,'start_ticks':info['start_ticks'],'output':output}
    # The stopped parent may have reaped a finished job since its last heartbeat.
    # Reconcile only final summaries; refuse ambiguous partially started output.
    done=list(state['done'])
    for name,j in jobs.items():
        if name in children or name in done:continue
        folder=Path(j['output'])
        if folder.exists() or (out/f'{name}.console.log').exists():
            raise ValueError(f'Heartbeat/reaping race for {name}; resume parent and retry capture')
    return {'created':time.time(),'output':str(out),'plan_sha256':file_hash(out/'plan.json'),
            'old_pid':pid,'old_start_ticks':parent['start_ticks'],'state':state,'children':children,
            'semantics':'Original child processes continue; only scheduler admission capacity changes.'}


def start_handoff(out, workers, env_file):
    from . import campaign
    state=load(out/'status.json');pid=state['pid']
    original_parent=process_info(pid)
    if not original_parent or original_parent['state'] in {'Z','X'}:
        raise ValueError('Original controller is not alive')
    # Validate all frozen inputs before touching the running controller.
    for name,sha in load(out/'source_hashes.json').items():
        if file_hash(out/'source_at_launch'/name)!=sha:raise ValueError('Frozen source changed')
    for name,sha in load(out/'input_hashes.json').items():
        if file_hash(out/name)!=sha:raise ValueError('Frozen input changed')
    plan=load(out/'plan.json')
    if file_hash(Path(plan['inventory'])/'manifest.json')!=plan['inventory_sha256']:
        raise ValueError('Inventory changed')
    if not 8<=workers<=32:raise ValueError('Invalid concurrency')
    # Import/credentials are verified before the parent is stopped.
    from dotenv import load_dotenv
    load_dotenv(env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):raise ValueError('Missing model credentials')
    migration=out/'runtime_concurrency'
    migration.mkdir(exist_ok=False)
    controller_source=migration/'source'
    shutil.copytree(Path(__file__).parent,controller_source/'mind2web_adapter',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write_json(migration/'controller_source_hashes.json',{
        str(p.relative_to(controller_source)):file_hash(p)
        for p in controller_source.rglob('*') if p.is_file()})
    if process_info(pid)['start_ticks']!=original_parent['start_ticks']:
        raise ValueError('Original controller PID changed')
    os.kill(pid,signal.SIGSTOP)
    killed=False
    try:
        for _ in range(100):
            info=process_info(pid)
            if info and info['state'] in {'T','t'}:break
            time.sleep(.02)
        handoff=capture(out,pid)
        write_json(migration/'handoff.json',handoff)
        write_json(migration/'capacity.json',{'workers':workers,'previous_workers':plan['workers'],
            'updated':time.time(),'reason':'User requested increased concurrency; training and learning protocols unchanged.'})
        # SIGKILL only the stopped scheduler PID: its SIGTERM handler would kill
        # the experiment children. No process group signal is used here.
        os.kill(pid,signal.SIGKILL);killed=True
        for _ in range(100):
            info=process_info(pid)
            if not info or info['state'] in {'Z','X'}:break
            time.sleep(.02)
        with (migration/'controller.console.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','mind2web_adapter.campaign_live',
                '--output',str(out),'--adopt','--env-file',str(env_file)],
                cwd=controller_source,stdout=log,stderr=log,
                env={**os.environ,'PYTHONPATH':str(controller_source)},
                stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(migration/'launch.json',{'pid':child.pid,'started':time.time(),'workers':workers})
        print(f'Controller PID {child.pid}; original experiment children preserved',flush=True)
    except BaseException:
        if not killed:os.kill(pid,signal.SIGCONT)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--adopt',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if not a.adopt:return start_handoff(out,a.workers,a.env_file)
    from dotenv import load_dotenv
    load_dotenv(a.env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):p.error('Missing model credentials')
    from .campaign import execute
    migration=out/'runtime_concurrency'
    for name,sha in load(migration/'controller_source_hashes.json').items():
        if file_hash(migration/'source'/name)!=sha:p.error('Controller source changed')
    raise SystemExit(execute(out,handoff=load(migration/'handoff.json'),
                             capacity=capacity_reader(migration/'capacity.json')))


if __name__=='__main__':main()
