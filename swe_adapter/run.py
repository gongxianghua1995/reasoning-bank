"""Frozen sequential Verified stream with optional ReasoningBank memory."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import ROOT
from .common import write_json, read_record, event, digest
from .vendor.workspace import Workspace
from .vendor.mini_benchmark import required_images


def load(path):return json.loads(path.read_text())


def preflight(args, records, config):
    from minisweagent.agents.default import DefaultAgent
    from minisweagent import __version__
    if __version__!='1.10.0':raise ValueError('Expected vendored mini 1.10.0')
    if len(set(args.instance_ids))!=len(args.instance_ids):raise ValueError('Duplicate stream task')
    for key in ['task_seconds','api_timeout','command_timeout','steps','max_tokens','judge_seconds','extract_seconds','heartbeat_seconds']:
        if config[key]<=0:raise ValueError('Limits must be positive')
    if config['top_k']!=1:raise ValueError('This adapter validates top-1 retrieval')
    version=subprocess.check_output([args.eval_python,'-c','import importlib.metadata as m; print(m.version("swebench"))'],text=True).strip()
    if version!='2.1.6':raise ValueError('Expected swebench 2.1.6 evaluator')
    images={}
    for record in records:
        for tag in required_images(record):
            images[tag]=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
    return dict(agent_version=__version__,eval_version=version,images=images,model_calls=0)


def prepare(args, records, config):
    verified=preflight(args,records,config)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_at_run';snapshot.mkdir()
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(ROOT/'swe_adapter',snapshot/'swe_adapter',ignore=ignore)
    shutil.copytree(ROOT/'third_party/src/minisweagent',snapshot/'third_party/src/minisweagent',ignore=ignore)
    write_json(out/'dataset.json',records)
    write_json(out/'config.json',config)
    hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snapshot.rglob('*') if p.is_file()}
    write_json(out/'source_hashes.json',hashes)
    embedding_files={}
    if args.memory=='reasoningbank':
        for p in args.embedding_cache.resolve().rglob('*'):
            if p.is_file() and p.suffix in {'.onnx','.json','.txt'}:
                embedding_files[str(p.relative_to(args.embedding_cache.resolve()))]=hashlib.sha256(p.read_bytes()).hexdigest()
        if not any(name.endswith('.onnx') for name in embedding_files):raise ValueError('Pre-download local embedding weights first')
    manifest=dict(created=time.time(),instance_ids=args.instance_ids,memory_mode=args.memory,
        repository_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        config=config,verified=verified,embedding_cache=str(args.embedding_cache.resolve()),embedding_files=embedding_files,
        input_sha256=hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
        frozen_dataset_sha256=hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
        eval_python=args.eval_python,python=sys.executable,network_mode='none',public_fields=['problem_statement'],
        memory_policy='empty start; prior completed tasks only; no official evaluation feedback',
        changes_from_official=['OpenAI-compatible transport','local BGE-small embedding instead of Gemini',
          'offline baseline/history isolation','host patch export on all exits','bounded stages and logs',
          'strict judge and memory parsing; atomic batch-local memory'])
    write_json(out/'manifest.json',manifest)
    return out,snapshot


async def solve(record, config, out, selected, memory_ids):
    ws=Workspace(record,config,out)
    proc=None
    stage={'status':'preparing','instance_id':record['instance_id']}
    async def heartbeat():
        while True:
            write_json(out/'status.json',{**stage,'elapsed_seconds':round(time.monotonic()-ws.started,1),
                'remaining_seconds':max(0,round(ws.deadline-time.monotonic(),1)),
                'worker_pid':proc.pid if proc else None})
            print(json.dumps({'heartbeat':record['instance_id'],**stage}),flush=True)
            await asyncio.sleep(config['heartbeat_seconds'])
    pulse=asyncio.create_task(heartbeat());error=None;code=None
    try:
        await ws.prepare()
        if (await ws.export(include_tests=True))['patch']:raise ValueError('Initial workspace not clean')
        seconds=ws.deadline-time.monotonic()-5
        if seconds<=5:raise TimeoutError('Workspace preparation consumed task budget')
        request=dict(task=record['problem_statement'],selected_memory=selected,memory_ids=memory_ids,
                     config=config,container=ws.main,seconds=seconds,output_dir=str(out))
        write_json(out/'request.json',request)
        stage['status']='generating'
        with (out/'worker.log').open('w') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-m','swe_adapter.worker',str(out/'request.json'),
                stdout=log,stderr=log,stdin=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(proc.wait(),seconds+3)
            code=proc.returncode
    except Exception as exc:
        error=type(exc).__name__
        event(out,'generation_error',error=error)
    finally:
        if proc and proc.returncode is None:
            proc.terminate()
            try:await asyncio.wait_for(proc.wait(),5)
            except asyncio.TimeoutError:proc.kill();await proc.wait()
        stage['status']='exporting'
        result=load(out/'result.json') if (out/'result.json').exists() else {'exit_status':error or 'WorkerProcessError'}
        try:
            if not ws.main:raise RuntimeError('No prepared workspace')
            actual=await ws.export()
            (out/'selected.patch').write_text(actual['patch'])
            write_json(out/'predictions.json',[dict(instance_id=record['instance_id'],model_patch=actual['patch'],
                         model_name_or_path='reasoningbank_'+config['model'])])
            result.update(patch_chars=len(actual['patch']),patch_sha256=actual['sha256'],excluded_paths=actual['excluded_paths'])
        except Exception as exc:
            result['export_error']=type(exc).__name__
        result.update(error=error,worker_returncode=code,generation_seconds=time.monotonic()-ws.started)
        write_json(out/'generation.json',result)
        try:await ws.close()
        finally:
            pulse.cancel()
            try:await pulse
            except asyncio.CancelledError:pass
            write_json(out/'status.json',dict(status='generation_finished',**result))
    return result


async def stream(args):
    from fastembed import TextEmbedding
    from .memory import Memory,learn
    out=args.output_dir.resolve();manifest=load(out/'manifest.json');config=load(out/'config.json')
    lock=(out/'stream.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,sha in load(out/'source_hashes.json').items():
        if hashlib.sha256((out/'source_at_run'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed')
    if hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest()!=manifest['frozen_dataset_sha256']:raise ValueError('Dataset changed')
    if hashlib.sha256((out/'config.json').read_bytes()).hexdigest()!=manifest['config_sha256']:raise ValueError('Config changed')
    records=load(out/'dataset.json')
    state=dict(status='preparing',pid=os.getpid(),started=time.time(),completed=[],active=None,total=len(records))
    def persist():
        state['updated']=time.time()
        state['elapsed_seconds']=round(state['updated']-state['started'],1)
        write_json(out/'status.json',state)
    persist()
    async def heartbeat():
        while state['status'] not in {'finished','memory_error'}:
            persist()
            print(json.dumps({'stream_heartbeat':state['status'],'phase':state.get('phase'),
                              'active':state['active'],'completed':len(state['completed'])}),flush=True)
            await asyncio.sleep(config['heartbeat_seconds'])
    pulse=asyncio.create_task(heartbeat())
    memory=None
    if args.memory=='reasoningbank':
        if (out/'memory.json').exists():raise ValueError('A stream must start from empty memory')
        # Runtime cannot download weights; only the explicit setup stage does so.
        os.environ['HF_HUB_OFFLINE']='1'
        for name,sha in manifest['embedding_files'].items():
            if hashlib.sha256((Path(manifest['embedding_cache'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('Embedding weights changed')
        embedder=TextEmbedding(config['embedding_model'],cache_dir=manifest['embedding_cache'],
                              threads=config['embedding_threads'],local_files_only=True)
        memory=Memory(out/'memory.json',embedder,config['embedding_model'])
        write_json(out/'memory.json',memory.state)
    for number,record in enumerate(records,1):
        iid=record['instance_id'];folder=out/'instances'/f'{out.name}__{iid}';folder.mkdir(parents=True)
        state.update(status='running',active=iid,phase='retrieval');persist()
        before=memory.state if memory else {'entries':[]}
        write_json(folder/'memory_before.json',before)
        selected_ids,selected,vector,scores=memory.select(iid,record['problem_statement']) if memory else ([], '', [], [])
        write_json(folder/'retrieval.json',dict(memory_ids=selected_ids,selected_memory=selected,scores=scores,
                   memory_before_sha256=digest(json.dumps(before,sort_keys=True))))
        event(folder,'memory_retrieved',memory_ids=selected_ids,memory_items_chars=len(selected))
        state['phase']='solve';persist()
        generation=await solve(record,config,folder,selected,selected_ids)
        memory_error=None
        if memory:
            state['phase']='memory';persist()
            write_json(folder/'status.json',{'status':'learning_memory'})
            try:
                entry=await asyncio.to_thread(learn,folder,iid,record['problem_statement'],config,vector)
                memory.commit(entry)
            except Exception as exc:
                memory_error=type(exc).__name__;event(folder,'memory_error',error=memory_error)
            write_json(folder/'memory_after.json',memory.state)
        state['phase']='evaluation';persist()
        write_json(folder/'status.json',{'status':'evaluating','memory_error':memory_error})
        evaluation_error=None
        if (folder/'predictions.json').exists():
            proc=None
            try:
                with (folder/'evaluation.log').open('w') as log:
                    proc=await asyncio.create_subprocess_exec(manifest['eval_python'],'-u','-m','swe_adapter.evaluate',
                        '--data-path',str(out/'dataset.json'),'--output-dir',str(folder),stdout=log,stderr=log)
                    await asyncio.wait_for(proc.wait(),2100)
                    if proc.returncode or not (folder/'report.json').exists():raise RuntimeError('Evaluation failed')
            except Exception as exc:evaluation_error=type(exc).__name__
            finally:
                if proc and proc.returncode is None:proc.kill();await proc.wait()
                found=await asyncio.create_subprocess_exec('docker','ps','-aq','--filter',
                    'label=reasoningbank.swe.run='+folder.name,stdout=asyncio.subprocess.PIPE)
                raw,_=await found.communicate()
                leftovers=raw.decode().split()
                if leftovers:
                    clean=await asyncio.create_subprocess_exec('docker','rm','-f',*leftovers,
                        stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                    await clean.wait()
                    event(folder,'leftover_containers_removed',containers=leftovers)
        else:evaluation_error='NoPrediction'
        report=load(folder/'report.json') if (folder/'report.json').exists() else {}
        outcome=('resolved' if iid in report.get('resolved_ids',[]) else 'unresolved' if iid in report.get('unresolved_ids',[]) else
                 'empty_patch' if iid in report.get('empty_patch_ids',[]) else 'evaluation_error')
        result=dict(instance_id=iid,outcome=outcome,memory_error=memory_error,evaluation_error=evaluation_error,
                    generation=generation,output_dir=str(folder),finished=time.time())
        write_json(folder/'status.json',dict(status='finished',**result))
        state['completed'].append(result);persist()
        if memory_error:
            state.update(status='memory_error',active=None);persist();return state
    state.update(status='finished',active=None,phase=None,finished=time.time());persist()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',required=True,type=Path)
    p.add_argument('--instance-ids',required=True,nargs='+')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--eval-python',required=True)
    p.add_argument('--memory',choices=['reasoningbank','none'],default='reasoningbank')
    p.add_argument('--config',type=Path,default=ROOT/'swe_adapter/config.json')
    p.add_argument('--embedding-cache',type=Path,default=ROOT/'.models')
    p.add_argument('--env-file',type=Path)
    p.add_argument('--check',action='store_true')
    p.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    from dotenv import load_dotenv
    if args.env_file:load_dotenv(args.env_file.resolve(),override=False)
    if args.frozen:
        try:
            state=asyncio.run(stream(args))
        except BaseException as exc:
            path=args.output_dir/'status.json'
            state=load(path) if path.exists() else {}
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'execution_error',
                         error=type(exc).__name__,updated=time.time())
            write_json(path,state)
            raise
        if state['status']!='finished':raise SystemExit(1)
        return
    records=[read_record(args.data_path,iid) for iid in args.instance_ids]
    config=load(args.config)
    if args.check:
        print(json.dumps(preflight(args,records,config),indent=2));return
    if not os.environ.get('OPENAI_API_KEY'):raise ValueError('OPENAI_API_KEY is missing')
    out,snapshot=prepare(args,records,config)
    os.environ['PYTHONPATH']=str(snapshot/'third_party/src')+os.pathsep+str(snapshot)
    os.environ['MSWEA_GLOBAL_CONFIG_DIR']=str(out/'mini_config')
    os.chdir(snapshot)
    command=[sys.executable,'-u','-m','swe_adapter.run','--frozen','--data-path',str(out/'dataset.json'),
             '--instance-ids',*args.instance_ids,'--output-dir',str(out),'--eval-python',args.eval_python,
             '--memory',args.memory,'--config',str(out/'config.json')]
    os.execv(sys.executable,command)


if __name__=='__main__':main()
