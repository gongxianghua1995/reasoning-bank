"""Frozen two-protocol campaign: paper stream, official-feedback training, shared no-memory control."""
import argparse
from collections import Counter
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

ROOT=Path(__file__).resolve().parents[1]
TEST_SPLITS=('test_task','test_website','test_domain')


def validation_rows(out, job):
    folder=Path(job['output'])
    rows=load(folder/'results.json') if (folder/'results.json').exists() else []
    ledger=out/'validation_retries.json'
    for retry in (load(ledger).get(job['name'],[]) if ledger.exists() else []):
        path=Path(retry['output'])/'results.json'
        if path.exists():rows+=load(path)
    if len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('Validation retry repeated a scored task')
    return rows


def choose_checkpoint(candidates):
    """Require complete, identical validation coverage; ties select the earlier epoch."""
    expected=None
    scored=[]
    for epoch,rows,ids in candidates:
        observed=[r['case_id'] for r in rows]
        if len(set(observed)) != len(observed) or set(observed)!=set(ids):
            raise ValueError('Checkpoint selection requires complete validation coverage')
        if expected is not None and set(ids)!=expected:
            raise ValueError('Validation populations differ')
        expected=set(ids)
        score=sum(r['metrics']['step_success_rate'] for r in rows)/len(rows)
        scored.append({'epoch':epoch,'step_success_rate':score,'n_cases':len(rows)})
    best=max(scored,key=lambda r:(r['step_success_rate'],-r['epoch']))
    return {'selected_epoch':best['epoch'],'candidates':scored,
            'rule':'Max validation task-macro Step SR; exact ties choose earlier epoch; no test scores.'}


def prepare(args):
    from swarmevo.benches.mind2web.dataset import load_inventory
    inventory=args.inventory.resolve();manifest,parts=load_inventory(inventory,seed=42)
    if manifest['diagnostic'] or {k:len(parts[k]) for k in ('train','validation',*TEST_SPLITS)} != {
            'train':303,'validation':101,'test_task':177,'test_website':142,'test_domain':694}:
        raise ValueError('Complete official seed42 inventory required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frozen=out/'source_at_launch'
    shutil.copytree(ROOT/'mind2web_adapter',frozen/'mind2web_adapter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write_json(out/'source_hashes.json',{str(p.relative_to(frozen)):file_hash(p) for p in frozen.rglob('*') if p.is_file()})
    for name in ('stream','aligned'):
        write_json(out/f'config_{name}.json',load(ROOT/f'mind2web_adapter/config_{name}.json'))
    write_json(out/'splits.json',{k:manifest[k] for k in ('parts','counts','split_sha256','source_files')})
    base=[sys.executable,'-u','-m','mind2web_adapter.run','--inventory',str(inventory),'--seed','42',
          '--embedding-cache',str(args.embedding_cache.resolve()),'--continue-case-errors']
    jobs=[]
    def add(name,group,split,command,depends=(),slots=1,ids=()):
        jobs.append({'name':name,'group':group,'split':split,'command':command,'depends':list(depends),
                     'slots':slots,'expected':len(ids),'case_ids':list(ids),'output':str(out/name)})
    train=[sys.executable,'-u','-m','mind2web_adapter.aligned','--inventory',str(inventory),
           '--config',str(out/'config_aligned.json'),'--embedding-cache',str(args.embedding_cache.resolve()),
           '--epochs','2','--batch-size','8','--workers',str(args.training_workers),
           '--continue-case-errors','--output',str(out/'aligned_train')]
    add('aligned_train','aligned_train','train',train,slots=args.training_workers,
        ids=manifest['parts']['train']*2)
    for split in TEST_SPLITS if getattr(args, 'include_paper', False) else ():
        name='paper_'+split
        add(name,'paper',split,base+['--mode','stream','--scope','global','--split',split,
            '--config',str(out/'config_stream.json'),'--output',str(out/name)],ids=manifest['parts'][split])
    def shards(group,split,memory=None,depends=()):
        names=[];ids=manifest['parts'][split]
        for index in range(8):
            selected=ids[index::8]
            name=f'{group}_{split}_{index:02d}';names.append(name)
            path=out/'case_lists'/f'{name}.json';write_json(path,selected)
            cmd=base+['--mode','eval','--scope','global' if memory else 'none','--split',split,
                '--case-ids-file',str(path),'--config',str(out/('config_aligned.json' if memory else 'config_stream.json')),
                '--output',str(out/name)]
            if memory:cmd+=['--memory-from',str(memory)]
            add(name,group,split,cmd,depends=depends,ids=selected)
        return names
    validation_jobs=[]
    for epoch in (1,2):
        validation_jobs+=shards(f'validation_epoch{epoch}','validation',
            out/'aligned_train/checkpoints'/f'epoch_{epoch}'/'memory.json',depends=['aligned_train'])
    add('select_checkpoint','selection','validation',[],depends=validation_jobs,slots=0)
    for split in TEST_SPLITS:
        shards('aligned',split,out/'selected_memory.json',depends=['select_checkpoint'])
    for split in ('validation',*TEST_SPLITS):
        shards('none',split)
    plan={'created':time.time(),'source_root':str(frozen),'inventory':str(inventory),
          'inventory_sha256':file_hash(inventory/'manifest.json'),'split_sha256':manifest['split_sha256'],
          'workers':args.workers,'training_workers':args.training_workers,'jobs':jobs,
          'expected_episodes':sum(j['expected'] for j in jobs),
          'protocol':'Swarm 90b6cd5 aligned B training with post-execution references, frozen tests, matched no-memory control; optional A',
          'model_note':'DeepSeek and BGE replacements declared; not exact Gemini reproduction.'}
    write_json(out/'plan.json',plan)
    write_json(out/'input_hashes.json',{str(p.relative_to(out)):file_hash(p) for p in
        [out/'plan.json',out/'config_stream.json',out/'config_aligned.json',out/'splits.json',*sorted((out/'case_lists').glob('*.json'))]})
    write_json(out/'status.json',{'status':'prepared','updated':time.time()})
    return plan


def collect(out,plan,done,failed,active):
    from swarmevo.benches.mind2web.metrics import aggregate
    groups={};all_rows={};usage=Counter();errors=Counter()
    for job in plan['jobs']:
        if job['group']=='selection':continue
        folder=Path(job['output']);key=job['group']+'/'+job['split']
        g=groups.setdefault(key,{'scheduled':0,'processed':0,'scored':0,'failed':0,'complete':True})
        g['scheduled']+=job['expected'];g['complete'] &= job['name'] in done
        rows=validation_rows(out,job) if job['group'] != 'aligned_train' else (
            load(folder/'results.json') if (folder/'results.json').exists() else [])
        all_rows.setdefault(key,[]).extend(rows)
        outcomes=load(folder/'case_outcomes.json') if (folder/'case_outcomes.json').exists() else []
        if job['group'] != 'aligned_train':
            ledger=load(out/'validation_retries.json') if (out/'validation_retries.json').exists() else {}
            folders=[folder]+[Path(r['output']) for r in ledger.get(job['name'],[])]
            attempted=set()
            errors=0
            for attempt_folder in folders:
                f=attempt_folder/'case_outcomes.json'
                for item in load(f) if f.exists() else []:
                    attempted.add(item.get('case_id') or item['episode_id'].split(':',1)[1])
                f=attempt_folder/'failures.json'
                errors+=len(load(f)) if f.exists() else 0
            scored={row['case_id'] for row in rows}
            g['processed']+=len(attempted|scored)
            g['failed']+=len(attempted-scored)
            g['historical_failure_events']=g.get('historical_failure_events',0)+errors
        else:
            g['processed']+=len(outcomes)
            g['failed']+=len(load(folder/'failures.json')) if (folder/'failures.json').exists() else 0
        g['scored']+=len(rows)
    for key,rows in all_rows.items():
        groups[key]['metrics']=aggregate([r['metrics'] for r in rows])
        groups[key]['full_coverage'] = groups[key]['scored'] == groups[key]['scheduled']
        groups[key]['metrics_population'] = 'full scheduled population' if groups[key]['full_coverage'] else 'partial scored population; not a final result'
    paired={}
    for method in ('paper','aligned'):
        for split in TEST_SPLITS:
            a={r['case_id']:r['metrics'] for r in all_rows.get(method+'/'+split,[])}
            b={r['case_id']:r['metrics'] for r in all_rows.get('none/'+split,[])}
            common=sorted(a.keys() & b.keys())
            paired[method+'/'+split]={'n_common':len(common),'step_sr_delta':
                sum(a[c]['step_success_rate']-b[c]['step_success_rate'] for c in common)/len(common) if common else None}
    report={'updated':time.time(),'groups':groups,'paired':paired,'failed_jobs':failed,
            'selection':load(out/'selection.json') if (out/'selection.json').exists() else None,
            'interpretation':'A and B differ in data/feedback/protocol; compare each with its matched no-memory control. Partial metrics require coverage.'}
    write_json(out/'comparison.json',report)
    lines=['# ReasoningBank 两组实验进度','',
           'B 为官方评分及训练失败步骤参考答案指导的经验提取，随后冻结测试；无记忆为同底座对照。A可选。thinking开启，执行上限64000，BGE-small。','',
           '| 组别 / split | 已处理 / 计划 | 完整评分 | 失败 | Step SR 宏平均 | 完成 |',
           '|---|---:|---:|---:|---:|---|']
    for key,g in groups.items():
        value=g['metrics'].get('step_success_rate')
        score=f'{value*100:.2f}%' if value is not None else '—'
        lines.append(f"| {key} | {g['processed']}/{g['scheduled']} | {g['scored']} | {g['failed']} | {score} | {g['complete']} |")
    fixed=all(j['group'] in {'aligned','none'} for j in plan['jobs'])
    lines+=['',
            '固定第一轮训练经验库；验证和测试均冻结，不再训练或选库。' if fixed else '训练轮数及快照选择以运行计划为准。',
            '泛化结论使用官方测试结果。缺测不补造分数，完整覆盖和共同完成集合同时报告。',
            '失败列为已尝试但仍未评分的题数；恢复成功的历史接口故障不计入该列。官方评分只进入 B 的训练后提炼。',
            '详细请求和 token 用量在各运行 events.jsonl / audit.json 中，结束后汇总为 usage.json。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return report


def audit_job(folder):
    usage={};requests=Counter();errors=Counter()
    for p in folder.glob('cases/*/events.jsonl'):
        for line in p.read_text().splitlines():
            e=json.loads(line);phase=e.get('phase','unknown')
            if e['event']=='model_request':requests[phase]+=1
            if e['event']=='model_error':errors[e['error']]+=1
            if e['event']=='model_response':
                u=usage.setdefault(phase,Counter())
                for k in ('prompt_tokens','completion_tokens','total_tokens'):
                    u[k]+=(e.get('usage') or {}).get(k,0) or 0
    result={'requests':dict(requests),'errors':dict(errors),'returned_usage_by_phase':{k:dict(v) for k,v in usage.items()},
            'note':'Failed requests may incur provider charges without returned usage.'}
    write_json(folder/'cost_audit.json',result)
    return result


def execute(out, *, handoff=None, capacity=None):
    plan=load(out/'plan.json');lock=(out/'.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if handoff is None and load(out/'status.json')['status']!='prepared':
        raise ValueError('Campaign already launched; existing outputs are immutable')
    for name,sha in load(out/'source_hashes.json').items():
        if file_hash(out/'source_at_launch'/name)!=sha:raise ValueError('Source changed')
    for name,sha in load(out/'input_hashes.json').items():
        if file_hash(out/name)!=sha:raise ValueError('Campaign input changed')
    if file_hash(Path(plan['inventory'])/'manifest.json')!=plan['inventory_sha256']:
        raise ValueError('Inventory changed')
    jobs={j['name']:j for j in plan['jobs']};pending=list(jobs);running={};done=[];failed={};blocked=[];costs={}
    started=time.time()
    if handoff is not None:
        from .campaign_live import restore_handoff
        pending,running,done,failed,blocked,costs,started=restore_handoff(out,plan,handoff)
    def stop(signum,frame):raise InterruptedError('Campaign interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or running:
            for name in list(pending):
                j=jobs[name]
                if any(dep in failed or dep in blocked for dep in j['depends']):
                    pending.remove(name);blocked.append(name);continue
                if not set(j['depends'])<=set(done):continue
                used=sum(jobs[n]['slots'] for n in running)
                limit=capacity() if capacity else plan['workers']
                if used+j['slots']>limit:continue
                pending.remove(name)
                if name=='select_checkpoint':
                    try:
                        candidates=[]
                        ids=load(out/'splits.json')['parts']['validation']
                        for epoch in (1,2):
                            rows=[row for v in plan['jobs'] if v['group']==f'validation_epoch{epoch}'
                                  for row in validation_rows(out,v)]
                            candidates.append((epoch,rows,ids))
                        selection=choose_checkpoint(candidates)
                        memory=out/'aligned_train/checkpoints'/f"epoch_{selection['selected_epoch']}"/'memory.json'
                        selection['memory_sha256']=file_hash(memory)
                        shutil.copy2(memory,out/'selected_memory.json')
                        write_json(out/'selection.json',selection);done.append(name)
                    except Exception as exc:
                        failed[name]=type(exc).__name__
                    continue
                log=(out/f'{name}.console.log').open('x')
                p=subprocess.Popen(j['command'],cwd=plan['source_root'],stdout=log,stderr=log,
                                   env={**os.environ,'PYTHONPATH':plan['source_root']},
                                   stdin=subprocess.DEVNULL,start_new_session=True)
                running[name]=(p,log)
            for name,(p,log) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[name]
                job=jobs[name]
                ledger=load(out/'validation_retries.json') if (out/'validation_retries.json').exists() else {}
                retries=ledger.get(name,[])
                folder=Path(retries[-1]['output']) if retries else Path(job['output'])
                costs[name+(f"_retry{len(retries)}" if retries else '')]=audit_job(folder)
                write_json(out/'usage.json',costs)
                if p.returncode:
                    failed[name]=f'exit_{p.returncode}'
                    # Do not amplify authentication/quota failure through more jobs.
                    raise RuntimeError(f'Job failed: {name}; saved outputs retained')
                summary=load(folder/'summary.json')
                expected=retries[-1]['expected'] if retries else job['expected']
                if summary['processed']!=expected:
                    raise ValueError('Job coverage differs from plan')
                if job['group'] != 'aligned_train' and '--mode' in job['command'] and job['command'][job['command'].index('--mode')+1] == 'eval':
                    observed={r['case_id'] for r in validation_rows(out,job)}
                    missing=[cid for cid in job['case_ids'] if cid not in observed]
                    if missing and len(retries)<2:
                        # SwarmEvolution permits up to 3 validation attempts. Only retry missing cases.
                        attempt=len(retries)+1
                        target=out/f'{name}_retry{attempt}'
                        ids_file=out/'retry_case_lists'/f'{name}_{attempt}.json'
                        write_json(ids_file,missing)
                        command=list(job['command'])
                        command[command.index('--output')+1]=str(target)
                        command[command.index('--case-ids-file')+1]=str(ids_file)
                        retries.append({'output':str(target),'case_ids':missing,'expected':len(missing)})
                        ledger[name]=retries;write_json(out/'validation_retries.json',ledger)
                        retry_log=(out/f'{name}_retry{attempt}.console.log').open('x')
                        proc=subprocess.Popen(command,cwd=plan['source_root'],stdout=retry_log,stderr=retry_log,
                                              env={**os.environ,'PYTHONPATH':plan['source_root']},
                                              stdin=subprocess.DEVNULL,start_new_session=True)
                        running[name]=(proc,retry_log)
                        continue
                    if missing:
                        failed[name] = f'missing_{len(missing)}_after_3_attempts'
                        raise RuntimeError(f'Incomplete evaluation coverage: {name}; completed predictions retained')
                if job['group'] == 'aligned_train' and summary.get('failed', 0):
                    raise RuntimeError('Training has unresolved failures; do not start frozen evaluations')
                done.append(name)
            report=collect(out,plan,done,failed,running)
            state={'status':'running','started':started,'updated':time.time(),'elapsed_seconds':time.time()-started,
                   'pid':os.getpid(),'active':{},'done':done,'failed':failed,'blocked':blocked,
                   'pending':len(pending),'expected_episodes':plan['expected_episodes'],
                   'processed':sum(g['processed'] for g in report['groups'].values())}
            if capacity:
                state['runtime_workers']=capacity()
                state['used_slots']=sum(jobs[n]['slots'] for n in running)
                state['controller_mode']='adopted_live_capacity'
            for name,(p,_) in running.items():
                ledger=load(out/'validation_retries.json') if (out/'validation_retries.json').exists() else {}
                retries=ledger.get(name,[])
                status=Path(retries[-1]['output'] if retries else jobs[name]['output'])/'status.json'
                state['active'][name]={'pid':p.pid,'progress':load(status) if status.exists() else None}
            write_json(out/'status.json',state)
            if pending or running:time.sleep(15)
        state.update(status='finished' if not failed and not blocked else 'finished_with_blocked_jobs',updated=time.time())
        write_json(out/'status.json',state)
        return 0 if not failed and not blocked else 1
    except BaseException as exc:
        for p,_ in running.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,log in running.values():
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        write_json(out/'status.json',{'status':'error','updated':time.time(),'error':type(exc).__name__,
                   'done':done,'failed':failed,'pending':pending,'interrupted_jobs':list(running)})
        raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path)
    p.add_argument('--embedding-cache',type=Path,default=ROOT/'.models')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--training-workers',type=int,default=4)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--include-paper',action='store_true',help='Schedule A streaming in addition to the priority B/control campaign')
    a=p.parse_args();out=a.output.resolve()
    if not a.execute:
        if not a.inventory or not 1<=a.training_workers<=a.workers<=16:
            p.error('Inventory and 1 <= training-workers <= workers <= 16 required')
        prepare(a)
        if a.prepare_only:return
    from dotenv import load_dotenv
    if a.env_file:load_dotenv(a.env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):p.error('OPENAI_API_KEY missing')
    os.environ.setdefault('REASONINGBANK_RATE_FILE', str(out/'request_admission.txt'))
    frozen=out/'source_at_launch'
    if ROOT!=frozen:
        os.environ['PYTHONPATH']=str(frozen);os.chdir(frozen)
        os.execv(sys.executable,[sys.executable,'-u','-m','mind2web_adapter.campaign','--execute','--output',str(out)])
    raise SystemExit(execute(out))


if __name__=='__main__':main()
