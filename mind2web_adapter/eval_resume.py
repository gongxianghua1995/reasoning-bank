"""Resume an interrupted no-memory evaluation using its original frozen agent.

Only completed steps are reused; original elapsed action time is charged. Queue
waiting is outside the task budget. Full original partial artifacts are archived.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time


def load(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def prepare(folder, stopped_at):
    """The caller must first stop this evaluation process, then terminate it."""
    folder=Path(folder);options=load(folder/'options.json')
    if options['mode']!='eval' or options['scope']!='none':
        raise ValueError('Only no-memory evaluation may use this recovery')
    if load(folder/'memory.json')['entries']:raise ValueError('Expected empty memory')
    archive=folder/'runtime_resume_archive';archive.mkdir(exist_ok=False)
    for n in ['options.json','config.json','manifest.json','memory.json','status.json',
              'results.json','case_outcomes.json','failures.json']:
        if (folder/n).exists():shutil.copy2(folder/n,archive/n)
    outcomes=load(folder/'case_outcomes.json') if (folder/'case_outcomes.json').exists() else []
    episodes={x['episode_id'] for x in outcomes}
    partial=[]
    for case in (folder/'cases').iterdir():
        identity=load(case/'case_identity.json')
        if identity['episode_id'] not in episodes:partial.append((case,identity))
    if len(partial)>1:raise ValueError('Multiple incomplete tasks')
    recovery={'source':str(folder),'failed_episode':None,'kind':'actions','previous_elapsed':0,
              'stopped_at':stopped_at,'completed_prefix':len(outcomes)}
    if partial:
        case,identity=partial[0];shutil.copytree(case,archive/case.name)
        actions=load(case/'predictions_local.json') if (case/'predictions_local.json').exists() else []
        trajectory=load(case/'trajectory.json') if (case/'trajectory.json').exists() else []
        trajectory=trajectory[:len(actions)]
        if [x.get('final_action') for x in trajectory]!=actions:
            raise ValueError('Partial action checkpoint is inconsistent')
        events=[json.loads(line) for line in (case/'events.jsonl').read_text().splitlines()]
        starts=[x['time'] for x in events if x['event']=='step_started']
        elapsed=max(0,stopped_at-min(starts)) if starts else 0
        recovery.update(failed_episode=identity['episode_id'],previous_elapsed=elapsed,
                        reused_steps=len(actions))
        write(case/'predictions_local.json',actions);write(case/'trajectory.json',trajectory)
    rows=load(folder/'results.json') if (folder/'results.json').exists() else []
    # A result written before its outcome ledger may be scored again from saved
    # predictions, but must never be counted twice or cause a repeated model call.
    write(folder/'results.json',[x for x in rows if x['episode_id'] in episodes])
    write(folder/'case_outcomes.json',outcomes)
    if not (folder/'failures.json').exists():write(folder/'failures.json',[])
    write(folder/'runtime_resume.json',recovery)
    return recovery


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--env-file',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();campaign=a.campaign.resolve()
    from dotenv import load_dotenv
    load_dotenv(a.env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):raise ValueError('Missing model credentials')
    # These four processes reserve existing scheduler slots while waiting. They
    # resume once four other slots have drained, so the actual load stays <=12.
    while True:
        state=load(campaign/'status.json')
        limit=load(campaign/'runtime_concurrency/capacity.json')['workers']
        if state.get('runtime_workers')==limit and state.get('used_slots',999)<=limit:break
        time.sleep(5)
    frozen=out/'source_at_run';sys.path.insert(0,str(frozen))
    import mind2web_adapter
    mind2web_adapter.__path__=[str(frozen/'mind2web_adapter')]
    from mind2web_adapter import run
    if Path(run.__file__).resolve()!=frozen/'mind2web_adapter/run.py':
        raise ValueError('Recovery must import original frozen runner')
    original_load=run.load;original_write=run.write_json
    options=original_load(out/'options.json');recovery=original_load(out/'runtime_resume.json')
    def runtime_load(path):
        if Path(path)==out/'options.json':return {**options,'resume_from':str(out)}
        if Path(path)==out/'resume.json':return recovery
        return original_load(path)
    def runtime_write(path,value):
        # Preserve original manifest; record the resumed invocation separately.
        target=out/'runtime_resume_manifest.json' if Path(path)==out/'manifest.json' else path
        return original_write(target,value)
    run.load=runtime_load;run.write_json=runtime_write
    write(out/'runtime_resume_started.json',{'time':time.time(),'queue_wait_excluded':True,
        'previous_elapsed':recovery['previous_elapsed'],'source':str(run.__file__)})
    raise SystemExit(run.execute_run(out))


if __name__=='__main__':main()
