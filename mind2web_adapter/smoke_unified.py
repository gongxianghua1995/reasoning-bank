"""Real two-epoch training and paired frozen/no-memory validation gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import file_hash, load, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(a.env_file,override=False)
    os.environ['REASONINGBANK_RATE_FILE']=str(out/'request_admission.txt')
    from swarmevo.benches.mind2web.dataset import load_inventory
    manifest,parts=load_inventory(a.inventory,seed=42)
    # Fixed small operational probe; no ranking by model outcomes.
    val=[case.id for case in parts['validation'] if 2<=case.payload['n_steps']<=6][:2]
    train_ids=[case.id for case in parts['train'] if 2<=case.payload['n_steps']<=6][:2]
    if len(val)!=2 or len(train_ids)!=2:raise ValueError('Need two complete short tasks per split')
    write_json(out/'validation_ids.json',val)
    write_json(out/'train_ids.json',train_ids)
    config=ROOT/'mind2web_adapter/config_aligned.json'
    write_json(out/'protocol.json',{'inventory':str(a.inventory.resolve()),'inventory_sha256':file_hash(a.inventory/'manifest.json'),
        'split_sha256':manifest['split_sha256'],'train_ids':train_ids,
        'validation_ids':val,'selection':'first two IDs with 2..6 published steps in each fixed split order',
        'config':load(config)})
    def run(name,cmd):
        write_json(out/f'{name}_command.json',cmd)
        with (out/f'{name}.console.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=os.environ,stdout=log,stderr=log,check=True)
        return load(out/name/'summary.json')
    base=[sys.executable,'-u','-m']
    train=run('train',base+['mind2web_adapter.aligned','--inventory',str(a.inventory.resolve()),
        '--config',str(config),'--epochs','2','--batch-size','1','--workers','1','--case-ids-file',str(out/'train_ids.json'),
        '--output',str(out/'train')])
    if train['processed']!=4 or train['completed']!=4 or train['failed']:
        raise ValueError('Training smoke did not complete every episode')
    memory=out/'train/checkpoints/epoch_2/memory.json';before=file_hash(memory)
    def evaluation(name):
        cmd=base+['mind2web_adapter.run','--inventory',str(a.inventory.resolve()),'--config',str(config),
            '--mode','eval','--scope','global' if name=='frozen' else 'none','--split','validation',
            '--case-ids-file',str(out/'validation_ids.json'),'--output',str(out/name)]
        if name=='frozen':cmd+=['--memory-from',str(memory)]
        return name,run(name,cmd)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=dict(pool.map(evaluation,['frozen','none']))
    for value in results.values():
        if value['processed']!=2 or value['completed']!=2 or value['failed']:
            raise ValueError('Evaluation smoke coverage incomplete')
    if file_hash(memory)!=before or not results['frozen']['frozen_memory_unchanged']:
        raise ValueError('Evaluation changed memory')
    def public_inputs(name):
        return {x['case_id']:x for f in (out/name/'cases').glob('*/public_case.json') for x in [load(f)]}
    if public_inputs('frozen')!=public_inputs('none'):
        raise ValueError('Paired evaluation inputs differ')
    events=[];references=0;retrieved=0
    for folder in (out/'train/cases').iterdir():
        signal=load(folder/'learning_signal.json')
        if signal['source']!='official' or not signal['reference_actions_supplied'] or (folder/'self_judgment.json').exists():
            raise ValueError('Wrong training feedback boundary')
        references+=len(load(folder/'reference_actions.json'))
        retrieved+=bool(load(folder/'retrieval.json')['episode_ids'])
    for f in out.glob('*/cases/*/events.jsonl'):
        events.extend(json.loads(line) for line in f.read_text().splitlines())
    errors=[e for e in events if e['event']=='model_error']
    responses=[e for e in events if e['event']=='model_response']
    if not retrieved or not any(e.get('reasoning_chars',0)>0 for e in responses):
        raise ValueError('Memory retrieval or active thinking was not exercised')
    report={'status':'passed','updated':time.time(),'train':train,'evaluation':results,
        'memory_sha256':before,'matched_public_inputs':True,'failed_step_references':references,
        'episodes_retrieving_memory':retrieved,'model_errors':len(errors),'model_responses':len(responses),
        'returned_tokens':sum((e.get('usage') or {}).get('total_tokens',0) for e in responses),
        'thinking_observed':True,'input_mode':'text','interpretation':'Operational protocol gate; small-sample scores are not a gain estimate.'}
    write_json(out/'gate.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
