"""Real native agents and Docker across two tasks; deterministic LLM only."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from fastembed import TextEmbedding
from swe_adapter import ROOT
from swe_adapter.common import read_record,write_json
from swe_adapter.memory import Memory,learn
from swe_adapter.vendor.workspace import Workspace
from swe_adapter.worker import main as worker


async def run(data,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'swe_adapter/config.json').read_text())
    embedder=TextEmbedding(config['embedding_model'],cache_dir=str(ROOT/'.models'),threads=2,local_files_only=True)
    bank=Memory(out/'memory.json',embedder,config['embedding_model']);results=[]
    for number,iid in enumerate(['django__django-10999','django__django-11141']):
        record=read_record(data,iid);folder=out/iid;folder.mkdir();ws=Workspace(record,config,folder)
        ids,memory,vector,scores=bank.select(iid,record['problem_statement'])
        if number==0:assert not ids and not memory
        else:assert ids==['django__django-10999'] and 'Assert behavior' in memory
        commands=["printf 'value = 17\\n' > reasoningbank_contract.py",
                  "python -c 'import reasoningbank_contract; assert reasoningbank_contract.value == 17; print(\"1 assertion passed\")'",
                  'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff']
        calls=[]
        def complete(**kwargs):
            messages=kwargs['messages'];system=messages[0]['content']
            if 'judges whether' in system:response='success' if number==0 else 'fail'
            elif 'You need to extract and summarize' in system:
                assert ('successfully resolved' in system) if number==0 else ('attempted to resolve' in system)
                response='# Memory Item 1\n## Title Assert behavior\n## Description Use reproducible checks.\n## Content Inspect source and run assertions before claiming success.'
            else:
                if number:assert memory in system
                else:assert 'Below are some memory items' not in system
                response='THOUGHT: contract action\n```bash\n'+commands[len(calls)]+'\n```';calls.append(messages)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response),finish_reason='stop')],
                usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)))
        fake=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        try:
            await ws.prepare();assert not (await ws.export())['patch']
            req=dict(task=record['problem_statement'],selected_memory=memory,memory_ids=ids,config=config,
                     container=ws.main,seconds=120,output_dir=str(folder))
            write_json(folder/'request.json',req)
            with patch('swe_adapter.model.OpenAI',return_value=fake):
                await asyncio.to_thread(worker,str(folder/'request.json'))
                assert json.loads((folder/'result.json').read_text())['exit_status']=='Submitted'
                entry=await asyncio.to_thread(learn,folder,iid,record['problem_statement'],config,vector)
            actual=await ws.export();assert 'reasoningbank_contract.py' in actual['patch']
            bank.commit(entry)
            write_json(folder/'memory_after.json',bank.state)
            results.append(dict(instance_id=iid,selected_ids=ids,memory_items=len(entry['memory_items']),self_judgment=entry['status']))
        finally:await ws.close()
    assert len(bank.state['entries'])==2
    write_json(out/'report.json',dict(passed=True,real_api_calls=0,actual_embeddings=True,
        agent='native mini 1.10.0',real_docker=True,results=results,
        success_and_failure_prompts_exercised=True,note='Deterministic responses; not a task-solving score'))
    print((out/'report.json').read_text())


if __name__=='__main__':asyncio.run(run(*sys.argv[1:]))
