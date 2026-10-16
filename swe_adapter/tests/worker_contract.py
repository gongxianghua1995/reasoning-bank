"""Native mini 1.10.0 plus actual Docker, deterministic API responses."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys

from swe_adapter import ROOT
from swe_adapter.common import read_record,write_json
from swe_adapter.vendor.workspace import Workspace
from swe_adapter.worker import main as worker


async def run(data,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    record=read_record(data,'django__django-10999')
    cfg=json.loads((ROOT/'swe_adapter/config.json').read_text());ws=Workspace(record,cfg,out)
    memory='Check literal template syntax {{ unknown.value }} {% load fixture %}.'
    try:
        await ws.prepare()
        req=dict(task=record['problem_statement'],selected_memory=memory,memory_ids=['prior'],config=cfg,
                 container=ws.main,seconds=120,output_dir=str(out))
        write_json(out/'request.json',req)
        commands=["printf 'value = 17\\n' > reasoningbank_contract.py",
                  "python -c 'import reasoningbank_contract; assert reasoningbank_contract.value == 17; print(\"1 assertion passed\")'",
                  'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff']
        calls=[]
        def complete(**kwargs):
            messages=kwargs['messages'];assert memory in messages[0]['content']
            assert 'past interaction' in messages[0]['content']  # native injection
            command=commands[len(calls)];calls.append(messages)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='THOUGHT: contract action\n```bash\n'+command+'\n```'),finish_reason='stop')],
                                   usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)))
        fake=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        with patch('swe_adapter.model.OpenAI',return_value=fake):
            await asyncio.to_thread(worker,str(out/'request.json'))
        result=json.loads((out/'result.json').read_text());assert result['exit_status']=='Submitted',result
        actual=await ws.export();assert 'reasoningbank_contract.py' in actual['patch']
        # git diff submission misses an untracked file; independent export recovers it.
        assert not result['submission'].strip()
        events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
        requests=[e for e in events if e['event']=='model_request'];assert len(requests)==3
        assert all(e['memory_in_system'] for e in requests)
        write_json(out/'report.json',dict(passed=True,agent='vendored DefaultAgent 1.10.0',real_api_calls=0,
            memory_in_all_requests=True,literal_template_preserved=True,real_source_edit=True,
            host_recovers_untracked_patch=True,native_exit_status=result['exit_status']))
        print((out/'report.json').read_text())
    finally:await ws.close()


if __name__=='__main__':asyncio.run(run(*sys.argv[1:]))
