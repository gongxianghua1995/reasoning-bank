"""Audit an existing stream without calling models or modifying predictions."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from .common import write_json


def summarize(out):
    state=json.loads((out/'status.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    rows=[];totals=defaultdict(Counter);network=[]
    for iid in manifest['instance_ids']:
        folder=out/'instances'/f'{out.name}__{iid}'
        if not (folder/'generation.json').exists():continue
        gen=json.loads((folder/'generation.json').read_text())
        report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else {}
        retrieval=json.loads((folder/'retrieval.json').read_text())
        entry=json.loads((folder/'memory_entry.json').read_text()) if (folder/'memory_entry.json').exists() else {}
        judgment=json.loads((folder/'self_judgment.json').read_text()) if (folder/'self_judgment.json').exists() else {}
        task_state=json.loads((folder/'status.json').read_text())
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        calls=Counter(e.get('phase') for e in events if e['event']=='model_response')
        errors=Counter(e.get('error') for e in events if e['event']=='model_error')
        solve_requests=[e for e in events if e['event']=='model_request' and e['phase']=='solve']
        for e in events:
            if e['event']=='model_response':
                for k in ['prompt_tokens','completion_tokens','total_tokens']:
                    totals[e['phase']][k]+=e.get('usage',{}).get(k,0) or 0
        for filename in ['runtime.jsonl','eval_network.jsonl']:
            if (folder/filename).exists():
                network.extend(json.loads(line)['network'] for line in (folder/filename).read_text().splitlines() if '"network"' in line)
        detail_files=list((folder/'logs').glob('run_evaluation/*/*/*/report.json'))
        detail=json.loads(detail_files[0].read_text()).get(iid,{}) if detail_files else {}
        tests=detail.get('tests_status',{})
        counts={name:{key:len(value) for key,value in result.items()} for name,result in tests.items()}
        resolved=iid in report.get('resolved_ids',[])
        row=dict(instance_id=iid,exit_status=gen['exit_status'],generation_seconds=gen['generation_seconds'],
            patch_chars=gen.get('patch_chars'),patch_sha256=gen.get('patch_sha256'),
            selected_memory_ids=retrieval['memory_ids'],memory_in_all_solve_requests=(
                all(e['memory_in_system'] for e in solve_requests) if retrieval['memory_ids'] and solve_requests else None),
            self_judgment=judgment.get('status'),memory_items=len(entry.get('memory_items',[])),
            memory_error=task_state.get('memory_error'),evaluation_error=task_state.get('evaluation_error'),
            model_requests=dict(Counter(e.get('phase') for e in events if e['event']=='model_request')),
            empty_model_responses=sum(e['event']=='model_response' and not e.get('content') for e in events),
            model_responses=dict(calls),model_errors=dict(errors),actions=sum(e['event']=='action' for e in events),
            official_resolved=resolved,official_completed=iid in report.get('completed_ids',[]),
            official_errors=report.get('error_instances'),tests=counts,
            patch_applied=detail.get('patch_successfully_applied'))
        rows.append(row)
    expected=json.loads((out/'source_hashes.json').read_text())
    changed=[name for name,sha in expected.items() if hashlib.sha256((out/'source_at_run'/name).read_bytes()).hexdigest()!=sha]
    memory=json.loads((out/'memory.json').read_text()) if (out/'memory.json').exists() else {'entries':[]}
    summary=dict(status=state['status'],instances=rows,returned_usage_by_phase={k:dict(v) for k,v in totals.items()},
        resolved=sum(r['official_resolved'] for r in rows),official_completed=sum(r['official_completed'] for r in rows),
        memory_entries=len(memory['entries']),source_files_checked=len(expected),changed_frozen_files=changed,
        network_records=len(network),all_recorded_containers_offline=bool(network) and set(network)=={'none'},
        elapsed_seconds=state.get('elapsed_seconds'),embedding_model=manifest['config']['embedding_model'],
        interpretation='Two-task integration smoke; no claim of accuracy improvement. LLM self-judgment is not official scoring.')
    write_json(out/'smoke_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output_dir',type=Path)
    args=p.parse_args();print(json.dumps(summarize(args.output_dir.resolve()),ensure_ascii=False,indent=2))
