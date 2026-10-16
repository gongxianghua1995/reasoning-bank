import hashlib
import json
from pathlib import Path
import time


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def event(out, name, **values):
    with (Path(out)/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name,time=time.time(),**values),ensure_ascii=False)+'\n')


def read_record(path, instance_id):
    rows=json.loads(Path(path).read_text())
    matches=[r for r in rows if r['instance_id']==instance_id]
    if len(matches)!=1:raise ValueError('Expected exactly one instance: '+instance_id)
    record=matches[0]
    for field in ['instance_id','repo','base_commit','problem_statement','version','test_patch','FAIL_TO_PASS','PASS_TO_PASS']:
        if field not in record:raise ValueError('Missing Verified field: '+field)
    if record.get('dockerhub_tag'):raise ValueError('Only Verified is supported')
    return record
