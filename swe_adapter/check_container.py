"""Real disposable-container contract checks with zero model calls."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from .run import read_record
from .vendor.workspace import Workspace
from .common import write_json


async def check(record,out):
    ws=Workspace(record,{'task_seconds':180},out,sys.executable)
    checks=[]
    try:
        await ws.prepare()
        assert not (await ws.export())['patch']
        checks.append('initial_empty_patch')
        result=await ws.shell(ws.main,'python -c "import sys; print(sys.executable)"')
        checks.append('activated_python: '+result['output'].strip())
        await ws.shell(ws.main,"printf 'value = 17\\n' > camel_smoke_probe.py\nmkdir -p tests\nprintf 'assert True\\n' > tests/test_camel_smoke_probe.py\ngit add camel_smoke_probe.py")
        result=await ws.shell(ws.main,'python -c "import camel_smoke_probe; assert camel_smoke_probe.value == 17"')
        assert result['returncode']==0
        actual=await ws.export()
        assert 'camel_smoke_probe.py' in actual['patch']
        assert 'test_camel_smoke_probe.py' not in actual['patch']
        checks.extend(['real_edit_and_assertion','staged_file_export','test_file_exclusion'])
        snap=await ws.start_workspace('snapshot',actual['patch'])
        await ws.shell(snap,'python -c "import camel_smoke_probe; assert camel_smoke_probe.value == 17"')
        await ws.shell(snap,"printf 'value = 99\\n' > camel_smoke_probe.py")
        await ws.shell(ws.main,'python -c "import camel_smoke_probe; assert camel_smoke_probe.value == 17"')
        checks.append('snapshot_patch_transfer_and_isolation')
        timeout=await ws.shell(ws.main,'sleep 3',seconds=1,check=False)
        assert timeout['returncode']==124
        checks.append('container_command_timeout')
        (out/'selected.patch').write_text(actual['patch'])
    finally:
        await ws.close()
    write_json(out/'report.json',dict(passed=True,checks=checks,model_calls=0,containers_remaining=list(ws.containers)))
    print(json.dumps({'passed':True,'checks':checks}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',type=Path,required=True)
    p.add_argument('--instance-id',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    asyncio.run(check(read_record(args.data_path,args.instance_id),args.output_dir))


if __name__=='__main__':
    main()
