"""Run upstream ReasoningBank DefaultAgent in an existing offline workspace."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent import package_dir
from .common import event, write_json
from .model import Model, DeadlineExceeded


def main(path):
    req=json.loads(Path(path).read_text());out=Path(req['output_dir']);config=req['config']
    deadline=time.monotonic()+req['seconds']

    class Environment:
        def get_template_vars(self):return {'cwd':'/testbed'}

        def execute(self, command, cwd=''):
            remaining=deadline-time.monotonic()
            if remaining<=1:raise DeadlineExceeded('Command deadline reached')
            seconds=max(1,min(config['command_timeout'],int(remaining)))
            event(out,'action_started',command=command)
            argv=['docker','exec','-w',cwd or '/testbed','-e','BASH_ENV=/root/.bashrc',req['container'],
                  'timeout','-k','2',str(seconds),'bash','-o','pipefail','-c',command]
            try:
                p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',timeout=seconds+5)
                result={'returncode':p.returncode,'output':p.stdout}
            except subprocess.TimeoutExpired as exc:
                value=exc.output or b''
                result={'returncode':124,'output':value.decode(errors='replace') if isinstance(value,bytes) else value}
            event(out,'action',command=command,result=result)
            return result

    class ObservedAgent(DefaultAgent):
        # Native run/query/parse_action/execute_action and selected_memory remain unchanged.
        def add_message(self, role, content, **kwargs):
            super().add_message(role,content,**kwargs)
            write_json(out/'trajectory.json',{'messages':self.messages,'agent_class':'minisweagent.agents.default.DefaultAgent'})

    cfg=yaml.safe_load((package_dir/'config/extra/swebench.yaml').read_text())['agent']
    cfg.update(step_limit=config['steps'],cost_limit=0)
    cfg['instance_template']+='''
Execution conditions: this container has no external network. Use installed dependencies.
Do not inspect future Git history, alter .git, or modify tracked tests.
Temporary checks go in /tmp. Generation has a bounded wall time; save source edits
as you work. The host independently exports actual source changes even on timeout.
'''
    model=Model(config,out,'solve',deadline,req['selected_memory'])
    agent=ObservedAgent(model,Environment(),**cfg)
    started=time.monotonic()
    event(out,'worker_started',agent_version='1.10.0',memory_ids=req['memory_ids'])
    try:
        status,submission=agent.run(req['task'],selected_memory=req['selected_memory'])
    except Exception as exc:
        status,submission=type(exc).__name__,''
        event(out,'worker_error',error=status)
    finally:
        write_json(out/'trajectory.json',{'messages':agent.messages,'agent_class':'minisweagent.agents.default.DefaultAgent'})
    write_json(out/'result.json',dict(exit_status=status,submission=submission,
               n_calls=model.n_calls,elapsed_seconds=time.monotonic()-started))
    event(out,'worker_finished',exit_status=status,n_calls=model.n_calls)


if __name__=='__main__':main(sys.argv[1])
