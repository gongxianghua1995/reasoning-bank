import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from mind2web_adapter.common import write_json,file_hash
from mind2web_adapter.campaign_live import AdoptedProcess,restore_handoff,capacity_reader


class LiveCampaignTest(unittest.TestCase):
    def test_adopted_process_keeps_live_child_and_requires_final_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p=AdoptedProcess(42,100,Path(d))
            with patch('mind2web_adapter.campaign_live.process_info',return_value={'start_ticks':100,'state':'S'}):
                self.assertIsNone(p.poll())
            with patch('mind2web_adapter.campaign_live.process_info',return_value=None):
                self.assertEqual(p.poll(),1)
                write_json(Path(d)/'summary.json',{'status':'finished_with_failures','processed':8})
                self.assertEqual(p.poll(),0)

    def test_reused_pid_is_not_adopted(self):
        with patch('mind2web_adapter.campaign_live.process_info',return_value={'start_ticks':200,'state':'S'}):
            with self.assertRaises(RuntimeError):AdoptedProcess(42,100,Path('/tmp')).poll()

    def test_capacity_changes_without_restart_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'capacity.json';read=capacity_reader(p)
            write_json(p,{'workers':16});self.assertEqual(read(),16)
            write_json(p,{'workers':24});self.assertEqual(read(),24)
            for value in [True,0,33,'16']:
                write_json(p,{'workers':value})
                with self.assertRaises(ValueError):read()

    def test_handoff_preserves_started_jobs_and_rejects_untracked_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            plan={'jobs':[{'name':n,'output':str(out/n)} for n in ['done','active','pending']]}
            write_json(out/'plan.json',plan)
            h={'output':str(out),'plan_sha256':file_hash(out/'plan.json'),
               'old_pid':1,'old_start_ticks':1,
               'state':{'done':['done'],'failed':{},'blocked':[],'started':50},
               'children':{'active':{'pid':42,'start_ticks':100,'output':str(out/'active')}}}
            with patch('mind2web_adapter.campaign_live.process_info',return_value=None):
                pending,running,done,failed,blocked,costs,started=restore_handoff(out,plan,h)
                self.assertEqual(pending,['pending']);self.assertEqual(done,['done'])
                self.assertEqual(running['active'][0].pid,42);self.assertEqual(started,50)
                running['active'][1].close()
                (out/'pending').mkdir()
                with self.assertRaises(ValueError):restore_handoff(out,plan,h)

    def test_handoff_refuses_two_live_controllers(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);write_json(out/'plan.json',{})
            h={'output':str(out),'plan_sha256':file_hash(out/'plan.json'),'old_pid':1,'old_start_ticks':2}
            with patch('mind2web_adapter.campaign_live.process_info',return_value={'start_ticks':2,'state':'T'}):
                with self.assertRaises(ValueError):restore_handoff(out,{},h)
