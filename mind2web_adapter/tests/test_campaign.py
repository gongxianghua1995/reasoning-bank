import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.common import load, write_json, file_hash
from mind2web_adapter.campaign import prepare,execute,choose_checkpoint,TEST_SPLITS


class CampaignTest(unittest.TestCase):
    def test_selection_requires_full_coverage_and_ties_choose_earlier(self):
        def rows(value):return [{'case_id':c,'metrics':{'step_success_rate':value}} for c in ['a','b']]
        self.assertEqual(choose_checkpoint([(1,rows(.5),['a','b']),(2,rows(.5),['a','b'])])['selected_epoch'],1)
        self.assertEqual(choose_checkpoint([(1,rows(.5),['a','b']),(2,rows(.7),['a','b'])])['selected_epoch'],2)
        with self.assertRaises(ValueError):choose_checkpoint([(1,rows(.5)[:1],['a','b'])])

    def scenario(self, retry=False, handoff=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'campaign';inventory=root/'inventory'
            write_json(inventory/'manifest.json',{})
            counts={'train':303,'validation':101,'test_task':177,'test_website':142,'test_domain':694}
            ids={k:[k+str(i) for i in range(n)] for k,n in counts.items()}
            manifest={'diagnostic':False,'parts':ids,'counts':counts,'source_files':{},'split_sha256':'fixed'}
            a=argparse.Namespace(inventory=inventory,output=out,embedding_cache=root/'models',workers=8,training_workers=4,include_paper=True)
            with patch('swarmevo.benches.mind2web.dataset.load_inventory',return_value=(manifest,ids)):
                plan=prepare(a)
            self.assertEqual(plan['expected_episodes'],3948)
            paper=[j for j in plan['jobs'] if j['group']=='paper']
            self.assertEqual(len(paper),3)
            self.assertEqual([j['case_ids'] for j in paper],[ids[s] for s in TEST_SPLITS])
            jobs={j['name']:j for j in plan['jobs']}
            launched=[]
            def popen(cmd,**kwargs):
                folder=Path(cmd[cmd.index('--output')+1]);name=folder.name;j=jobs[name.split('_retry')[0]]
                selected=load(cmd[cmd.index('--case-ids-file')+1]) if '--case-ids-file' in cmd else j['case_ids']
                for dep in j['depends']:
                    if dep=='select_checkpoint':self.assertTrue((out/'selection.json').exists())
                    else:self.assertTrue((Path(jobs[dep]['output'])/'summary.json').exists())
                launched.append(name)
                if j['group']=='aligned_train':
                    for e in (1,2):write_json(folder/'checkpoints'/f'epoch_{e}'/'memory.json',{'epoch':e})
                rows=[{'case_id':c,'metrics':{'step_success_rate':.7 if j['group']=='validation_epoch2' else .5,
                    'element_accuracy':.6,'operation_f1':.8,'candidate_recall':1.,'task_success_rate':0.,'n_steps':1}} for c in selected]
                if retry and name=='validation_epoch1_validation_00':rows=rows[1:]
                write_json(folder/'results.json',rows)
                write_json(folder/'case_outcomes.json',[{'case_id':c} for c in selected])
                write_json(folder/'summary.json',{'status':'finished','processed':len(selected)})
                class Process:
                    pid=123
                    returncode=0
                    def poll(self):return 0
                return Process()
            adopted=None
            if handoff:
                popen(jobs['aligned_train']['command'])
                adopted={'output':str(out),'plan_sha256':file_hash(out/'plan.json'),
                         'old_pid':99,'old_start_ticks':1,
                         'state':{'done':[],'failed':{},'blocked':[],'started':1},
                         'children':{'aligned_train':{'pid':123,'start_ticks':2,
                                     'output':jobs['aligned_train']['output']}}}
            with patch('mind2web_adapter.campaign.subprocess.Popen',side_effect=popen), \
                 patch('mind2web_adapter.campaign.time.sleep'),patch('mind2web_adapter.campaign.signal.signal'), \
                 patch('mind2web_adapter.campaign_live.process_info',return_value=None):
                self.assertEqual(execute(out,handoff=adopted,capacity=(lambda:16) if handoff else None),0)
            self.assertEqual(load(out/'selection.json')['selected_epoch'],2)
            self.assertEqual(load(out/'status.json')['status'],'finished')
            self.assertEqual(load(out/'status.json')['processed'],3948)
            self.assertEqual(len(launched),len(jobs)-1+int(retry))

    def test_full_dag_keeps_streams_whole_and_selects_before_frozen_tests(self):
        self.scenario()

    def test_validation_retries_only_missing_cases_before_selection(self):
        self.scenario(retry=True)

    def test_adopted_training_finishes_dag_without_repeating_it(self):
        self.scenario(retry=True,handoff=True)
