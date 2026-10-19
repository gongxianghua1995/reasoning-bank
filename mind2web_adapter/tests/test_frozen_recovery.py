from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from mind2web_adapter.common import load, write_json, file_hash, digest
from mind2web_adapter.frozen_recovery import restore_prefix, execute_windows
from mind2web_adapter.model import DeadlineExceeded
from mind2web_adapter.continue_frozen import execute as continue_campaign


class FrozenRecoveryTest(unittest.TestCase):
    def test_preserves_earliest_prefix_rejects_different_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public={'case_id':'a','observations':[{'html':str(i)} for i in range(3)]}
            config={'model':'same'};identity={'case_id':'a'}
            write_json(root/'memory.json',{'entries':[]})
            retrieval={'episode_ids':[],'selected_memory':'','scope':'none',
                       'memory_before_sha256':file_hash(root/'memory.json')}
            def source(name,actions):
                folder=root/name;case=folder/'cases'/f"01_{digest('a')[:16]}"
                for filename,value in [('config.json',config),('options.json',{'mode':'eval'}),('memory.json',{'entries':[]})]:
                    write_json(folder/filename,value)
                for filename,value in [('public_case.json',public),('case_identity.json',identity),
                    ('retrieval.json',retrieval),('predictions_local.json',actions),
                    ('trajectory.json',[{'observation':obs,'final_action':act} for obs,act in zip(public['observations'],actions)])]:
                    write_json(case/filename,value)
                return folder
            first=source('first',[None]);independent=source('independent',[{'element_id':'x'},None])
            extension=source('extension',[None,{'element_id':'y'}])
            out=root/'out';out.mkdir()
            self.assertEqual(restore_prefix([first,independent,extension],out,public,config,retrieval,identity),2)
            self.assertEqual(load(out/'predictions_local.json'),[None,{'element_id':'y'}])
            self.assertFalse(load(out/'eval_recovery.json')['candidates'][1]['accepted'])
            write_json(extension/'memory.json',{'entries':['changed']})
            with self.assertRaisesRegex(ValueError,'memory differs'):
                restore_prefix([extension],out,public,config,retrieval,identity)

    def test_timeout_keeps_finished_steps_and_fresh_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);calls=[]
            def actor(*args,**kw):
                calls.append(kw)
                if len(calls)==1:
                    write_json(folder/'predictions_local.json',[None])
                    raise DeadlineExceeded('window exhausted')
                return [None,{'element_id':'next'}]
            result=execute_windows({}, {}, folder,'',None,actor,lambda e:isinstance(e,DeadlineExceeded))
            self.assertEqual(len(result),2)
            self.assertEqual(calls,[{'resume_steps':0,'previous_elapsed':0},{'resume_steps':1,'previous_elapsed':0}])

    def test_incomplete_shard_does_not_stop_others_or_repeat_scored_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);recovery=out/'recovery';source=recovery/'source';source.mkdir(parents=True)
            inv=out/'inventory';write_json(inv/'manifest.json',{})
            jobs=[]
            for name,ids in [('bad',['done','missing']),('good',['new'])]:
                ids_file=out/f'{name}.json';write_json(ids_file,ids)
                jobs.append({'name':name,'group':'aligned','split':'validation','case_ids':ids,'expected':len(ids),
                    'output':str(out/name),'command':['python','-m','mind2web_adapter.run','--output',str(out/name),
                        '--case-ids-file',str(ids_file)],'depends':[]})
            metric={'step_success_rate':.5,'element_accuracy':.5,'operation_f1':1.,'task_success_rate':0.,'candidate_recall':1.,'n_steps':1}
            write_json(out/'bad/results.json',[{'case_id':'done','metrics':metric}])
            plan={'jobs':jobs,'workers':1,'expected_episodes':3,'inventory':str(inv),'inventory_sha256':file_hash(inv/'manifest.json')}
            write_json(out/'plan.json',plan);write_json(out/'input_hashes.json',{'plan.json':file_hash(out/'plan.json')})
            write_json(recovery/'source_hashes.json',{})
            commands=[]
            def popen(command,**kwargs):
                target=Path(command[command.index('--output')+1]);ids=load(command[command.index('--case-ids-file')+1]);commands.append(ids)
                self.assertNotIn('done',ids)
                rows=[{'case_id':cid,'metrics':metric} for cid in ids if cid!='missing']
                write_json(target/'results.json',rows)
                write_json(target/'summary.json',{'processed':len(ids)})
                write_json(target/'case_outcomes.json',[{'episode_id':'epoch-1:'+cid} for cid in ids])
                write_json(target/'failures.json',[{'case_id':cid} for cid in ids if cid=='missing'])
                return SimpleNamespace(pid=123,returncode=0,poll=lambda:0)
            with patch('mind2web_adapter.continue_frozen.subprocess.Popen',side_effect=popen), \
                 patch('mind2web_adapter.continue_frozen.time.sleep'),patch('mind2web_adapter.continue_frozen.signal.signal'):
                self.assertEqual(continue_campaign(out,recovery),1)
            state=load(out/'status.json')
            self.assertEqual(state['status'],'incomplete');self.assertIn('good',state['done'])
            self.assertEqual(state['processed'],2)
            self.assertEqual(commands.count(['missing']),3)
