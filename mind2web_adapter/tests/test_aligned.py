import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.common import load, write_json, file_hash
from mind2web_adapter.memory import Bank, learn, official_feedback
from mind2web_adapter.aligned import execute, epoch_order
from mind2web_adapter.tests.test_contract import CONFIG, ITEM, FakeModel, Embeddings, task_fixture
from mind2web_adapter.agent import public_case
from swarmevo.benches.protocol import Case
from swarmevo.benches.mind2web.metrics import score_case


class AlignedTest(unittest.TestCase):
    def test_feedback_whitelist_and_no_self_judge(self):
        public=public_case(task_fixture())
        metrics=score_case([None,None],task_fixture())
        metrics['gold_action']='SECRET_GOLD'
        metrics['steps'][0]['gold_value']='SECRET_GOLD'
        feedback=official_feedback(metrics)
        self.assertNotIn('SECRET_GOLD',json.dumps(feedback))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);write_json(out/'trajectory.json',[])
            FakeModel.responses=[ITEM];FakeModel.requests=[]
            entry=learn(public,out,{**CONFIG,'feedback_source':'official'},[1.,0.],'epoch-1:sample',
                        model_factory=FakeModel,metrics=metrics)
            self.assertEqual([p for p,_ in FakeModel.requests],['extract'])
            self.assertEqual(entry['learning_signal'],{'source':'official','status':'fail'})
            self.assertIsNone(entry['self_judgment'])
            request=json.dumps(FakeModel.requests)
            self.assertIn('official_score',request)
            self.assertNotIn('SECRET_GOLD',request)
            self.assertFalse((out/'self_judgment.json').exists())
            with self.assertRaises(ValueError):
                learn(public,out,CONFIG,[1.,0.],'x',model_factory=FakeModel,metrics=metrics)
            with self.assertRaises(ValueError):
                learn(public,out,{**CONFIG,'feedback_source':'official'},[1.,0.],'x',model_factory=FakeModel)

    def test_order_matches_family_grouping_then_epoch_shuffle(self):
        cases=[Case('a','z',{}),Case('b','a',{}),Case('c','z',{})]
        expected=[cases[1],cases[0],cases[2]]
        random.Random(42).shuffle(expected)
        self.assertEqual(epoch_order(cases,42,1),expected)

    def test_previous_epoch_same_case_is_available_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            bank=Bank(Path(d)/'memory.json',Embeddings(),{},'global',query_format='plain')
            bank.commit({'episode_id':'epoch-1:a','case_id':'a','website':'shop',
                         'embedding':[1.,0.],'memory_items':[ITEM]})
            self.assertEqual(bank.select('a','shop','search')[0],[])
            bank.allow_same_case=True
            self.assertEqual(bank.select('a','shop','search')[0],['epoch-1:a'])

    def scenario(self, crash=False):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);inventory=out/'inventory';weights=out/'weights'
            write_json(inventory/'manifest.json',{})
            weights.mkdir();(weights/'fixture').write_text('fake')
            cases=[Case(cid,site,{'sha256':'fixture'}) for cid,site in [('a','z'),('b','a'),('c','z')]]
            manifest={'diagnostic':False,'split_sha256':'fixture','cache_identity':{},
                      'parts':{'train':['a','b','c'],'validation':['v']},'counts':{},'source_files':{}}
            config={**CONFIG,'model':'fake','embedding_model':'fake','embedding_threads':1,
                    'embedding_query_format':'plain','feedback_source':'official'}
            options={'inventory':str(inventory),'embedding_cache':str(weights),'seed':42,
                     'epochs':2,'batch_size':2,'workers':2,'continue_case_errors':False}
            write_json(out/'config.json',config);write_json(out/'options.json',options)
            write_json(out/'source_hashes.json',{})
            write_json(out/'input_hashes.json',{n:file_hash(out/n) for n in ['config.json','options.json']})
            calls=[]
            def read(case):
                task=task_fixture();task['annotation_id']=case.id;task['website']=case.family_id
                return task
            def complete(model,messages,temperature):
                calls.append(model.phase)
                request=json.dumps(messages)
                for secret in ['SECRET_CURRENT_VALUE','SECRET_NEXT_VALUE','HIDDEN_FUTURE_ANSWER']:
                    self.assertNotIn(secret,request)
                if model.phase=='extract':
                    self.assertIn('official_score',request)
                    # Batch barrier: every action in this batch completed before extraction.
                    cid=load(model.out/'case_identity.json')['case_id']
                    epoch=int(model.out.name.split('_')[0])
                    order=epoch_order(cases,42,epoch)
                    i=next(i for i,c in enumerate(order) if c.id==cid)
                    from mind2web_adapter.common import digest
                    for c in order[i//2*2:i//2*2+2]:
                        self.assertTrue((out/'cases'/f'{epoch:02d}_{digest(c.id)[:16]}'/'solve_receipt.json').exists())
                    return ITEM
                self.assertEqual(model.phase,'solve')
                self.assertNotIn('official_score',request)
                obs=json.loads(messages[-1]['content'])
                return json.dumps({'element_id':obs['candidate_ids'][0],'operation':'CLICK','value':''})
            original=write_json;thrown=[]
            def write(path,value):
                if crash and Path(path).name=='committed.json' and not thrown:
                    thrown.append(True);raise RuntimeError('simulated crash after bank replace')
                return original(path,value)
            with patch('swarmevo.benches.mind2web.dataset.load_inventory',return_value=(manifest,{'train':cases})), \
                 patch('swarmevo.benches.mind2web.dataset.read_case',side_effect=read), \
                 patch('fastembed.TextEmbedding',return_value=Embeddings()), \
                 patch('mind2web_adapter.model.Model.complete',new=complete), \
                 patch('mind2web_adapter.aligned.write_json',side_effect=write), \
                 patch.dict('os.environ',{'OPENAI_API_KEY':'fake'}):
                if crash:
                    with self.assertRaises(RuntimeError):execute(out)
                self.assertEqual(execute(out),0)
                self.assertEqual(len(calls),18) # 6 tasks: 2 actions + 1 extraction.
                before=file_hash(out/'memory.json')
                self.assertEqual(execute(out),0)
                self.assertEqual(len(calls),18)
                self.assertEqual(file_hash(out/'memory.json'),before)
            self.assertEqual(len(load(out/'memory.json')['entries']),6)
            for batch in (out/'batches').iterdir():
                initial=load(batch/'memory_before.json')
                allowed={e['episode_id'] for e in initial['entries']}
                tx=load(batch/'transaction.json')
                for row in tx['rows']:
                    self.assertTrue(set(row['selected_memory_ids']) <= allowed)
            self.assertEqual(len(load(out/'checkpoints/epoch_1/memory.json')['entries']),3)
            self.assertEqual(load(out/'summary.json')['processed'],6)

    def test_batch_visibility_barriers_and_idempotent_resume(self):
        self.scenario()

    def test_recovery_after_bank_write_before_commit_receipt(self):
        self.scenario(crash=True)


if __name__=='__main__':unittest.main()
