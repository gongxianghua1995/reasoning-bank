import tempfile
from pathlib import Path
import unittest
from mind2web_adapter.eval_resume import prepare,write,load


class EvalResumeTest(unittest.TestCase):
    def test_reuses_only_committed_steps_and_preserves_elapsed_and_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case=root/'cases'/'one'
            write(root/'options.json',{'mode':'eval','scope':'none'})
            write(root/'memory.json',{'entries':[]})
            write(root/'results.json',[{'episode_id':'epoch-1:a'}])
            write(case/'case_identity.json',{'episode_id':'epoch-1:a'})
            write(case/'predictions_local.json',[{'element_id':'1'}])
            write(case/'trajectory.json',[{'final_action':{'element_id':'1'}},{'attempts':[]}])
            (case/'events.jsonl').write_text('{"event":"step_started","time":100}\n')
            r=prepare(root,150)
            self.assertEqual(r['previous_elapsed'],50);self.assertEqual(r['reused_steps'],1)
            self.assertEqual(len(load(case/'trajectory.json')),1)
            self.assertEqual(len(load(root/'runtime_resume_archive/one/trajectory.json')),2)
            self.assertEqual(load(root/'results.json'),[])

    def test_refuses_memory_learning_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'options.json',{'mode':'stream','scope':'global'})
            with self.assertRaises(ValueError):prepare(root,1)
