import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np

from swe_adapter.memory import Memory,parse_items,extract_memory
from swe_adapter.vendor.mini_benchmark import task_text
from swe_adapter.model import Model


class Embeddings:
    def embed(self, texts):
        for text in texts:yield np.array([1,0] if 'parser' in text else [0,1],dtype=float)


class ContractTest(unittest.TestCase):
    def test_memory_recovers_from_shell_action_without_continuing_task(self):
        invalid='THOUGHT: Run more tests.\n```bash\npython tests/runtests.py\n```'
        valid='# Memory Item 1\n## Title Verify regressions\n## Description Test related behavior\n## Content Run local regression tests.'
        with tempfile.TemporaryDirectory() as d, patch('swe_adapter.memory.Model') as model:
            model.return_value.complete.side_effect=[invalid,valid]
            out=Path(d)
            entry=extract_memory(out,'task','public issue',{'extract_seconds':180,'extract_temperature':1},
                                 [1.0],[{'role':'assistant','content':'Continue running commands.'}],'success')
            self.assertEqual(len(entry['memory_items']),1)
            self.assertEqual(model.call_count,1)  # One shared deadline, not a fresh retry budget.
            self.assertEqual(model.return_value.complete.call_count,2)
            self.assertEqual((out/'memory_raw_attempt_1.md').read_text(),invalid)
            request=json.loads((out/'extract_request.json').read_text())
            self.assertIn('END RECORDED TRAJECTORY',request[1]['content'])
            self.assertIn('historical messages',request[1]['content'])

    def test_invalid_memory_retry_is_bounded_and_never_creates_entry(self):
        with tempfile.TemporaryDirectory() as d, patch('swe_adapter.memory.Model') as model:
            model.return_value.complete.return_value='```bash\necho keep working\n```'
            out=Path(d)
            with self.assertRaises(ValueError):
                extract_memory(out,'task','issue',{'extract_seconds':180,'extract_temperature':1},
                               [1.0],[{'role':'assistant','content':'history'}],'fail')
            self.assertEqual(model.return_value.complete.call_count,2)
            self.assertFalse((out/'memory_entry.json').exists())

    def test_memory_uses_prior_tasks_and_persists_query_embedding_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'memory.json';bank=Memory(p,Embeddings(),'fake')
            ids,text,vector,_=bank.select('first','parser issue')
            self.assertEqual((ids,text),([],''))
            bank.commit(dict(task_id='first',query='parser issue',embedding=vector,memory_items=['parser lesson']))
            bank.commit(dict(task_id='second',query='database issue',embedding=[0.,1.],memory_items=['database lesson']))
            restored=Memory(p,Embeddings(),'fake')
            ids,text,_,_=restored.select('third','parser problem')
            self.assertEqual((ids,text),(['first'],'parser lesson'))
            with self.assertRaises(ValueError):restored.select('first','parser issue')
            before=p.read_bytes()
            with self.assertRaises(ValueError):restored.commit(restored.state['entries'][0])
            self.assertEqual(p.read_bytes(),before)

    def test_memory_parser_keeps_complete_items_not_blank_line_fragments(self):
        text='# Memory Item 1\n## Title Test\n\n## Description Verify behavior\n## Content Use assertions.\n'
        self.assertEqual(len(parse_items(text)),1)
        with self.assertRaises(ValueError):parse_items('# Memory Item 1\n## Title incomplete')

    def test_public_task_excludes_evaluation_metadata(self):
        record={'problem_statement':'PUBLIC','test_patch':'PRIVATE','patch':'GOLD','FAIL_TO_PASS':['PRIVATE_TEST']}
        self.assertEqual(task_text(record),'PUBLIC')

    def test_exhausted_provider_budget_stops_without_retry(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError('Budget has been exceeded!')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as d:
            model=Model(dict(model='fake',temperature=0,max_tokens=10,api_timeout=5),
                        Path(d),'solve',time.monotonic()+60,client=client)
            with self.assertRaises(RuntimeError):model.query([{'role':'user','content':'task'}])
            self.assertEqual(len(calls),1)
            events=[json.loads(line) for line in (Path(d)/'events.jsonl').read_text().splitlines()]
            self.assertTrue(events[-1]['quota_exhausted'])


if __name__=='__main__':unittest.main()
