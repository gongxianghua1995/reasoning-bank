"""Streaming must learn only from preceding tasks, without evaluator feedback."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.agent import parse_action
from mind2web_adapter.common import load, write_json
from mind2web_adapter.memory import Bank
from mind2web_adapter.run import execute_run, validate_protocol
from mind2web_adapter.tests.test_contract import CONFIG, ITEM, Embeddings, task_fixture
from swarmevo.benches.protocol import Case


class StreamTest(unittest.TestCase):
    def test_stream_rejects_import_repeat_and_shards(self):
        base = {'mode': 'stream', 'epochs': 1}
        validate_protocol(base)
        for key, value in [('epochs', 2), ('memory_from', 'previous.json'),
                           ('case_ids_file', 'shard.json')]:
            with self.assertRaises(ValueError):
                validate_protocol({**base, key: value})

    def test_memory_applicability_covers_every_item(self):
        action = {'element_id': '1', 'operation': 'CLICK', 'value': ''}
        with self.assertRaises(ValueError):
            parse_action(json.dumps(action), ['1'], 2)
        action['memory_usage'] = [{'item': 1, 'use': False, 'reason': 'Different page'},
                                  {'item': 2, 'use': True, 'reason': 'Matching search input'}]
        self.assertEqual(parse_action(json.dumps(action), ['1'], 2), action)
        action['memory_usage'][1]['item'] = 1
        with self.assertRaises(ValueError):
            parse_action(json.dumps(action), ['1'], 2)

    def test_plain_query_does_not_embed_instruction(self):
        class Spy(Embeddings):
            texts = []
            def embed(self, texts):
                self.texts.extend(texts)
                return super().embed(texts)
        with tempfile.TemporaryDirectory() as root:
            spy = Spy()
            bank = Bank(Path(root)/'memory.json', spy, {'retrieval': 'plain'}, 'global', query_format='plain')
            bank.commit({'case_id': 'old', 'episode_id': 'epoch-1:old', 'website': 'shop',
                         'embedding': [1., 0.], 'memory_items': [ITEM]})
            bank.select('new', 'shop', 'search')
            self.assertEqual(spy.texts, ['search', 'search'])

    def scenario(self, scope):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inventory, weights = root/'inventory', root/'weights'
            write_json(inventory/'manifest.json', {})
            weights.mkdir(); (weights/'fixture').write_text('local fake weights')
            ids = ['first', 'second', 'third']
            cases = [Case(cid, 'shop', {'sha256': 'fixture'}) for cid in ids]
            manifest = {'diagnostic': False, 'seed': 42, 'split_sha256': 'fixture', 'cache_identity': {},
                        'parts': {'train': ['unrelated-training'], 'validation': ids},
                        'counts': {}, 'website_counts': {}, 'source_files': {}}
            config = {**CONFIG, 'model': 'fake', 'embedding_model': 'fake', 'embedding_threads': 1,
                      'embedding_query_format': 'plain', 'memory_usage_review': True, 'offline_memory_guard': True}
            options = {'mode': 'stream', 'scope': scope, 'inventory': str(inventory), 'seed': 42,
                       'split': 'validation', 'max_cases': None, 'epochs': 1,
                       'embedding_cache': str(weights), 'memory_from': None}
            def read(case):
                task = task_fixture(); task['annotation_id'] = case.id
                return task
            # Each independent stream starts with no previous run's memory.
            for run in range(2):
                out = root / f'run{run}'
                write_json(out/'options.json', options)
                write_json(out/'config.json', config)
                write_json(out/'source_hashes.json', {})
                calls = []
                def complete(model, messages, temperature):
                    cid = load(model.out/'case_identity.json')['case_id']
                    calls.append((cid,model.phase))
                    stored = load(out/'memory.json')['entries']
                    expected = ids[:ids.index(cid)] if scope != 'none' else []
                    self.assertEqual([e['case_id'] for e in stored], expected)
                    if model.phase in {'judge', 'extract'}:
                        self.assertFalse((model.out/'metrics.json').exists())
                        self.assertIn('Never infer that a prediction caused', messages[0]['content'] if model.phase=='judge' else messages[-1]['content'])
                    if model.phase == 'judge':return 'fail'
                    if model.phase == 'extract':return ITEM
                    obs = json.loads(messages[-1]['content'])
                    action = {'element_id': obs['candidate_ids'][0], 'operation': 'CLICK', 'value': ''}
                    if stored:
                        self.assertIn('assess each supplied memory', messages[0]['content'])
                        action['memory_usage'] = [{'item':1,'use':False,'reason':'Not applicable to current page'}]
                    return json.dumps(action)
                with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest,{'validation':cases})), \
                     patch('swarmevo.benches.mind2web.dataset.read_case',side_effect=read), \
                     patch('fastembed.TextEmbedding',return_value=Embeddings()), \
                     patch('mind2web_adapter.model.Model.complete',new=complete), \
                     patch.dict('os.environ',{'OPENAI_API_KEY':'fake'}):
                    self.assertEqual(execute_run(out),0)
                rows = load(out/'results.json')
                self.assertEqual([r['case_id'] for r in rows],ids)
                self.assertEqual(load(out/'summary.json')['learning_protocol'],'stream')
                if scope == 'none':
                    self.assertTrue(all(phase=='solve' for _,phase in calls))
                    self.assertEqual(load(out/'memory.json')['entries'],[])
                else:
                    self.assertEqual([phase for _,phase in calls],['solve','solve','judge','extract']*3)
                    self.assertEqual(rows[0]['selected_memory_ids'],[])
                    self.assertEqual(rows[1]['selected_memory_ids'],['epoch-1:first'])
                    self.assertTrue(set(rows[2]['selected_memory_ids']) <= {'epoch-1:first','epoch-1:second'})

    def test_sequential_learning_no_future_memory_and_split_reset(self):
        self.scenario('global')

    def test_no_memory_stream_has_no_learning_calls(self):
        self.scenario('none')


if __name__ == '__main__':
    unittest.main()
