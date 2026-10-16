import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil

import numpy as np

from mind2web_adapter.agent import execute, original_predictions, public_case
from mind2web_adapter.common import load, write_json
from mind2web_adapter.memory import Bank, learn
from mind2web_adapter.common import file_hash
from mind2web_adapter.run import execute_run
from swarmevo.benches.protocol import Case
from swarmevo.benches.mind2web.dataset import normalize_task
from swarmevo.benches.mind2web.metrics import score_case

CONFIG = {'case_seconds': 60, 'step_seconds': 30, 'temperature': 0,
          'judge_seconds': 30, 'extract_seconds': 30, 'judge_temperature': 0, 'extract_temperature': 0}
ITEM = '# Memory Item 1\n## Title Check the label\n## Description Before selecting\n## Content Match the visible label to the requested operation.'


class FakeModel:
    responses = []
    requests = []

    def __init__(self, config, out, phase, deadline, **kwargs):
        self.phase = phase

    def complete(self, messages, temperature):
        self.requests.append((self.phase, json.loads(json.dumps(messages))))
        return self.responses.pop(0)


class Embeddings:
    def embed(self, texts):
        for text in texts:
            yield np.array([1., 0.] if 'search' in text else [0., 1.])


def task_fixture():
    return normalize_task({'annotation_id': 'sample', 'website': 'shop', 'domain': 'shopping',
        'subdomain': 'general', 'confirmed_task': 'Search for a product',
        'action_reprs': ['HUMAN_PREVIOUS_ACTION', 'HIDDEN_FUTURE_ANSWER'],
        'actions': [
            {'action_uid': 'a', 'cleaned_html': '<html><button backend_node_id="10">Search</button></html>',
             'operation': {'op': 'CLICK', 'value': 'SECRET_CURRENT_VALUE'},
             'pos_candidates': [{'backend_node_id': '10', 'tag': 'button', 'attributes': {}}],
             'neg_candidates': []},
            {'action_uid': 'b', 'cleaned_html': '<html><input backend_node_id="20" placeholder="FUTURE_PAGE"/></html>',
             'operation': {'op': 'TYPE', 'value': 'SECRET_NEXT_VALUE'},
             'pos_candidates': [{'backend_node_id': '20', 'tag': 'input', 'attributes': {}}],
             'neg_candidates': []}]}, {'sample_a': {'10': 0}, 'sample_b': {'20': 0}})


class ContractTest(unittest.TestCase):
    def setUp(self):
        FakeModel.requests = []
        FakeModel.responses = []

    def test_actor_receives_only_current_observation_and_past_human_history(self):
        task = task_fixture()
        public = public_case(task)
        ids = [o['candidate_ids'][0] for o in public['observations']]
        FakeModel.responses = [json.dumps({'element_id': i, 'operation': 'CLICK', 'value': ''}) for i in ids]
        with tempfile.TemporaryDirectory() as d:
            predictions = execute(public, CONFIG, Path(d), 'MEMORY_MARKER', model_factory=FakeModel)
            requests = [json.dumps(m) for _, m in FakeModel.requests]
            self.assertNotIn('FUTURE_PAGE', requests[0])
            self.assertNotIn('HUMAN_PREVIOUS_ACTION', requests[0])
            self.assertIn('HUMAN_PREVIOUS_ACTION', requests[1])
            for request in requests:
                for secret in ['SECRET_CURRENT_VALUE', 'SECRET_NEXT_VALUE', 'HIDDEN_FUTURE_ANSWER', 'pos_candidates']:
                    self.assertNotIn(secret, request)
                self.assertIn('MEMORY_MARKER', request)
            original = original_predictions(predictions, task)
            self.assertEqual(original[0]['element_id'], '10')
            metrics = score_case(original, task)
            self.assertEqual(metrics['step_success_rate'], 0.5)

    def test_invalid_candidate_gets_one_repair_then_scores_as_invalid(self):
        public = public_case(task_fixture())
        public['observations'] = public['observations'][:1]
        FakeModel.responses = ['not JSON', '{"element_id":"999","operation":"CLICK","value":""}']
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(execute(public, CONFIG, Path(d), model_factory=FakeModel), [None])
            self.assertEqual(len(FakeModel.requests), 2)
            self.assertIn('not JSON', FakeModel.requests[1][1][-2]['content'])

    def test_bank_scope_self_exclusion_and_frozen_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'memory.json'
            bank = Bank(path, Embeddings(), {'id': 'fixed'}, 'global')
            for eid, site, vector in [('one', 'site-a', [1., 0.]), ('two', 'site-b', [0., 1.])]:
                bank.commit({'case_id': eid, 'episode_id': eid, 'website': site,
                             'embedding': vector, 'memory_items': [eid]})
            before = path.read_bytes()
            frozen = Bank(path, Embeddings(), {'id': 'fixed'}, 'global', frozen=True)
            ids, _, vector, _ = frozen.select('third', 'site-b', 'search')
            self.assertEqual(ids, ['one'])
            self.assertEqual(vector, [])
            self.assertEqual(frozen.select('one', 'site-a', 'search')[0], ['two'])
            with self.assertRaises(ValueError):
                frozen.commit(bank.state['entries'][0])
            self.assertEqual(path.read_bytes(), before)
            local = Bank(Path(d) / 'local.json', Embeddings(), {'id': 'fixed'}, 'local')
            local.commit(bank.state['entries'][0])
            self.assertEqual(local.select('new', 'site-b', 'search')[0], [])

    def test_learning_uses_web_prompts_and_repairs_without_evaluator_fields(self):
        public = public_case(task_fixture())
        FakeModel.responses = ['fail', '```bash\nkeep working\n```', ITEM]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_json(out / 'trajectory.json', [{'step': 0, 'observation': public['observations'][0],
                                                'final_action': None}])
            # Host scoring files must never be picked up by the memory learner.
            write_json(out / 'metrics.json', {'private_gold': 'NEVER_SEND_EVALUATION'})
            entry = learn(public, out, CONFIG, [1., 0.], 'epoch-1:sample', model_factory=FakeModel)
            self.assertEqual(entry['self_judgment'], 'fail')
            self.assertEqual(len(entry['memory_items']), 1)
            requests = json.dumps(FakeModel.requests)
            self.assertNotIn('NEVER_SEND_EVALUATION', requests)
            self.assertNotIn('SECRET_CURRENT_VALUE', requests)
            self.assertIn('expert in web navigation', requests)
            self.assertIn('not live browsing', requests)
            self.assertEqual(len(FakeModel.requests), 3)

    def test_train_then_frozen_evaluation_has_no_learning_or_memory_mutation(self):
        calls = []
        def complete(model, messages, temperature):
            calls.append(model.phase)
            if model.phase == 'judge':
                return 'fail'
            if model.phase == 'extract':
                return ITEM
            observation = json.loads(messages[-1]['content'])
            return json.dumps({'element_id': observation['candidate_ids'][0], 'operation': 'CLICK', 'value': ''})

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory = root / 'inventory'
            inventory.mkdir()
            write_json(inventory / 'manifest.json', {'fixture': True})
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'fixture').write_text('fake embedding for no-API test')
            cases = [Case(cid, 'shop', {'sha256': 'fixture'}) for cid in ['train-a', 'train-b', 'valid']]
            manifest = {'diagnostic': False, 'seed': 42, 'split_sha256': 'fixture',
                        'cache_identity': {'fixture': True}, 'parts': {'train': ['train-a', 'train-b'],
                        'validation': ['valid']}, 'counts': {}, 'website_counts': {}, 'source_files': {}}
            def read(case):
                task = task_fixture()
                task['annotation_id'] = case.id
                return task
            for mode in ['train', 'eval']:
                out = root / mode
                out.mkdir()
                write_json(out / 'source_hashes.json', {})
                write_json(out / 'config.json', {**CONFIG, 'model': 'fake', 'embedding_model': 'fake',
                                                'embedding_threads': 1})
                memory_from = str(root / 'train/memory.json') if mode == 'eval' else None
                options = {'mode': mode, 'scope': 'global', 'inventory': str(inventory), 'seed': 42,
                           'split': 'validation', 'max_cases': None, 'epochs': 1,
                           'embedding_cache': str(weights), 'memory_from': memory_from}
                write_json(out / 'options.json', options)
                if memory_from:
                    shutil.copy2(memory_from, out / 'memory.json')
                    write_json(out / 'memory_source.json', {'sha256': file_hash(memory_from)})
                with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest,
                            {'train': cases[:2], 'validation': cases[2:]})), \
                     patch('swarmevo.benches.mind2web.dataset.read_case', side_effect=read), \
                     patch('fastembed.TextEmbedding', return_value=Embeddings()), \
                     patch('mind2web_adapter.model.Model.complete', new=complete), \
                     patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-contract-key'}):
                    calls.clear()
                    self.assertEqual(execute_run(out), 0)
                if mode == 'train':
                    self.assertEqual(calls.count('extract'), 2)
                    self.assertEqual(len(load(out / 'memory.json')['entries']), 2)
                else:
                    self.assertEqual(calls, ['solve', 'solve'])
                    self.assertEqual((out / 'memory.json').read_bytes(), Path(memory_from).read_bytes())
                    result = load(out / 'results.json')[0]
                    self.assertTrue(result['selected_memory_ids'])
                    self.assertTrue(load(out / 'summary.json')['frozen_memory_unchanged'])


if __name__ == '__main__':
    unittest.main()
