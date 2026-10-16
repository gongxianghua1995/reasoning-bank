"""Recoverable case failures must not stop the queue or inflate coverage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.common import file_hash, load, write_json
from mind2web_adapter.model import DeadlineExceeded
from mind2web_adapter.resume import folder_name, restore_learning_failure
from mind2web_adapter.run import execute_run
from mind2web_adapter.tests.test_contract import CONFIG, ITEM, Embeddings, task_fixture
from swarmevo.benches.protocol import Case


class ContinueTest(unittest.TestCase):
    def scenario(self, phase, fatal_third=False, quota=False, mode='train'):
        calls = []
        trigger = [True]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory, weights = root / 'inventory', root / 'weights'
            write_json(inventory / 'manifest.json', {})
            weights.mkdir()
            (weights / 'fake').write_text('fake')
            ids = ['one', 'two', 'three']
            cases = [Case(cid, 'shop', {'sha256': 'fixture'}) for cid in ids]
            folders = {folder_name(f'epoch-1:{cid}'): cid for cid in ids}
            manifest = {'diagnostic': False, 'seed': 42, 'split_sha256': 'fixture',
                        'cache_identity': {}, 'parts': {'train': ids, 'validation': ids}, 'counts': {},
                        'website_counts': {}, 'source_files': {}}
            options = {'mode': mode, 'scope': 'global', 'inventory': str(inventory), 'seed': 42,
                       'split': 'validation', 'max_cases': None, 'epochs': 1,
                       'embedding_cache': str(weights), 'memory_from': None, 'continue_case_errors': True}
            config = {**CONFIG, 'model': 'fake', 'embedding_model': 'fake', 'embedding_threads': 1}
            source = root / 'source'
            write_json(source / 'options.json', options)
            write_json(source / 'config.json', config)
            write_json(source / 'source_hashes.json', {})

            def read(case):
                task = task_fixture()
                task['annotation_id'] = case.id
                return task

            def complete(model, messages, temperature):
                cid = folders[model.out.name]
                calls.append((cid, model.phase))
                if trigger[0] and cid == 'two' and model.phase == phase:
                    if quota:
                        raise RuntimeError('insufficient_quota')
                    raise DeadlineExceeded('Synthetic case timeout')
                if trigger[0] and fatal_third and cid == 'three':
                    raise RuntimeError('insufficient_quota')
                if model.phase == 'judge':
                    return 'fail'
                if model.phase == 'extract':
                    return ITEM
                obs = json.loads(messages[-1]['content'])
                return json.dumps({'element_id': obs['candidate_ids'][0], 'operation': 'CLICK', 'value': ''})

            with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest, {'train': cases, 'validation': cases})), \
                 patch('swarmevo.benches.mind2web.dataset.read_case', side_effect=read), \
                 patch('fastembed.TextEmbedding', return_value=Embeddings()), \
                 patch('mind2web_adapter.model.Model.complete', new=complete), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-key'}):
                result = execute_run(source)
                if quota:
                    self.assertEqual(result, 1)
                    self.assertFalse(any(cid == 'three' for cid, _ in calls))
                    return
                if fatal_third:
                    self.assertEqual(result, 1)
                    self.assertEqual(len(load(source / 'case_outcomes.json')), 2)
                    original = {p.relative_to(source): file_hash(p) for p in source.rglob('*') if p.is_file()}
                    out = root / 'resumed'
                    resumed_options = {**options, 'resume_from': str(source)}
                    write_json(out / 'options.json', resumed_options)
                    write_json(out / 'config.json', config)
                    write_json(out / 'source_hashes.json', {})
                    restore_learning_failure(source, out, config, resumed_options)
                    trigger[0] = False
                    calls.clear()
                    self.assertEqual(execute_run(out), 0)
                    self.assertEqual({cid for cid, _ in calls}, {'three'})
                    self.assertEqual(original, {p.relative_to(source): file_hash(p) for p in source.rglob('*') if p.is_file()})
                else:
                    self.assertEqual(result, 0)
                    out = source
                summary = load(out / 'summary.json')
                self.assertEqual(summary['status'], 'finished_with_failures')
                self.assertEqual(summary['processed'], 3)
                self.assertEqual(summary['failed'], 1)
                self.assertEqual(summary['completed'], 2 if phase == 'solve' else 3)
                self.assertEqual(summary['coverage']['unscored'], 1 if phase == 'solve' else 0)
                self.assertEqual(len(load(out / 'memory.json')['entries']), 2)
                self.assertEqual(load(out / 'failures.json')[0]['case_id'], 'two')
                self.assertFalse((out / 'cases' / folder_name('epoch-1:two') / 'memory_entry.json').exists())

    def test_action_timeout_continues_without_fabricated_scores(self):
        self.scenario('solve')

    def test_learning_timeout_preserves_score_and_continues_without_memory(self):
        self.scenario('judge')

    def test_quota_stops_instead_of_skipping_entire_dataset(self):
        self.scenario('solve', quota=True)

    def test_resume_keeps_prior_failure_and_skips_handled_cases(self):
        self.scenario('solve', fatal_third=True)

    def test_stream_resume_keeps_order_and_does_not_relearn_previous_tasks(self):
        self.scenario('solve', fatal_third=True, mode='stream')


if __name__ == '__main__':
    unittest.main()
