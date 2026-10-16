"""Recovery must retain successful memory and avoid repeating action calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.common import file_hash, load, write_json
from mind2web_adapter.model import DeadlineExceeded
from mind2web_adapter.resume import restore_learning_failure
from mind2web_adapter.run import execute_run
from mind2web_adapter.tests.test_contract import CONFIG, ITEM, Embeddings, task_fixture
from swarmevo.benches.protocol import Case


class ResumeTest(unittest.TestCase):
    def test_failed_judge_recovery_reuses_actions_and_preserves_source(self):
        self.recovery_scenario('judge')

    def test_failed_action_recovery_skips_completed_steps_and_preserves_source(self):
        self.recovery_scenario('solve')

    def recovery_scenario(self, failed_phase):
        calls = []
        failed_once = False

        def complete(model, messages, temperature):
            nonlocal failed_once
            calls.append(model.phase)
            if failed_phase == 'solve' and model.phase == 'solve' and calls.count('solve') == 4 and not failed_once:
                failed_once = True
                raise DeadlineExceeded('Synthetic action timeout')
            if model.phase == 'judge':
                if failed_phase == 'judge' and calls.count('judge') == 2 and not failed_once:
                    failed_once = True
                    raise DeadlineExceeded('Synthetic timeout')
                return 'fail'
            if model.phase == 'extract':
                return ITEM
            observation = json.loads(messages[-1]['content'])
            return json.dumps({'element_id': observation['candidate_ids'][0], 'operation': 'CLICK', 'value': ''})

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory = root / 'inventory'
            write_json(inventory / 'manifest.json', {})
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'fixture').write_text('fake weights')
            cases = [Case(cid, 'shop', {'sha256': 'fixture'}) for cid in ['first', 'second']]
            manifest = {'diagnostic': False, 'seed': 42, 'split_sha256': 'fixture',
                        'cache_identity': {'fixture': True}, 'parts': {'train': ['first', 'second']},
                        'counts': {}, 'website_counts': {}, 'source_files': {}}

            def read(case):
                task = task_fixture()
                task['annotation_id'] = case.id
                return task

            options = {'mode': 'train', 'scope': 'global', 'inventory': str(inventory), 'seed': 42,
                       'split': 'validation', 'max_cases': None, 'epochs': 1,
                       'embedding_cache': str(weights), 'memory_from': None}
            config = {**CONFIG, 'model': 'fake', 'embedding_model': 'fake', 'embedding_threads': 1}
            source, resumed = root / 'source', root / 'resumed'
            for out in (source, resumed):
                write_json(out / 'config.json', config)
                write_json(out / 'source_hashes.json', {})
            write_json(source / 'options.json', options)
            with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest, {'train': cases})), \
                 patch('swarmevo.benches.mind2web.dataset.read_case', side_effect=read), \
                 patch('fastembed.TextEmbedding', return_value=Embeddings()), \
                 patch('mind2web_adapter.model.Model.complete', new=complete), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-contract-key'}):
                self.assertEqual(execute_run(source), 1)
                original_hashes = {p.relative_to(source): file_hash(p) for p in source.rglob('*') if p.is_file()}
                before = load(source / 'results.json')
                self.assertEqual(len(before), 2 if failed_phase == 'judge' else 1)
                self.assertEqual(len(load(source / 'memory.json')['entries']), 1)
                continued_options = {**options, 'resume_from': str(source)}
                write_json(resumed / 'options.json', continued_options)
                with self.assertRaises(ValueError):
                    restore_learning_failure(source, resumed, {**config, 'temperature': 0.5}, continued_options)
                restore_learning_failure(source, resumed, config, continued_options)
                calls.clear()
                self.assertEqual(execute_run(resumed), 0)
                self.assertEqual(calls, ['judge', 'extract'] if failed_phase == 'judge' else ['solve', 'judge', 'extract'])
            after = load(resumed / 'results.json')
            self.assertEqual(after[0], before[0])
            if failed_phase == 'judge':
                self.assertEqual(after[1]['metrics'], before[1]['metrics'])
            self.assertIsNone(after[1]['learning_error'])
            self.assertEqual(len(load(resumed / 'memory.json')['entries']), 2)
            self.assertEqual(load(resumed / 'summary.json')['completed'], 2)
            self.assertEqual(original_hashes,
                {p.relative_to(source): file_hash(p) for p in source.rglob('*') if p.is_file()})
            archive_pattern = ('*/learning_recovery_*/judge_request_1.json' if failed_phase == 'judge'
                               else '*/actions_recovery_*/001_1.json')
            archives = list((resumed / 'cases').glob(archive_pattern))
            self.assertEqual(len(archives), 1)


if __name__ == '__main__':
    unittest.main()
