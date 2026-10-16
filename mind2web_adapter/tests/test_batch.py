"""Exercise batch sequencing and fail-stop behavior without model calls."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.batch import execute
from mind2web_adapter.common import file_hash, load, write_json


class BatchTest(unittest.TestCase):
    def fixture(self, root, failure=False, case_failure=False):
        inventory = root / 'inventory'
        write_json(inventory / 'manifest.json', {})
        write_json(root / 'config.json', {})
        write_json(root / 'source_hashes.json', {})
        write_json(root / 'batch_status.json', {'status': 'prepared'})
        stages = []
        for name in ('train', 'frozen', 'none'):
            out = root / name
            script = ('from pathlib import Path; import json; '
                      f'p=Path({str(out)!r}); p.mkdir(); '
                      '(p/"summary.json").write_text(json.dumps({"status":"finished","completed":1})); '
                      '(p/"status.json").write_text(json.dumps({"status":"finished","completed":1}))')
            if failure and name == 'frozen':
                script = 'raise SystemExit(7)'
            if case_failure and name == 'train':
                script = script.replace('{"status":"finished","completed":1}',
                    '{"status":"finished_with_failures","completed":0,"processed":1,"failed":1}')
            stages.append({'name': name, 'output': str(out), 'expected_episodes': 1,
                           'command': [sys.executable, '-c', script]})
        write_json(root / 'plan.json', {'stages': stages, 'source_root': str(root),
            'inventory': str(inventory), 'inventory_sha256': file_hash(inventory / 'manifest.json'),
            'config_sha256': file_hash(root / 'config.json'), 'expected_episodes': 3,
            'python': sys.executable})

    def test_complete_sequence_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            with patch('mind2web_adapter.batch.time.sleep'), \
                 patch('mind2web_adapter.batch.signal.signal'), \
                 patch('mind2web_adapter.batch.subprocess.run') as audit, \
                 patch('builtins.print'):
                self.assertEqual(execute(root), 0)
                self.assertEqual(audit.call_count, 3)
                with self.assertRaises(ValueError):
                    execute(root)
            state = load(root / 'batch_status.json')
            self.assertEqual(state['status'], 'finished')
            self.assertEqual(state['completed_episodes'], 3)
            self.assertEqual(state['finished_stages'], ['train', 'frozen', 'none'])

    def test_failure_preserves_completed_stage_and_prevents_later_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, failure=True)
            with patch('mind2web_adapter.batch.time.sleep'), \
                 patch('mind2web_adapter.batch.signal.signal'), \
                 patch('mind2web_adapter.batch.subprocess.run') as audit, \
                 patch('builtins.print'):
                self.assertEqual(execute(root), 7)
                self.assertEqual(audit.call_count, 1)
            state = load(root / 'batch_status.json')
            self.assertEqual(state['status'], 'error')
            self.assertEqual(state['completed_episodes'], 1)
            self.assertEqual(state['active_stage'], 'frozen')
            self.assertFalse((root / 'none').exists())
            self.assertEqual(len(load(root / 'results.json')), 1)

    def test_recorded_case_failure_allows_later_stages_with_honest_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, case_failure=True)
            with patch('mind2web_adapter.batch.time.sleep'), \
                 patch('mind2web_adapter.batch.signal.signal'), \
                 patch('mind2web_adapter.batch.subprocess.run'), patch('builtins.print'):
                self.assertEqual(execute(root), 0)
            state = load(root / 'batch_status.json')
            self.assertEqual(state['status'], 'finished_with_failures')
            self.assertEqual(state['processed_episodes'], 3)
            self.assertEqual(state['completed_episodes'], 2)
            self.assertEqual(state['failed_episodes'], 1)
            self.assertEqual(state['finished_stages'], ['train', 'frozen', 'none'])


if __name__ == '__main__':
    unittest.main()
