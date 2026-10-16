import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mind2web_adapter.common import load, write_json
from mind2web_adapter.run import select_cases
from mind2web_adapter.validation import collect, prepare


class ValidationTest(unittest.TestCase):
    def test_eval_selection_rejects_training_duplicates_and_wrong_split(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ids.json'
            cases = [SimpleNamespace(id=i) for i in ['one', 'two', 'three']]
            opts = {'mode': 'eval', 'case_ids_file': str(path), 'max_cases': None}
            write_json(path, ['three', 'one'])
            self.assertEqual([c.id for c in select_cases(cases, opts)], ['one', 'three'])
            with self.assertRaises(ValueError):
                select_cases(cases, {**opts, 'mode': 'train'})
            for ids in [[], ['one', 'one'], ['outside'], [1]]:
                write_json(path, ids)
                with self.assertRaises(ValueError):
                    select_cases(cases, opts)

    def test_merge_rejects_duplicate_results_and_missing_final_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            row = {'case_id': 'one', 'episode_id': 'epoch-1:one', 'metrics': {}}
            write_json(out / 'global_0/results.json', [row, row])
            plan = {'validation_ids': ['one'], 'jobs': [{'name': 'global_0', 'scope': 'global', 'case_ids': ['one']}]}
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                collect(out, plan)
            write_json(out / 'global_0/results.json', [])
            write_json(out / 'global_0/summary.json', {'status': 'finished', 'frozen_memory_unchanged': True})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                collect(out, plan, final=True)


if __name__ == '__main__':
    unittest.main()
