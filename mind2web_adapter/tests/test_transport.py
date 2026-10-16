"""Bounded timeout recovery must preserve requests and case deadlines."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mind2web_adapter.model import DeadlineExceeded, Model


class APITimeoutError(Exception):
    pass


class TransportTest(unittest.TestCase):
    def run_scenario(self, failures, hard_deadline=None, quota=False):
        clock = [0.0]
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            if len(requests) <= failures:
                clock[0] += kwargs['timeout']
                raise APITimeoutError('insufficient_quota' if quota else 'synthetic timeout')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'),
                                     finish_reason='stop')], usage=None, id='fake', model='fake')

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = {'model': 'fake', 'max_tokens': 10, 'api_timeout': 180, 'timeout_recoveries': 2}
        with tempfile.TemporaryDirectory() as d, \
             patch('mind2web_adapter.model.time.monotonic', side_effect=lambda: clock[0]), \
             patch('mind2web_adapter.model.time.sleep', side_effect=lambda n: clock.__setitem__(0, clock[0] + n)):
            model = Model(config, Path(d), 'solve', 180, client=client, hard_deadline=hard_deadline)
            try:
                result = model.complete([{'role': 'user', 'content': 'same request'}], 0)
                error = None
            except Exception as exc:
                result, error = None, exc
            return requests, result, error, clock[0]

    def test_timeout_retries_identical_request(self):
        requests, result, error, _ = self.run_scenario(1, hard_deadline=1200)
        self.assertIsNone(error)
        self.assertEqual(result, 'ok')
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['messages'], requests[1]['messages'])

    def test_repeated_timeout_is_bounded_to_three_calls(self):
        requests, _, error, elapsed = self.run_scenario(10)
        self.assertIsInstance(error, APITimeoutError)
        self.assertEqual(len(requests), 3)
        self.assertLessEqual(elapsed, 540)

    def test_case_deadline_and_quota_do_not_get_retry(self):
        requests, _, error, _ = self.run_scenario(1, hard_deadline=180)
        self.assertIsInstance(error, DeadlineExceeded)
        self.assertEqual(len(requests), 1)
        requests, _, error, _ = self.run_scenario(1, quota=True)
        self.assertIsInstance(error, APITimeoutError)
        self.assertEqual(len(requests), 1)


if __name__ == '__main__':
    unittest.main()
