"""Offline harness tests: no API calls or API key required."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent


def proposal(value=2):
    return {'rationale': 'Try an explicit coefficient.', 'edits': [{'parameter': 'x', 'value': value}]}


class AgentTests(unittest.TestCase):
    def test_edits_are_absolute_and_do_not_mutate_best(self):
        best = {'x': 1, 'y': 4}
        self.assertEqual(agent.apply_proposal(best, proposal()), {'x': 2, 'y': 4})
        self.assertEqual(best['x'], 1)

    def test_invalid_edits(self):
        for value in [float('nan'), float('inf'), True, '2']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.apply_proposal({'x': 0}, proposal(value))
        p = proposal(); p['edits'] *= 2
        with self.assertRaises(ValueError):
            agent.apply_proposal({'x': 0}, p)
        with self.assertRaises(ValueError):
            agent.apply_proposal({'y': 0}, proposal())

    def test_search_keeps_best_and_recovers_invalid_edits(self):
        attempts = iter([proposal(2), proposal(9), proposal(float('nan'))])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = agent.search({'objective': 'Find x=2', 'parameters': {'x': 0}},
                                  lambda p: {'score': -(p['x']-2)**2},
                                  lambda payload: (next(attempts), {}), 3, out)
            self.assertEqual(result, {'x': 2})
            self.assertEqual(json.loads((out/'best_metrics.json').read_text())['score'], 0)
            self.assertFalse(json.loads((out/'round_002.json').read_text())['accepted'])
            self.assertIn('error', json.loads((out/'round_003.json').read_text()))

    def test_evaluator_contract_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            cmd = [sys.executable, '-c', 'import sys,json; p=json.load(sys.stdin); print(json.dumps({"score":p["x"]}))']
            evaluate = agent.command_evaluator(cmd, directory, 5)
            self.assertEqual(evaluate({'x': 3})['score'], 3)
            bad = agent.command_evaluator([sys.executable, '-c', 'print("{}")'], directory, 5)
            with self.assertRaises(ValueError):
                bad({'x': 0})
            timed = agent.command_evaluator([sys.executable, '-c', 'import time; time.sleep(5)'], directory, .05)
            with self.assertRaises(agent.subprocess.TimeoutExpired):
                timed({'x': 0})

    def test_api_contract_and_incomplete_response(self):
        calls = []
        response = SimpleNamespace(status='completed', output_text=json.dumps(proposal()), id='mock', usage=None)
        def create(**kwargs):
            calls.append(kwargs)
            return response
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch.dict('sys.modules', {'openai': SimpleNamespace(OpenAI=lambda **kwargs: client)}):
            propose = agent.api_proposer('mock-model', 1000, agent.proposal_schema({'x': 0}))
            self.assertEqual(propose({})[0], proposal())
            self.assertTrue(calls[0]['text']['format']['strict'])
            self.assertFalse(calls[0]['store'])
            response.status = 'incomplete'
            with self.assertRaises(RuntimeError):
                propose({})

    def test_final_evaluation_runs_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root/'run'; task_path = root/'task.json'
            agent.save(task_path, {'objective': 'Find x=2', 'parameters': {'x': 0},
                                  'evaluator': ['calibration'], 'final_evaluator': ['holdout']})
            def evaluator(command, cwd, timeout):
                def evaluate(parameters):
                    if command == ['holdout']:
                        self.assertTrue((out/'frozen.json').exists())
                    return {'score': -(parameters['x']-2)**2}
                return evaluate
            args = ['agent.py', '--task', str(task_path), '--output', str(out), '--rounds', '1', '--evaluate-final']
            with patch('sys.argv', args), patch.dict('os.environ', {'OPENAI_API_KEY': 'mock-only'}), \
                 patch.object(agent, 'command_evaluator', side_effect=evaluator), \
                 patch.object(agent, 'api_proposer', return_value=lambda payload: (proposal(), {})):
                agent.main()
            self.assertTrue((out/'final_results.json').exists())
            self.assertNotIn('holdout', (out/'round_001_input.json').read_text())


if __name__ == '__main__':
    unittest.main()
