"""Offline tests: no API key, no network, no CIFAR download (random tensors stand in)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import agent
from wsp import data as datasets, toolkit
from wsp.evaluation import evaluate, weight_stats
from wsp.model import initial_weights, state_shapes
from wsp.tasks import TASKS, start_weights

CLASSES = [f'c{i}' for i in range(10)]


def random_split(n=40, k=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g),
            torch.arange(n) % k)


class ProgramRunner:
    """Writes a data subset once and runs programs against it on CPU."""

    def __init__(self, directory):
        images, labels = random_split()
        self.path = Path(directory) / 'subset.pt'
        torch.save({'images': images, 'labels': labels, 'classes': CLASSES}, self.path)
        self.info = {'num_classes': 10}

    def __call__(self, program, weights=None, allow_gradients=False, timeout=120):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
            return agent.run_program(weights or initial_weights(10), program, self.info,
                                     self.path, timeout, allow_gradients)


def program(body):
    return 'def build(weights, data, info):\n' + ''.join(f'    {line}\n' for line in body.splitlines())


class RestrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runner = ProgramRunner(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assertBlocked(self, body, pattern):
        with self.assertRaisesRegex(agent.ProgramError, pattern):
            self.runner(program(body))

    def test_valid_program_uses_data_and_toolkit(self):
        result = self.runner(program(
            "f = toolkit.capture(weights, data['images'], ['avgpool'], 10, reduce='pool')['avgpool']\n"
            "weights['fc.weight'], weights['fc.bias'] = toolkit.ridge(f, data['labels'], 10)\n"
            "return weights"))
        self.assertEqual(list(result['fc.weight'].shape), [10, 512])
        self.assertFalse(torch.equal(result['fc.weight'], initial_weights(10)['fc.weight']))

    def test_gradients_blocked(self):
        for body in ["x = torch.ones(2, requires_grad=True)\n(x * 2).sum().backward()",
                     "with torch.enable_grad():\n    pass",
                     "torch.set_grad_enabled(True)",
                     "torch.optim.SGD([torch.ones(1)], lr=1)",
                     "import torch.optim\ntorch.optim.Adam",
                     "torch.func.grad(lambda x: x.sum())(torch.ones(2))"]:
            with self.subTest(body):
                self.assertBlocked(body + '\nreturn weights', 'Gradients are disabled|does not require grad')

    def test_gradients_allowed_with_flag(self):
        body = ("with torch.enable_grad():\n    x = torch.ones(2, requires_grad=True)\n"
                "    (x * 3).sum().backward()\nweights['fc.bias'][:] = x.grad[0]\nreturn weights")
        self.assertTrue(torch.all(self.runner(program(body), allow_gradients=True)['fc.bias'] == 3))

    def test_network_subprocess_threads_blocked(self):
        self.assertBlocked("import socket\nsocket.create_connection(('example.com', 80), timeout=2)", 'not allowed')
        self.assertBlocked("import urllib.request\nurllib.request.urlopen('http://example.com')", 'not allowed')
        self.assertBlocked("import subprocess\nsubprocess.run(['ls'])", 'not allowed')
        self.assertBlocked("import os\nos.system('ls')", 'not allowed')
        self.assertBlocked("import threading\nthreading.Thread(target=print).start()", 'not allowed')

    def test_file_access_limited_to_subset(self):
        repo_file = str(Path(agent.__file__).resolve())
        self.assertBlocked(f"open({repo_file!r}).read()", 'Reading .* is not allowed')
        self.assertBlocked(f"torch.load({str(Path.home() / 'x.pt')!r})", 'not allowed|No such file')
        self.assertBlocked("open('/tmp/escape.txt', 'w')", 'Writing .* is not allowed')

    def test_api_key_hidden(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'secret', 'OPENROUTER_API_KEY': 'secret'}):
            result = self.runner(program("import os\nweights['fc.bias'][:] = float(any('API_KEY' in k for k in os.environ))\n"
                                      "return weights"))
        self.assertTrue(torch.all(result['fc.bias'] == 0))

    def test_bad_outputs(self):
        self.assertBlocked("weights['fc.weight'] = torch.zeros(3, 512)\nreturn weights", 'shape')
        self.assertBlocked("del weights['fc.bias']\nreturn weights", 'missing')
        self.assertBlocked("weights['conv1.weight'][0] = float('nan')\nreturn weights", 'NaN')
        self.assertBlocked("weights['bn1.running_var'][0] = -1.\nreturn weights", 'non-negative')
        self.assertBlocked("raise RuntimeError('boom')", 'boom')
        with self.assertRaisesRegex(agent.ProgramError, 'define build'):
            self.runner('x = 1\n')
        with self.assertRaisesRegex(agent.ProgramError, 'timed out'):
            self.runner(program("while True:\n    pass"), timeout=8)


class ToolkitAndEvalTests(unittest.TestCase):
    def test_capture_mean_var_matches_full(self):
        images, _ = random_split(n=12)
        w = initial_weights(10)
        full = toolkit.capture(w, images, ['layer1.0.conv1'], 10, batch_size=5)['layer1.0.conv1']
        stats = toolkit.capture(w, images, ['layer1.0.conv1'], 10, reduce='mean_var', batch_size=5)['layer1.0.conv1']
        self.assertTrue(torch.allclose(stats['mean'], full.mean(dim=(0, 2, 3)), atol=1e-4))
        self.assertTrue(torch.allclose(stats['var'], full.var(dim=(0, 2, 3), unbiased=False), atol=1e-4))

    def test_patches_ridge_kmeans_whiten(self):
        images, _ = random_split(n=8)
        p = toolkit.patches(images, size=3, n=500)
        self.assertEqual(list(p.shape), [500, 27])
        W, mean = toolkit.whiten(p)
        white = (p - mean) @ W.T
        self.assertLess((torch.cov(white.T) - torch.eye(27)).abs().max().item(), 0.2)
        self.assertEqual(list(toolkit.kmeans(p, 16, iters=3).shape), [16, 27])
        X = torch.randn(200, 5)
        y = (X[:, 0] > 0).long()
        Wr, b = toolkit.ridge(X, y, 2, lam=1e-4)
        self.assertGreater(((X @ Wr.T + b).argmax(1) == y).float().mean().item(), 0.9)

    def test_evaluate_metrics_json(self):
        w = initial_weights(10)
        metrics = evaluate(w, random_split(), CLASSES)
        self.assertEqual(sum(map(sum, metrics['confusion'])), 40)
        json.dumps({'m': metrics, 's': weight_stats(w)}, allow_nan=False)
        m100 = evaluate(initial_weights(100), random_split(n=200, k=100), [f'c{i}' for i in range(100)])
        self.assertIn('worst_classes', m100)
        self.assertNotIn('confusion', m100)

    def test_stratified_subset(self):
        labels = torch.arange(1000) % 10
        idx = datasets.stratified_subset(labels, 0.1, seed=1)
        self.assertEqual(len(idx), 100)
        self.assertTrue(torch.all(torch.bincount(labels[idx]) == 10))
        self.assertTrue(torch.equal(idx, datasets.stratified_subset(labels, 0.1, seed=1)))

    def test_transfer_start_weights(self):
        source = initial_weights(10, seed=5)
        start = start_weights(TASKS['cifar10-to-cifar100'], source)
        self.assertEqual({k: list(v.shape) for k, v in start.items()}, state_shapes(100))
        self.assertTrue(torch.equal(start['layer3.1.conv2.weight'], source['layer3.1.conv2.weight']))
        with self.assertRaises(ValueError):
            start_weights(TASKS['cifar10-to-cifar100'], None)


class SearchTests(unittest.TestCase):
    def test_keeps_best_and_survives_bad_programs(self):
        scores = iter([0.5, 0.7, 0.6])
        proposals = iter([{'rationale': r, 'program': p} for r, p in
                          [('good', 'a'), ('bad', 'b'), ('broken', 'c'), ('worse', 'd')]])

        def apply_fn(weights, prog, info):
            if prog == 'c':
                raise agent.ProgramError('Program failed: boom')
            return {k: v + 1 for k, v in weights.items()}

        seen = []
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            best, metrics = agent.search(
                {'w': torch.zeros(2)}, lambda w: {'accuracy': next(scores, 0.1), 'loss': 1.0},
                lambda payload: (seen.append(payload) or next(proposals), {}), apply_fn, 4, out, {'task': 't'})
            self.assertEqual(metrics['accuracy'], 0.7)
            self.assertTrue(torch.equal(best['w'], torch.ones(2)))
            self.assertEqual([e['round'] for e in json.loads((out / 'lineage.json').read_text())], [1])
            self.assertIn('boom', seen[3]['previous_attempts'][-1]['error'])

    def test_api_contract(self):
        calls = []
        response = SimpleNamespace(status='completed', id='mock', usage=None,
                                   output_text=json.dumps({'rationale': 'r', 'program': 'p'}))
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: calls.append(kw) or response))
        with patch.dict('sys.modules', {'openai': SimpleNamespace(OpenAI=lambda **kw: client)}):
            propose = agent.api_proposer('mock-model', 1000)
            self.assertEqual(propose({})[0]['program'], 'p')
            self.assertTrue(calls[0]['text']['format']['strict'])
            self.assertFalse(calls[0]['store'])
            response.status = 'incomplete'
            with self.assertRaises(RuntimeError):
                propose({})


class OpenRouterTests(unittest.TestCase):
    def test_openrouter_contract_and_fallbacks(self):
        calls, captured = [], {}
        message = SimpleNamespace(content='```json\n' + json.dumps({'rationale': 'r', 'program': 'p'}) + '\n```')
        response = SimpleNamespace(id='mock', model='x/y', usage=None,
                                   choices=[SimpleNamespace(finish_reason='stop', message=message)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: calls.append(kw) or response)))

        def make_client(**kw):
            captured.update(kw)
            return client
        with patch.dict('sys.modules', {'openai': SimpleNamespace(OpenAI=make_client)}), \
             patch.dict('os.environ', {'OPENROUTER_API_KEY': 'or-key'}):
            propose = agent.openrouter_proposer('x/y', 1000)
            self.assertEqual(propose({})[0]['program'], 'p')
            self.assertEqual(captured['base_url'], agent.OPENROUTER_URL)
            self.assertEqual(captured['api_key'], 'or-key')
            self.assertTrue(calls[0]['response_format']['json_schema']['strict'])
            self.assertEqual(calls[0]['messages'][0]['content'], agent.INSTRUCTIONS)
            response.choices[0].finish_reason = 'length'
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                propose({})
            response.choices[0].finish_reason = 'stop'
            message.content = 'no json here'
            with self.assertRaisesRegex(RuntimeError, 'did not return JSON'):
                propose({})


class EndToEndTests(unittest.TestCase):
    def test_transfer_run_with_fraction_and_late_test(self):
        loads = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, source, prog = root / 'run', root / 'source.pt', root / 'p.py'
            torch.save(initial_weights(10, seed=3), source)
            prog.write_text(Path('examples/programs/linear_probe.py').read_text())
            splits = {'train': random_split(n=1000, k=100, seed=1), 'val': random_split(n=200, k=100, seed=2),
                      'test': random_split(n=200, k=100, seed=3)}

            def load_split(dataset, name, root_dir):
                self.assertEqual(dataset, 'cifar100')
                loads.append((name, (out / 'frozen.json').exists()))
                return splits[name]

            argv = ['agent.py', '--task', 'cifar10-to-cifar100', '--data-fraction', '0.2', '--source', str(source),
                    '--output', str(out), '--device', 'cpu', '--programs', str(prog), '--evaluate-test']
            with patch('sys.argv', argv), patch.object(agent.datasets, 'load_split', side_effect=load_split), \
                 patch.object(agent.datasets, 'class_names', return_value=[f'c{i}' for i in range(100)]), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
                agent.main()
            self.assertEqual(loads[-1], ('test', True))
            self.assertTrue(all(not frozen for name, frozen in loads if name != 'test'))
            self.assertEqual(len(json.loads((out / 'subset_indices.json').read_text())), 200)
            self.assertEqual(json.loads((out / 'round_001' / 'input.json').read_text())['data_budget']['examples'], 200)
            self.assertTrue(json.loads((out / 'round_001' / 'result.json').read_text())['accepted'])
            self.assertTrue((out / 'final_test.json').exists())


if __name__ == '__main__':
    unittest.main()
