"""Offline tests: no API key, no network, no CIFAR download (tiny random splits stand in)."""
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from wsp import data as datasets
from wsp.__main__ import main
from wsp.agent import ScriptedAgent, run_episode
from wsp.episode import Episode, audit, pixels_text
from wsp.model import initial_weights, state_shapes
from wsp.prompt import tool_specs
from wsp.tasks import TASKS, apply_diff, start_weights, writable_names

ROOT = Path(__file__).resolve().parent
CLASSES = [f'c{i}' for i in range(10)]
NO_GPU = patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''})


def tiny(n=20):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g), torch.arange(n) % 10


def episode(mode, tmp):
    images, labels = tiny() if mode != 'blind' else (torch.zeros(0, 3, 32, 32, dtype=torch.uint8), torch.zeros(0, dtype=torch.long))
    return Episode(mode, initial_weights(10), images, labels, CLASSES, Path(tmp) / 'ws',
                   partial(apply_diff, max_numbers=10000), 'cpu', timeout=120, pool=10 if mode == 'reasoned' else 0)


def fake_data_dir(root, n=60):
    g = torch.Generator().manual_seed(0)
    for split in datasets.SPLITS:
        torch.save((torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g), torch.arange(n) % 10),
                   Path(root) / f'cifar10_{split}.pt')
    (Path(root) / 'cifar10_classes.json').write_text(json.dumps(CLASSES))


BIAS = {'weights': [{'name': 'fc.bias', 'values': [0.0] * 10}]}


class LockTests(unittest.TestCase):
    """reasoned: once labels() or evaluate() has executed, tools() never executes again."""

    def test_tools_refused_after_labels_and_code_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            text, err = ep.call('tools', {'code': "open('before.txt', 'w').write('x'); print('ran')"})
            self.assertFalse(err)
            self.assertIn('ran', text)
            self.assertFalse(ep.call('labels', {'indices': [10, 11]})[1])
            text, err = ep.call('tools', {'code': "open('after.txt', 'w').write('x')"})
            self.assertTrue(err)
            self.assertIn('permanently disabled', text)
            self.assertTrue((ep.workspace / 'before.txt').exists())
            self.assertFalse((ep.workspace / 'after.txt').exists())
            # data, labels and evaluate still work in phase 2
            for name, args in (('data', {'indices': [0, 15]}), ('labels', {'indices': [12]}), ('evaluate', BIAS)):
                self.assertFalse(ep.call(name, args)[1], name)
            self.assertEqual([e['phase'] for e in ep.log], [1, 1, 2, 2, 2, 2])
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])

    def test_evaluate_locks_tools(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            self.assertTrue(ep.call('evaluate', {'weights': [{'name': 'fc.bias', 'values': [0.0]}]})[1])
            self.assertFalse(ep.locked)  # a rejected diff scores nothing: no lock
            self.assertFalse(ep.call('evaluate', BIAS)[1])
            self.assertIn('permanently disabled', ep.call('tools', {'code': 'print(1)'})[0])
            self.assertFalse(ep.call('data', {'indices': [10]})[1])  # labeled images visible after the lock
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])
            self.assertIn('after labels/evaluate', audit('reasoned', [{'call': 'evaluate', 'executed': True},
                                                                       {'call': 'tools', 'executed': True}])[0])
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:  # full mode: no lock
            ep = episode('full', tmp)
            self.assertFalse(ep.call('evaluate', BIAS)[1])
            self.assertFalse(ep.call('tools', {'code': 'print(1)'})[1])

    def test_failed_labels_call_does_not_lock(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            self.assertTrue(ep.call('labels', {'indices': [999]})[1])
            self.assertFalse(ep.call('tools', {'code': 'print(1)'})[1])

    def test_same_turn_order_respected(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            agent = ScriptedAgent([[{'name': 'tools', 'args': {'code': 'print(1)'}},
                                    {'name': 'labels', 'args': {'indices': [10]}},
                                    {'name': 'tools', 'args': {'code': 'print(2)'}}],
                                   [{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep)
            self.assertEqual([(e['call'], e['executed']) for e in ep.log],
                             [('tools', True), ('labels', True), ('tools', False), ('submit', True)])
            self.assertIsNotNone(ep.submission)

    def test_pool_and_labeled_set_rules(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            self.assertIn('have no labels', ep.call('labels', {'indices': [3]})[0])
            self.assertIn('only after labels()', ep.call('data', {'indices': [10]})[0])
            self.assertFalse(ep.call('data', {'indices': [9]})[1])
            self.assertFalse(ep.locked)  # refused calls don't lock
            self.assertFalse(ep.call('labels', {'indices': [19]})[1])
            self.assertIn('"accuracy"', ep.call('evaluate', BIAS)[0])
            self.assertFalse(ep.call('data', {'indices': [10]})[1])
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])

    def test_evaluate_scores_labeled_set_only(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            m = json.loads(episode('reasoned', tmp).call('evaluate', BIAS)[0])
            self.assertEqual(sum(m['prediction_counts'].values()), 10)

    def test_audit_catches_broken_lock_and_images(self):
        log = [{'call': 'labels', 'executed': True}, {'call': 'tools', 'executed': True}]
        self.assertIn('tools executed after labels/evaluate', audit('reasoned', log)[0])
        self.assertIn('pool images', audit('reasoned', [{'call': 'labels', 'executed': True, 'args': {'indices': [1]}}], pool=5)[0])
        self.assertIn('before the lock', audit('reasoned', [{'call': 'data', 'executed': True, 'args': {'indices': [7]}}], pool=5)[0])
        self.assertEqual(audit('full', log), [])
        self.assertIn('image', audit('full', [], ['text', 'image'])[0])
        self.assertIn('blind', audit('blind', [{'call': 'data', 'executed': True}])[0])


class DataIsolationTests(unittest.TestCase):
    def test_reasoned_sandbox_has_images_but_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            text, err = ep.call('tools', {'code': "print(sorted(data), len(data['images']))"})
            self.assertEqual(text.strip(), "['classes', 'images'] 10")  # pool only
            self.assertNotIn('labels', torch.load(ep.workspace / 'data.pt'))

    def test_sandbox_cannot_read_repo_data_or_outside(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            for path in (ROOT / 'data' / 'cifar10_train.pt', ROOT / 'README.md', Path.home() / '.zshrc', Path(tmp) / 'x'):
                with self.subTest(path):
                    text, _ = ep.call('tools', {'code': f'open({str(path)!r}).read()'})
                    self.assertIn('not allowed', text)

    def test_sandbox_blocks_network_subprocess_ctypes_and_hides_keys(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU, patch.dict('os.environ', {'ANTHROPIC_API_KEY': 's'}):
            ep = episode('full', tmp)
            for code in ("import socket; socket.create_connection(('example.com', 80))",
                         "import subprocess; subprocess.run(['ls'])", "import ctypes; ctypes.CDLL(None)"):
                with self.subTest(code):
                    self.assertIn('not allowed', ep.call('tools', {'code': code})[0])
            self.assertIn('False', ep.call('tools', {'code': "import os; print('ANTHROPIC_API_KEY' in os.environ)"})[0])

    def test_gradients_blocked_in_every_mode(self):
        attempts = ["x = torch.ones(2, requires_grad=True); (x * 2).sum().backward()",
                    "x = torch.ones(2); x.requires_grad_(); print(x.requires_grad)",
                    "with torch.enable_grad(): pass",
                    "torch.set_grad_enabled(True)",
                    "with torch.inference_mode(False): pass",
                    "torch.optim.Adam([torch.ones(1)])",
                    "from torch.optim import SGD",
                    "import torch.optim; torch.optim.SGD",
                    "torch.func.grad(lambda x: x.sum())(torch.ones(2))",
                    "torch.autograd.grad(torch.ones(1), torch.ones(1))"]
        for mode in ('blind', 'reasoned', 'full'):
            with tempfile.TemporaryDirectory() as tmp, NO_GPU:
                ep = episode(mode, tmp)
                for code in attempts:
                    with self.subTest(mode=mode, code=code):
                        out = ep.call('tools', {'code': code + "\nprint('GRADIENT ESCAPED')"})[0]
                        self.assertNotIn('GRADIENT ESCAPED', out)

    def test_full_sandbox_has_labels_and_blind_has_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            self.assertIn('20', episode('full', tmp).call('tools', {'code': "print(len(data['labels']))"})[0])
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('blind', tmp)
            self.assertIn('None', ep.call('tools', {'code': 'print(data)'})[0])
            for name in ('data', 'labels', 'evaluate'):
                self.assertTrue(ep.call(name, {'indices': [0]})[1], name)

    def test_data_is_hex_pixels_only(self):
        images, _ = tiny()
        text = pixels_text(images[0])
        self.assertEqual(len(text.splitlines()), 96)
        self.assertEqual(bytes.fromhex(text.replace('\n', '')), images[0].numpy().tobytes())
        with tempfile.TemporaryDirectory() as tmp:
            ep = episode('reasoned', tmp)
            out, _ = ep.call('data', {'indices': [0, 1]})
            import re  # only 'image i:' headers, blank lines and 64-hex-digit pixel rows
            self.assertTrue(all(re.fullmatch(r'image \d+:|[0-9a-f]{64}|', l) for l in out.splitlines()))
            self.assertTrue(ep.call('data', {'indices': list(range(17))})[1])  # per-call cap
            self.assertFalse(ep.locked)


class DiffTests(unittest.TestCase):
    def test_apply_and_reject(self):
        start = initial_weights(10)
        conv = torch.arange(64 * 27, dtype=torch.float32).view(64, 3, 3, 3)
        w, n = apply_diff(start, [{'name': 'conv1.weight', 'values': conv.tolist()}], 10000)
        self.assertTrue(torch.equal(w['conv1.weight'], conv))
        self.assertTrue(torch.equal(w['fc.weight'], start['fc.weight']))
        self.assertEqual(n, 1728)
        for diff, pattern in [([{'name': 'bn1.num_batches_tracked', 'values': [1]}], 'not a writable'),
                              ([{'name': 'fc.bias', 'values': [0] * 9}], 'needs 10'),
                              ([{'name': 'fc.bias', 'values': [float('inf')] * 10}], 'NaN or inf'),
                              ([{'name': 'bn1.running_var', 'values': [-1] * 64}], 'non-negative'),
                              (BIAS['weights'] * 2, 'twice'),
                              ([{'name': 'conv1.weight', 'values': [0] * 1728}], 'limit is 100')]:
            with self.subTest(pattern), self.assertRaisesRegex(ValueError, pattern):
                apply_diff(start, diff, 100)

    def test_empty_submit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = episode('blind', tmp)
            text, err = ep.call('submit', {'weights': []})
            self.assertTrue(err)
            self.assertIn('empty', text)
            self.assertIsNone(ep.submission)

    def test_tool_specs_per_mode(self):
        names = writable_names(initial_weights(10))
        self.assertEqual([t['name'] for t in tool_specs('blind', names)], ['tools', 'submit'])
        self.assertEqual(len(tool_specs('reasoned', names)), 5)

    def test_transfer_start(self):
        source = initial_weights(10, seed=1)
        start = start_weights(TASKS['cifar10-to-cifar100'], source)
        self.assertEqual({k: list(v.shape) for k, v in start.items()}, state_shapes(100))
        self.assertTrue(torch.equal(start['conv1.weight'], source['conv1.weight']))


class EndToEndTests(unittest.TestCase):
    def test_scripted_reasoned_run(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            script = Path(tmp) / 'calls.json'
            script.write_text(json.dumps([[{'name': 'tools', 'args': {'code': 'print(1)'}}],
                                          [{'name': 'labels', 'args': {'indices': [59]}}],
                                          [{'name': 'tools', 'args': {'code': 'print(2)'}}],
                                          [{'name': 'submit', 'args': BIAS}]]))
            out = Path(tmp) / 'run'
            main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--script', str(script),
                  '--data-dir', tmp, '--device', 'cpu', '--output', str(out), '--evaluate-test'])
            self.assertEqual(json.loads((out / 'audit.json').read_text()), {'clean': True, 'problems': []})
            calls = json.loads((out / 'calls.json').read_text())
            self.assertEqual([(c['call'], c['executed']) for c in calls],
                             [('tools', True), ('labels', True), ('tools', False), ('submit', True)])
            split = json.loads((out / 'subset_indices.json').read_text())
            labels = datasets.load_split('cifar10', 'train', tmp)[1][split['indices']]
            pool, labeled = labels[:split['pool']], labels[split['pool']:]
            self.assertEqual((len(pool), len(labeled)), (30, 30))
            self.assertEqual(len(set(split['indices'])), 60)  # disjoint
            self.assertEqual(torch.bincount(labeled, minlength=10).tolist(), [3] * 10)
            self.assertEqual(set(json.loads((out / 'result.json').read_text())), {'calls', 'val', 'test'})
            self.assertTrue(torch.equal(torch.load(out / 'weights.pt')['fc.bias'], torch.zeros(10)))

    def test_mode_fraction_mismatch(self):
        for args in (['--mode', 'blind', '--data-fraction', '0.1'], ['--mode', 'full']):
            with self.subTest(args), self.assertRaises(SystemExit):
                main(['--task', 'scratch-cifar10', '--dry-run'] + args)


if __name__ == '__main__':
    unittest.main()
