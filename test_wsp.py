"""Offline tests: no API key, no network, no CIFAR download (tiny random splits stand in)."""
import asyncio
from contextlib import redirect_stdout
from functools import partial
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from wsp import data as datasets
from wsp.__main__ import describe, main, turn_line
from wsp.agent import ScriptedAgent, run_episode
from wsp.episode import Episode, audit, clip, pixels_text
from wsp.evaluation import evaluate, layer_outputs
from wsp.model import initial_weights, state_shapes
from wsp.prompt import system_prompt, tool_specs, user_prompt
from wsp.tasks import TASKS, apply_diff, load_start, start_weights, writable_names

ROOT = Path(__file__).resolve().parent
CLASSES = [f'c{i}' for i in range(10)]
NO_GPU = patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''})


def tiny(n=20):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g), torch.arange(n) % 10


def episode(mode, tmp):
    images, labels = tiny() if mode != 'blind' else (torch.zeros(0, 3, 32, 32, dtype=torch.uint8), torch.zeros(0, dtype=torch.long))
    return Episode(mode, initial_weights(10), images, labels, CLASSES, Path(tmp) / 'ws',
                   partial(apply_diff, max_numbers=10000), 'cpu', timeout=120)


def fake_data_dir(root, n=60):
    g = torch.Generator().manual_seed(0)
    for split in datasets.SPLITS:
        torch.save((torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g), torch.arange(n) % 10),
                   Path(root) / f'cifar10_{split}.pt')
    (Path(root) / 'cifar10_classes.json').write_text(json.dumps(CLASSES))


BIAS = {'weights': [{'name': 'fc.bias', 'values': [0.0] * 10}]}


class LockTests(unittest.TestCase):
    """reasoned: labels(), evaluate() and submit() only after end_tools(); after it, tools() never executes again."""

    def test_tools_refused_after_end_tools_and_code_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            text, err = ep.call('tools', {'code': "open('before.txt', 'w').write('x'); print('ran')"})
            self.assertFalse(err)
            self.assertIn('ran', text)
            self.assertIn('Phase 2', ep.call('end_tools', {})[0])
            text, err = ep.call('tools', {'code': "open('after.txt', 'w').write('x')"})
            self.assertTrue(err)
            self.assertIn('permanently disabled', text)
            self.assertTrue((ep.workspace / 'before.txt').exists())
            self.assertFalse((ep.workspace / 'after.txt').exists())
            for name, args in (('data', {'indices': [0, 15]}), ('labels', {'indices': [12]}), ('evaluate', BIAS),
                               ('submit', BIAS)):
                self.assertFalse(ep.call(name, args)[1], name)
            self.assertEqual([e['phase'] for e in ep.log], [1, 1, 2, 2, 2, 2, 2])
            self.assertIsNotNone(ep.submission)
            self.assertEqual(audit('reasoned', ep.log), [])

    def test_labels_evaluate_submit_refused_in_phase_1(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            for name, args in (('labels', {'indices': [10]}), ('evaluate', BIAS), ('submit', BIAS)):
                text, err = ep.call(name, args)
                self.assertTrue(err, name)
                self.assertIn('only in phase 2', text)
            self.assertFalse(ep.locked)
            self.assertIsNone(ep.submission)
            self.assertFalse(any(e['executed'] for e in ep.log))
            self.assertFalse(ep.call('tools', {'code': 'print(1)'})[1])
            self.assertEqual(audit('reasoned', ep.log), [])
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('full', tmp)
            self.assertFalse(ep.call('evaluate', BIAS)[1])
            self.assertFalse(ep.call('tools', {'code': 'print(1)'})[1])
            self.assertIn('not available', ep.call('end_tools', {})[0])

    def test_end_tools_is_one_way(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            self.assertFalse(ep.call('end_tools', {})[1])
            text, err = ep.call('end_tools', {})
            self.assertTrue(err)
            self.assertIn('already in phase 2', text)
            self.assertTrue(ep.locked)
            self.assertTrue(ep.call('tools', {'code': 'print(1)'})[1])

    def test_same_turn_order_respected(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            agent = ScriptedAgent([[{'name': 'tools', 'args': {'code': 'print(1)'}},
                                    {'name': 'end_tools', 'args': {}},
                                    {'name': 'tools', 'args': {'code': 'print(2)'}}],
                                   [{'name': 'labels', 'args': {'indices': [10]}},
                                    {'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep)
            self.assertEqual([(e['call'], e['executed']) for e in ep.log],
                             [('tools', True), ('end_tools', True), ('tools', False), ('labels', True),
                              ('submit', True)])
            self.assertIsNotNone(ep.submission)

    def test_nudge_names_end_tools_in_phase_1(self):
        class Recorder(ScriptedAgent):
            def next(self, results):
                self.seen = getattr(self, 'seen', []) + [results]
                return super().next(results)
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            agent = Recorder([[], [{'name': 'end_tools', 'args': {}}], [], [{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep)
            nudges = [r for r in agent.seen if isinstance(r, str)]
            self.assertEqual([n.split('\n')[0] for n in nudges], ['Call end_tools(), then submit() with your diff to finish.',
                                                                  'Call submit() with your diff to finish.'])
            self.assertIn('[harness] Turn 1 of 10 done; 9 turns left.', nudges[0])
            self.assertIsNotNone(ep.submission)

    def test_every_image_has_pixels_in_phase_1_and_labels_in_phase_2(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            for i in (0, 10, 19):
                self.assertFalse(ep.call('data', {'indices': [i]})[1], i)
                self.assertFalse(ep.call('predict', {'weights': [], 'indices': [i], 'layer': 'fc',
                                                     'channels': None})[1], i)
            self.assertIn('only in phase 2', ep.call('labels', {'indices': [0]})[0])
            self.assertFalse(ep.call('end_tools', {})[1])
            self.assertEqual(json.loads(ep.call('labels', {'indices': [0, 10, 19]})[0]),
                             {'0': 'c0', '10': 'c0', '19': 'c9'})
            self.assertIn('"accuracy"', ep.call('evaluate', BIAS)[0])
            self.assertEqual(audit('reasoned', ep.log), [])

    def test_evaluate_scores_whole_subset_and_returns_accuracy_only(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            ep.call('end_tools', {})
            m = json.loads(ep.call('evaluate', BIAS)[0])
            self.assertEqual(set(m), {'accuracy'})
            weights, _ = apply_diff(ep.start, BIAS['weights'], 10000)
            whole = evaluate(weights, (ep.images, ep.labels), CLASSES, 'cpu')['accuracy']
            self.assertEqual(m['accuracy'], whole)
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            m = json.loads(episode('full', tmp).call('evaluate', BIAS)[0])
            self.assertIn('confusion', m)

    def test_audit_catches_broken_phases_and_images(self):
        log = [{'call': 'end_tools', 'executed': True}, {'call': 'tools', 'executed': True}]
        self.assertIn('tools executed after end_tools', audit('reasoned', log)[0])
        for call in ('labels', 'evaluate', 'submit'):
            self.assertIn(f'{call} executed before end_tools', audit('reasoned', [{'call': call, 'executed': True}])[0])
        self.assertEqual(audit('reasoned', [{'call': 'data', 'executed': True, 'args': {'indices': [7]}},
                                            {'call': 'predict', 'executed': True, 'args': {'indices': [7]}}]), [])
        self.assertIn('end_tools executed in full', audit('full', log)[0])
        self.assertEqual(audit('full', [{'call': 'labels', 'executed': True}, {'call': 'tools', 'executed': True}]), [])
        self.assertIn('image', audit('full', [], ['text', 'image'])[0])
        self.assertIn('blind', audit('blind', [{'call': 'data', 'executed': True}])[0])


class DataIsolationTests(unittest.TestCase):
    def test_reasoned_sandbox_has_images_but_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            text, err = ep.call('tools', {'code': "print(sorted(data), len(data['images']))"})
            self.assertEqual(text.strip(), "['classes', 'images'] 20")
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
            for name in ('data', 'labels', 'evaluate', 'end_tools', 'predict'):
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
                              ([{'name': 'conv1.weight', 'values': [0] * 1728}], 'limit is 100'),
                              ([{'name': 'fc.bias', 'values': [0] * 10, 'fill': 0}], 'exactly one of'),
                              ([{'name': 'fc.bias'}], 'exactly one of'),
                              ([{'name': 'fc.bias', 'values': [0] * 10, 'scale': 2}], 'unknown keys'),
                              ([{'name': 'fc.weight', 'index': [[0, 11]], 'fill': 0}], 'axis 0 range'),
                              ([{'name': 'fc.weight', 'index': [[3, 3]], 'fill': 0}], 'start < stop'),
                              ([{'name': 'fc.weight', 'index': [[0, 1.5]], 'fill': 0}], 'integers'),
                              ([{'name': 'fc.weight', 'index': [[0, 1]] * 3, 'fill': 0}], 'list of 1 to 2'),
                              ([{'name': 'fc.weight', 'index': [], 'fill': 0}], 'list of 1 to 2'),
                              ([{'name': 'fc.weight', 'index': [[0, 1]], 'values': [0] * 51}],
                               r'fc.weight \[0:1\]: got 51 numbers, block \[1, 512\] needs 512'),
                              ([{'name': 'bn1.running_var', 'index': [[0, 4]], 'fill': -1}], 'non-negative'),
                              ([{'name': 'fc.bias', 'fill': float('nan')}], 'NaN or inf'),
                              ([{'name': 'fc.bias', 'fill': 'x'}], 'fill must be a number'),
                              ([{'name': 'fc.bias', 'fill': True}], 'fill must be a number')]:
            with self.subTest(pattern), self.assertRaisesRegex(ValueError, pattern):
                apply_diff(start, diff, 100)

    def test_blocks_and_fill_apply_in_order(self):
        start = initial_weights(10)
        before = {k: v.clone() for k, v in start.items()}
        diff = [{'name': 'layer2.0.shortcut.0.weight', 'fill': 0},
                {'name': 'layer2.0.shortcut.0.weight', 'index': [[0, 10]], 'values': list(range(640))},
                {'name': 'fc.weight', 'index': [[0, 10], [37, 38]], 'values': [1.0] * 10},
                {'name': 'fc.bias', 'index': [[2, 4]], 'fill': 5},
                {'name': 'conv1.weight', 'index': [[5, 6], [1, 2]], 'values': [list(range(3))] * 3}]
        w, n = apply_diff(start, diff, 10000)
        self.assertEqual(n, 1 + 640 + 10 + 1 + 9)
        shortcut = w['layer2.0.shortcut.0.weight']
        self.assertTrue(torch.equal(shortcut[:10].flatten(), torch.arange(640, dtype=torch.float32)))
        self.assertTrue((shortcut[10:] == 0).all())
        self.assertTrue((w['fc.weight'][:, 37] == 1).all())
        others = torch.ones(512, dtype=torch.bool)
        others[37] = False
        self.assertTrue(torch.equal(w['fc.weight'][:, others], start['fc.weight'][:, others]))
        self.assertTrue((w['fc.bias'][2:4] == 5).all())
        self.assertTrue(torch.equal(w['fc.bias'][[0, 1, 4, 5, 6, 7, 8, 9]], start['fc.bias'][[0, 1, 4, 5, 6, 7, 8, 9]]))
        self.assertTrue(torch.equal(w['conv1.weight'][5, 1], torch.tensor([[0., 1, 2]] * 3)))
        rest = w['conv1.weight'].clone()
        rest[5, 1] = start['conv1.weight'][5, 1]
        self.assertTrue(torch.equal(rest, start['conv1.weight']))
        self.assertTrue(all(torch.equal(start[k], before[k]) for k in start))
        self.assertTrue(torch.equal(w['bn1.weight'], start['bn1.weight']))

    def test_fill_counts_one_and_null_keys_are_absent(self):
        start = initial_weights(10)
        w, n = apply_diff(start, [{'name': 'conv1.weight', 'fill': 0.5}], 1)
        self.assertEqual(n, 1)
        self.assertTrue((w['conv1.weight'] == 0.5).all())
        w, n = apply_diff(start, [{'name': 'fc.bias', 'index': None, 'values': None, 'fill': 3}], 10)
        self.assertTrue((w['fc.bias'] == 3).all())
        with self.assertRaisesRegex(ValueError, 'limit is 1'):
            apply_diff(start, [{'name': 'conv1.weight', 'fill': 0}, {'name': 'fc.bias', 'fill': 0}], 1)

    def test_submit_counts_distinct_tensors_and_log_labels_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = episode('blind', tmp)
            diff = [{'name': 'fc.weight', 'fill': 0}, {'name': 'fc.weight', 'index': [[0, 10], [3, 4]],
                                                       'values': [1] * 10}, {'name': 'fc.bias', 'fill': 2}]
            text, err = ep.call('submit', {'weights': diff})
            self.assertFalse(err)
            self.assertEqual(text, 'accepted: 12 numbers in 2 tensors')
            self.assertEqual(ep.log[-1]['args']['weights'], ['fc.weight = 0', 'fc.weight [0:10, 3:4]', 'fc.bias = 2'])

    def test_openai_strict_schema_makes_optional_keys_nullable(self):
        from wsp.agent import strict_schema
        spec = next(t for t in tool_specs('reasoned', writable_names(initial_weights(10))) if t['name'] == 'predict')
        entry = strict_schema(spec['input_schema'])['properties']['weights']['items']
        self.assertEqual(entry['required'], ['name', 'index', 'values', 'fill'])
        self.assertEqual(entry['properties']['fill']['type'], ['number', 'null'])
        self.assertEqual(entry['properties']['name']['type'], 'string')
        self.assertEqual(strict_schema(spec['input_schema'])['properties']['channels']['type'], ['array', 'null'])

    def test_all_problems_reported_together(self):
        start = initial_weights(10)
        diff = [{'name': 'layer4.1.bn2.weight', 'values': [0] * 504}, {'name': 'fc.bias', 'values': [0.0] * 10},
                {'name': 'layer3.1.bn2.weight', 'values': [0] * 258}, {'name': 'nope', 'values': [1]}]
        with self.assertRaises(ValueError) as ctx:
            apply_diff(start, diff, 10000)
        message = str(ctx.exception)
        self.assertTrue(message.startswith('3 problems, nothing applied: '))
        for part in ('layer4.1.bn2.weight: got 504', 'layer3.1.bn2.weight: got 258', "'nope' is not a writable"):
            self.assertIn(part, message)
        self.assertNotIn('fc.bias', message)

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
        self.assertEqual([t['name'] for t in tool_specs('reasoned', names)],
                         ['data', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit'])
        self.assertEqual(len(tool_specs('full', names)), 6)

    def test_transfer_start(self):
        source = initial_weights(10, seed=1)
        start = start_weights(TASKS['cifar10-to-cifar100'], source)
        self.assertEqual({k: list(v.shape) for k, v in start.items()}, state_shapes(100))
        self.assertTrue(torch.equal(start['conv1.weight'], source['conv1.weight']))


class LoggingTests(unittest.TestCase):
    def test_call_status_and_clipped_result(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            ep.call('submit', BIAS)
            ep.call('tools', {'code': "print('a' * 3000 + 'END')"})
            ep.call('tools', {'code': "raise ValueError('boom')"})
            ep.call('end_tools', {})
            ep.call('evaluate', {'weights': [{'name': 'fc.bias', 'values': [0.0]}]})
            ep.call('tools', {'code': 'print(1)'})
            self.assertEqual([e['status'] for e in ep.log], ['refused', 'ok', 'ok', 'ok', 'format', 'refused'])
            long = ep.log[1]['result']
            self.assertLess(len(long), 2100)
            self.assertIn('chars omitted', long)
            self.assertTrue(long.rstrip().endswith('END'))
            self.assertEqual(clip('short'), 'short')
            self.assertEqual([describe(e) for e in ep.log][:2], [
                'submit (refused: submit() is available only in phase 2. Call end_tools() to enter it; that '
                'permanently disables tools())', 'tools'])
            self.assertEqual(describe(ep.log[2]), 'tools (exit code 1)')
            self.assertEqual(describe(ep.log[4]), 'evaluate (format: fc.bias: got 1 numbers, shape [10] needs 10)')
            self.assertTrue(ep.log[4]['result'].startswith('evaluate() not run, the diff has format problems'))

    def test_turn_line(self):
        entries = [{'call': 'tools', 'phase': 1, 'status': 'ok', 'result': 'x'},
                   {'call': 'end_tools', 'phase': 1, 'status': 'ok', 'result': 'Phase 2 has started.'}]
        openrouter = {'prompt_tokens': 52000, 'completion_tokens': 3100, 'cost': 0.061,
                      'prompt_tokens_details': {'cached_tokens': 50000}}
        self.assertEqual(turn_line(36, entries, 'reasoned', True, 81.4, openrouter),
                         'turn 36 [phase 1->2]: tools, end_tools | 81s | in 52.0k (50.0k cached), out 3.1k, $0.061')
        anthropic = {'input_tokens': 200, 'cache_read_input_tokens': 40000, 'cache_creation_input_tokens': 1800,
                     'output_tokens': 900}
        self.assertEqual(turn_line(3, [], 'full', False, 5, anthropic),
                         'turn 3: no calls | 5s | in 42.0k (40.0k cached), out 0.9k')
        self.assertEqual(turn_line(2, [], 'reasoned', False, 1, None), 'turn 2 [phase 1]: no calls | 1s')
        rejected = [{'call': 'submit', 'phase': 2, 'status': 'format', 'reason': 'fc.bias: got 9', 'result': 'x'}]
        self.assertEqual(turn_line(4, rejected, 'reasoned', True, 30, None, counted=False),
                         'turn 5 retry (format only, not counted) [phase 2]: submit (format: fc.bias: got 9) | 30s')

    def test_no_call_turns_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep, turns = episode('reasoned', tmp), []
            agent = ScriptedAgent([[], [{'name': 'end_tools', 'args': {}}], [{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep, on_turn=lambda t, counted, r: turns.append((t, counted)))
            self.assertEqual(turns, [(1, True), (2, True), (3, True)])


class TurnTests(unittest.TestCase):
    BAD = {'weights': [{'name': 'fc.bias', 'values': [0.0] * 9}]}

    def recorder(self, turns):
        class Recorder(ScriptedAgent):
            def next(self, results):
                self.seen = getattr(self, 'seen', []) + [results]
                return super().next(results)
        return Recorder(turns)

    def test_sdk_turn_rules_in_the_episode_loop(self):
        from wsp.agent import CUT_OFF

        class CutThenWork:
            def __init__(self):
                self.cut_off, self.seen, self.step = False, [], 0

            def next(self, results):
                self.seen.append(results)
                self.step += 1
                if self.step == 1:
                    self.cut_off = True
                    return []
                if self.step == 2:
                    return [('a', 'evaluate', TurnTests.BAD), ('b', 'tools', {'code': 'print(1)'})]
                return [('c', 'labels', {'indices': [0]})]

        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            agent, turns = CutThenWork(), []
            run_episode(agent, episode('reasoned', tmp), 3, lambda t, counted, r: turns.append((t, counted)), 3, 'sdk')
            self.assertEqual(agent.seen[1], CUT_OFF)
            self.assertEqual(turns, [(1, True), (2, True), (3, True)])
            self.assertTrue(all(text.endswith('\n[harness] turn 2 of 3; 1 turns left after this one.')
                                for _, text, _ in agent.seen[2]))
            self.assertEqual(len(agent.seen[2]), 2)

    def test_openrouter_cut_off_reply_continues(self):
        from wsp.agent import OpenRouterAgent

        class Response:
            def __init__(self, finish):
                message = {'role': 'assistant', 'content': None, 'reasoning': 'r',
                           'tool_calls': [{'id': 'x', 'type': 'function',
                                           'function': {'name': 'tools', 'arguments': '{"code": "pri'}}]}
                self.choices = [type('C', (), {'finish_reason': finish,
                                               'message': type('M', (), {'model_dump': lambda s, **k: dict(message)})()})()]
                self.usage = None

        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'x'}):
            agent = OpenRouterAgent('s', 'p', [], continue_on_cap=True)
            classic = OpenRouterAgent('s', 'p', [])
        for a in (agent, classic):
            a.client = type('Client', (), {})()
            a.client.chat = type('Chat', (), {})()
            a.client.chat.completions = type('Completions', (), {'create': lambda s, **k: Response('length')})()
        self.assertEqual(agent.next(None), [])
        self.assertTrue(agent.cut_off)
        self.assertEqual(agent.messages[-1], {'role': 'assistant', 'content': '', 'reasoning': 'r'})
        with self.assertRaisesRegex(RuntimeError, 'length'):
            classic.next(None)

    def test_openrouter_sdk_turn_rules_prompt(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            out = Path(tmp) / 'run'
            with redirect_stdout(io.StringIO()):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--provider',
                      'openrouter', '--turn-rules', 'sdk', '--max-turns', '100', '--data-dir', tmp, '--output',
                      str(out), '--dry-run'])
            user = json.loads((out / 'prompt.json').read_text())['user']
            for part in ('# Turns and budget\nYou have 100 turns.', 'within about 960,000 output tokens',
                         'calling end_tools() by about turn 50, and before', 'a reply that reaches it is cut off'):
                self.assertIn(part, user)
            self.assertNotIn('# Turns\n', user)
            with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--provider',
                      'anthropic', '--turn-rules', 'sdk', '--data-dir', tmp, '--output', str(out) + '2', '--dry-run'])

    def test_format_only_replies_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep, turns = episode('reasoned', tmp), []
            agent = self.recorder([[{'name': 'end_tools', 'args': {}}]] + [[{'name': 'submit', 'args': self.BAD}]] * 4
                                  + [[{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep, max_turns=3, on_turn=lambda t, c, r: turns.append((t, c)), format_retries=3)
            self.assertEqual(turns, [(1, True), (1, False), (1, False), (1, False), (2, True), (3, True)])
            self.assertIsNotNone(ep.submission)
            notes = [r[-1][1].split('\n\n')[-1] for r in agent.seen[1:]]
            self.assertEqual(notes[0], '[harness] Turn 1 of 3 done; 2 turns left.')
            self.assertEqual(notes[1], '[harness] Every call in this reply was rejected for format problems, so it did '
                                       'not count as a turn (uncounted format retries left: 2). Still 2 turns left.')
            self.assertEqual(notes[4], '[harness] Turn 2 of 3 done; 1 turn left. Your next reply is your last turn: '
                                       'it must contain a valid submit().')
            self.assertIn('got 9 numbers', agent.seen[2][-1][1])

    def test_mixed_reply_counts_and_turn_limit_ends_episode(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep, turns = episode('reasoned', tmp), []
            agent = self.recorder([[{'name': 'tools', 'args': {'code': 'print(1)'}}],
                                   [{'name': 'tools', 'args': {'code': 'print(2)'}}, {'name': 'submit', 'args': BIAS}],
                                   [{'name': 'end_tools', 'args': {}}], [{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep, max_turns=2, on_turn=lambda t, c, r: turns.append((t, c)))
            self.assertEqual(turns, [(1, True), (2, True)])
            self.assertIsNone(ep.submission)
            self.assertIn('call end_tools() before submit() in that same reply', agent.seen[1][-1][1])

    def test_prompts_state_turns_phase2_reasoning_and_limits(self):
        start = initial_weights(10)
        prompt = user_prompt(TASKS['scratch-cifar10'], 'reasoned', start, 20, 10000, max_turns=10, format_retries=3,
                             max_tokens=96000)
        for part in ('# How this episode works (mode reasoned)', '20 images, indices 0..19',
                     'data["images"][i] in tools() code is image i', '- PHASE 1, no labels. Available: tools(), data(), '
                     'predict()', 'end_tools() ends phase 1. It is one-way', '- PHASE 2, labels. Available: labels() on '
                     'any subset image, evaluate()', '# What we want you to do', 'call predict() with your diff on many '
                     'images, and labels() on the same indices', 'calling end_tools() by about turn 5',
                     '# Diffs and costs', 'at most 10,000 numbers in total', 'each take a whole diff',
                     'limited to 96,000 output tokens, thinking included', 'You have 10 turns. A turn is one reply',
                     'once submit() is accepted, the rest of that reply is skipped', 'A reply with no calls still uses',
                     'at most 3 such replies'):
            self.assertIn(part, prompt)
        self.assertNotIn('Pool', prompt)
        long = user_prompt(TASKS['scratch-cifar10'], 'reasoned', start, 20, 10000, max_turns=100, format_retries=3,
                           max_tokens=96000)
        self.assertIn('calling end_tools() by about turn 50.', long)
        self.assertIn('You have 100 turns.', long)
        self.assertIn('reason over the (representation, label)\npairs yourself', system_prompt('reasoned'))
        self.assertNotIn('representation', system_prompt('full'))
        specs = {t['name']: t['description'] for t in tool_specs('reasoned', writable_names(start), 300, 'cpu')}
        expected = {'data': ['96 lines', 'At most 16 images per call', 'Never includes labels'],
                    'labels': ['{"index": "class name"}', 'Phase 2 only'],
                    'tools': ['at most 300 s on cpu', 'last 10,000 characters', 'uint8 tensor [N, 3, 32, 32]'],
                    'end_tools': ['One-way', 'calling it again is an error'],
                    'predict': ['an empty list means the start weights', '"i [predicted class]: v1 ... v10"',
                                'At most 500 images and 6,000 numbers', 'Works in both phases'],
                    'evaluate': ['{"accuracy": x}', 'no per-class results', 'Phase 2 only'],
                    'submit': ['Phase 2 only', '"accepted: N numbers in K tensors"', 'An empty diff is rejected']}
        for name, parts in expected.items():
            for part in parts:
                with self.subTest(name=name, part=part):
                    self.assertIn(part, specs[name])
        blind = user_prompt(TASKS['scratch-cifar10'], 'blind', start, 0, 10000)
        self.assertIn('submit() takes a whole diff', blind)
        self.assertNotIn('output tokens, thinking included', blind)


class PredictTests(unittest.TestCase):
    @staticmethod
    def parse(text):
        head, *rows = text.splitlines()
        return head, {int(r.split(' ')[0].rstrip(':')): [float(v) for v in r.split(': ', 1)[1].split()] for r in rows}

    def test_fc_and_intermediate_layers_match_the_network(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            weights, _ = apply_diff(ep.start, BIAS['weights'], 10000)
            text, err = ep.call('predict', {**BIAS, 'indices': [0, 3], 'layer': 'fc', 'channels': None})
            self.assertFalse(err)
            head, rows = self.parse(text)
            self.assertIn('10 logits per image', head)
            want = layer_outputs(weights, ep.images[[0, 3]], 'fc', 10)
            for k, i in enumerate([0, 3]):
                self.assertTrue(torch.allclose(torch.tensor(rows[i]), want[k], rtol=6e-3, atol=1e-6))
                self.assertIn(f'{i} [{CLASSES[int(want[k].argmax())]}]:', text)
            text, err = ep.call('predict', {'weights': [], 'indices': [1, 2, 4], 'layer': 'layer4', 'channels': [5, 13]})
            head, rows = self.parse(text)
            self.assertIn('channels 5..12', head)
            self.assertEqual([len(v) for v in rows.values()], [8, 8, 8])
            pooled = layer_outputs(ep.start, ep.images[[1, 2, 4]], 'avgpool', 10)[:, 5:13]
            self.assertTrue(torch.allclose(torch.tensor([rows[i] for i in (1, 2, 4)]), pooled, rtol=6e-3, atol=1e-6))
            _, rows = self.parse(ep.call('predict', {'weights': [], 'indices': [0], 'layer': 'layer2.0.bn1',
                                                     'channels': None})[0])
            self.assertEqual(len(rows[0]), 128)
            self.assertEqual(audit('reasoned', ep.log), [])

    def test_diff_is_used_and_works_in_both_phases(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            push = {'weights': [{'name': 'fc.bias', 'values': [0] * 3 + [1e4] + [0] * 6}]}
            text, _ = ep.call('predict', {**push, 'indices': [0, 1, 2], 'layer': 'fc', 'channels': None})
            self.assertEqual(text.count('[c3]'), 3)
            self.assertFalse(ep.call('predict', {'weights': [], 'indices': [12], 'layer': 'fc', 'channels': None})[1])
            ep.call('end_tools', {})
            self.assertFalse(ep.call('predict', {'weights': [], 'indices': [12, 19], 'layer': 'fc',
                                                 'channels': None})[1])
            self.assertEqual(audit('reasoned', ep.log), [])

    def test_bad_requests(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('full', tmp)
            base = {'weights': [], 'indices': [0], 'layer': 'fc', 'channels': None}
            cases = [({'layer': 'nope'}, 'failed', 'unknown layer'),
                     ({'layer': 'layer4', 'indices': list(range(12))}, 'failed', 'at most 6000'),
                     ({'layer': 'layer4', 'channels': [8, 3]}, 'failed', 'channels must be'),
                     ({'indices': []}, 'failed', 'give 1 to 500'),
                     ({'weights': [{'name': 'fc.bias', 'values': [0.0] * 9}]}, 'format', 'got 9 numbers')]
            for change, status, message in cases:
                with self.subTest(change):
                    text, err = ep.call('predict', {**base, **change})
                    self.assertTrue(err)
                    self.assertEqual(ep.log[-1]['status'], status)
                    self.assertIn(message, text)
            self.assertFalse(ep.call('predict', {**base, 'indices': [0, 19]})[1])
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            self.assertIn('not available', episode('blind', tmp).call('predict', base)[0])

    def test_predict_schema_and_docs(self):
        specs = {t['name']: t for t in tool_specs('reasoned', writable_names(initial_weights(10)))}
        schema = specs['predict']['input_schema']
        self.assertEqual(schema['required'], ['weights', 'indices', 'layer', 'channels'])
        self.assertIn('Works in both phases, on any subset image', specs['predict']['description'])
        self.assertIn('labels() on any image (indices 0..19)', episode_text_end_tools())


@unittest.skipUnless(importlib.util.find_spec('claude_agent_sdk'), 'claude-agent-sdk not installed')
class SDKBackendTests(unittest.TestCase):
    def test_tools_route_to_the_episode_and_stop_after_submit(self):
        from wsp.sdk_agent import episode_tools
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            tools = {t.name: t for t in episode_tools(ep, tool_specs('reasoned', writable_names(ep.start)))}
            self.assertEqual(set(tools), {'data', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit'})
            self.assertTrue(all(t.annotations.maxResultSizeChars >= 200_000 for t in tools.values()))
            run = lambda name, args: asyncio.run(tools[name].handler(args))
            out = run('labels', {'indices': [0]})
            self.assertTrue(out['is_error'])
            self.assertIn('only in phase 2', out['content'][0]['text'])
            self.assertFalse(run('end_tools', {})['is_error'])
            self.assertEqual(json.loads(run('labels', {'indices': [3]})['content'][0]['text']), {'3': 'c3'})
            self.assertFalse(run('submit', BIAS)['is_error'])
            late = run('labels', {'indices': [4]})
            self.assertTrue(late['is_error'])
            self.assertIn('episode is over', late['content'][0]['text'])
            self.assertEqual([e['call'] for e in ep.log], ['labels', 'end_tools', 'labels', 'submit'])

    def test_read_fence_allows_only_saved_tool_outputs(self):
        from wsp.sdk_agent import read_allowed, saved_outputs_dir
        with tempfile.TemporaryDirectory() as tmp:
            root = saved_outputs_dir(tmp)
            self.assertTrue(read_allowed(root / 'abc-session' / 'tool-results' / 'x.txt', tmp))
            self.assertFalse(read_allowed(root / 'abc-session.jsonl', tmp))
            self.assertFalse(read_allowed('/lfs/local/0/vedanga/scratch/ricursive-wsp/data/cifar10_train.pt', tmp))
            self.assertFalse(read_allowed(root / '..' / 'other' / 'tool-results' / 'x.txt', tmp))

    def test_transcript_serialisation_and_content_types(self):
        from claude_agent_sdk import AssistantMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage
        from wsp.sdk_agent import content_types, to_json
        messages = [to_json(AssistantMessage(content=[TextBlock('hi'), ToolUseBlock('t1', 'mcp__wsp__data', {})],
                                             model='m')),
                    to_json(UserMessage(content=[ToolResultBlock('t1', [{'type': 'text', 'text': 'ok'}])]))]
        self.assertEqual(messages[0]['type'], 'AssistantMessage')
        self.assertEqual(content_types(messages), ['text', 'tool_use', 'tool_result', 'text'])
        self.assertIn('image', audit('reasoned', [], content_types(
            [to_json(UserMessage(content=[ToolResultBlock('t2', [{'type': 'image'}])]))]))[0])

    def test_soft_budget_prompt(self):
        start = initial_weights(10)
        prompt = user_prompt(TASKS['scratch-cifar10'], 'reasoned', start, 20, 10000, max_tokens=96000,
                             soft_budget=960000)
        self.assertIn('# Budget\nNothing enforces a limit on replies or tokens, but please keep the whole episode '
                      'within about 960,000 output tokens', prompt)
        self.assertIn('before you have used about half of the budget', prompt)
        self.assertIn('a reply that reaches it is cut off', prompt)
        self.assertNotIn('# Turns', prompt)
        self.assertNotIn('by about turn 5', prompt)

    def test_turn_limit_notes_and_closing(self):
        from wsp.sdk_agent import SDKRun, episode_tools
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            turns = {'reply': 3, 'limit': 5, 'closed': False}
            tools = {t.name: t for t in episode_tools(ep, tool_specs('reasoned', writable_names(ep.start)), turns)}
            run = lambda name, args: asyncio.run(tools[name].handler(args))['content'][0]['text']
            self.assertTrue(run('labels', {'indices': [0]}).endswith(
                '\n[harness] turn 3 of 5; 2 turns left after this one.'))
            turns['reply'] = 5
            self.assertTrue(run('end_tools', {}).endswith(
                '[harness] turn 5 of 5: this is your last turn. Without an accepted submit() in this reply the '
                'episode ends with no score.'))
            turns['closed'] = True
            self.assertIn('turn limit was reached', run('labels', {'indices': [1]}))
            self.assertEqual([e['call'] for e in ep.log], ['labels', 'end_tools'])
            turns['closed'] = False
            self.assertEqual(run('submit', BIAS), 'accepted: 10 numbers in 1 tensors')
            agent = SDKRun(episode('reasoned', Path(tmp) / 'b'), 's', 'p', [], 'm', tmp, max_replies=2)
            self.assertEqual([agent.new_reply() for _ in range(3)], [False, False, True])
            self.assertTrue(agent.turns['closed'])
            done = SDKRun(ep, 's', 'p', [], 'm', tmp, max_replies=1)
            self.assertEqual([done.new_reply() for _ in range(3)], [False, False, False])
            self.assertFalse(SDKRun(ep, 's', 'p', [], 'm', tmp).new_reply())

    def test_usage_hold(self):
        from wsp.sdk_agent import SDKRun, episode_tools, five_hour_window, hold_for_usage
        event = {'rate_limit_info': {'raw': {'unifiedWindows': {'five_hour': {'utilization': 0.45, 'resetsAt': 1000}}}}}
        self.assertEqual(five_hour_window(event), (0.45, 1000))
        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        window = {'limit': 0.4, 'used': 0.45, 'resets': 1000, 'holds': []}
        asyncio.run(hold_for_usage(window, fake_sleep, lambda: 400.0))
        self.assertEqual(slept, [720.0])
        self.assertIsNone(window['used'])
        self.assertEqual(window['holds'], [{'at': 400.0, 'used': 0.45, 'seconds': 720.0}])
        for w in ({'limit': 0.4, 'used': 0.39, 'resets': 1000, 'holds': []},
                  {'limit': None, 'used': 0.9, 'resets': 1000, 'holds': []}, None):
            asyncio.run(hold_for_usage(w, fake_sleep, lambda: 400.0))
        self.assertEqual(len(slept), 1)
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            agent = SDKRun(ep, 's', 'p', [], 'm', tmp, usage_limit=0.4)
            agent.rate_event(event)
            self.assertEqual((agent.window['used'], agent.window['resets']), (0.45, 1000))
            self.assertEqual(agent.options().env['MCP_TOOL_TIMEOUT'], str(6 * 3600 * 1000))
            self.assertEqual(SDKRun(ep, 's', 'p', [], 'm', tmp).options().env['MCP_TOOL_TIMEOUT'], str(30 * 60 * 1000))
            window = {'limit': 0.4, 'used': 0.5, 'resets': 0, 'holds': []}
            tools = {t.name: t for t in episode_tools(ep, tool_specs('reasoned', writable_names(ep.start)),
                                                      None, window)}
            with patch('wsp.sdk_agent.asyncio.sleep', fake_sleep), redirect_stdout(io.StringIO()) as out:
                text = asyncio.run(tools['labels'].handler({'indices': [0]}))['content'][0]['text']
            self.assertIn('only in phase 2', text)
            self.assertIn('usage hold: five-hour window 50% used >= limit 40%', out.getvalue())
            self.assertEqual(len(window['holds']), 1)

    def test_full_mode_prompt_has_no_reasoned_language(self):
        from wsp.sdk_agent import sdk_system
        start = initial_weights(10)
        prompt = user_prompt(TASKS['scratch-cifar10'], 'full', start, 20, 10000, max_tokens=96000,
                             soft_budget=960000, turn_limit=100)
        specs = tool_specs('full', writable_names(start))
        text = (system_prompt('full') + prompt + json.dumps(specs) + sdk_system([t['name'] for t in specs])).lower()
        for word in ('end_tools', 'phase 1', 'phase 2', 'two phases', 'reasoned', 'labeled set', 'in any mode'):
            self.assertNotIn(word, text)
        for part in ('# How this episode works (mode full)', 'There is a single phase', 'from the start, as often and '
                     'in any order you like', 'Your tools() code can read every label', 'the start network scores '
                     'about 10%, chance', '# What we want you to do\nSubmit the network with the highest held-out'):
            self.assertIn(part, prompt)
        tools_text = next(t['description'] for t in specs if t['name'] == 'tools')
        self.assertIn('"labels": int64 tensor [N] of class indices', tools_text)
        reasoned_tools = next(t['description'] for t in tool_specs('reasoned', writable_names(start))
                              if t['name'] == 'tools')
        self.assertIn('; no labels', reasoned_tools)
        self.assertNotIn('"labels"', reasoned_tools)
        self.assertIn('mcp__wsp__end_tools', sdk_system([t['name'] for t in tool_specs('reasoned', writable_names(start))]))

    def test_turn_limit_prompt(self):
        start = initial_weights(10)
        prompt = user_prompt(TASKS['scratch-cifar10'], 'reasoned', start, 20, 10000, max_tokens=96000,
                             soft_budget=960000, turn_limit=100)
        for part in ('# Turns and budget\nYou have 100 turns. A turn is one reply of yours (one model response)',
                     'a reply cut off at the per-reply output-token limit (its cut-off part is discarded)',
                     'Every call result ends with a [harness] note giving the current turn and how many are left',
                     'by the end of turn 100, the episode ends with no score',
                     'please keep the whole episode within about 960,000 output tokens',
                     'calling end_tools() by about turn 50, and before you have used about half of the token budget'):
            self.assertIn(part, prompt)
        self.assertNotIn('Nothing enforces a limit on replies', prompt)
        self.assertNotIn('# Budget', prompt)
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            out = Path(tmp) / 'run'
            with redirect_stdout(io.StringIO()):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--provider',
                      'claude-sdk', '--max-turns', '100', '--data-dir', tmp, '--output', str(out), '--dry-run'])
            self.assertIn('You have 100 turns', json.loads((out / 'prompt.json').read_text())['user'])
            config = json.loads((out / 'config.json').read_text())
            self.assertEqual((config['max_turns'], config['sdk_max_turns']), (100, 300))

    def test_dry_run_uses_soft_budget_prompt(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            out = Path(tmp) / 'run'
            with redirect_stdout(io.StringIO()):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--provider',
                      'claude-sdk', '--data-dir', tmp, '--output', str(out), '--dry-run'])
            user = json.loads((out / 'prompt.json').read_text())['user']
            self.assertIn('within about 960,000 output tokens', user)


class WarmStartTests(unittest.TestCase):
    def test_default_task_text_unchanged(self):
        self.assertEqual(TASKS['scratch-cifar10'].description,
                         'Write every weight of a ResNet-18 CIFAR-10 classifier yourself. The starting weights are a '
                         'random PyTorch initialization (BatchNorm running stats mean 0, var 1).')
        self.assertEqual(TASKS['cifar10-to-cifar100'].description,
                         'Turn a ResNet-18 trained on CIFAR-10 into a CIFAR-100 classifier. The starting weights are '
                         'the trained CIFAR-10 network, except fc (now 512 -> 100) which is a random initialization. '
                         'You may rebuild the head and edit any other weights.')

    def test_load_start_checks_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'w.pt'
            torch.save(initial_weights(100), path)
            with self.assertRaisesRegex(ValueError, 'not a 10-class'):
                load_start(path, 10)

    def test_warm_start_run_uses_given_network_and_states_its_score(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            given = initial_weights(10, seed=3)
            given['fc.bias'] = torch.arange(10, dtype=torch.float32)
            torch.save(given, Path(tmp) / 'given.pt')
            script = Path(tmp) / 'calls.json'
            script.write_text(json.dumps([[{'name': 'tools', 'args': {'code': "print(float(start['fc.bias'][9]))"}}],
                                          [{'name': 'end_tools', 'args': {}}],
                                          [{'name': 'submit', 'args': {'weights': [{'name': 'bn1.bias',
                                                                                    'values': [0.0] * 64}]}}]]))
            out = Path(tmp) / 'run'
            with redirect_stdout(io.StringIO()):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--script',
                      str(script), '--data-dir', tmp, '--device', 'cpu', '--output', str(out), '--start-weights',
                      str(Path(tmp) / 'given.pt')])
            user = json.loads((out / 'prompt.json').read_text())['user']
            self.assertIn('The starting weights are a network that another agent produced for this same task. That '
                          'agent was given an identical prompt, except that its starting weights were a random PyTorch '
                          'initialization (BatchNorm running stats mean 0, var 1). You are not told how it arrived at '
                          'this network.', user)
            val_start = json.loads((out / 'result.json').read_text())['val']['start']['accuracy']
            self.assertIn(f'the start network scores {val_start:.1%} on the held-out validation images', user)
            self.assertNotIn('about 10%, chance', user)
            self.assertIn('9.0', json.loads((out / 'calls.json').read_text())[0]['result'])
            submitted = torch.load(out / 'weights.pt')
            self.assertTrue(torch.equal(submitted['fc.bias'], given['fc.bias']))
            self.assertTrue(torch.equal(submitted['bn1.bias'], torch.zeros(64)))


def episode_text_end_tools():
    with tempfile.TemporaryDirectory() as tmp:
        return episode('reasoned', tmp).call('end_tools', {})[0]


class EndToEndTests(unittest.TestCase):
    def test_scripted_reasoned_run(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            fake_data_dir(tmp)
            script = Path(tmp) / 'calls.json'
            script.write_text(json.dumps([[{'name': 'tools', 'args': {'code': 'print(1)'}}],
                                          [{'name': 'submit', 'args': BIAS}],
                                          [{'name': 'end_tools', 'args': {}}],
                                          [{'name': 'tools', 'args': {'code': 'print(2)'}}],
                                          [{'name': 'labels', 'args': {'indices': [59]}}],
                                          [{'name': 'submit', 'args': BIAS}]]))
            out = Path(tmp) / 'run'
            printed = io.StringIO()
            with redirect_stdout(printed):
                main(['--task', 'scratch-cifar10', '--mode', 'reasoned', '--data-fraction', '1.0', '--script',
                      str(script), '--data-dir', tmp, '--device', 'cpu', '--output', str(out), '--evaluate-test'])
            lines = [l.split(' | ')[0] for l in printed.getvalue().splitlines() if l.startswith('turn ')]
            self.assertEqual(lines, [
                'turn 1 [phase 1]: tools',
                'turn 2 [phase 1]: submit (refused: submit() is available only in phase 2. Call end_tools() to enter '
                'it; that permanently disables tools())',
                'turn 3 [phase 1->2]: end_tools',
                'turn 4 [phase 2]: tools (refused: tools() is permanently disabled: end_tools() has been called)',
                'turn 5 [phase 2]: labels',
                'turn 6 [phase 2]: submit'])
            self.assertTrue((out / 'usage.json').exists())
            pending = json.loads((out / 'transcript.json').read_text())[-1]
            self.assertEqual(pending['role'], 'pending_tool_results')
            self.assertTrue(pending['content'][-1]['content'].startswith('accepted'))
            self.assertEqual(json.loads((out / 'audit.json').read_text()), {'clean': True, 'problems': []})
            calls = json.loads((out / 'calls.json').read_text())
            self.assertEqual([(c['call'], c['executed']) for c in calls],
                             [('tools', True), ('submit', False), ('end_tools', True), ('tools', False),
                              ('labels', True), ('submit', True)])
            split = json.loads((out / 'subset_indices.json').read_text())
            self.assertEqual(set(split), {'indices'})
            self.assertEqual(sorted(split['indices']), list(range(60)))
            self.assertEqual(len(torch.load(out / 'workspace' / 'data.pt')['images']), 60)
            self.assertEqual(set(json.loads((out / 'result.json').read_text())), {'calls', 'val', 'test'})
            self.assertTrue(torch.equal(torch.load(out / 'weights.pt')['fc.bias'], torch.zeros(10)))

    def test_mode_fraction_mismatch(self):
        for args in (['--mode', 'blind', '--data-fraction', '0.1'], ['--mode', 'full']):
            with self.subTest(args), self.assertRaises(SystemExit):
                main(['--task', 'scratch-cifar10', '--dry-run'] + args)


if __name__ == '__main__':
    unittest.main()
