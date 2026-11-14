"""Offline tests: no API key, no network, no CIFAR download (tiny random splits stand in)."""
from contextlib import redirect_stdout
from functools import partial
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
from wsp.evaluation import evaluate
from wsp.model import initial_weights, state_shapes
from wsp.prompt import system_prompt, tool_specs, user_prompt
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
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])

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
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])
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

    def test_pool_and_labeled_set_rules(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            self.assertIn('only in phase 2', ep.call('data', {'indices': [10]})[0])
            self.assertFalse(ep.call('data', {'indices': [9]})[1])
            self.assertFalse(ep.locked)
            self.assertFalse(ep.call('end_tools', {})[1])
            self.assertIn('have no labels', ep.call('labels', {'indices': [3]})[0])
            self.assertFalse(ep.call('labels', {'indices': [19]})[1])
            self.assertIn('"accuracy"', ep.call('evaluate', BIAS)[0])
            self.assertFalse(ep.call('data', {'indices': [10]})[1])
            self.assertEqual(audit('reasoned', ep.log, pool=10), [])

    def test_evaluate_scores_labeled_set_only_and_returns_accuracy_only(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep = episode('reasoned', tmp)
            ep.call('end_tools', {})
            m = json.loads(ep.call('evaluate', BIAS)[0])
            self.assertEqual(set(m), {'accuracy'})
            weights, _ = apply_diff(ep.start, BIAS['weights'], 10000)
            labeled = evaluate(weights, (ep.images[10:], ep.labels[10:]), CLASSES, 'cpu')['accuracy']
            self.assertEqual(m['accuracy'], labeled)
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            m = json.loads(episode('full', tmp).call('evaluate', BIAS)[0])
            self.assertIn('confusion', m)

    def test_audit_catches_broken_phases_and_images(self):
        log = [{'call': 'end_tools', 'executed': True}, {'call': 'tools', 'executed': True}]
        self.assertIn('tools executed after end_tools', audit('reasoned', log)[0])
        for call in ('labels', 'evaluate', 'submit'):
            self.assertIn(f'{call} executed before end_tools', audit('reasoned', [{'call': call, 'executed': True}])[0])
        self.assertIn('pool images', audit('reasoned', [{'call': 'end_tools', 'executed': True},
                                                        {'call': 'labels', 'executed': True, 'args': {'indices': [1]}}],
                                           pool=5)[0])
        self.assertIn('before end_tools', audit('reasoned', [{'call': 'data', 'executed': True, 'args': {'indices': [7]}}], pool=5)[0])
        self.assertIn('end_tools executed in full', audit('full', log)[0])
        self.assertEqual(audit('full', [{'call': 'labels', 'executed': True}, {'call': 'tools', 'executed': True}]), [])
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
            for name in ('data', 'labels', 'evaluate', 'end_tools'):
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
                         ['data', 'labels', 'tools', 'end_tools', 'evaluate', 'submit'])
        self.assertEqual(len(tool_specs('full', names)), 5)

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
            run_episode(agent, ep, on_turn=lambda t, counted: turns.append((t, counted)))
            self.assertEqual(turns, [(1, True), (2, True), (3, True)])


class TurnTests(unittest.TestCase):
    BAD = {'weights': [{'name': 'fc.bias', 'values': [0.0] * 9}]}

    def recorder(self, turns):
        class Recorder(ScriptedAgent):
            def next(self, results):
                self.seen = getattr(self, 'seen', []) + [results]
                return super().next(results)
        return Recorder(turns)

    def test_format_only_replies_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp, NO_GPU:
            ep, turns = episode('reasoned', tmp), []
            agent = self.recorder([[{'name': 'end_tools', 'args': {}}]] + [[{'name': 'submit', 'args': self.BAD}]] * 4
                                  + [[{'name': 'submit', 'args': BIAS}]])
            run_episode(agent, ep, max_turns=3, on_turn=lambda t, c: turns.append((t, c)), format_retries=3)
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
            run_episode(agent, ep, max_turns=2, on_turn=lambda t, c: turns.append((t, c)))
            self.assertEqual(turns, [(1, True), (2, True)])
            self.assertIsNone(ep.submission)
            self.assertIn('call end_tools() before submit() in that same reply', agent.seen[1][-1][1])

    def test_prompts_state_turns_phase2_reasoning_and_limits(self):
        start = initial_weights(10)
        prompt = user_prompt(TASKS['scratch-cifar10'], 'reasoned', start, 20, 10000, pool=10, max_turns=10,
                             format_retries=3)
        self.assertIn('You have 10 turns. A turn is one reply of yours', prompt)
        self.assertIn('at most 3 such replies', prompt)
        self.assertIn('reasoning in your own context', system_prompt('reasoned'))
        self.assertNotIn('reasoning in your own context', system_prompt('full'))
        specs = {t['name']: t['description'] for t in tool_specs('reasoned', writable_names(start))}
        self.assertTrue(specs['evaluate'].startswith('Accuracy (one number)'))
        self.assertIn('last 10,000 characters', specs['tools'])
        self.assertIn('evaluate() and submit() check', prompt)
        self.assertIn('submit() check', user_prompt(TASKS['scratch-cifar10'], 'blind', start, 0, 10000))


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
            self.assertTrue((out / 'transcript.json').exists() and (out / 'usage.json').exists())
            self.assertEqual(json.loads((out / 'audit.json').read_text()), {'clean': True, 'problems': []})
            calls = json.loads((out / 'calls.json').read_text())
            self.assertEqual([(c['call'], c['executed']) for c in calls],
                             [('tools', True), ('submit', False), ('end_tools', True), ('tools', False),
                              ('labels', True), ('submit', True)])
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
