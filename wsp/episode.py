"""The calls an agent can make, the reasoned-mode phases, and the audit.

    data(indices)     training images as hex text (never labels)
    labels(indices)   class labels
    tools(code)       Python in the sandbox
    end_tools()       reasoned only: leave phase 1 for good
    evaluate(weights) training-subset accuracy of a diff (aggregate numbers only)
    submit(weights)   final diff; ends the episode if valid

reasoned mode splits the training subset into a pool (indices 0..pool-1) and a labeled set
(pool..N-1), and runs in two phases:
  1. phase 1: tools() and data() on the pool; labels(), evaluate() and submit() are refused;
  2. end_tools() switches to phase 2 for good: tools() never executes again, and labels(),
     evaluate(), submit() and data() on the labeled set become available;
  3. the sandbox holds only pool images; pool labels are never returned by any call;
  4. labels() covers only the labeled set, and evaluate() scores on it, returning the accuracy only.
evaluate() and submit() check a diff's format first; a format rejection is logged with status "format".
So nothing derived from a label ever reaches code, and no number computed by code describes an
image whose label the agent sees.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from wsp.evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent
MODES = ('blind', 'reasoned', 'full')
ALLOWED = {'blind': {'tools', 'submit'},
           'reasoned': {'data', 'labels', 'tools', 'end_tools', 'evaluate', 'submit'},
           'full': {'data', 'labels', 'tools', 'evaluate', 'submit'}}
PHASE2_CALLS = {'labels', 'evaluate', 'submit'}
MAX_IMAGES_PER_CALL = 16
MAX_OUTPUT_CHARS = 10000


class FormatError(ValueError):
    pass


def pixels_text(image):
    """uint8 [3,32,32] -> 96 lines (R rows 0-31, G rows, B rows) of 64 hex digits."""
    return '\n'.join(image[c, r].numpy().tobytes().hex() for c in range(3) for r in range(32))


class Episode:
    def __init__(self, mode, start, images, labels, classes, workspace, apply_diff, device='cpu',
                 timeout=300, log=None, pool=0):
        assert mode in MODES and (mode == 'reasoned') == (pool > 0) and pool < max(len(labels), 1)
        self.mode, self.start, self.images, self.labels, self.classes = mode, start, images, labels, classes
        self.pool = pool
        self.workspace, self.apply_diff, self.device, self.timeout = Path(workspace), apply_diff, device, timeout
        self.locked, self.submission, self.log = False, None, log if log is not None else []
        self.workspace.mkdir(parents=True, exist_ok=True)
        torch.save(start, self.workspace / 'start.pt')
        if mode == 'reasoned':  # pool images only, no labels, ever
            torch.save({'images': images[:pool], 'classes': classes}, self.workspace / 'data.pt')
        elif mode == 'full':
            torch.save({'images': images, 'labels': labels, 'classes': classes}, self.workspace / 'data.pt')

    def call(self, name, args):
        """Run one call; returns (text, is_error). Every call is logged with its phase."""
        entry = {'call': name, 'phase': 2 if self.locked else 1, 'executed': False}
        status = 'refused'
        if name not in ALLOWED[self.mode]:
            text = reason = f'{name}() is not available in {self.mode} mode'
        elif name == 'tools' and self.locked:
            text = reason = 'tools() is permanently disabled: end_tools() has been called'
        elif self.mode == 'reasoned' and not self.locked and name in PHASE2_CALLS:
            text = reason = (f'{name}() is available only in phase 2. Call end_tools() to enter it; that '
                             'permanently disables tools()')
        else:
            try:
                text, status, reason = getattr(self, f'_{name}')(args), 'ok', None
                entry['executed'] = True
            except FormatError as e:
                status, reason = 'format', str(e)
                text = f'{name}() not run, the diff has format problems (fix them and call again): {e}'
            except (ValueError, KeyError, TypeError) as e:
                status, reason = 'failed', str(e)
                text = f'{name}() failed: {e}'
        error = status != 'ok'
        entry.update(status=status, error=error, args=_summary(args), result=clip(text))
        if reason is not None:
            entry['reason'] = reason
        self.log.append(entry)
        return text, error

    def _indices(self, args):
        idx = args['indices']
        if not isinstance(idx, list) or not all(isinstance(i, int) and 0 <= i < len(self.labels) for i in idx):
            raise ValueError(f'indices must be a list of ints in [0, {len(self.labels)})')
        return idx

    def _data(self, args):
        idx = self._indices(args)
        if len(idx) > MAX_IMAGES_PER_CALL:
            raise ValueError(f'at most {MAX_IMAGES_PER_CALL} images per call')
        if self.pool and not self.locked and any(i >= self.pool for i in idx):
            raise ValueError(f'labeled-set images (indices >= {self.pool}) are visible only in phase 2, after end_tools()')
        return '\n\n'.join(f'image {i}:\n{pixels_text(self.images[i])}' for i in idx)

    def _end_tools(self, args):
        if self.locked:
            raise ValueError('already in phase 2')
        self.locked = True
        return ('Phase 2 has started. tools() is permanently disabled. Available now: labels() and evaluate() on the '
                f'labeled set (indices {self.pool}..{len(self.labels) - 1}), data() on any index, and submit().')

    def _labels(self, args):
        idx = self._indices(args)
        if self.pool and any(i < self.pool for i in idx):
            raise ValueError(f'pool images (indices < {self.pool}) have no labels; labeled set is {self.pool}..{len(self.labels) - 1}')
        return json.dumps({i: self.classes[self.labels[i]] for i in idx})

    def _tools(self, args):
        (self.workspace / '_call.py').write_text(args['code'])
        env = {k: v for k, v in os.environ.items() if 'API_KEY' not in k.upper()}
        env['PYTHONPATH'] = str(ROOT)
        try:
            p = subprocess.run([sys.executable, '-m', 'wsp.sandbox', str(self.workspace)], env=env,
                               capture_output=True, text=True, timeout=self.timeout)
            out = (p.stdout + p.stderr)[-MAX_OUTPUT_CHARS:] or '(no output)'
            return out if p.returncode == 0 else f'exit code {p.returncode}\n{out}'
        except subprocess.TimeoutExpired:
            return f'timed out after {self.timeout}s'

    def _diff(self, args):
        try:
            return self.apply_diff(self.start, args.get('weights'))
        except ValueError as e:
            raise FormatError(str(e)) from None

    def _evaluate(self, args):
        weights, _ = self._diff(args)
        m = evaluate(weights, (self.images[self.pool:], self.labels[self.pool:]), self.classes, self.device)
        if self.mode == 'reasoned':
            return json.dumps({'accuracy': m['accuracy']})
        m.pop('activations', None)
        return json.dumps(m)

    def _submit(self, args):
        if not args.get('weights'):
            raise FormatError('the diff is empty: it would leave the start network unchanged. Write the tensors you '
                              'want to set, every number in them')
        weights, numbers = self._diff(args)
        self.submission = weights
        return f'accepted: {numbers} numbers in {len(args["weights"])} tensors'


def clip(text, limit=2000):
    if len(text) <= limit:
        return text
    return f'{text[:limit // 2]}\n[... {len(text) - limit} chars omitted ...]\n{text[-(limit // 2):]}'


def _summary(args):
    if 'weights' in args and isinstance(args['weights'], list):
        return {'weights': [e.get('name') if isinstance(e, dict) else '?' for e in args['weights']]}
    return args


def audit(mode, log, transcript_blocks=(), pool=0):
    """Re-check the invariants from the log; returns a list of violations (empty = clean)."""
    problems, locked = [], False
    for i, e in enumerate(log):
        idx = e.get('args', {}).get('indices', []) if isinstance(e.get('args'), dict) else []
        if mode == 'reasoned' and e['executed'] and e['call'] == 'labels' and any(j < pool for j in idx):
            problems.append(f'call {i}: labels returned for pool images')
        if mode == 'reasoned' and e['executed'] and e['call'] == 'data' and not locked and any(j >= pool for j in idx):
            problems.append(f'call {i}: labeled-set images shown before end_tools')
        if e['executed'] and e['call'] not in ALLOWED[mode]:
            problems.append(f'call {i}: {e["call"]} executed in {mode} mode')
        if mode == 'reasoned' and e['executed'] and e['call'] in PHASE2_CALLS and not locked:
            problems.append(f'call {i}: {e["call"]} executed before end_tools')
        if mode == 'reasoned' and e['executed'] and e['call'] == 'tools' and locked:
            problems.append(f'call {i}: tools executed after end_tools')
        locked |= e['executed'] and e['call'] == 'end_tools'
    for t in transcript_blocks:
        if t in ('image', 'document'):
            problems.append(f'transcript contains a {t} block')
    return problems
