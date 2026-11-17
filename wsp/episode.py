"""The calls an agent can make, the reasoned-mode phases, and the audit.

    data(indices)                  training images as hex text (never labels)
    labels(indices)                class labels
    tools(code)                    Python in the sandbox
    end_tools()                    reasoned only: leave phase 1 for good
    predict(weights, indices, ..)  one layer's output of the network a diff defines, per image
    evaluate(weights)              training-subset accuracy of a diff (aggregate numbers only)
    submit(weights)                final diff; ends the episode if valid

reasoned mode runs in two phases over one training subset (no split):
  1. phase 1: tools(), data() and predict(); the sandbox holds every subset image and no labels;
     labels(), evaluate() and submit() are refused;
  2. end_tools() switches to phase 2 for good: tools() never executes again, and labels() (any
     subset image), evaluate() (accuracy only, on the whole subset) and submit() become available;
     data() and predict() stay available.
predict(), evaluate() and submit() check a diff's format first; a format rejection is logged with
status "format". So nothing derived from a label ever reaches code: labels appear only after code
is gone, and the agent combines them with images and representations in its own context.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from wsp.evaluation import evaluate, layer_outputs
from wsp.tasks import entry_label

ROOT = Path(__file__).resolve().parent.parent
MODES = ('blind', 'reasoned', 'full')
ALLOWED = {'blind': {'tools', 'submit'},
           'reasoned': {'data', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit'},
           'full': {'data', 'labels', 'tools', 'predict', 'evaluate', 'submit'}}
PHASE2_CALLS = {'labels', 'evaluate', 'submit'}
MAX_IMAGES_PER_CALL = 16
MAX_OUTPUT_CHARS = 10000
PREDICT_MAX_NUMBERS = 6000
PREDICT_MAX_IMAGES = 500


class FormatError(ValueError):
    pass


def pixels_text(image):
    """uint8 [3,32,32] -> 96 lines (R rows 0-31, G rows, B rows) of 64 hex digits."""
    return '\n'.join(image[c, r].numpy().tobytes().hex() for c in range(3) for r in range(32))


SANDBOX_ENV = ('PATH', 'LANG', 'LC_ALL', 'LC_CTYPE')
SANDBOX_ENV_PREFIXES = ('OMP_', 'MKL_', 'PYTORCH_', 'CUDA_', 'NVIDIA_', 'LD_LIBRARY_PATH', 'DYLD_')


def sandbox_env(workspace):
    """A minimal environment: nothing that names this machine's sessions, sockets, keys or other runs."""
    env = {k: v for k, v in os.environ.items() if k in SANDBOX_ENV or k.startswith(SANDBOX_ENV_PREFIXES)}
    work = str(Path(workspace).resolve())
    env.update(PYTHONPATH=str(ROOT), HOME=work, TMPDIR=work, PYTHONDONTWRITEBYTECODE='1')
    return env


class Episode:
    def __init__(self, mode, start, images, labels, classes, workspace, apply_diff, device='cpu',
                 timeout=300, log=None):
        assert mode in MODES
        self.mode, self.start, self.images, self.labels, self.classes = mode, start, images, labels, classes
        self.workspace, self.apply_diff, self.device, self.timeout = Path(workspace), apply_diff, device, timeout
        self.locked, self.submission, self.log = False, None, log if log is not None else []
        self.workspace.mkdir(parents=True, exist_ok=True)
        torch.save(start, self.workspace / 'start.pt')
        if mode == 'reasoned':
            torch.save({'images': images, 'classes': classes}, self.workspace / 'data.pt')
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
        return '\n\n'.join(f'image {i}:\n{pixels_text(self.images[i])}' for i in idx)

    def _predict(self, args):
        idx = self._indices(args)
        if not idx or len(idx) > PREDICT_MAX_IMAGES:
            raise ValueError(f'give 1 to {PREDICT_MAX_IMAGES} image indices')
        weights, _ = self._diff(args)
        layer = args.get('layer') or 'fc'
        out = layer_outputs(weights, self.images[idx], layer, len(self.classes), self.device)
        channels = args.get('channels')
        if channels is not None:
            if not (isinstance(channels, list) and len(channels) == 2 and all(isinstance(c, int) for c in channels)
                    and 0 <= channels[0] < channels[1] <= out.shape[1]):
                raise ValueError(f'channels must be [start, end) with 0 <= start < end <= {out.shape[1]}, or null')
            out = out[:, channels[0]:channels[1]]
        first = channels[0] if channels is not None else 0
        if out.numel() > PREDICT_MAX_NUMBERS:
            raise ValueError(f'{len(idx)} images x {out.shape[1]} values = {out.numel()} numbers; at most '
                             f'{PREDICT_MAX_NUMBERS} per call: ask for fewer images or a channels range')
        if layer == 'fc':
            head = f'layer fc: {out.shape[1]} logits per image, in class order {self.classes}; predicted class first'
            rows = [f'{i} [{self.classes[int(o.argmax())]}]: ' + ' '.join(f'{v:.3g}' for v in o.tolist())
                    for i, o in zip(idx, out)]
        else:
            head = (f'layer {layer}: channels {first}..{first + out.shape[1] - 1}, one value per channel per image '
                    f'(spatial outputs averaged over space)')
            rows = [f'{i}: ' + ' '.join(f'{v:.3g}' for v in o.tolist()) for i, o in zip(idx, out)]
        return '\n'.join([head] + rows)

    def _end_tools(self, args):
        if self.locked:
            raise ValueError('already in phase 2')
        self.locked = True
        return ('Phase 2 has started. tools() is permanently disabled. Available now: labels() on any image '
                f'(indices 0..{len(self.labels) - 1}), evaluate() (accuracy on all of them), data(), predict() and '
                'submit(). Learn from the labels in your own context: predict() your network on images, call '
                'labels() on the same indices, and reason over the (representation, label) pairs to decide the '
                'numbers you write.')

    def _labels(self, args):
        idx = self._indices(args)
        return json.dumps({i: self.classes[self.labels[i]] for i in idx})

    def _tools(self, args):
        (self.workspace / '_call.py').write_text(args['code'])
        env = sandbox_env(self.workspace)
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
        m = evaluate(weights, (self.images, self.labels), self.classes, self.device)
        if self.mode == 'reasoned':
            return json.dumps({'accuracy': m['accuracy']})
        m.pop('activations', None)
        return json.dumps(m)

    def _submit(self, args):
        if not args.get('weights'):
            raise FormatError('the diff is empty: it would leave the start network unchanged. Write the tensors or '
                              'blocks you want to set')
        weights, numbers = self._diff(args)
        self.submission = weights
        return f'accepted: {numbers} numbers in {len({e["name"] for e in args["weights"]})} tensors'


def clip(text, limit=2000):
    if len(text) <= limit:
        return text
    return f'{text[:limit // 2]}\n[... {len(text) - limit} chars omitted ...]\n{text[-(limit // 2):]}'


def _summary(args):
    if 'weights' in args and isinstance(args['weights'], list):
        return {**{k: v for k, v in args.items() if k != 'weights'},
                'weights': [entry_label(e) for e in args['weights']]}
    return args


def audit(mode, log, transcript_blocks=()):
    """Re-check the invariants from the log; returns a list of violations (empty = clean)."""
    problems, locked = [], False
    for i, e in enumerate(log):
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
        elif isinstance(t, str) and t.startswith('forbidden:'):
            problems.append(f'transcript contains a {t.removeprefix("forbidden:")} item')
    return problems
