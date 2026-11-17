"""The calls an agent can make, the reasoned-mode phases, and the audit.

    data(indices)                  training images as hex text (never labels)
    view(indices, grid, scale)     --visual only: training images as PNG image blocks (never labels)
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
import base64
import io
import json
import math
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
VISUAL_MODES = ('reasoned', 'full')
MAX_IMAGES_PER_CALL = 16
VIEW_MAX_IMAGES = 16
VIEW_MAX_GRID_IMAGES = 64
VIEW_DEFAULT_SCALE, VIEW_DEFAULT_GRID_SCALE, VIEW_MAX_SCALE = 8, 4, 16
MAX_OUTPUT_CHARS = 10000
PREDICT_MAX_NUMBERS = 6000
PREDICT_MAX_IMAGES = 500


def allowed(mode, visual=False):
    """The calls a mode permits; --visual adds view() (reasoned and full only)."""
    return ALLOWED[mode] | ({'view'} if visual and mode in VISUAL_MODES else set())


class FormatError(ValueError):
    pass


class Visual(str):
    """A call result that carries images. The str value is the text (images shown as placeholders, which is
    what logs and transcripts keep); .parts is the ordered list of str and PNG bytes the agent is sent."""

    def __new__(cls, parts):
        text = '\n'.join(p if isinstance(p, str) else f'[image: {len(p)}-byte PNG]' for p in parts)
        obj = super().__new__(cls, text)
        obj.parts = list(parts)
        return obj


def add_note(text, note):
    """Append a harness note to a result without losing its images."""
    return Visual(text.parts + [note]) if isinstance(text, Visual) else f'{text}\n\n{note}'


def mcp_content(text):
    """MCP tool-result content for a result: one text block, or text and image blocks in order (view())."""
    if not hasattr(text, 'parts'):
        return [{'type': 'text', 'text': text}]
    return [{'type': 'text', 'text': p} if isinstance(p, str) else
            {'type': 'image', 'data': base64.b64encode(p).decode(), 'mimeType': 'image/png'} for p in text.parts]


def image_png(image, scale):
    """uint8 [3,32,32] -> PNG bytes of the image upscaled `scale` times (bicubic)."""
    from PIL import Image
    im = Image.fromarray(image.permute(1, 2, 0).contiguous().numpy()).resize((32 * scale, 32 * scale), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()


def grid_png(images, indices, scale):
    """One PNG tiling the images row by row (4 per row, or 8 for more than 16), each tile captioned with its index."""
    from PIL import Image, ImageDraw
    cols = min(len(indices), 4 if len(indices) <= 16 else 8)
    rows = math.ceil(len(indices) / cols)
    tile, gap = 32 * scale, 4
    sheet = Image.new('RGB', (cols * (tile + gap) + gap, rows * (tile + gap) + gap), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(indices):
        x, y = gap + (k % cols) * (tile + gap), gap + (k // cols) * (tile + gap)
        sheet.paste(Image.fromarray(images[i].permute(1, 2, 0).contiguous().numpy()).resize((tile, tile), Image.BICUBIC),
                    (x, y))
        label = str(i)
        draw.rectangle([x, y, x + 7 * len(label) + 4, y + 12], fill=(0, 0, 0))
        draw.text((x + 2, y + 1), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, format='PNG')
    return buf.getvalue()


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
                 timeout=300, log=None, visual=False):
        assert mode in MODES and not (visual and mode not in VISUAL_MODES)
        self.visual = visual
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
        if name not in allowed(self.mode, self.visual):
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

    def _view(self, args):
        idx = self._indices(args)
        grid, scale = bool(args.get('grid')), args.get('scale')
        limit = VIEW_MAX_GRID_IMAGES if grid else VIEW_MAX_IMAGES
        if not idx or len(idx) > limit:
            raise ValueError(f'give 1 to {limit} image indices')
        if scale is None:
            scale = VIEW_DEFAULT_GRID_SCALE if grid else VIEW_DEFAULT_SCALE
        if not isinstance(scale, int) or not 1 <= scale <= VIEW_MAX_SCALE:
            raise ValueError(f'scale must be an integer in [1, {VIEW_MAX_SCALE}], or null')
        if grid:
            return Visual([f'{len(idx)} images in one grid, row by row; the number in each tile\'s corner is its '
                           f'image index: {idx}', grid_png(self.images, idx, scale)])
        parts = []
        for i in idx:
            parts += [f'image {i}:', image_png(self.images[i], scale)]
        return Visual(parts)

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
                f'(indices 0..{len(self.labels) - 1}), evaluate() (accuracy on all of them), data(), ' +
                ('view(), ' if self.visual else '') + 'predict() and '
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


def audit(mode, log, transcript_blocks=(), visual=False):
    """Re-check the invariants from the log; returns a list of violations (empty = clean).
    Image blocks are a violation unless the run is --visual (then they may come only from view())."""
    problems, locked = [], False
    for i, e in enumerate(log):
        if e['executed'] and e['call'] not in allowed(mode, visual):
            problems.append(f'call {i}: {e["call"]} executed in {mode} mode')
        if mode == 'reasoned' and e['executed'] and e['call'] in PHASE2_CALLS and not locked:
            problems.append(f'call {i}: {e["call"]} executed before end_tools')
        if mode == 'reasoned' and e['executed'] and e['call'] == 'tools' and locked:
            problems.append(f'call {i}: tools executed after end_tools')
        locked |= e['executed'] and e['call'] == 'end_tools'
    for t in transcript_blocks:
        if t == 'document' or (t == 'image' and not visual):
            problems.append(f'transcript contains a {t} block')
        elif isinstance(t, str) and t.startswith('forbidden:'):
            problems.append(f'transcript contains a {t.removeprefix("forbidden:")} item')
    if visual and 'image' in transcript_blocks and not any(e['call'] == 'view' and e['executed'] for e in log):
        problems.append('transcript contains image blocks but no view() call executed')
    return problems
