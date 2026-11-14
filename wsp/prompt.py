"""What the agent is told, and the tool definitions it calls. Text only: no image blocks, ever."""
import json

from wsp.evaluation import weight_stats
from wsp.model import ARCHITECTURE
from wsp.tasks import DATASET_DESCRIPTIONS
from wsp.toolkit import DOC as TOOLKIT_DOC

SYSTEM = """You are doing weight-space programming: you write the weights of a ResNet-18 directly,
as literal numbers, instead of training it. Your final answer is one submit() call with a diff:
the tensors you want to change and every number in them. Tensors you leave out keep their start
values. The submitted network is then scored on held-out images you never see.

Write each tensor's values flat in row-major (C) order of its shape, e.g. a conv weight
[out, in, kh, kw] goes out-channel by out-channel. Every number costs output tokens."""

MODE_RULES = {
    'blind': 'Mode blind: you have no data. Available: tools(), submit(). Everything must come from your own '
             'knowledge. In tools() code, `data` is None. No gradients in any mode.',
    'reasoned': 'Mode reasoned: the training subset has two disjoint parts. The POOL: unlabeled images; the sandbox '
                'behind tools() holds them (`data` = {"images", "classes"}) and data() shows them; their labels are '
                'never available. The LABELED SET: labels() gives their labels, evaluate() scores a diff on them, '
                'and data() shows them in phase 2. The episode has two phases. PHASE 1: tools() and data() on the '
                'pool; labels(), evaluate() and submit() are not available. PHASE 2 starts when you call end_tools(). '
                'IMPORTANT: end_tools() is one-way: tools() is then permanently disabled for the rest of the '
                'episode, and labels(), evaluate() and data() (pool and labeled set) are available as often as you '
                'like, and submit() ends the episode. Do all the code work you need in phase 1, then call '
                'end_tools(). No gradients in any mode.',
    'full': 'Mode full: available: data(), labels(), tools(), evaluate(), submit(). In tools() code, `data` = '
            '{"images", "labels", "classes"}. No gradients in any mode (autograd and torch.optim are disabled).',
}

CALL_DOCS = {
    'data': 'Training images by index, as text: per image 96 lines (R rows 0-31, then G, then B), each line 64 hex '
            'digits = 32 uint8 pixels. At most 16 images per call. Never includes labels.',
    'labels': 'Class labels of training images by index (reasoned mode: labeled set only, phase 2 only).',
    'end_tools': 'End phase 1 for good: tools() is permanently disabled, and labels(), evaluate(), submit() and '
                 'data() on the labeled set become available. Takes no arguments.',
    'tools': 'Run Python in a sandbox (fresh process; files in the working directory persist across calls). '
             'Preloaded: torch, math, toolkit, `data`, `start` (start state_dict). No network, no subprocesses, '
             'no files outside the working directory. Returns the end of stdout/stderr.\n' + TOOLKIT_DOC,
    'evaluate': 'Accuracy, loss, per-class accuracy and confusion of a candidate diff on the training subset '
                '(reasoned mode: the labeled set, phase 2 only).',
    'submit': 'Submit the final diff (reasoned mode: phase 2 only). Ends the episode if it is valid; otherwise you '
              'are told why.',
}


def tool_specs(mode, names):
    diff = {'type': 'object', 'required': ['weights'], 'additionalProperties': False, 'properties': {'weights': {
        'type': 'array', 'items': {'type': 'object', 'required': ['name', 'values'], 'additionalProperties': False,
                                   'properties': {'name': {'type': 'string', 'enum': list(names)},
                                                  'values': {'type': 'array', 'items': {'type': 'number'}}}}}}}
    indices = {'type': 'object', 'required': ['indices'], 'additionalProperties': False,
               'properties': {'indices': {'type': 'array', 'items': {'type': 'integer'}}}}
    schemas = {'data': indices, 'labels': indices, 'evaluate': diff, 'submit': diff,
               'tools': {'type': 'object', 'required': ['code'], 'additionalProperties': False,
                         'properties': {'code': {'type': 'string'}}},
               'end_tools': {'type': 'object', 'required': [], 'additionalProperties': False, 'properties': {}}}
    from wsp.episode import ALLOWED
    return [{'name': n, 'description': CALL_DOCS[n], 'input_schema': schemas[n], 'eager_input_streaming': True}
            for n in ('data', 'labels', 'tools', 'end_tools', 'evaluate', 'submit') if n in ALLOWED[mode]]


def user_prompt(task, mode, start, n_examples, max_numbers, pool=0):
    tensors = '\n'.join(f'{n} {s["shape"]} mean={s["mean"]} std={s["std"]}' for n, s in weight_stats(start).items())
    return '\n\n'.join([
        f'# Task: {task.name}\n{task.description}',
        f'# Dataset\n{json.dumps(DATASET_DESCRIPTIONS[task.dataset], indent=1)}',
        f'# Architecture\n{ARCHITECTURE}',
        f'# Writable tensors (name, shape, start-weight stats)\n{tensors}',
        f'# {MODE_RULES[mode]}',
        (f'# Training subset\nPool: indices 0..{pool - 1} ({pool} unlabeled images). Labeled set: indices '
         f'{pool}..{n_examples - 1} ({n_examples - pool} images). Each part is class-balanced and not in class order.'
         if pool else f'# Training subset\n{n_examples} images, indices 0..{n_examples - 1} (class-balanced, not in '
                       f'class order).') if n_examples else '# Training subset\nnone',
        f'# Budget\nThe diff may hold at most {max_numbers:,} numbers in total.',
    ])
