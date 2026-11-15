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
    'reasoned': 'Mode reasoned: available: data(), labels(), tools(), evaluate(), submit(). The training subset '
                'has two disjoint parts. The POOL: unlabeled images; the sandbox behind tools() holds them '
                '(`data` = {"images", "classes"}) and data() shows them; their labels are never available. The '
                'LABELED SET: labels() gives their labels, and data() shows their images only after your first labels() '
                'call. evaluate() returns a diff\'s accuracy on the labeled set (one number) and may be used at any '
                'time. IMPORTANT: the first labels() call permanently disables tools() for the rest of the episode. '
                'Do all the code work you need first, then stop using tools() and call labels(). No gradients in '
                'any mode.',
    'full': 'Mode full: available: data(), labels(), tools(), evaluate(), submit(). In tools() code, `data` = '
            '{"images", "labels", "classes"}. No gradients in any mode (autograd and torch.optim are disabled).',
}

CALL_DOCS = {
    'data': 'Training images by index, as text: per image 96 lines (R rows 0-31, then G, then B), each line 64 hex '
            'digits = 32 uint8 pixels. At most 16 images per call. Never includes labels.',
    'labels': 'Class labels of training images by index.',
    'tools': 'Run Python in a sandbox (fresh process; files in the working directory persist across calls). '
             'Preloaded: torch, math, toolkit, `data`, `start` (start state_dict). No network, no subprocesses, '
             'no files outside the working directory. Returns the end of stdout/stderr.\n' + TOOLKIT_DOC,
    'evaluate': 'Accuracy of a candidate diff on the training subset. In reasoned mode: accuracy on the labeled set '
                'only, one number. Otherwise also loss, per-class accuracy and the top confusions.',
    'submit': 'Submit the final diff. Ends the episode if it is valid; otherwise you are told why.',
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
                         'properties': {'code': {'type': 'string'}}}}
    from wsp.episode import ALLOWED
    return [{'name': n, 'description': CALL_DOCS[n], 'input_schema': schemas[n], 'eager_input_streaming': True}
            for n in ('data', 'labels', 'tools', 'evaluate', 'submit') if n in ALLOWED[mode]]


def user_prompt(task, mode, start, n_examples, max_numbers, pool=0, max_turns=100):
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
        f'# Budget\nThe diff may hold at most {max_numbers:,} numbers in total. You have {max_turns} turns (replies); '
        f'each call result shows the turn count. If you have not submitted by the last turn, you get nothing.',
    ])
