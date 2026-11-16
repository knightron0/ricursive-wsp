"""What the agent is told, and the tool definitions it calls. Text only: no image blocks, ever."""
import json

from wsp.episode import ALLOWED, MAX_IMAGES_PER_CALL, MAX_OUTPUT_CHARS, PREDICT_MAX_IMAGES, PREDICT_MAX_NUMBERS
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

REASONED_SYSTEM = """In this episode, phase 2 (after end_tools()) is where you learn from labels, and we expect you
to do that learning in your own context. No code runs in phase 2, but predict() does: give it a diff and image
indices, and it returns your network's representation of each image, either the final logits or the output of any
intermediate layer. Call labels() on the same indices, and reason over the (representation, label)
pairs yourself: which features or outputs go with which class, which of your output rows or clusters stand for which
class, what weights would separate the classes. Then write the resulting numbers in your diff. Plan phase 1 so that
this works: build representations in phase 1 that you can read and reason about in phase 2, and print every number
you will want to reuse before end_tools(). data() shows the images themselves. evaluate() returns only an accuracy,
so it can tell you whether a diff is better, not why."""


def system_prompt(mode):
    return SYSTEM + ('\n\n' + REASONED_SYSTEM if mode == 'reasoned' else '')

MODE_RULES = {
    'blind': 'Mode blind: you have no data. Available: tools(), submit(). Everything must come from your own '
             'knowledge. In tools() code, `data` is None. No gradients in any mode.',
    'full': 'Mode full: available: data(), labels(), tools(), predict(), evaluate(), submit(). In tools() code, '
            '`data` = {"images", "labels", "classes"}. No gradients in any mode (autograd and torch.optim are '
            'disabled).',
}

CHECKED = ("The diff is checked first (tensor names, number counts, finite values, running_var >= 0, the budget); if "
           'anything is wrong, every problem is listed and nothing runs.')

CALL_DOCS = {
    'data': 'Pixels of training-subset images by index, as text. For each image: a line "image i:", then 96 lines '
            '(red rows 0-31, then green rows 0-31, then blue rows 0-31), each 64 hex digits = 32 uint8 pixels left to '
            f'right. At most {MAX_IMAGES_PER_CALL} images per call; each image is about 6,250 characters of text. '
            'Never includes labels.',
    'labels': 'Class names of training-subset images by index, as a JSON object {"index": "class name"}. No limit per '
              'call; every label costs input tokens.',
    'end_tools': 'End phase 1. One-way: tools() is disabled for the rest of the episode and no code of yours runs '
                 'again. labels(), evaluate() and submit() become available; data() and predict() stay available. '
                 'Takes no arguments; calling it again is an error.',
    'predict': "Run the network defined by a diff (same format as submit(); an empty list means the start weights) on "
               "training-subset images by index, and return one layer's output for each image. Never returns labels. "
               'layer: a module name from the architecture, e.g. "fc" (the logits), "avgpool" (the 512 features fc '
               'reads), "layer4", "layer3.1", "layer2.0.bn1", "bn1", "conv1". Spatial outputs are averaged over '
               'space, one value per channel; BatchNorm outputs are before the ReLU that follows them. channels: '
               '[start, end) to return only those channels, or null for all. Output: a header line, then one line per '
               'image, "i [predicted class]: v1 ... v10" for fc (logits in class order) and "i: v1 v2 ..." for other '
               'layers, values with 3 significant digits. At most '
               f'{PREDICT_MAX_IMAGES} images and {PREDICT_MAX_NUMBERS:,} numbers (images x values) per call, e.g. the '
               'fc logits of 500 images, 60 channels of 100 images, or all 512 avgpool features of 11 images. '
               + CHECKED,
    'evaluate': 'Accuracy, loss, per-class accuracy and confusion of a diff on the training subset, as JSON. '
                + CHECKED,
    'submit': 'Submit your final diff: if it passes the check, the episode ends ("accepted: N numbers in K '
              'tensors") and the network is scored once on held-out images. An empty diff is rejected. ' + CHECKED,
}

REASONED_CALL_DOCS = {
    'labels': CALL_DOCS['labels'] + ' Phase 2 only; any subset image.',
    'predict': CALL_DOCS['predict'] + ' Works in both phases, on any subset image.',
    'evaluate': 'Accuracy of a diff on all training-subset images, using their labels, as JSON {"accuracy": x}: one '
                'number, no per-class results, loss or confusion matrix. These are the same images labels() covers, '
                'so it measures fit to them, not held-out accuracy. Phase 2 only. ' + CHECKED,
    'submit': 'Phase 2 only. ' + CALL_DOCS['submit'],
}


def tools_doc(timeout, device):
    return ('Run Python in a sandbox. Each call is a fresh process that runs for at most '
            f'{timeout:g} s on {device}; files you write in the working directory persist across calls (add "." to '
            'sys.path to import your own modules from there). Preloaded: torch, math, toolkit; `start`, the start '
            'state_dict {name: tensor}; `data`: {"images": uint8 tensor [N, 3, 32, 32], "classes": list of class names} '
            'where data["images"][i] is image i of every call (in full mode also "labels"; in blind mode `data` is '
            'None). No network, no subprocesses, no threads, no files outside the working directory, no gradients. '
            f'Returns only the last {MAX_OUTPUT_CHARS:,} characters of stdout+stderr, so print long outputs in several '
            'calls.\n' + TOOLKIT_DOC)


def tool_specs(mode, names, timeout=300, device='cpu'):
    diff = {'type': 'object', 'required': ['weights'], 'additionalProperties': False, 'properties': {'weights': {
        'type': 'array', 'items': {'type': 'object', 'required': ['name', 'values'], 'additionalProperties': False,
                                   'properties': {'name': {'type': 'string', 'enum': list(names)},
                                                  'values': {'type': 'array', 'items': {'type': 'number'}}}}}}}
    indices = {'type': 'object', 'required': ['indices'], 'additionalProperties': False,
               'properties': {'indices': {'type': 'array', 'items': {'type': 'integer'}}}}
    predict = {'type': 'object', 'required': ['weights', 'indices', 'layer', 'channels'], 'additionalProperties': False,
               'properties': {'weights': diff['properties']['weights'], 'indices': indices['properties']['indices'],
                              'layer': {'type': 'string'},
                              'channels': {'type': ['array', 'null'], 'items': {'type': 'integer'}}}}
    schemas = {'data': indices, 'labels': indices, 'evaluate': diff, 'submit': diff, 'predict': predict,
               'tools': {'type': 'object', 'required': ['code'], 'additionalProperties': False,
                         'properties': {'code': {'type': 'string'}}},
               'end_tools': {'type': 'object', 'required': [], 'additionalProperties': False, 'properties': {}}}
    docs = {**CALL_DOCS, 'tools': tools_doc(timeout, device), **(REASONED_CALL_DOCS if mode == 'reasoned' else {})}
    return [{'name': n, 'description': docs[n], 'input_schema': schemas[n], 'eager_input_streaming': True}
            for n in ('data', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit') if n in ALLOWED[mode]]


def turns_rules(max_turns, format_retries):
    return (f'# Turns\nYou have {max_turns} turns. A turn is one reply of yours; it can contain several calls, which '
            'run in order (once submit() is accepted, the rest of that reply is skipped). A reply with no calls still '
            'uses a turn. After each turn, the last result ends with a [harness] note saying how many turns are left. '
            f'If there is no accepted submit() by the end of turn {max_turns}, the episode ends with no score. A call '
            'the harness rejects returns an error message and has no effect (a tools() call whose code raises still '
            'returns its output, and files it wrote before the error stay). A reply whose calls were all rejected for '
            f'diff problems alone does not count as a turn (at most {format_retries} such replies).')


def diff_rules(mode, max_numbers, max_tokens):
    calls = 'submit()' if mode == 'blind' else 'predict(), evaluate() and submit()'
    lines = ['# Diffs and costs',
             '- A diff is a list of {"name", "values"}: each tensor you change, with every one of its numbers, flat in '
             'row-major (C) order of its shape (nested lists are flattened). Tensors you leave out keep their start '
             f'values. A diff may hold at most {max_numbers:,} numbers in total; num_batches_tracked entries are not '
             'writable.',
             f'- {calls} {"takes" if mode == "blind" else "each take"} a whole diff, so every call carries all of its '
             'numbers again. Every number costs output tokens: keep diffs compact (small integers are cheap).',
             f'- {calls} {"checks" if mode == "blind" else "check"} the diff first (tensor names, number counts, finite '
             'values, running_var >= 0, the budget) and list every problem; nothing runs until the diff passes.']
    if max_tokens:
        lines.append(f'- Each reply of yours is limited to {max_tokens:,} output tokens, thinking included. A reply '
                     'that hits the limit ends the episode with no score.')
    return '\n'.join(lines)


def reasoned_rules(n_examples):
    return '\n\n'.join([
        f'# How this episode works (mode reasoned)\nYou have a training subset of {n_examples:,} images, indices '
        f'0..{n_examples - 1} (class-balanced, not in class order). Every call and the sandbox use the same indices: '
        'data["images"][i] in tools() code is image i. The episode has two phases.\n'
        '- PHASE 1, no labels. Available: tools(), data(), predict(). The sandbox behind tools() holds every subset '
        'image and no labels. labels(), evaluate() and submit() are refused.\n'
        '- end_tools() ends phase 1. It is one-way: tools() is disabled for the rest of the episode and no code of '
        'yours runs again.\n'
        f'- PHASE 2, labels. Available: labels() on any subset image, evaluate() (accuracy of a diff on all '
        f'{n_examples:,} images), data(), predict(), submit(). submit() ends the episode.\n'
        'The submitted network is scored once on held-out images that you never see; the start network scores about '
        '10%, chance. No gradients anywhere: autograd, torch.optim and torch.func transforms are disabled.',
        '# What we want you to do\nSubmit the network with the highest held-out accuracy you can, and learn from the '
        'labels in your own context in phase 2:\n'
        '1. Phase 1: with code on the unlabeled images, build a network (a diff) whose representations carry the '
        'information you will need, in a form you can read and reason about in phase 2: for example fc outputs that '
        'score unsupervised clusters, or a small set of channels you understand. Print every number you will want to '
        'reuse, since no code runs after end_tools().\n'
        '2. Call end_tools().\n'
        '3. Phase 2: call predict() with your diff on many images, and labels() on the same indices. Reason over the '
        '(representation, label) pairs in your own context: which features or outputs go with which class, which '
        'output rows or clusters stand for which class, what weights would separate the classes. Write the resulting '
        "numbers into your diff. evaluate() gives a diff's accuracy (one number) if you want to compare candidates.\n"
        '4. submit() your final diff.\n'
        'Plan your turns so that phase 2 has room: we suggest calling end_tools() by about turn 5.',
    ])


def user_prompt(task, mode, start, n_examples, max_numbers, max_turns=10, format_retries=3, max_tokens=None):
    tensors = '\n'.join(f'{n} {s["shape"]} mean={s["mean"]} std={s["std"]}' for n, s in weight_stats(start).items())
    if mode == 'reasoned':
        rules = [reasoned_rules(n_examples)]
    else:
        rules = [f'# {MODE_RULES[mode]}',
                 (f'# Training subset\n{n_examples} images, indices 0..{n_examples - 1} (class-balanced, not in class '
                  'order).' if n_examples else '# Training subset\nnone')]
    return '\n\n'.join([
        f'# Task: {task.name}\n{task.description}',
        f'# Dataset\n{json.dumps(DATASET_DESCRIPTIONS[task.dataset], indent=1)}',
        f'# Architecture\n{ARCHITECTURE}',
        f'# Writable tensors (name, shape, start-weight stats)\n{tensors}',
        *rules,
        diff_rules(mode, max_numbers, max_tokens),
        turns_rules(max_turns, format_retries),
    ])
