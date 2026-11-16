"""What the agent is told, and the tool definitions it calls. Text only: no image blocks, ever."""
import json

from wsp.episode import ALLOWED, MAX_OUTPUT_CHARS, PREDICT_MAX_IMAGES, PREDICT_MAX_NUMBERS
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
    'reasoned': 'Mode reasoned: the episode has two phases over the same training subset. PHASE 1: no labels. The '
                'sandbox behind tools() holds every subset image (`data` = {"images", "classes"}, no labels), and '
                'data() and predict() work on any of them; labels(), evaluate() and submit() are not available. '
                'PHASE 2 starts when you call end_tools(). IMPORTANT: end_tools() is one-way: tools() is then '
                'permanently disabled for the rest of the episode, and labels() (any subset image), evaluate() '
                '(accuracy on the whole subset), data() and predict() are available as often as you like, and '
                'submit() ends the episode. Do all the code work you need in phase 1, then call end_tools(); print '
                'every number you will need before that, since no code runs afterwards. In phase 2, learn from the '
                'labels in your own context: predict() your network on images, labels() on the same indices, and '
                'reason over the pairs. No gradients in any mode.',
    'full': 'Mode full: available: data(), labels(), tools(), predict(), evaluate(), submit(). In tools() code, '
            '`data` = {"images", "labels", "classes"}. No gradients in any mode (autograd and torch.optim are '
            'disabled).',
}

CALL_DOCS = {
    'data': 'Training images by index, as text: per image 96 lines (R rows 0-31, then G, then B), each line 64 hex '
            'digits = 32 uint8 pixels. At most 16 images per call. Never includes labels.',
    'labels': 'Class labels of training images by index (reasoned mode: phase 2 only).',
    'end_tools': 'End phase 1 for good: tools() is permanently disabled, and labels(), evaluate() and submit() become '
                 'available; data() and predict() stay available. Takes no arguments.',
    'predict': 'Run the network defined by a diff (same format as evaluate(); an empty list means the start weights) '
               'on training images by index, and return one layer\'s output for each image. Never returns labels. '
               'layer: a module name from the architecture, e.g. "fc" (the logits, with the predicted class), '
               '"avgpool" (the 512 features fc reads), "layer4", "layer3.1", "layer2.0.bn1", "bn1", "conv1". Spatial '
               'outputs are averaged over space, one value per channel; BatchNorm outputs are before the ReLU that '
               'follows them. channels: [start, end) to return only those channels, or null for all. At most '
               f'{PREDICT_MAX_IMAGES} images and {PREDICT_MAX_NUMBERS:,} numbers (images x values) per call. Values '
               "have 3 significant digits. The diff's format is checked first, as in evaluate().",
    'tools': 'Run Python in a sandbox (fresh process; files in the working directory persist across calls). '
             'Preloaded: torch, math, toolkit, `data`, `start` (start state_dict). No network, no subprocesses, '
             f'no files outside the working directory. Only the last {MAX_OUTPUT_CHARS:,} characters of stdout+stderr '
             'are returned, so print long outputs in several calls.\n' + TOOLKIT_DOC,
    'evaluate': 'Accuracy, loss, per-class accuracy and confusion of a candidate diff on the training subset. The '
                "diff's format is checked first (tensor names, number counts, finite values, budget); if anything is "
                'wrong, every problem is listed and nothing is scored.',
    'submit': "Submit the final diff. Ends the episode if it is valid. The diff's format is checked first (tensor "
              'names, number counts, finite values, budget); if anything is wrong, every problem is listed and '
              'nothing is submitted.',
}

REASONED_CALL_DOCS = {
    'predict': CALL_DOCS['predict'] + ' Works in both phases, on any subset image.',
    'evaluate': "Accuracy (one number) of a candidate diff on the whole training subset. Phase 2 only. The diff's format is "
                'checked first (tensor names, number counts, finite values, budget); if anything is wrong, every '
                'problem is listed and nothing is scored.',
    'submit': "Submit the final diff. Phase 2 only. Ends the episode if it is valid. The diff's format is checked "
              'first (tensor names, number counts, finite values, budget); if anything is wrong, every problem is '
              'listed and nothing is submitted.',
}


def tool_specs(mode, names):
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
    docs = {**CALL_DOCS, **(REASONED_CALL_DOCS if mode == 'reasoned' else {})}
    return [{'name': n, 'description': docs[n], 'input_schema': schemas[n], 'eager_input_streaming': True}
            for n in ('data', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit') if n in ALLOWED[mode]]


def turns_rules(mode, max_turns, format_retries):
    checked = 'submit()' if mode == 'blind' else 'predict(), evaluate() and submit()'
    return (f'# Turns\nYou have {max_turns} turns. A turn is one reply of yours; one reply can contain several calls, '
            f'which run in order. After each turn, the last result says how many turns are left. If there is no '
            f'valid submit() by the end of turn {max_turns}, the episode ends with no score. {checked} check the '
            f"diff's format first and list every problem; a reply whose calls were all rejected for format alone "
            f'does not count as a turn (at most {format_retries} such replies).')


def user_prompt(task, mode, start, n_examples, max_numbers, max_turns=10, format_retries=3):
    tensors = '\n'.join(f'{n} {s["shape"]} mean={s["mean"]} std={s["std"]}' for n, s in weight_stats(start).items())
    return '\n\n'.join([
        f'# Task: {task.name}\n{task.description}',
        f'# Dataset\n{json.dumps(DATASET_DESCRIPTIONS[task.dataset], indent=1)}',
        f'# Architecture\n{ARCHITECTURE}',
        f'# Writable tensors (name, shape, start-weight stats)\n{tensors}',
        f'# {MODE_RULES[mode]}',
        (f'# Training subset\n{n_examples} images, indices 0..{n_examples - 1} (class-balanced, not in class order).'
         if n_examples else '# Training subset\nnone'),
        f'# Budget\nThe diff may hold at most {max_numbers:,} numbers in total.',
        turns_rules(mode, max_turns, format_retries),
    ])
