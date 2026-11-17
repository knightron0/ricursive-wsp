"""What the agent is told, and the tool definitions it calls. Text only: no image blocks, unless the run is
--visual, in which case images arrive only as view() results."""
import json

from wsp.episode import (ALLOWED, MAX_IMAGES_PER_CALL, MAX_OUTPUT_CHARS, PREDICT_MAX_IMAGES, PREDICT_MAX_NUMBERS,
                         VIEW_DEFAULT_GRID_SCALE, VIEW_DEFAULT_SCALE, VIEW_MAX_GRID_IMAGES, VIEW_MAX_IMAGES,
                         VIEW_MAX_SCALE, allowed)
from wsp.evaluation import weight_stats
from wsp.model import ARCHITECTURE
from wsp.tasks import DATASET_DESCRIPTIONS

SYSTEM = """You are doing weight-space programming: you write the weights of a ResNet-18 directly,
as literal numbers, instead of training it. Your final answer is one submit() call with a diff:
the tensors, or blocks of tensors, you want to change and the numbers to put there, or a constant
to fill them with. Everything you leave out keeps its start values. The submitted network is then
scored on held-out images you never see.

Write values flat in row-major (C) order of the tensor's or block's shape, e.g. a conv weight
[out, in, kh, kw] goes out-channel by out-channel. Every number costs output tokens.

No training of any kind: do not use backpropagation or gradient-based updates of any weights, in
any form. That includes gradients you derive and compute by hand and optimizer steps you write
yourself (SGD, Adam and the like), not only autograd and torch.optim, which the sandbox disables.
An episode that trains weights this way breaks the task, whatever it scores."""

REASONED_SYSTEM = """In this episode, phase 2 (after end_tools()) is where you learn from labels, and we expect you
to do that learning in your own context. No code runs in phase 2, but predict() does: give it a diff and image
indices, and it returns your network's representation of each image, either the final logits or the output of any
intermediate layer. Call labels() on the same indices, and reason over the (representation, label)
pairs yourself: which features or outputs go with which class, which of your output rows stand for which
class, what weights would separate the classes. Then write the resulting numbers in your diff. Plan phase 1 so that
this works: build representations in phase 1 that you can read and reason about in phase 2, and print every number
you will want to reuse before end_tools(). data() shows the images themselves. evaluate() returns only an accuracy,
so it can tell you whether a diff is better, not why."""


VISUAL_SYSTEM = """You can also look at the images. view() returns training images as real images, not text, and you
can see several at once. Use your own visual judgment: what the images in a class have in common (colours, textures,
shapes, backgrounds, the layout of an object), what separates classes that the network confuses, which images are
atypical. Let what you see decide which weights you write: which filters, channels or fc rows to build, which class
they should favour, and what to fix. Look at many images, in groups, rather than a few: a pattern you can see across
a dozen images is worth more than one example. Images cost input tokens, so ask for what you need."""


def system_prompt(mode, visual=False):
    return (SYSTEM + ('\n\n' + REASONED_SYSTEM if mode == 'reasoned' else '')
            + ('\n\n' + VISUAL_SYSTEM if visual else ''))

MODE_RULES = {
    'blind': 'Mode blind: you have no data. Available: tools(), submit(). Everything must come from your own '
             'knowledge. In tools() code, `data` is None. No gradients in any mode.',
}

CHECKED = ('The diff is checked first (tensor names, block ranges, number counts, finite values, running_var >= 0, the '
           'budget); if anything is wrong, every problem is listed and nothing runs.')

CALL_DOCS = {
    'data': 'Pixels of training-subset images by index, as text. For each image: a line "image i:", then 96 lines '
            '(red rows 0-31, then green rows 0-31, then blue rows 0-31), each 64 hex digits = 32 uint8 pixels left to '
            f'right. At most {MAX_IMAGES_PER_CALL} images per call; each image is about 6,250 characters of text. '
            'Never includes labels.',
    'view': 'Look at training-subset images by index: they are returned as images (PNG, 32x32 pixels upscaled '
            f'with bicubic interpolation), several per call, each preceded by a line "image i:". scale: the integer '
            f'upscaling factor, 1 to {VIEW_MAX_SCALE} (default {VIEW_DEFAULT_SCALE}, i.e. {32 * VIEW_DEFAULT_SCALE}x'
            f'{32 * VIEW_DEFAULT_SCALE} pixels; {VIEW_DEFAULT_GRID_SCALE} with grid). At most {VIEW_MAX_IMAGES} images '
            f'per call. grid: true returns all the images as ONE picture, tiled row by row (4 per row, 8 if more than '
            f'16), each tile marked with its index in the corner; up to {VIEW_MAX_GRID_IMAGES} images per call; this '
            'is the cheapest way to compare many images side by side. Never includes labels; use labels() on the same '
            'indices to learn the classes.',
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
    'view': CALL_DOCS['view'][:-1] + ' (phase 2). Works in both phases, on any subset image.',
    'evaluate': 'Accuracy of a diff on all training-subset images, using their labels, as JSON {"accuracy": x}: one '
                'number, no per-class results, loss or confusion matrix. These are the same images labels() covers, '
                'so it measures fit to them, not held-out accuracy. Phase 2 only. ' + CHECKED,
    'submit': 'Phase 2 only. ' + CALL_DOCS['submit'],
}


DATA_DOCS = {
    'blind': '`data` is None (there is no training data)',
    'reasoned': '`data`: {"images": uint8 tensor [N, 3, 32, 32], "classes": list of class names}, where '
                'data["images"][i] is image i of every call; no labels',
    'full': '`data`: {"images": uint8 tensor [N, 3, 32, 32], "labels": int64 tensor [N] of class indices into '
            '"classes", "classes": list of class names}, where data["images"][i] is image i of every call and '
            'data["labels"][i] is its class',
}


def tools_doc(timeout, device, mode='reasoned'):
    return ('Run Python in a sandbox. Each call is a fresh process that runs for at most '
            f'{timeout:g} s on {device}; files you write in the working directory persist across calls (add "." to '
            'sys.path to import your own modules from there). Preloaded: torch, math; `start`, the start '
            f'state_dict {{name: tensor}}; {DATA_DOCS[mode]}. '
            'No network, no subprocesses, no threads, no files outside the working directory, no gradients. '
            f'Returns only the last {MAX_OUTPUT_CHARS:,} characters of stdout+stderr, so print long outputs in several '
            'calls.')


def tool_specs(mode, names, timeout=300, device='cpu', visual=False):
    entry = {'type': 'object', 'required': ['name'], 'additionalProperties': False,
             'properties': {'name': {'type': 'string', 'enum': list(names)},
                            'index': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}}},
                            'values': {'type': 'array', 'items': {'type': 'number'}},
                            'fill': {'type': 'number'}}}
    diff = {'type': 'object', 'required': ['weights'], 'additionalProperties': False,
            'properties': {'weights': {'type': 'array', 'items': entry}}}
    indices = {'type': 'object', 'required': ['indices'], 'additionalProperties': False,
               'properties': {'indices': {'type': 'array', 'items': {'type': 'integer'}}}}
    predict = {'type': 'object', 'required': ['weights', 'indices', 'layer', 'channels'], 'additionalProperties': False,
               'properties': {'weights': diff['properties']['weights'], 'indices': indices['properties']['indices'],
                              'layer': {'type': 'string'},
                              'channels': {'type': ['array', 'null'], 'items': {'type': 'integer'}}}}
    view = {'type': 'object', 'required': ['indices', 'grid', 'scale'], 'additionalProperties': False,
            'properties': {'indices': indices['properties']['indices'], 'grid': {'type': ['boolean', 'null']},
                           'scale': {'type': ['integer', 'null']}}}
    schemas = {'data': indices, 'view': view, 'labels': indices, 'evaluate': diff, 'submit': diff, 'predict': predict,
               'tools': {'type': 'object', 'required': ['code'], 'additionalProperties': False,
                         'properties': {'code': {'type': 'string'}}},
               'end_tools': {'type': 'object', 'required': [], 'additionalProperties': False, 'properties': {}}}
    docs = {**CALL_DOCS, 'tools': tools_doc(timeout, device, mode), **(REASONED_CALL_DOCS if mode == 'reasoned' else {})}
    return [{'name': n, 'description': docs[n], 'input_schema': schemas[n], 'eager_input_streaming': True}
            for n in ('data', 'view', 'labels', 'tools', 'end_tools', 'predict', 'evaluate', 'submit')
            if n in allowed(mode, visual)]


def turns_rules(max_turns, format_retries):
    return (f'# Turns\nYou have {max_turns} turns. A turn is one reply of yours; it can contain several calls, which '
            'run in order (once submit() is accepted, the rest of that reply is skipped). A reply with no calls still '
            'uses a turn. After each turn, the last result ends with a [harness] note saying how many turns are left. '
            f'If there is no accepted submit() by the end of turn {max_turns}, the episode ends with no score. A call '
            'the harness rejects returns an error message and has no effect (a tools() call whose code raises still '
            'returns its output, and files it wrote before the error stay). A reply whose calls were all rejected for '
            f'diff problems alone does not count as a turn (at most {format_retries} such replies).')


def budget_rules(soft_budget):
    return (f'# Budget\nNothing enforces a limit on replies or tokens, but please keep the whole episode within about '
            f'{soft_budget:,} output tokens, thinking included, and finish with an accepted submit(): without one the '
            'episode has no score. A reply can contain several calls, which run in order; once submit() is accepted '
            'the episode is over. A call the harness rejects returns an error message and has no effect (a tools() '
            'call whose code raises still returns its output, and files it wrote before the error stay).')


def sdk_turn_rules(turn_limit, soft_budget):
    return (f'# Turns and budget\nYou have {turn_limit} turns. A turn is one reply of yours (one model response); it '
            'can contain several calls, which run in order, and once submit() is accepted the episode is over. Every '
            'reply uses a turn, including a reply with no calls, a reply whose calls were all rejected, and a reply '
            'cut off at the per-reply output-token limit (its cut-off part is discarded). Every call result ends with '
            'a [harness] note giving the current turn and how many are left. If there is no accepted submit() by the '
            f'end of turn {turn_limit}, the episode ends with no score. A call the harness rejects returns an error '
            'message and has no effect (a tools() call whose code raises still returns its output, and files it '
            'wrote before the error stay). Nothing enforces a token limit, but please keep the whole episode within '
            f'about {soft_budget:,} output tokens, thinking included.')


def diff_rules(mode, max_numbers, max_tokens, hard_cap=True):
    calls = 'submit()' if mode == 'blind' else 'predict(), evaluate() and submit()'
    lines = ['# Diffs and costs',
             '- A diff is a list of entries, applied in order; where two entries write the same element, the later one '
             'wins. Each entry names one writable tensor and sets all of it or one block of it:\n'
             '  - {"name", "values"}: every number of the tensor, flat in row-major (C) order of its shape (nested '
             'lists are flattened).\n'
             '  - {"name", "index", "values"}: a block. index is a list of [start, stop] integer pairs (stop '
             'excluded), one per leading axis; axes after the last pair are taken whole. values are the block\'s '
             'numbers, row-major over the block. E.g. {"name": "fc.weight", "index": [[0, 10], [37, 38]], "values": '
             '[10 numbers]} sets column 37 of fc.weight, and {"name": "conv1.weight", "index": [[5, 6]], "values": '
             '[27 numbers]} sets filter 5 of conv1.\n'
             '  - {"name", "fill": c} or {"name", "index", "fill": c}: every element of the tensor or block set to '
             'the number c.\n'
             '  Elements no entry writes keep their start values. num_batches_tracked entries are not writable.',
             f'- A diff may hold at most {max_numbers:,} numbers in total, counting every number in "values" and 1 '
             'for each "fill".',
             f'- {calls} {"takes" if mode == "blind" else "each take"} a whole diff, so every call carries all of its '
             'numbers again. Every number costs output tokens: keep diffs compact (small integers are cheap; a fill '
             'is one number however large its block).',
             f'- {calls} {"checks" if mode == "blind" else "check"} the diff first (tensor names, block ranges, number '
             'counts, finite values, running_var >= 0, the budget) and list every problem, entry by entry; nothing '
             'runs until the diff passes.']
    if max_tokens and hard_cap:
        lines.append(f'- Each reply of yours is limited to {max_tokens:,} output tokens, thinking included. A reply '
                     'that hits the limit ends the episode with no score.')
    elif max_tokens:
        lines.append(f'- Each reply of yours is limited to {max_tokens:,} output tokens, thinking included; a reply that '
                     'reaches it is cut off, so keep every reply well under it.')
    return '\n'.join(lines)


VISUAL_RULES = """# Looking at the images (view)
view() shows you images as images, several at a time (grid=true puts up to 64 in one picture, each tile marked with
its index). It works in both phases and never shows a label. Use it as a second instrument next to code:
- Phase 1 (no labels): look at batches of images to see what varies in this data: colours, backgrounds, textures,
  shapes, object size and position. Let that decide what your code measures and what features you build into the
  network. Look at the images the network groups together (same cluster, same fc argmax) and judge by eye whether
  they belong together.
- Phase 2 (labels): look at images together with their labels() and with the network's predict() outputs for them.
  Look at what the images of one class share, and at the images the network gets wrong, and decide by what you see
  which weights to write or change. You may look at as many images as you have turns and budget for.
Your visual judgment is the point of this variant: base your weight decisions on what you see, and say what you saw."""


def reasoned_rules(n_examples, soft_budget=None, start_score=None, turn_limit=None, max_turns=10, visual=False):
    if soft_budget and turn_limit:
        pacing = (f'Leave room for phase 2: we suggest calling end_tools() by about turn {turn_limit // 2}, and before '
                  'you have used about half of the token budget.')
    elif soft_budget:
        pacing = 'Leave room for phase 2: we suggest calling end_tools() before you have used about half of the budget.'
    else:
        pacing = f'Plan your turns so that phase 2 has room: we suggest calling end_tools() by about turn {max_turns // 2}.'
    return '\n\n'.join([
        f'# How this episode works (mode reasoned)\nYou have a training subset of {n_examples:,} images, indices '
        f'0..{n_examples - 1} (class-balanced, not in class order). Every call and the sandbox use the same indices: '
        'data["images"][i] in tools() code is image i. The episode has two phases.\n'
        '- PHASE 1, no labels. Available: tools(), data(), '
        + ('view(), ' if visual else '') + 'predict(). The sandbox behind tools() holds every subset '
        'image and no labels. labels(), evaluate() and submit() are refused.\n'
        '- end_tools() ends phase 1. It is one-way: tools() is disabled for the rest of the episode and no code of '
        'yours runs again.\n'
        f'- PHASE 2, labels. Available: labels() on any subset image, evaluate() (accuracy of a diff on all '
        f'{n_examples:,} images), data(), ' + ('view(), ' if visual else '') + 'predict(), submit(). submit() ends '
        'the episode.\n'
        'The submitted network is scored once on held-out images that you never see; the start network scores '
        + (f'{start_score:.1%} on the held-out validation images. ' if start_score is not None
           else 'about 10%, chance. ') +
        'No gradients anywhere: autograd, torch.optim and torch.func transforms are disabled.',
        '# What we want you to do\nSubmit the network with the highest held-out accuracy you can, and learn from the '
        'labels in your own context in phase 2:\n'
        '1. Phase 1: with code on the unlabeled images, build a network (a diff) whose representations carry the '
        'information you will need, in a form you can read and reason about in phase 2: for example a small set of fc '
        'outputs or channels whose meaning you understand. Print every number you will want to '
        'reuse, since no code runs after end_tools().\n'
        '2. Call end_tools().\n'
        '3. Phase 2: call predict() with your diff on many images, and labels() on the same indices. Reason over the '
        '(representation, label) pairs in your own context: which features or outputs go with which class, which '
        'output rows stand for which class, what weights would separate the classes. Write the resulting '
        "numbers into your diff. evaluate() gives a diff's accuracy (one number) if you want to compare candidates.\n"
        '4. submit() your final diff.\n' + pacing,
    ])


def full_rules(n_examples, start_score=None, visual=False):
    return '\n\n'.join([
        f'# How this episode works (mode full)\nYou have a training subset of {n_examples:,} labeled images, indices '
        f'0..{n_examples - 1} (class-balanced, not in class order). Every call and the sandbox use the same indices: '
        'data["images"][i] and data["labels"][i] in tools() code are image i and its class. There is a single phase: '
        'data(), ' + ('view(), ' if visual else '') + 'labels(), tools(), predict(), evaluate() and submit() are all '
        'available from the start, as often and '
        'in any order you like, and only an accepted submit() ends the episode. Your tools() code can read every '
        'label, and evaluate() reports accuracy, loss, per-class accuracy and confusion on the training subset.\n'
        'The submitted network is scored once on held-out images that you never see; the start network scores '
        + (f'{start_score:.1%} on the held-out validation images. ' if start_score is not None
           else 'about 10%, chance. ') +
        'No gradients anywhere: autograd, torch.optim and torch.func transforms are disabled.',
        '# What we want you to do\nSubmit the network with the highest held-out accuracy you can.',
    ])


def user_prompt(task, mode, start, n_examples, max_numbers, max_turns=10, format_retries=3, max_tokens=None,
                soft_budget=None, warm=False, start_score=None, turn_limit=None, visual=False):
    tensors = '\n'.join(f'{n} {s["shape"]} mean={s["mean"]} std={s["std"]}' for n, s in weight_stats(start).items())
    if soft_budget:
        pace = sdk_turn_rules(turn_limit, soft_budget) if turn_limit else budget_rules(soft_budget)
    else:
        pace = turns_rules(max_turns, format_retries)
    if mode == 'reasoned':
        rules = [reasoned_rules(n_examples, soft_budget, start_score, turn_limit, max_turns, visual)]
        rules += [VISUAL_RULES] if visual else []
    elif mode == 'full':
        rules = [full_rules(n_examples, start_score, visual)]
    else:
        rules = [f'# {MODE_RULES[mode]}',
                 (f'# Training subset\n{n_examples} images, indices 0..{n_examples - 1} (class-balanced, not in class '
                  'order).' if n_examples else '# Training subset\nnone')]
    return '\n\n'.join([
        f'# Task: {task.name}\n{task.warm_description if warm else task.description}',
        f'# Dataset\n{json.dumps(DATASET_DESCRIPTIONS[task.dataset], indent=1)}',
        f'# Architecture\n{ARCHITECTURE}',
        f'# Writable tensors (name, shape, start-weight stats)\n{tensors}',
        *rules,
        diff_rules(mode, max_numbers, max_tokens, hard_cap=soft_budget is None),
        pace,
    ])
