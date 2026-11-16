"""The two weight-space programming tasks."""
from dataclasses import dataclass

import torch

from wsp.model import initial_weights, state_shapes


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str          # train subset, val and test all come from here
    num_classes: int
    goal: str
    origin: str
    needs_source: bool
    note: str = ''

    @property
    def description(self):
        return f'{self.goal} The starting weights are {self.origin}.' + (f' {self.note}' if self.note else '')

    @property
    def warm_description(self):
        return (f'{self.goal} The starting weights are a network that another agent produced for this same task. That '
                f'agent was given an identical prompt, except that its starting weights were {self.origin}. You are '
                'not told how it arrived at this network.' + (f' {self.note}' if self.note else ''))


TASKS = {
    'scratch-cifar10': Task(
        'scratch-cifar10', 'cifar10', 10,
        'Write every weight of a ResNet-18 CIFAR-10 classifier yourself.',
        'a random PyTorch initialization (BatchNorm running stats mean 0, var 1)',
        needs_source=False),
    'cifar10-to-cifar100': Task(
        'cifar10-to-cifar100', 'cifar100', 100,
        'Turn a ResNet-18 trained on CIFAR-10 into a CIFAR-100 classifier.',
        'the trained CIFAR-10 network, except fc (now 512 -> 100) which is a random initialization',
        needs_source=True, note='You may rebuild the head and edit any other weights.'),
}


def load_start(path, num_classes):
    weights = torch.load(path, weights_only=True)
    shapes = {k: list(v.shape) for k, v in weights.items()}
    if shapes != state_shapes(num_classes):
        raise ValueError(f'{path} is not a {num_classes}-class wsp ResNet-18 state_dict')
    return {k: v.clone() for k, v in weights.items()}


DATASET_DESCRIPTIONS = {
    'cifar10': {
        'name': 'CIFAR-10 (Krizhevsky, 2009)',
        'images': '32x32 RGB natural photos (uint8), downsampled from the 80 Million Tiny Images '
                  'collection; one object per image, usually roughly centered and filling much of '
                  'the frame, with varied pose, scale, viewpoint and background.',
        'channel_mean': (0.4914, 0.4822, 0.4465), 'channel_std': (0.2470, 0.2435, 0.2616),
        'splits': '50,000 official train images (5,000 per class; here 45,000 train + 5,000 val) '
                  'and 10,000 test images (1,000 per class). Classes are balanced and mutually exclusive.',
        'classes': {
            'airplane': 'aircraft of all kinds, often against sky (blue/gray/white backgrounds)',
            'automobile': 'sedans, SUVs and similar cars (not pickup trucks)',
            'bird': 'many species, perched or flying; often against foliage or sky',
            'cat': 'domestic cats, often indoors or close-up',
            'deer': 'deer and similar animals, often in green/brown natural scenes',
            'dog': 'domestic dogs of many breeds; most often confused with cat',
            'frog': 'frogs and toads, often green/brown on ground or leaves',
            'horse': 'horses, often side-on in fields',
            'ship': 'boats and ships, often on water (blue backgrounds)',
            'truck': 'big trucks only (not pickup trucks); most often confused with automobile',
        },
    },
    'cifar100': {
        'name': 'CIFAR-100 (Krizhevsky, 2009)',
        'images': '32x32 RGB natural photos (uint8), same style and source as CIFAR-10.',
        'splits': '50,000 official train images (500 per class; here 45,000 train + 5,000 val) '
                  'and 10,000 test images (100 per class). 100 fine classes in 20 superclasses.',
    },
}


def start_weights(task, source=None, seed=0):
    weights = initial_weights(task.num_classes, seed)
    if task.needs_source:
        if source is None:
            raise ValueError(f'{task.name} needs a trained CIFAR-10 source checkpoint')
        source_shapes = {k: list(v.shape) for k, v in source.items()}
        if source_shapes != state_shapes(10):
            raise ValueError('Source checkpoint is not a 10-class wsp ResNet-18')
        weights.update({k: v.clone() for k, v in source.items() if not k.startswith('fc.')})
    return weights


def writable_names(weights):
    """Tensors a diff may set: every float entry (num_batches_tracked is unused)."""
    return [name for name, t in weights.items() if t.is_floating_point()]


def apply_diff(start, diff, max_numbers):
    """Return start weights with the diff's literals written in, plus how many numbers it set.

    `diff` is [{"name": ..., "values": [...]}]: every number of that tensor, flat in row-major
    order (nested lists are flattened). Tensors not listed keep their start values."""
    if not isinstance(diff, list):
        raise ValueError('weights must be a list of {"name", "values"}')
    writable, result, total = set(writable_names(start)), {k: v.clone() for k, v in start.items()}, 0
    seen, problems = set(), []
    for entry in diff:
        name = entry.get('name') if isinstance(entry, dict) else None
        if name not in writable:
            problems.append(f'{name!r} is not a writable tensor name')
            continue
        if name in seen:
            problems.append(f'{name} is listed twice')
            continue
        seen.add(name)
        try:
            tensor = torch.tensor(entry.get('values'), dtype=torch.float32).flatten()
        except (TypeError, ValueError, RuntimeError) as e:
            problems.append(f'{name}: values must be numbers ({e})')
            continue
        ref = start[name]
        if tensor.numel() != ref.numel():
            problems.append(f'{name}: got {tensor.numel()} numbers, shape {list(ref.shape)} needs {ref.numel()}')
        elif not torch.isfinite(tensor).all():
            problems.append(f'{name}: contains NaN or inf')
        elif name.endswith('running_var') and (tensor < 0).any():
            problems.append(f'{name}: running_var must be non-negative')
        else:
            result[name] = tensor.view(ref.shape).contiguous()
            total += tensor.numel()
    if total > max_numbers:
        problems.append(f'{total} numbers in the diff; the limit is {max_numbers}')
    if len(problems) > 1:
        raise ValueError(f'{len(problems)} problems, nothing applied: ' + '; '.join(problems))
    if problems:
        raise ValueError(problems[0])
    return result, total
