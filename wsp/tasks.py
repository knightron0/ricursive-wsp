"""The two weight-space programming tasks."""
from dataclasses import dataclass

import torch

from wsp.model import initial_weights, state_shapes


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str          # target dataset: data subset, val and test all come from here
    num_classes: int
    description: str
    needs_source: bool


TASKS = {
    'scratch-cifar10': Task(
        'scratch-cifar10', 'cifar10', 10,
        'Write every weight of a ResNet-18 CIFAR-10 classifier yourself. The starting '
        'weights are a random PyTorch initialization (BatchNorm running stats mean 0, var 1). '
        'You may compute weights from your data subset (statistics, patches, PCA/whitening, '
        'clustering, least squares, BatchNorm statistics, ...) but there are no gradients.',
        needs_source=False),
    'cifar10-to-cifar100': Task(
        'cifar10-to-cifar100', 'cifar100', 100,
        'Turn a ResNet-18 trained on CIFAR-10 into a CIFAR-100 classifier. The starting '
        'weights are the trained CIFAR-10 network, except fc (now 512 -> 100) which is a random '
        'initialization. You may rebuild the head and edit any other weights, computing from '
        'your CIFAR-100 data subset, but there are no gradients.',
        needs_source=True),
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


def validate_weights(reference, candidate, error=ValueError):
    if not isinstance(candidate, dict) or set(candidate) != set(reference):
        missing = set(reference) - set(candidate or {})
        extra = set(candidate or {}) - set(reference)
        raise error(f'build() must return exactly the ResNet-18 state_dict names '
                    f'(missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]})')
    result = {}
    for name, ref in reference.items():
        tensor = candidate[name]
        if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != list(ref.shape):
            raise error(f'{name}: expected a tensor of shape {list(ref.shape)}')
        if ref.is_floating_point():
            tensor = tensor.detach().to(torch.float32).contiguous()
            if not torch.isfinite(tensor).all():
                raise error(f'{name}: contains NaN or inf')
            if name.endswith('running_var') and (tensor < 0).any():
                raise error(f'{name}: running_var must be non-negative')
        else:
            tensor = ref.clone()
        result[name] = tensor
    return result
