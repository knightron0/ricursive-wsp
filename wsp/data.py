"""CIFAR-10 / CIFAR-100 with fixed train/val/test splits, cached as uint8 tensors.

For each dataset: 5k of the official train set is held out as val (the search
signal), the remaining 45k is train (the pool the agent's X% budget is drawn
from), and the official test set is only touched after the result is frozen.
"""
import json
import math
from pathlib import Path

import torch

DATASETS = ('cifar10', 'cifar100')
SPLITS = ('train', 'val', 'test')
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
VAL_SIZE = 5000
SPLIT_SEED = 0


def _paths(dataset, root):
    root = Path(root)
    return {name: root / f'{dataset}_{name}.pt' for name in SPLITS}, root / f'{dataset}_classes.json'


def prepare(dataset, root='data'):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}')
    paths, classes_path = _paths(dataset, root)
    if all(p.exists() for p in paths.values()) and classes_path.exists():
        return paths
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = CIFAR10 if dataset == 'cifar10' else CIFAR100

    def tensors(ds):
        return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)

    train_ds = cls(root, train=True, download=True)
    x_train, y_train = tensors(train_ds)
    x_test, y_test = tensors(cls(root, train=False, download=True))
    order = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(SPLIT_SEED))
    val, train = order[:VAL_SIZE], order[VAL_SIZE:]
    torch.save((x_train[train], y_train[train]), paths['train'])
    torch.save((x_train[val], y_train[val]), paths['val'])
    torch.save((x_test, y_test), paths['test'])
    classes_path.write_text(json.dumps(list(train_ds.classes)))
    return paths


def load_split(dataset, name, root='data'):
    """Return (images uint8 [N,3,32,32], labels int64 [N])."""
    if name not in SPLITS:
        raise ValueError(f'Unknown split {name!r}')
    return torch.load(prepare(dataset, root)[name], weights_only=True)


def class_names(dataset, root='data'):
    prepare(dataset, root)
    return json.loads(_paths(dataset, root)[1].read_text())


def stratified_subset(labels, fraction, seed=0):
    """Indices of a class-balanced random `fraction` of `labels` (at least 1 per class)."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for c in labels.unique().tolist():
        members = (labels == c).nonzero().flatten()
        take = max(1, math.floor(fraction * len(members) + 0.5))
        chosen.append(members[torch.randperm(len(members), generator=generator)[:take]])
    return torch.cat(chosen).sort().values


def normalize(images):
    mean = torch.tensor(MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std
