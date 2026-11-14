"""Scoring and diagnostics: what the agent is judged on and what it gets to see."""
import math

import torch
import torch.nn.functional as F

from wsp.data import normalize
from wsp.model import ResNet18

STAGES = ('bn1', 'layer1', 'layer2', 'layer3', 'layer4')


def _r(value):
    return float(f'{float(value):.4g}')


@torch.no_grad()
def evaluate(weights, split, classes, device='cpu', batch_size=500):
    """Accuracy, loss, per-class accuracy, confusions and activation health on a split."""
    k = len(classes)
    model = ResNet18(k)
    model.load_state_dict(weights)
    model.to(device).eval()
    modules = dict(model.named_modules())
    current = {}
    hooks = [modules[s].register_forward_hook(lambda m, i, o, s=s: current.__setitem__(s, o)) for s in STAGES]
    images, labels = split
    confusion = torch.zeros(k, k, dtype=torch.long)
    loss_sum, peak, act_sum = 0.0, {}, {}
    try:
        for start in range(0, len(labels), batch_size):
            x = normalize(images[start:start + batch_size].to(device))
            y = labels[start:start + batch_size].to(device)
            logits = model(x).float()
            loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
            confusion += torch.bincount((y * k + logits.argmax(1)).cpu(), minlength=k * k).view(k, k)
            for s in STAGES:
                a = F.relu(current[s]) if s == 'bn1' else current[s]
                p = a.amax(dim=(0, 2, 3)).cpu()
                peak[s] = torch.maximum(peak[s], p) if s in peak else p
                act_sum[s] = act_sum.get(s, 0.0) + a.float().mean().item() * len(y)
    finally:
        for h in hooks:
            h.remove()
    n = len(labels)
    loss = loss_sum / n
    if not math.isfinite(loss):
        raise ValueError('Network produced non-finite loss')
    correct = confusion.diag()
    per_class = {c: _r(correct[i] / max(confusion[i].sum().item(), 1)) for i, c in enumerate(classes)}
    off = confusion.clone().fill_diagonal_(0)
    top = torch.topk(off.flatten(), 15)
    metrics = {
        'accuracy': correct.sum().item() / n,
        'loss': round(loss, 6),
        'top_confusions': [{'true': classes[i // k], 'predicted': classes[i % k], 'count': v.item()}
                           for v, i in zip(top.values, top.indices.tolist()) if v > 0],
        'prediction_counts': None,
        'activations': {s: {'dead_channel_fraction': _r((peak[s] == 0).float().mean()),
                            'mean': _r(act_sum[s] / n)} for s in STAGES},
    }
    predicted = confusion.sum(0)
    if k <= 10:
        metrics['per_class_accuracy'] = per_class
        metrics['confusion'] = confusion.tolist()
        metrics['prediction_counts'] = dict(zip(classes, predicted.tolist()))
    else:
        ranked = sorted(per_class.items(), key=lambda kv: kv[1])
        metrics['worst_classes'] = dict(ranked[:15])
        metrics['best_classes'] = dict(ranked[-10:])
        metrics['prediction_counts'] = {'max': predicted.max().item(), 'min': predicted.min().item(),
                                        'classes_never_predicted': int((predicted == 0).sum())}
    return metrics


def summarize(metrics):
    return {'accuracy': metrics['accuracy'], 'loss': metrics['loss']}


def weight_stats(weights):
    stats = {}
    for name, tensor in weights.items():
        if not tensor.is_floating_point():
            continue
        t = tensor.float()
        stats[name] = {'shape': list(t.shape), 'mean': _r(t.mean()),
                       'std': _r(t.std()) if t.numel() > 1 else 0.0, 'norm': _r(t.norm())}
    return stats
