"""Gradient-free helpers available to weight-building programs as `toolkit`.

Everything here runs without autograd. All functions take uint8 images
[N,3,32,32] and handle normalization and batching themselves.
"""
import torch
import torch.nn.functional as F

from wsp.data import normalize
from wsp.model import ResNet18, default_device

DOC = """toolkit functions (all gradient-free):
- toolkit.normalize(images) -> float [N,3,32,32] normalized network input.
- toolkit.capture(weights, images, names, num_classes, reduce=None, batch_size=500)
    Run ResNet-18 with `weights` (eval-mode BatchNorm) and return {name: value} for the
    OUTPUT of each named module (e.g. "conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1",
    "layer2", "layer3.1", "avgpool", "fc"). Also accepts "input" for the normalized input.
    reduce=None -> full tensors on CPU (memory heavy for early layers!)
    reduce="pool" -> global average over H,W -> [N, C]
    reduce="mean_var" -> per-channel {"mean": [C], "var": [C]} over N,H,W (biased var),
                         exactly what BatchNorm running stats need.
    Note: modules after an unset layer see that layer's current weights, so build
    layer by layer and re-capture after each change.
- toolkit.logits(weights, images, num_classes) -> [N, num_classes]
- toolkit.accuracy(weights, images, labels, num_classes) -> float, on YOUR data subset.
- toolkit.patches(images, size=3, n=100000, seed=0) -> [n, 3*size*size] random normalized patches
    (flattened in the same order as a conv weight [out, 3, size, size].flatten(1)).
- toolkit.whiten(x, eps=1e-2) -> (W, mean): ZCA whitening matrix for rows of x.
- toolkit.kmeans(x, k, iters=20, seed=0) -> [k, D] centroids (spherical k-means on rows of x).
- toolkit.ridge(features, labels, num_classes, lam=1e-3) -> (W [C, D], b [C]) least-squares
    classifier on one-hot targets.
- toolkit.device: "mps"/"cuda"/"cpu" used for the forward passes."""

device = default_device()


def _model(weights, num_classes):
    model = ResNet18(num_classes)
    model.load_state_dict({k: v.to(torch.float32) if v.is_floating_point() else v for k, v in weights.items()})
    return model.to(device).eval()


@torch.no_grad()
def capture(weights, images, names, num_classes, reduce=None, batch_size=500):
    if reduce not in (None, 'pool', 'mean_var'):
        raise ValueError('reduce must be None, "pool" or "mean_var"')
    model = _model(weights, num_classes)
    modules = dict(model.named_modules())
    missing = [n for n in names if n != 'input' and n not in modules]
    if missing:
        raise KeyError(f'Unknown module names: {missing}')
    chunks = {n: [] for n in names}
    sums = {n: [0.0, 0.0, 0] for n in names}
    current = {}
    hooks = [modules[n].register_forward_hook(lambda m, i, o, n=n: current.__setitem__(n, o))
             for n in names if n != 'input']
    try:
        for start in range(0, len(images), batch_size):
            x = normalize(images[start:start + batch_size].to(device))
            current.clear()
            current['input'] = x
            model(x)
            for n in names:
                out = current[n].float()
                if out.dim() == 4 and reduce is not None:
                    if reduce == 'pool':
                        chunks[n].append(out.mean(dim=(2, 3)).cpu())
                    else:
                        s = sums[n]
                        s[0] = s[0] + out.sum(dim=(0, 2, 3)).cpu().double()
                        s[1] = s[1] + (out ** 2).sum(dim=(0, 2, 3)).cpu().double()
                        s[2] += out.shape[0] * out.shape[2] * out.shape[3]
                elif reduce == 'mean_var':
                    s = sums[n]
                    s[0] = s[0] + out.sum(0).cpu().double()
                    s[1] = s[1] + (out ** 2).sum(0).cpu().double()
                    s[2] += out.shape[0]
                else:
                    chunks[n].append(out.flatten(1).cpu() if reduce == 'pool' else out.cpu())
    finally:
        for h in hooks:
            h.remove()
    if reduce == 'mean_var':
        result = {}
        for n, (s1, s2, count) in sums.items():
            mean = s1 / count
            result[n] = {'mean': mean.float(), 'var': (s2 / count - mean ** 2).clamp_min(0).float()}
        return result
    return {n: torch.cat(c) for n, c in chunks.items()}


def logits(weights, images, num_classes, batch_size=500):
    return capture(weights, images, ['fc'], num_classes, batch_size=batch_size)['fc']


def accuracy(weights, images, labels, num_classes):
    return (logits(weights, images, num_classes).argmax(1) == labels).float().mean().item()


def patches(images, size=3, n=100000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(images), (n,), generator=g)
    ys = torch.randint(0, 33 - size, (n,), generator=g)
    xs = torch.randint(0, 33 - size, (n,), generator=g)
    padded = F.pad(normalize(images), (size // 2,) * 4) if size % 2 else normalize(images)
    offsets = torch.arange(size)
    rows = (ys[:, None] + offsets)[:, :, None].expand(n, size, size)
    cols = (xs[:, None] + offsets)[:, None, :].expand(n, size, size)
    out = padded[idx[:, None, None, None], torch.arange(3)[None, :, None, None],
                 rows[:, None], cols[:, None]]
    return out.flatten(1)


def whiten(x, eps=1e-2):
    mean = x.mean(0)
    cov = torch.cov((x - mean).T.double())
    values, vectors = torch.linalg.eigh(cov)
    W = vectors @ torch.diag((values.clamp_min(0) + eps).rsqrt()) @ vectors.T
    return W.float(), mean


def kmeans(x, k, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(x.float(), dim=1)
    centroids = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (x @ centroids.T).argmax(1)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        sums[empty] = x[torch.randint(0, len(x), (int(empty.sum()),), generator=g)]
        centroids = F.normalize(sums, dim=1)
    return centroids


def ridge(features, labels, num_classes, lam=1e-3):
    X = features.double()
    X = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    Y = F.one_hot(labels, num_classes).double()
    A = X.T @ X + lam * len(X) * torch.eye(X.shape[1], dtype=X.dtype)
    sol = torch.linalg.solve(A, X.T @ Y)
    return sol[:-1].T.float().contiguous(), sol[-1].float().contiguous()
