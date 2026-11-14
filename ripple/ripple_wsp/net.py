import numpy as np
import torch
from torch import nn

ALLOWED = (
    nn.Sequential, nn.Conv2d, nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.MaxPool2d, nn.AvgPool2d,
    nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Flatten, nn.BatchNorm1d, nn.BatchNorm2d, nn.Softmax, nn.Identity,
)
ALLOWED_NAMES = ", ".join(c.__name__ for c in ALLOWED if c is not nn.Sequential)
HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks")
MAX_PARAMS = 25_000_000


def check(net) -> dict:
    if type(net) is not nn.Sequential:
        raise TypeError(f"build() must return a torch.nn.Sequential, got {type(net).__name__}")
    for name, m in net.named_modules():
        where = name or "the Sequential"
        if type(m) not in ALLOWED:
            raise TypeError(f"{where} is a {type(m).__module__}.{type(m).__name__}; only these layers are allowed: {ALLOWED_NAMES}, Sequential")
        if "forward" in m.__dict__:
            raise TypeError(f"{where} overrides forward")
        if any(getattr(m, h, None) for h in HOOKS):
            raise TypeError(f"{where} has hooks registered")
    n = sum(p.numel() for p in net.parameters()) + sum(b.numel() for b in net.buffers())
    if n > MAX_PARAMS:
        raise ValueError(f"the network holds {n:,} numbers; the limit is {MAX_PARAMS:,}")
    return {"numbers": int(n), "layers": layers(net)}


def layers(net) -> str:
    parts = []
    for m in net.modules():
        if type(m) is nn.Sequential:
            continue
        if isinstance(m, nn.Conv2d):
            parts.append(f"Conv2d({m.in_channels}->{m.out_channels}, k{m.kernel_size[0]}x{m.kernel_size[1]}, s{m.stride[0]}, p{m.padding if isinstance(m.padding, str) else m.padding[0]}, g{m.groups})")
        elif isinstance(m, nn.Linear):
            parts.append(f"Linear({m.in_features}->{m.out_features})")
        else:
            parts.append(type(m).__name__)
    return ", ".join(parts)


def prepare(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(255.0)


def classify(net, x: np.ndarray, batch: int = 500) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = net(prepare(x[i : i + batch]))
            if logits.dim() != 2 or logits.shape != (len(x[i : i + batch]), 10):
                raise ValueError(f"the network output has shape {tuple(logits.shape)}; it must be [N, 10]")
            out.append(logits.argmax(1).numpy())
    return np.concatenate(out).astype(np.int64)
