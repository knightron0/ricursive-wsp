import importlib
import json
import os
import sys

import torch

from ripple_wsp.guard import lock_torch
from ripple_wsp.net import check
from ripple_wsp.runner import lock


def main() -> None:
    box = os.getcwd()
    lock_torch(int(os.environ.get("LAB_THREADS", "8")))
    lock(box, [])
    sys.path.insert(0, box)
    model = importlib.import_module("model")
    if not callable(getattr(model, "build", None)):
        raise AttributeError("model.py must define build() returning the network")
    net = model.build()
    info = check(net)
    with torch.no_grad():
        probe = net.eval()(torch.rand(2, 3, 32, 32))
    if tuple(probe.shape) != (2, 10):
        raise ValueError(f"the network maps [2, 3, 32, 32] to {tuple(probe.shape)}; it must give [N, 10]")
    net.eval()
    torch.save(net, os.path.join(box, "net.pt"))
    with open(os.path.join(box, "export.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
