import json
import os
import sys
import time

import numpy as np
import torch

from ripple_wsp.guard import lock_torch
from ripple_wsp.net import ALLOWED, check, classify
from ripple_wsp.runner import lock


def main() -> None:
    net_path, data_path, out_path = sys.argv[1:4]
    with np.load(data_path) as d:
        x, y = d["x"].copy(), d["y"].copy()
    out_f = open(out_path, "w")
    lock_torch(int(os.environ.get("LAB_THREADS", "8")))
    lock(os.path.dirname(os.path.abspath(net_path)), [])
    torch.serialization.add_safe_globals(list(ALLOWED))
    net = torch.load(net_path, weights_only=True)
    info = check(net)
    t0 = time.time()
    pred = classify(net, x)
    k = 10
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    json.dump(
        {
            "accuracy": float((pred == y).mean()),
            "per_class": [float((pred[y == c] == c).mean()) for c in range(k)],
            "confusion": confusion.tolist(),
            "n": int(len(y)),
            "seconds": time.time() - t0,
            **info,
        },
        out_f,
    )
    out_f.close()


if __name__ == "__main__":
    main()
