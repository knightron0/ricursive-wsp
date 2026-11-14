import json
import time
from pathlib import Path




def main(net: str, sets: dict, out: str, threads: int = 8) -> None:
    import numpy as np
    import torch

    from ripple_wsp.net import ALLOWED, check, classify

    torch.set_num_threads(threads)
    torch.serialization.add_safe_globals(list(ALLOWED))
    model = torch.load(Path(net).expanduser(), weights_only=True)
    info = check(model)
    result = {"net": str(Path(net).expanduser()), "numbers": info["numbers"], "layers": info["layers"]}
    for name, path in sets.items():
        with np.load(Path(path).expanduser()) as d:
            x, y = d["x"], d["y"]
        t0 = time.time()
        pred = classify(model, x)
        result[name] = {
            "accuracy": float((pred == y).mean()),
            "per_class": [float((pred[y == c] == c).mean()) for c in range(10)],
            "n": int(len(y)),
            "seconds": time.time() - t0,
        }
        print(f"evaluate: {name} {result[name]['accuracy']:.4f} ({len(y)} images, {time.time() - t0:.0f}s)", flush=True)
    Path(out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(out).expanduser().write_text(json.dumps(result, indent=1) + "\n")
