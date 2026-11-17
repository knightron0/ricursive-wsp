"""Forward-pass-only helpers: load MNIST, pool, run a hand-written network, view images.
No fitting of any kind happens here."""
import gzip, os, struct, sys, urllib.request
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download():
    os.makedirs(D, exist_ok=True)
    for p in ("train", "t10k"):
        for kind in ("images-idx3", "labels-idx1"):
            f = f"{p}-{kind}-ubyte.gz"
            if not os.path.exists(os.path.join(D, f)):
                urllib.request.urlretrieve(MIRROR + f, os.path.join(D, f))


def load(split):
    download()
    p = "train" if split == "train" else "t10k"
    with gzip.open(f"{D}/{p}-images-idx3-ubyte.gz") as f:
        _, n, h, w = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), np.uint8).reshape(n, 28, 28).astype(np.float32) / 255
    with gzip.open(f"{D}/{p}-labels-idx1-ubyte.gz") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8).astype(int)
    return x, y


def pool(x, start=4, size=21, k=3):
    """Layer 1: crop [start, start+size) and average-pool k x k, stride k."""
    g = size // k
    c = x[:, start:start + size, start:start + size]
    return c.reshape(len(x), g, k, g, k).mean(axis=(2, 4))


def show(grids, labels=None, per_row=8):
    """Print pooled grids as ASCII, ink 0..1 -> ' .:-=+*#%@'."""
    ramp = " .:-=+*#%@"
    g = grids.shape[1]
    for s in range(0, len(grids), per_row):
        chunk = grids[s:s + per_row]
        if labels is not None:
            print("  ".join(f"{str(l):<{g}}" for l in labels[s:s + per_row]))
        for r in range(g):
            print("  ".join("".join(ramp[min(9, int(v * 10))] for v in im[r]) for im in chunk))
        print()
