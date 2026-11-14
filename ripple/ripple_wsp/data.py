import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def parse(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    rec = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 1 + 3072)
    y = rec[:, 0].astype(np.int64)
    x = rec[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return x, y


def prepare(out: Path, fraction: float = 0.1, seed: int = 0, val_per_class: int = 500, val_seed: int = 1) -> None:
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "cifar-10-binary.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive.with_suffix(".part"))
        archive.with_suffix(".part").rename(archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    assert digest == MD5, f"md5 {digest} != {MD5}"
    with tarfile.open(archive) as tar:
        blobs = {Path(m.name).name: tar.extractfile(m).read() for m in tar.getmembers() if m.name.endswith(".bin")}
    train = [parse(blobs[f"data_batch_{i}.bin"]) for i in range(1, 6)]
    x_train = np.concatenate([t[0] for t in train])
    y_train = np.concatenate([t[1] for t in train])
    x_test, y_test = parse(blobs["test_batch.bin"])
    assert x_train.shape == (50000, 32, 32, 3) and x_test.shape == (10000, 32, 32, 3)

    rng = np.random.default_rng(seed)
    per_class = int(round(fraction * len(y_train) / 10))
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y_train == c), per_class, replace=False) for c in range(10)]))

    free = np.setdiff1d(np.arange(len(y_train)), idx)
    vrng = np.random.default_rng(val_seed)
    vidx = np.sort(np.concatenate([vrng.choice(free[y_train[free] == c], val_per_class, replace=False) for c in range(10)]))
    assert not np.intersect1d(idx, vidx).size

    np.savez(out / "train.npz", x=x_train, y=y_train)
    np.savez(out / "test.npz", x=x_test, y=y_test)
    np.savez(out / "sample.npz", x=x_train[idx], y=y_train[idx], idx=idx)
    np.savez(out / "val.npz", x=x_train[vidx], y=y_train[vidx], idx=vidx)
    print(f"train {x_train.shape}, test {x_test.shape} -> {out}")
    print(f"split sample.npz: {len(idx)} images ({per_class}/class, seed {seed})")
    print(f"val val.npz: {len(vidx)} images ({val_per_class}/class, seed {val_seed}), disjoint from the split")
