"""v0: the first attempt, written with no data at all (7x7 hand-drawn grids).
Evaluated once on the MNIST test set: 32.00%. The forward pass is plain Python (no numpy)."""
import gzip, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mn

mn.download()
D = mn.D

TEMPLATES = {
0: """
..+++..
.+.-.+.
.+---+.
.+---+.
.+---+.
.+.-.+.
..+++..""",
1: """
.-.+.-.
.-.+.-.
.-.+.-.
.-.+.-.
.-.+.-.
.-.+.-.
.-.+.-.""",
2: """
..+++..
...-.+.
.---.+.
.--.+..
.-.+.-.
.++....
.+++++.""",
3: """
..+++..
...-.+.
.---.+.
.-.++..
.---.+.
...-.+.
.++++..""",
4: """
.+.-+..
.+.-+..
.+.-+..
.+++++.
....+..
.--.+..
.--.+..""",
5: """
..++++.
..+---.
..+---.
..+++..
.---.+.
...-.+.
.++++..""",
6: """
...++..
..++.-.
.++.--.
.++++..
.+.-.+.
.+.-.+.
..+++..""",
7: """
.+++++.
.---.+.
.--.+..
.--.+..
.-.+.-.
...+.-.
...+...""",
8: """
..+++..
.+.-.+.
..+.+..
.-.+.-.
..+.+..
.+.-.+.
..+++..""",
9: """
..+++..
.+.-.+.
.+.-.+.
..++++.
.--.+..
.--.+..
..-.+..""",
}

# Build W[k] as a flat list of 49 weights: +1/P on '+', -1/N on '-', 0 on '.'
W = {}
for k, s in TEMPLATES.items():
    rows = s.strip().split("\n")
    assert len(rows) == 7 and all(len(r) == 7 for r in rows), k
    cells = "".join(rows)
    P, N = cells.count("+"), cells.count("-")
    W[k] = [1 / P if ch == "+" else (-1 / N if ch == "-" else 0.0) for ch in cells]
    assert abs(sum(W[k])) < 1e-9
    print(f"digit {k}: P={P:2d} N={N:2d}")

with gzip.open(f"{D}/t10k-images-idx3-ubyte.gz") as f:
    _, n, h, w = struct.unpack(">IIII", f.read(16))
    pix = f.read()
with gzip.open(f"{D}/t10k-labels-idx1-ubyte.gz") as f:
    f.read(8)
    labels = f.read()
assert (n, h, w) == (10000, 28, 28) and len(labels) == n

correct = 0
conf = [[0] * 10 for _ in range(10)]  # conf[true][pred]
for i in range(n):
    img = pix[i * 784:(i + 1) * 784]
    # Layer 1: crop pixels 4..24, average-pool 3x3 stride 3 -> 7x7, pixels in [0,1]
    hgrid = []
    for r in range(7):
        for c in range(7):
            s = 0
            for di in range(3):
                base = (4 + 3 * r + di) * 28 + 4 + 3 * c
                s += img[base] + img[base + 1] + img[base + 2]
            hgrid.append(s / (9 * 255))
    # Layer 2: template logits, argmax
    z = [sum(wk * hv for wk, hv in zip(W[k], hgrid)) for k in range(10)]
    pred = max(range(10), key=z.__getitem__)
    y = labels[i]
    conf[y][pred] += 1
    correct += pred == y

print(f"\nTest accuracy: {correct}/{n} = {100 * correct / n:.2f}%\n")
print("per-class accuracy and most common confusion:")
for y in range(10):
    tot = sum(conf[y])
    wrong = max((p for p in range(10) if p != y), key=lambda p: conf[y][p])
    print(f"  {y}: {100 * conf[y][y] / tot:5.1f}%  (n={tot}; most often called {wrong}: {conf[y][wrong]})")
print("\nconfusion matrix (rows = true, cols = predicted):")
print("     " + " ".join(f"{p:5d}" for p in range(10)))
for y in range(10):
    print(f"  {y}: " + " ".join(f"{v:5d}" for v in conf[y]))
