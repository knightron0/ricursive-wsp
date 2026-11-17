"""Hand-specified MNIST network: digits drawn as pen strokes -> template weights.
Forward passes only. No weight is computed from data.

Coordinates are 28x28 pixel space, (x = column, y = row, y grows downward).
Arcs: points (cx + rx cos t, cy + ry sin t), t in degrees; t = -90 is the top.
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mn

# ---------------------------------------------------------------- hand-drawn digits
L = lambda *pts: ("line", pts)                       # polyline through points
A = lambda cx, cy, rx, ry, t0, t1: ("arc", (cx, cy, rx, ry, t0, t1))
E = lambda cx, cy, rx, ry: A(cx, cy, rx, ry, 0, 360)  # closed ellipse
NEG = lambda st: ("neg", st)                          # inhibitory stroke: ink here counts against

DIGITS = {
    0: [[E(14, 14, 5.5, 9)]],
    1: [[L((14, 4), (14, 24))],
        [L((19, 4), (9, 24))]],                                         # strongly slanted 1
    2: [[A(14, 9, 5, 4.5, -170, 20), L((18.7, 10.5), (8, 20), (21, 19.5))],
        [A(14, 9, 5, 4.5, -170, 20), L((18.7, 10.5), (9, 19)), E(10, 19.5, 2.5, 2),
         L((12, 20.5), (21, 20))],                                      # 2 with a bottom loop
        [A(14, 9, 5, 4.5, -170, 20), L((18.7, 10.5), (12, 16)),
         A(10, 18.5, 4, 3, -40, 330), L((13, 21), (21, 20))]],           # 2 with a big bottom loop
    3: [[A(13.5, 9, 5, 4.5, -160, 90), A(13.5, 18.5, 5.5, 5, -90, 160),
         NEG(L((8.5, 9.5), (8.5, 11.5))), NEG(L((8, 16), (8, 18)))]],     # no left sides (vs 8)
    4: [[L((8, 5), (7, 16), (21, 16)), L((18, 5), (17, 24)),
         NEG(L((11, 5), (15, 5)))]],                                    # open top (vs 9)
    5: [[L((21, 6), (12, 6), (10, 13)), A(13, 18, 5.5, 4.5, -130, 150),
         NEG(L((18.5, 9), (18.5, 12)))]],                               # empty upper right (vs 3)
    6: [[L((16, 4), (11, 9), (7.5, 15), (7, 18.5)), E(11, 18.5, 4.5, 4)]],
    7: [[L((7, 8), (19, 8), (12, 25)), NEG(L((8, 14), (12, 15.5)))]],  # no loop bottom (vs 9)
    8: [[E(14, 9, 4.5, 4.5), E(14, 19, 5.5, 5)],
        [E(16, 9, 4, 4), E(12.5, 19, 3.5, 4)]],                         # slanted 8, small lower loop
    9: [[E(12.5, 11, 4.5, 4), L((17, 11), (14, 25)),
         NEG(L((9, 18), (9.5, 22))),                                    # no lower-left loop (vs 8)
         NEG(L((19.5, 14.5), (22.5, 15)))]],                            # no bar right of stem (vs 4)
}
SHEARS = [-0.4, -0.2, 0.0, 0.2, 0.4]   # x += s * (y - 14); negative = top leans right
WIDTHS = [0.6, 0.8, 1.0, 1.2]          # x scaled about the centre column
HEIGHTS = [0.85, 1.0]                        # y scaled about the centre row
SHIFTS = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]                      # (dx, dy) translations in pixels
NEG_W = 1.0                            # weight of inhibitory strokes
PEN = 0.9                              # stroke profile sigma in pixels
BLUR = 1.0                             # fixed blur applied to image and template
BIAS = {k: 0.0 for k in range(10)}     # per-class bias, set by hand
FEAT = "orient"                          # "blur" or "orient" (hand-set conv features)
N_ORI = 8                              # orientations of the line detectors
ORI_LONG, ORI_SHORT = 2.0, 0.8         # bar filter sigmas along / across the line
POOL = 2                               # max-pool size and stride for "orient"
CENTER = 0.5                           # subtract this times the mean of all my templates


def stroke_points(stroke, n=200):
    kind, p = stroke
    if kind == "arc":
        cx, cy, rx, ry, t0, t1 = p
        t = np.radians(np.linspace(t0, t1, n))
        return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
    pts = np.array(p, float)
    segs = [np.linspace(pts[i], pts[i + 1], n) for i in range(len(pts) - 1)]
    return np.concatenate(segs)


def _draw(strokes, shear, width, height):
    if not strokes:
        return np.zeros((28, 28))
    pts = np.concatenate([stroke_points(s) for s in strokes])
    x = 14 + (pts[:, 0] - 14) * width + shear * (pts[:, 1] - 14)
    y = 14 + (pts[:, 1] - 14) * height
    yy, xx = np.mgrid[0:28, 0:28]
    d2 = (xx[None] - x[:, None, None]) ** 2 + (yy[None] - y[:, None, None]) ** 2
    return np.exp(-d2.min(0) / (2 * PEN ** 2))


def render(strokes, shear, width, height=1.0, neg=False):
    pos = [s for s in strokes if s[0] != "neg"]
    img = _draw(pos, shear, width, height)
    if neg:
        img = img - NEG_W * _draw([s[1] for s in strokes if s[0] == "neg"], shear, width, height)
    return img


def gauss_blur(img, s):
    """Fixed separable Gaussian blur over the last two axes (a hand-set conv layer)."""
    i = np.arange(28)
    K = np.exp(-(i[:, None] - i[None]) ** 2 / (2 * s ** 2))
    K[np.abs(i[:, None] - i[None]) > 3] = 0
    K /= K.sum(1, keepdims=True)
    return (K @ img @ K.T).astype(np.float32)


def ori_kernels():
    """Fixed bar-shaped line detectors: elongated Gaussian, zero-sum, one per angle."""
    r = np.arange(-3, 4)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    ks = []
    for i in range(int(N_ORI)):
        th = np.pi * i / N_ORI
        u = xx * np.cos(th) + yy * np.sin(th)          # along the line
        v = -xx * np.sin(th) + yy * np.cos(th)         # across the line
        g = np.exp(-u ** 2 / (2 * ORI_LONG ** 2) - v ** 2 / (2 * ORI_SHORT ** 2))
        g2 = np.exp(-u ** 2 / (2 * ORI_LONG ** 2) - v ** 2 / (2 * (3 * ORI_SHORT) ** 2))
        k = g / g.sum() - g2 / g2.sum()                # centre-surround across the line
        ks.append(k)
    return np.array(ks, np.float32)


def features(x):
    """x: (n,28,28) -> (n, d). The fixed, hand-set feature layers."""
    if FEAT == "blur":
        f = gauss_blur(x, BLUR).reshape(len(x), -1)
    else:
        K = ori_kernels()
        xp = np.pad(x, ((0, 0), (3, 3), (3, 3)))
        win = np.lib.stride_tricks.sliding_window_view(xp, (7, 7), axis=(1, 2))  # (n,28,28,7,7)
        r = np.maximum(np.einsum("nhwij,kij->nkhw", win, K, optimize=True), 0)    # conv + ReLU
        P = int(POOL)
        g = 28 // P
        r = r[:, :, :g * P, :g * P].reshape(len(x), len(K), g, P, g, P).max((3, 5))  # max-pool
        f = gauss_blur_n(r, BLUR / P).reshape(len(x), -1)
    return f.astype(np.float32)


def gauss_blur_n(img, s):
    n = img.shape[-1]
    i = np.arange(n)
    K = np.exp(-(i[:, None] - i[None]) ** 2 / (2 * max(s, 1e-3) ** 2))
    K[np.abs(i[:, None] - i[None]) > 3] = 0
    K /= K.sum(1, keepdims=True)
    return K @ img @ K.T


def build_templates():
    T, N, owner = [], [], []
    for k, variants in DIGITS.items():
        for strokes in variants:
            for s in SHEARS:
                for w in WIDTHS:
                    for h in HEIGHTS:
                        base = render(strokes, s, w, h)
                        nb = _draw([q[1] for q in strokes if q[0] == "neg"], s, w, h)
                        for dx, dy in SHIFTS:
                            T.append(np.roll(base, (int(dy), int(dx)), (0, 1)))
                            N.append(np.roll(nb, (int(dy), int(dx)), (0, 1)))
                            owner.append(k)
    # templates = my drawings passed through the same fixed feature layers
    T = np.concatenate([features(np.array(T[i:i + 2000])) for i in range(0, len(T), 2000)])
    N = np.concatenate([features(np.array(N[i:i + 2000])) for i in range(0, len(N), 2000)])
    T = T - T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # remove the shared "ink in the middle" component (computed from my templates, not data)
    cls_mean = np.array([T[np.array(owner) == k].mean(0) for k in range(10)]).mean(0)
    T = T - CENTER * cls_mean
    T = T - T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # inhibitory strokes added after normalisation, scaled to the template's own peak
    T = T - NEG_W * N * T.max(1, keepdims=True)
    return T.astype(np.float32), np.array(owner)


def forward(x, T, owner):
    """x: (n,28,28) in [0,1] -> class scores (n,10)."""
    scores = np.full((len(x), 10), -np.inf, np.float32)
    for a in range(0, len(x), 4000):                            # chunks keep memory small
        xb = features(x[a:a + 4000])
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8  # cosine scale, doesn't change argmax
        s = xb @ T.T                                            # (chunk, n_templates)
        for k in range(10):
            scores[a:a + 4000, k] = s[:, owner == k].max(1) + BIAS[k]
    return scores


def report(x, y, T, owner, name):
    pred = forward(x, T, owner).argmax(1)
    conf = np.zeros((10, 10), int)
    np.add.at(conf, (y, pred), 1)
    print(f"{name}: accuracy {100 * (pred == y).mean():.2f}%  (n={len(y)})")
    print("      " + "".join(f"{p:6d}" for p in range(10)) + "   recall")
    for k in range(10):
        print(f"  {k}: " + "".join(f"{v:6d}" for v in conf[k]) + f"   {100 * conf[k, k] / conf[k].sum():5.1f}%")
    return pred


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    assert split == "train" or "--final" in sys.argv, "test set is only for the final evaluation"
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("="); globals()[k] = v if k == "FEAT" else float(v)
    x, y = mn.load(split)
    T, owner = build_templates()
    print(f"{len(T)} templates")
    report(x, y, T, owner, split)
