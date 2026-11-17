"""Final network = stroke-template scores (strokenet) + hand-set hole-feature table (holes)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mn, strokenet as sn, holes

# Hand-set penalties: (top hole present, top absent, bottom present, bottom absent), in units of BETA.
HOLE_W = {
    0: (0, -1, 0, -1),
    1: (-2, 0, -2, 0),
    2: (-2, 0, 0, 0),
    3: (-2, 0, -2, 0),
    4: (-1, 0, -2, 0),
    5: (-2, 0, -2, 0),
    6: (-2, 0, 0, -1),
    7: (-2, 0, -2, 0),
    8: (0, -1, 0, -1),
    9: (0, -1, -1, 0),
}
BETA = 0.05
MIN_PIX = 2

# Hand-set bay (concavity) penalties. Columns: open-Left/Right/Up/Down in the top half, then bottom half.
# First tuple: penalty when that bay is present; second: penalty when it is absent. Units of GAMMA.
BAY_W = {         # L_t R_t U_t D_t L_b R_b U_b D_b
    0: ((-2, -2, -1, -2, -2, -2, -2, -2), (0, 0, 0, 0, 0, 0, 0, 0)),
    1: ((-1, -2, -2, -2, -2, -1, -2, -2), (0, 0, 0, 0, 0, 0, 0, 0)),
    2: ((0, 0, -2, -2, 0, 0, -1, 0), (-2, 0, 0, 0, 0, 0, 0, 0)),
    3: ((0, 0, -2, -2, 0, -1, -1, -2), (-2, 0, 0, 0, -2, 0, 0, 0)),
    4: ((-2, 0, 0, -2, 0, -1, -1, -1), (0, 0, -1, 0, 0, 0, 0, 0)),
    5: ((-1, 0, -1, -2, 0, 0, -1, -2), (0, -1, 0, 0, -1, 0, 0, 0)),
    6: ((-2, 0, -1, -2, -2, -1, -1, -2), (0, -1, 0, 0, 0, 0, 0, 0)),
    7: ((0, -2, -1, -1, 0, -1, -2, -1), (0, 0, 0, 0, -1, 0, 0, 0)),
    8: ((-1, 0, -1, -2, 0, 0, -2, -2), (0, 0, 0, 0, 0, 0, 0, 0)),
    9: ((-2, -1, -2, -2, 0, -1, -2, -1), (0, 0, 0, 0, 0, 0, 0, 0)),
}
GAMMA = 0.03
BAY_PIX = 4

# Hand-set per-class offsets (3 was under-predicted, 2/0/8/9 over-predicted on train).
CLASS_BIAS = np.array([-0.015, 0, -0.02, 0.03, 0, 0, 0.005, 0.015, -0.01, -0.01], np.float32)


def bay_scores(B):
    P = (B >= BAY_PIX).astype(np.float32)                              # (n, 8)
    Wp = np.array([BAY_W[k][0] for k in range(10)], np.float32)        # (10, 8)
    Wa = np.array([BAY_W[k][1] for k in range(10)], np.float32)
    return GAMMA * (P @ Wp.T + (1 - P) @ Wa.T)


def hole_scores(H):
    top, bot = H[:, 0] >= MIN_PIX, H[:, 2] >= MIN_PIX
    W = np.array([HOLE_W[k] for k in range(10)], np.float32)          # (10, 4)
    F = np.stack([top, ~top, bot, ~bot], 1).astype(np.float32)        # (n, 4)
    return BETA * F @ W.T


def scores(x, cache=None):
    if cache and os.path.exists(cache):
        S = np.load(cache)
    else:
        T, o = sn.build_templates(); S = sn.forward(x, T, o)
        if cache: np.save(cache, S)
    return S, holes.hole_features(x), holes.bay_features(x)


if __name__ == "__main__":
    split = sys.argv[1]
    assert split == "train" or "--final" in sys.argv, "test set is only for the final evaluation"
    x, y = mn.load(split)
    S, H, B = scores(x, None)
    grid = lambda name, d: [float(a.split("=")[1]) for a in sys.argv if a.startswith(name + "=")] or [d]
    for b in grid("BETA", BETA):
        for g in grid("GAMMA", GAMMA):
            BETA, GAMMA = b, g
            p = (S + hole_scores(H) + bay_scores(B) + CLASS_BIAS).argmax(1)
            print(f"{split}: BETA={b} GAMMA={g}: accuracy {100 * (p == y).mean():.2f}%")
    conf = np.zeros((10, 10), int); np.add.at(conf, (y, p), 1)
    for k in range(10):
        print(f"  {k}: " + "".join(f"{v:6d}" for v in conf[k]) + f"   {100 * conf[k, k] / conf[k].sum():5.1f}%")
