"""Hole detector as a fixed recurrent conv net: flood-fill the background from the border.
Forward pass only; no learned parameters."""
import numpy as np

INK_T = 0.3        # binarisation threshold
ITERS = 40         # flood-fill steps (enough to reach every border-connected pixel)
BANDS = (12, 16)   # rows < 12 = top band, 12..15 = middle, >= 16 = bottom


def maxpool3(m):
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    out = m.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            if abs(dy - 1) + abs(dx - 1) == 2:   # 4-connected spread only, so diagonal gaps don't leak
                continue
            out |= p[:, dy:dy + 28, dx:dx + 28]
    return out


def hole_map(x):
    bg = x < INK_T
    reach = np.zeros_like(bg)
    reach[:, 0, :] = reach[:, -1, :] = reach[:, :, 0] = reach[:, :, -1] = True
    reach &= bg
    for _ in range(ITERS):
        reach = maxpool3(reach) & bg
    return bg & ~reach


def hole_features(x):
    """-> (n, 3) hole pixel counts in the top, middle and bottom bands."""
    out = []
    for a in range(0, len(x), 5000):
        h = hole_map(x[a:a + 5000])
        t, b = BANDS
        out.append(np.stack([h[:, :t].sum((1, 2)), h[:, t:b].sum((1, 2)), h[:, b:].sum((1, 2))], 1))
    return np.concatenate(out).astype(np.float32)


BAY_MID = 14       # top half = rows < 14, bottom half = rows >= 14


def bay_features(x):
    """Concavity features via fixed ray casts (cumulative max along each direction).
    -> (n, 8) counts of background pixels blocked on exactly three sides, by open side
       (L, R, U, D) and half (top, bottom): [L_t, R_t, U_t, D_t, L_b, R_b, U_b, D_b]."""
    out = []
    for a in range(0, len(x), 5000):
        ink = x[a:a + 5000] >= INK_T
        cm = np.maximum.accumulate
        left = np.concatenate([np.zeros_like(ink[:, :, :1]), cm(ink, 2)[:, :, :-1]], 2)       # ink somewhere to the left
        right = np.concatenate([cm(ink[:, :, ::-1], 2)[:, :, ::-1][:, :, 1:], np.zeros_like(ink[:, :, :1])], 2)
        up = np.concatenate([np.zeros_like(ink[:, :1]), cm(ink, 1)[:, :-1]], 1)
        down = np.concatenate([cm(ink[:, ::-1], 1)[:, ::-1][:, 1:], np.zeros_like(ink[:, :1])], 1)
        bg = ~ink
        n_hit = left.astype(int) + right + up + down
        three = bg & (n_hit == 3)
        feats = []
        for half in (slice(0, BAY_MID), slice(BAY_MID, 28)):
            for blocked in (left, right, up, down):
                feats.append((three & ~blocked)[:, half].sum((1, 2)))   # open towards that side
        out.append(np.stack(feats, 1))
    return np.concatenate(out).astype(np.float32)
