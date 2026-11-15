"""Show my base template (shear 0, width 1) for digit k next to random training examples of k."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np, mn, strokenet as sn
x, y = mn.load("train")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
for k in map(int, sys.argv[1].split(",")):
    tm = [sn.render(v, 0, 1) for v in sn.DIGITS[k]]
    ex = x[rng.choice(np.where(y == k)[0], 9 - len(tm), replace=False)]
    g = np.array(tm + list(ex)).reshape(-1, 14, 2, 14, 2).mean((2, 4))
    print(f"=== {k}: my template(s) then real examples")
    mn.show(g / g.max((1, 2), keepdims=True), per_row=9)
