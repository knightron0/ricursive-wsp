"""train_controller.py -- SETUP TRAINING (the only gradient use in this project).

Fits the 4-16-16-2 tanh controller to the conventional teacher (sim.teacher) by imitation:
supervised regression on (a) inputs sampled uniformly over the input domain and (b) states
visited by the network itself in closed-loop intact episodes (DAgger-style relabelling by the
teacher). Hand-written backprop + Adam in numpy. Then validates in closed loop.

    python3 train_controller.py            # writes checkpoints/original.json (+ .sha256)
    python3 train_controller.py --out /tmp/check.json   # same weights elsewhere, to compare the hash

The repair phase never calls this file and uses no gradients.
"""
import json, math, os, sys, time
import numpy as np
import sim

OUT = os.path.join(sim.HERE, "checkpoints", "original.json")
TRAIN_SEEDS = range(100000, 100300)      # setup-training episodes, disjoint from every split


def init(rng):
    p = {}
    for (w, b), (o, i) in zip([("W1", "b1"), ("W2", "b2"), ("W3", "b3")], [(16, 4), (16, 16), (2, 16)]):
        p[w] = rng.normal(0, 1 / math.sqrt(i), (o, i))
        p[b] = np.zeros(o)
    return p


def forward(p, X):
    h1 = np.tanh(X @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    y = np.tanh(h2 @ p["W3"].T + p["b3"])
    return h1, h2, y


def grads(p, X, T):
    h1, h2, y = forward(p, X)
    n = len(X)
    d3 = (y - T) * (1 - y * y) * (2 / n)
    g = {"W3": d3.T @ h2, "b3": d3.sum(0)}
    d2 = (d3 @ p["W3"]) * (1 - h2 * h2)
    g["W2"], g["b2"] = d2.T @ h1, d2.sum(0)
    d1 = (d2 @ p["W2"]) * (1 - h1 * h1)
    g["W1"], g["b1"] = d1.T @ X, d1.sum(0)
    return g, float(np.mean((y - T) ** 2))


def uniform_inputs(rng, n):
    d = rng.uniform(0.0, 4.0, n)
    a = rng.uniform(-math.pi, math.pi, n)
    return np.stack([d * np.cos(a) / sim.CFG["dist_scale"], d * np.sin(a) / sim.CFG["dist_scale"],
                     rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], axis=1)


def rollout_inputs(p, seeds):
    X = []
    for s in seeds:
        r = sim.run_episode(sim.nn_policy(p), sim.make_episode(s), record=True)
        X += [row[4:8] for row in r["rows"]]
    return np.array(X, dtype=float)


def fit(p, X, T, rng, steps, lr0):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(X), 1024)
        g, loss = grads(p, X[idx], T[idx])
        lr = lr0 * 0.5 * (1 + math.cos(math.pi * t / steps))
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return loss


def main():
    global OUT
    if len(sys.argv) > 2 and sys.argv[1] == "--out":      # e.g. to check that the hash reproduces
        OUT = sys.argv[2]
    t0 = time.time()
    rng = np.random.default_rng(0)
    p = init(rng)
    X = uniform_inputs(rng, 60000)
    log = []
    for rnd in range(4):
        T = sim.teacher(X)
        loss = fit(p, X, T, rng, 15000 if rnd == 0 else 8000, 3e-3 if rnd == 0 else 1e-3)
        s, _ = sim.evaluate(sim.nn_policy(p), range(100300, 100400))   # setup-validation seeds
        log.append({"round": rnd, "n": len(X), "mse": loss, **s})
        print(f"round {rnd}: n={len(X)} mse={loss:.5f} closed-loop(setup-val) {s}", flush=True)
        X = np.concatenate([X, rollout_inputs(p, TRAIN_SEEDS)])
    h = sim.save_params(p, OUT, {"architecture": "4-16-16-2 tanh", "note": "imitation of sim.teacher; see train_controller.py"})
    open(OUT + ".sha256", "w").write(h + "\n")
    if OUT.startswith(os.path.join(sim.HERE, "checkpoints")):
        json.dump({"log": log, "sha256": h, "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(sim.HERE, "logs", "setup_training.json"), "w"), indent=1)
    print("saved", OUT, "sha256", h, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
