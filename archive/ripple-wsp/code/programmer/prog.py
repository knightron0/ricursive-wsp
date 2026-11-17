#!/usr/bin/env python3
"""P1/P2: Ripple as weight programmer on tiny ReLU MLPs. No tools, no training, one reply per task.

usage: prog.py p1|p2 [--model ricursiveai/ripple-65b-preview] [--reps 3]
Writes results/<rung>/<task>_<rep>.json (prompt, reply, parsed weights, score).
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SECRET = os.path.expanduser("~/.lor/extensions/secure-paste/secret")


def key():
    k = os.environ.get("OPENROUTER_API_KEY") or subprocess.run([SECRET, "OPENROUTER_API_KEY"], capture_output=True, text=True).stdout.strip()
    if not k:
        sys.exit("no OPENROUTER_API_KEY")
    return k


def chat(model, prompt, max_tokens=32000):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=1200))
            if "choices" in d:
                m = d["choices"][0]["message"]
                return m.get("content") or "", m.get("reasoning") or "", d.get("usage", {})
            err = d
        except Exception as e:  # network / 5xx: retry
            err = repr(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat failed: {err}")


def forward(w, X):
    h = np.maximum(0, X @ np.array(w["W1"], float).T + np.array(w["b1"], float))
    return h @ np.array(w["W2"], float).T + np.array(w["b2"], float)


def parse(reply):
    """Last JSON object in the reply that has W1,b1,W2,b2."""
    for m in reversed(list(re.finditer(r"\{", reply))):
        try:
            obj, _ = json.JSONDecoder().raw_decode(reply[m.start():])
            if all(k in obj for k in ("W1", "b1", "W2", "b2")):
                return obj
        except Exception:
            pass
    return None


FMT = """The network is a one-hidden-layer ReLU MLP: y = W2 @ relu(W1 @ x + b1) + b2.
W1 has shape [hidden, n_in], b1 [hidden], W2 [n_out, hidden], b2 [n_out].
Reply with your reasoning, then finish with one JSON object with exactly the keys W1, b1, W2, b2 (nested lists of numbers)."""

rng = np.random.default_rng(0)
# name -> (n_in, n_out, spec, sampler of unseen test inputs, target fn)
TASKS = {
    "abs": (1, 1, "y = |x| for x in [-5, 5]", lambda n: rng.uniform(-5, 5, (n, 1)), lambda X: np.abs(X)),
    "clamp": (1, 1, "y = min(max(x, 0), 1) for x in [-5, 5]", lambda n: rng.uniform(-5, 5, (n, 1)), lambda X: np.clip(X, 0, 1)),
    "max2": (2, 1, "y = max(x1, x2) for x1, x2 in [-5, 5]", lambda n: rng.uniform(-5, 5, (n, 2)), lambda X: X.max(1, keepdims=True)),
    "xor": (2, 1, "y = XOR(x1, x2) for x1, x2 in {0, 1}", lambda n: rng.integers(0, 2, (n, 2)).astype(float), lambda X: (X[:, :1] != X[:, 1:]).astype(float)),
    "sort2": (2, 2, "y = (min(x1, x2), max(x1, x2)) for x1, x2 in [-5, 5]", lambda n: rng.uniform(-5, 5, (n, 2)), lambda X: np.sort(X, 1)),
    "median3": (3, 1, "y = median(x1, x2, x3) for inputs in [0, 5] (you may need a wider hidden layer; exactness matters)", lambda n: rng.uniform(0, 5, (n, 3)), lambda X: np.median(X, 1, keepdims=True)),
    "band": (1, 1, "y = 1 if 1 <= x <= 2 else 0, for x in [-5, 5]; inputs within 0.05 of the edges 1 and 2 are not tested", lambda n: (lambda x: x[(np.abs(x - 1) > .05) & (np.abs(x - 2) > .05)][:, None])(rng.uniform(-5, 5, 2 * n)), lambda X: ((X >= 1) & (X <= 2)).astype(float)),
}
TOL = 0.05


def score(w, name):
    n_in, n_out, _, sample, f = TASKS[name]
    try:
        X = sample(2000)
        err = np.abs(forward(w, X).reshape(len(X), -1) - f(X))
        return {"max_err": float(err.max()), "frac_ok": float((err.max(1) < TOL).mean()), "pass": bool(err.max() < TOL)}
    except Exception as e:
        return {"error": repr(e), "pass": False}


def p1_prompt(name):
    n_in, n_out, spec, _, _ = TASKS[name]
    return f"Write the weights of a neural network by hand. No training, no code execution.\n\nTarget: {spec}. Inputs: {n_in}. Outputs: {n_out}. Use at most 16 hidden units.\n{FMT}\nThe network is scored on unseen inputs; it passes if every output is within {TOL} of the target."


# P2: correct hand-written nets, hidden units permuted and rescaled, then one fault injected.
GOLD = {
    "abs": {"W1": [[1], [-1]], "b1": [0, 0], "W2": [[1, 1]], "b2": [0]},
    "clamp": {"W1": [[1], [1]], "b1": [0, -1], "W2": [[1, -1]], "b2": [0]},
    "max2": {"W1": [[1, -1], [0, 1], [0, -1]], "b1": [0, 0, 0], "W2": [[1, 1, -1]], "b2": [0]},
    "sort2": {"W1": [[1, -1], [0, 1], [0, -1], [1, 0], [-1, 0]], "b1": [0] * 5, "W2": [[-1, 0, 0, 1, -1], [1, 1, -1, 0, 0]], "b2": [0, 0]},
}
FAULTS = [("abs", "sign"), ("clamp", "bias"), ("max2", "sign"), ("max2", "zero"), ("sort2", "sign"), ("sort2", "scale")]


def obfuscate(w, seed):
    r = np.random.default_rng(seed)
    W1, b1, W2 = np.array(w["W1"], float), np.array(w["b1"], float), np.array(w["W2"], float)
    # pad with two dead/distractor units, rescale each unit by s (and W2 by 1/s), permute
    h = len(b1)
    W1 = np.vstack([W1, r.normal(0, .3, (2, W1.shape[1]))]); b1 = np.concatenate([b1, [-9, -9]]); W2 = np.hstack([W2, r.normal(0, .5, (W2.shape[0], 2))])
    s = r.uniform(.5, 3, h + 2); W1 *= s[:, None]; b1 *= s; W2 /= s[None, :]
    p = r.permutation(h + 2)
    return {"W1": W1[p].round(4).tolist(), "b1": b1[p].round(4).tolist(), "W2": W2[:, p].round(4).tolist(), "b2": w["b2"]}, h, p


def inject(w, kind, seed, h, p):
    r = np.random.default_rng(seed + 1)
    w = json.loads(json.dumps(w))
    live = [i for i, src in enumerate(p) if src < h]  # only break a live unit
    i = int(r.choice(live))
    if kind == "sign":
        j = int(np.argmax(np.abs(w["W1"][i]))); w["W1"][i][j] *= -1; where = f"W1[{i}][{j}] sign flipped"
    elif kind == "zero":
        o = int(np.argmax(np.abs(np.array(w["W2"])[:, i]))); w["W2"][o][i] = 0.0; where = f"W2[{o}][{i}] zeroed"
    elif kind == "scale":
        o = int(np.argmax(np.abs(np.array(w["W2"])[:, i]))); w["W2"][o][i] = round(w["W2"][o][i] * 2.5, 4); where = f"W2[{o}][{i}] x2.5"
    elif kind == "bias":
        i = int(np.argmax(np.abs(w["b1"][:]) * (np.array(w["b1"]) > -5))); w["b1"][i] = round(w["b1"][i] * 0.3, 4); where = f"b1[{i}] x0.3"
    return w, where


def p2_prompt(name, broken):
    n_in, n_out, spec, sample, f = TASKS[name]
    X = sample(6).round(2)
    ex = "\n".join(f"  x={x.tolist()}  network={np.round(y, 3).tolist()}  target={np.round(t, 3).tolist()}" for x, y, t in zip(X, forward(broken, X).reshape(len(X), -1), f(X)))
    return (f"A neural network that used to work has exactly one corrupted weight. Find it by reading the weights and repair it. No training, no code execution.\n\n"
            f"Intended function: {spec}.\n{FMT}\n\nCurrent weights:\n{json.dumps(broken)}\n\nA few of its current outputs:\n{ex}\n\n"
            f"Change as few weights as possible. State which weight was corrupted. The repaired network is scored on unseen inputs (tolerance {TOL}).")


def n_changed(a, b):
    try:
        return int(sum((np.abs(np.array(a[k], float) - np.array(b[k], float)) > 1e-9).sum() for k in ("W1", "b1", "W2", "b2")))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rung"); ap.add_argument("--model", default="ricursiveai/ripple-65b-preview"); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", a.rung, a.model.replace("/", "_")); os.makedirs(out, exist_ok=True)
    jobs = []
    if a.rung == "p1":
        for name in TASKS:
            for rep in range(a.reps):
                jobs.append((f"{name}_{rep}", name, p1_prompt(name), None, None))
    else:
        for fi, (name, kind) in enumerate(FAULTS):
            ob, h, p = obfuscate(GOLD[name], 100 + fi)
            assert score(ob, name)["pass"], name
            broken, where = inject(ob, kind, 100 + fi, h, p)
            assert not score(broken, name)["pass"], (name, kind)
            for rep in range(a.reps):
                jobs.append((f"{name}-{kind}_{rep}", name, p2_prompt(name, broken), broken, where))
    if a.dry:
        print(jobs[0][2]); print(len(jobs), "jobs"); return
    from concurrent.futures import ThreadPoolExecutor

    def run(job):
        jid, name, prompt, broken, where = job
        path = os.path.join(out, jid + ".json")
        if os.path.exists(path):
            return json.load(open(path))
        try:
            reply, reasoning, usage = chat(a.model, prompt)
        except Exception as e:
            return {"id": jid, "task": name, "score": {"pass": False, "error": repr(e)}, "api_error": True}
        w = parse(reply)
        rec = {"id": jid, "task": name, "prompt": prompt, "reply": reply, "reasoning": reasoning, "usage": usage, "weights": w,
               "score": score(w, name) if w else {"pass": False, "error": "no weights parsed"}}
        if broken is not None:
            rec["fault"] = where; rec["n_changed"] = n_changed(w, broken) if w else None
        json.dump(rec, open(path, "w"), indent=1)
        return rec

    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(run, jobs))
    for r in recs:
        s = r["score"]
        print(f"{r['id']:16s} pass={s['pass']!s:5s} max_err={s.get('max_err', float('nan')):.3g} " + (f"changed={r.get('n_changed')} fault={r.get('fault')}" if "fault" in r else "") + (" API_ERROR" if r.get("api_error") else "") + (f" {s['error']}" if "error" in s else ""))
    ok = [r for r in recs if not r.get("api_error")]
    print(f"\n{a.rung} {a.model}: {sum(r['score']['pass'] for r in ok)}/{len(ok)} passed ({len(recs) - len(ok)} API errors excluded)")


if __name__ == "__main__":
    main()
