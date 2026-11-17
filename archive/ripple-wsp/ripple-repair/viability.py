"""viability.py -- section 4: intact performance, motor-gain sweep, conventional compensation.

    python3 viability.py [--durations 20 9 8]     # writes logs/viability.json

Uses the 100 development seeds only (splits.json: dev100). The compensation here is the
oracle version (true gain) -- it only shows the failure is physically recoverable; the
baseline in evaluate.py estimates the gain from observed trajectories instead.
"""
import argparse, json, os
import numpy as np
import sim

DEV100 = list(range(1000, 1100))


def comp_filter(lg, rg):
    return lambda l, r: (l / lg, r / rg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", type=float, nargs="*", default=[sim.CFG["duration"]])
    ap.add_argument("--ckpt", default=os.path.join(sim.HERE, "checkpoints", "original.json"))
    a = ap.parse_args()
    p = sim.load_params(a.ckpt)
    pol = sim.nn_policy(p)
    full = sim.CFG["duration"]
    sim.CFG["duration"] = max(a.durations + [full])
    out = []
    conds = [("intact", {}, "nn"), ("intact", {}, "teacher")]
    for side in ("left", "right"):
        for g in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
            conds.append((f"{side}_gain={g}", {f"{side}_gain": g}, "nn"))
            conds.append((f"{side}_gain={g}", {f"{side}_gain": g}, "nn+oracle_comp"))
    print(f"{'condition':<16}{'controller':<16}" + "".join(f"succ@{d:<5g}" for d in a.durations)
          + f"{'mean_t(succ)':>13}{'p95_t':>7}{'max_t':>7}{'final_d@max':>12}")
    for name, dist, ctrl in conds:
        g = {**sim.INTACT, **dist}
        filt = comp_filter(g["left_gain"], g["right_gain"]) if ctrl == "nn+oracle_comp" else None
        policy = sim.teacher if ctrl == "teacher" else pol
        _, rs = sim.evaluate(policy, DEV100, dist, action_filter=filt)
        t = np.array([r["time"] if r["success"] else np.inf for r in rs])
        row = {"condition": name, "controller": ctrl,
               "success_by_duration": {str(d): float(np.mean(t <= d)) for d in a.durations},
               "mean_capped_time_by_duration": {str(d): float(np.mean(np.minimum(t, d))) for d in a.durations},
               "mean_time_successes": float(np.mean(t[np.isfinite(t)])) if np.isfinite(t).any() else None,
               "p95_time": float(np.percentile(np.minimum(t, sim.CFG["duration"]), 95)),
               "max_time": float(np.max(np.minimum(t, sim.CFG["duration"]))),
               "mean_final_dist": float(np.mean([r["final_dist"] for r in rs]))}
        out.append(row)
        print(f"{name:<16}{ctrl:<16}" + "".join(f"{row['success_by_duration'][str(d)]:<10.2f}" for d in a.durations)
              + f"{row['mean_time_successes'] or float('nan'):>13.2f}{row['p95_time']:>7.2f}{row['max_time']:>7.2f}{row['mean_final_dist']:>12.3f}")
    sim.CFG["duration"] = full
    json.dump({"config": sim.CFG, "durations": a.durations, "seeds": "1000..1099", "rows": out},
              open(os.path.join(sim.HERE, "logs", "viability.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
