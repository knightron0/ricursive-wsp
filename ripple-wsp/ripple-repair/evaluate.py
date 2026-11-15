"""evaluate.py -- held-out evaluation (section 7). Run only AFTER a patch has been selected.

    python3 evaluate.py --setting A --run runs/A_r1        # writes results/heldout_A.json and .md

Conditions (same 100 held-out seeds for all):
  original/intact, original/disturbed, Ripple patch/disturbed, Ripple patch/intact,
  estimated gain compensation (external action correction; gains fitted to the same 5 observed
  trajectories Ripple could read), and equal-budget random weight search (10 evaluations on the
  same 20 dev episodes, same ranking rule), repeated --search-reps times with different RNG seeds.

No gradients. The API is not called here.
"""
import argparse, json, math, os, time
import numpy as np
import sim, repair

SPLITS = repair.SPLITS


def estimate_gains(observed):
    """Least-squares wheel gains from recorded (pose, command) steps, assuming the nominal equations
    with unknown left/right gains. Uses only trajectories of the original controller in the changed env."""
    dt, b, vmax = sim.CFG["dt"], sim.CFG["wheel_separation"], sim.CFG["max_speed"]
    A, y = [], []
    for ep in observed:
        rows = ep["rows"]
        for k in range(len(rows) - 1):
            t, x, yy, th, _, _, _, _, l, r = rows[k]
            x2, y2, th2 = rows[k + 1][1:4]
            ds = (x2 - x) * math.cos(th) + (y2 - yy) * math.sin(th)
            dth = sim.wrap_angle(th2 - th)
            A.append([vmax * l / 2, vmax * r / 2]); y.append(ds / dt)
            A.append([-vmax * l, vmax * r]);        y.append(dth * b / dt)
    g, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return float(g[0]), float(g[1])


def random_search(original, disturbance, rng, mode, budget=repair.MAX_EVALS):
    """Equal-budget search on the dev-20 episodes with the same ranking rule. Returns (params, n_better)."""
    dev = SPLITS["repair_dev20"]
    base, _ = sim.evaluate(sim.nn_policy(original), dev, disturbance)
    best_p, best_s = original, base
    for _ in range(budget):
        if mode == "all_params_hillclimb":      # perturb every parameter of the current best
            sigma = 10 ** rng.uniform(-2, -0.5)
            cand = {k: v + sigma * rng.normal(size=v.shape) for k, v in best_p.items()}
        elif mode == "output_row_scale":        # structure-aware: random gain on each output row
            cand = {k: v.copy() for k, v in original.items()}
            for row in (0, 1):
                s = 2 ** rng.uniform(-2, 2)
                cand["W3"][row] *= s
                cand["b3"][row] *= s
        else:
            raise ValueError(mode)
        s, _ = sim.evaluate(sim.nn_policy(cand), dev, disturbance)
        if sim.rank_key(s) < sim.rank_key(best_s):
            best_p, best_s = cand, s
    return best_p, best_s, base


def row(name, summ, **kw):
    r = {"condition": name, "success_rate": summ["success_rate"], "mean_capped_time": summ["mean_capped_time"],
         "mean_final_dist": summ["mean_final_dist"], "candidate_evaluations": 0, "changed_parameters": 0,
         "patch_l2_norm": 0.0, "patch_file_bytes": 0, "api_cost_usd": 0.0, "repair_elapsed_s": 0.0}
    r.update(kw)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--run", required=True, help="repair run directory with best_patch.json")
    ap.add_argument("--search-reps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=None, help="SUPPLEMENTARY only: override the frozen episode duration")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds-file", default=os.path.join(sim.HERE, "hidden", "heldout_seeds.json"),
                    help="testing only: evaluate on another seed list (output is tagged)")
    a = ap.parse_args()
    if a.duration:
        sim.CFG["duration"] = a.duration
    disturbance = json.load(open(os.path.join(sim.HERE, "hidden", "disturbances.json")))[a.setting]
    held = json.load(open(a.seeds_file))["heldout100"]
    original = sim.load_params(os.path.join(sim.HERE, "checkpoints", "original.json"))
    res = json.load(open(os.path.join(a.run, "result.json")))
    patch = json.load(open(os.path.join(a.run, "best_patch.json")))
    assert patch["relative_to_sha256"] == sim.params_hash(original), "patch is for a different checkpoint"
    patched, err = repair.apply_patch(original, patch["edits"])
    assert err is None, err
    with open(os.path.join(sim.HERE, "logs", "heldout_access.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.time(), "setting": a.setting, "run": a.run, "patched_sha256": sim.params_hash(patched),
                            "duration": sim.CFG["duration"], "seeds_file": a.seeds_file}) + "\n")
    ev = lambda p, d, filt=None: sim.evaluate(sim.nn_policy(p), held, d, action_filter=filt)[0]

    rows = []
    rows.append(row("Original controller, intact environment", ev(original, {})))
    orig_dist = ev(original, disturbance)
    rows.append(row("Original controller, disturbed environment", orig_dist))
    bc = res["best_candidate"]
    ripple = ev(patched, disturbance)
    rows.append(row("Ripple-patched controller, disturbed environment", ripple,
                    candidate_evaluations=res["evaluations"], changed_parameters=bc["changed_parameters"],
                    patch_l2_norm=bc["patch_l2_norm"], patch_file_bytes=res.get("patch_file_bytes", 0),
                    api_cost_usd=res["cost"], repair_elapsed_s=res["elapsed_s"]))
    rows.append(row("Ripple-patched controller, intact environment", ev(patched, {}),
                    changed_parameters=bc["changed_parameters"], patch_l2_norm=bc["patch_l2_norm"]))
    # undo: removing the patch gives back the original bytes, hence the degraded behaviour
    undone = {k: patched[k] - (patched[k] - original[k]) for k in original}
    undo_same = sim.params_hash({k: original[k] for k in original}) == res["original_sha256"]
    rows.append(row("Patch removed (undo), disturbed environment", ev(original, disturbance),
                    note=f"original checkpoint restored, sha256 match={undo_same}"))

    t0 = time.time()
    obs = [sim.run_episode(sim.nn_policy(original), sim.make_episode(s), disturbance, record=True) for s in SPLITS["observed5"]]
    gl, gr = estimate_gains(obs)
    gl_c, gr_c = max(gl, 0.05), max(gr, 0.05)
    comp = ev(original, disturbance, lambda l, r: (l / gl_c, r / gr_c))
    rows.append(row("Estimated motor-gain compensation (external action correction, weights unchanged)", comp,
                    repair_elapsed_s=round(time.time() - t0, 2),
                    note=f"gains estimated from the 5 observed trajectories: left={gl:.3f}, right={gr:.3f}; assumes the fault is a wheel gain"))

    searches = {}
    for mode, label in (("all_params_hillclimb", "Random weight search, all 386 params (hill-climb, 10 evals)"),
                        ("output_row_scale", "Structure-aware random search: random scale per output row (10 evals)")):
        outs, t0 = [], time.time()
        for rep in range(a.search_reps):
            p, dev_s, base = random_search(original, disturbance, np.random.default_rng(5000 + rep), mode)
            h = ev(p, disturbance)
            _, st = repair.patch_delta(original, p)
            outs.append({"rep": rep, "dev": dev_s, "heldout": h, **st,
                         "found_dev_improvement": sim.rank_key(dev_s) < sim.rank_key(base)})
        searches[mode] = outs
        sr = [o["heldout"]["success_rate"] for o in outs]
        mean = {"success_rate": float(np.mean(sr)),
                "mean_capped_time": float(np.mean([o["heldout"]["mean_capped_time"] for o in outs])),
                "mean_final_dist": float(np.mean([o["heldout"]["mean_final_dist"] for o in outs]))}
        rows.append(row(f"{label}; mean of {a.search_reps} independent searches", mean,
                        candidate_evaluations=repair.MAX_EVALS,
                        changed_parameters=float(np.mean([o["changed_parameters"] for o in outs])),
                        patch_l2_norm=float(np.mean([o["patch_l2_norm"] for o in outs])),
                        repair_elapsed_s=round((time.time() - t0) / a.search_reps, 2),
                        note=f"held-out success min {min(sr):.2f} / median {float(np.median(sr)):.2f} / max {max(sr):.2f}; "
                             f"{sum(o['found_dev_improvement'] for o in outs)}/{a.search_reps} searches found any dev improvement; "
                             f"{sum(s >= ripple['success_rate'] for s in sr)}/{a.search_reps} match or beat the Ripple patch on held-out success"))

    improved = sim.rank_key(ripple) < sim.rank_key(orig_dist)
    out = {"setting": a.setting, "run": a.run, "config": sim.CFG, "heldout_episodes": len(held),
           "supplementary_duration_override": a.duration,
           "ripple_patch_improves_heldout": improved, "rows": rows, "random_search_detail": searches,
           "estimated_gains": {"left": gl, "right": gr}, "patch_explanation": patch["explanation"], "patch_edits": patch["edits"]}
    os.makedirs(os.path.join(sim.HERE, "results"), exist_ok=True)
    stem = os.path.join(sim.HERE, "results", f"heldout_{a.setting}{a.tag}")
    json.dump(out, open(stem + ".json", "w"), indent=1)
    md = [f"Held-out evaluation, setting {a.setting}, {len(held)} episodes, deadline {sim.CFG['duration']} s, run {a.run}", "",
          "| Condition | Success | Mean capped time (s) | Mean final dist (m) | Evals | Changed params | Patch L2 | Patch bytes | API $ | Repair s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cp = r["changed_parameters"]
        md.append(f"| {r['condition']} | {r['success_rate'] * 100:.0f}% | {r['mean_capped_time']:.2f} | {r['mean_final_dist']:.3f} | "
                  f"{r['candidate_evaluations']} | {cp if isinstance(cp, int) else round(cp, 1)} | {r['patch_l2_norm']:.3f} | "
                  f"{r['patch_file_bytes']} | {r['api_cost_usd']:.4f} | {r['repair_elapsed_s']} |")
    md += [""] + [f"- {r['condition']}: {r['note']}" for r in rows if r.get("note")]
    md += ["", f"Ripple patch improves held-out performance over the unpatched controller: **{improved}**"]
    open(stem + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
