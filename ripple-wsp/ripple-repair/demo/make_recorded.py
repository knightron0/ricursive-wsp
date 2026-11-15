"""demo/make_recorded.py -- freeze a finished repair run as the presentation backup.

    python3 demo/make_recorded.py --run runs/A_r1 --setting A

Writes demo/recorded/recorded.json (used by the server's "Load recorded run") and
demo/recorded/recorded.js (lets demo/index.html play the run when opened as a file, with no
server). Both are labelled RECORDED in the page. Trajectories are produced by sim.py with the
recorded patch; nothing is hand-edited.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sim, repair   # noqa: E402
import server        # noqa: E402  (reuses traj/board so the recorded view matches the live one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--setting", required=True)
    a = ap.parse_args()
    dist = json.load(open(os.path.join(sim.HERE, "hidden", "disturbances.json")))[a.setting]
    (key, gain), = dist.items()
    setting = {"side": key.split("_")[0], "gain": gain}
    patch = json.load(open(os.path.join(a.run, "best_patch.json")))
    events = [{k: v for k, v in json.loads(l).items() if k not in ("type", "t")}
              for l in open(os.path.join(a.run, "transcript.jsonl")) if '"type": "event"' in l]
    for e in events:       # display only: drop scratchpad text the provider leaked before "</think>"
        for k in ("text", "final_report"):
            if isinstance(e.get(k), str) and "</think>" in e[k]:
                e[k] = e[k].rsplit("</think>", 1)[1].strip()
    final = open(os.path.join(a.run, "final.md")).read()      # full final answer (the event keeps only a prefix)
    for e in events:
        if e.get("kind") == "end" and final.strip():
            e["final_report"] = final.rsplit("</think>", 1)[-1].strip()
    recorded_at = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(os.path.getmtime(os.path.join(a.run, "result.json"))))
    rec = {"recorded_at": recorded_at, "run": a.run, "setting": setting, "events": events,
           "patch": {"edits": patch["edits"], "explanation": patch["explanation"], "candidate": patch["candidate"]}}
    os.makedirs(os.path.join(HERE, "recorded"), exist_ok=True)
    json.dump(rec, open(os.path.join(HERE, "recorded", "recorded.json"), "w"), indent=1)

    params, err = repair.apply_patch(server.ORIGINAL, patch["edits"])
    assert err is None, err
    _, st = repair.patch_delta(server.ORIGINAL, params)
    cfg = {k: sim.CFG[k] for k in ("dt", "duration", "success_radius")}
    playbacks = []
    for seed in range(7000, 7008):
        ep = sim.make_episode(seed)
        playbacks.append({"episode": ep, "config": cfg, "intact": server.traj(server.ORIGINAL, ep, {}),
                          "disturbed": server.traj(server.ORIGINAL, ep, dist), "repaired": server.traj(params, ep, dist)})
    standalone = {**rec, "playbacks": playbacks,
                  "patch_info": {"source": "recorded", "candidate": patch["candidate"], **st},
                  "scoreboard": {"intact": server.board(server.ORIGINAL, {}), "disturbed": server.board(server.ORIGINAL, dist),
                                 "repaired": server.board(params, dist), "repaired_in_intact_env": server.board(params, {})}}
    with open(os.path.join(HERE, "recorded", "recorded.js"), "w") as f:
        f.write("window.RECORDED = " + json.dumps(standalone) + ";\n")
    print("wrote demo/recorded/recorded.json and recorded.js for", setting, "from", a.run)


if __name__ == "__main__":
    main()
