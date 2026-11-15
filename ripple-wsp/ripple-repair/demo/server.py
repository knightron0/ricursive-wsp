"""demo/server.py -- local server for the Ripple Repair browser demo (standard library only).

    export OPENROUTER_API_KEY=...          # only needed for live "Diagnose and repair"
    python3 demo/server.py                 # then open http://127.0.0.1:8765

All rover motion comes from sim.py running the 4-16-16-2 network (original, or original + patch).
Ripple is called only inside a repair job, through repair.run_repair -- never per simulation step.
The damage chosen with the slider is passed to the simulator; it is not sent to Ripple.
A live patch belongs to the slider setting it was made for and is dropped when the setting changes.
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sim, repair, ripple_api   # noqa: E402

ORIGINAL = sim.load_params(os.path.join(sim.HERE, "checkpoints", "original.json"))
BOARD_SEEDS = list(range(4000, 4050))     # scoreboard episodes for the demo (not the held-out split)
LOCK = threading.Lock()
STATE = {"patch": None,      # {"edits", "explanation", "setting": {side, gain}, "source": "live"|"recorded", "candidate"}
         "job": None}        # {"id", "setting", "events": [], "done": bool, "result": {...}}


def disturbance(side, gain):
    gain = min(1.0, max(0.2, float(gain)))
    return {("right_gain" if side == "right" else "left_gain"): round(gain, 3)}


def same_setting(a, b):
    return a is not None and b is not None and a["side"] == b["side"] and abs(a["gain"] - b["gain"]) < 1e-9


def traj(params, ep, dist):
    r = sim.run_episode(sim.nn_policy(params), ep, dist, record=True)
    return {"success": r["success"], "time": r["time"], "final_dist": round(r["final_dist"], 3),
            "rows": [[round(v, 4) for v in row[:4]] + [None if row[8] is None else round(row[8], 3),
                                                        None if row[9] is None else round(row[9], 3)] for row in r["rows"]]}


def board(params, dist):
    s, _ = sim.evaluate(sim.nn_policy(params), BOARD_SEEDS, dist)
    return {"success_rate": s["success_rate"], "mean_capped_time": round(s["mean_capped_time"], 2),
            "mean_final_dist": round(s["mean_final_dist"], 3), "episodes": s["n"]}


def playback(body):
    setting = {"side": body.get("side", "left"), "gain": round(float(body.get("gain", 1.0)), 3)}
    dist = disturbance(setting["side"], setting["gain"])
    ep = sim.make_episode(int(body.get("seed", 7000)))
    with LOCK:
        if STATE["patch"] and not same_setting(STATE["patch"]["setting"], setting):
            STATE["patch"] = None          # a patch is only valid for the setting it was made for
        patch = STATE["patch"]
    out = {"setting": setting, "episode": ep, "config": {k: sim.CFG[k] for k in ("dt", "duration", "success_radius")},
           "intact": traj(ORIGINAL, ep, {}), "disturbed": traj(ORIGINAL, ep, dist), "repaired": None, "patch": None,
           "scoreboard": {"intact": board(ORIGINAL, {}), "disturbed": board(ORIGINAL, dist), "repaired": None,
                          "repaired_in_intact_env": None}}
    if patch:
        params, err = repair.apply_patch(ORIGINAL, patch["edits"])
        if err is None:
            out["repaired"] = traj(params, ep, dist)
            out["scoreboard"]["repaired"] = board(params, dist)
            out["scoreboard"]["repaired_in_intact_env"] = board(params, {})
            _, st = repair.patch_delta(ORIGINAL, params)
            out["patch"] = {"source": patch["source"], "candidate": patch.get("candidate"), "explanation": patch["explanation"],
                            "edits": patch["edits"], **st}
    return out


def start_repair(body):
    setting = {"side": body.get("side", "left"), "gain": round(float(body.get("gain", 1.0)), 3)}
    with LOCK:
        if STATE["job"] and not STATE["job"]["done"]:
            return {"error": "a repair is already running"}
        job = {"id": f"live_{time.strftime('%Y%m%d_%H%M%S')}", "setting": setting, "events": [], "done": False, "result": None}
        STATE["job"] = job
        STATE["patch"] = None
    out_dir = os.path.join(sim.HERE, "runs", "demo_" + job["id"])

    def work():
        try:
            env = repair.RepairEnv(ORIGINAL, disturbance(setting["side"], setting["gain"]))

            def on_event(ev):
                job["events"].append(ev)
                # make the best-so-far patch playable as soon as it beats the original on the dev episodes
                if ev["kind"] == "candidate_evaluated" and ev.get("is_best") and sim.rank_key(env.best) < sim.rank_key(env.baseline):
                    with LOCK:
                        if STATE["job"] is job:
                            STATE["patch"] = {"edits": env.best["edits"], "explanation": env.best["explanation"],
                                              "setting": setting, "source": "live", "candidate": env.best["candidate"]}

            res = repair.run_repair(env, out_dir, on_event=on_event, run_name="demo_" + job["id"])
            json.dump({"setting_chosen_in_ui": setting}, open(os.path.join(out_dir, "demo_setting.json"), "w"))
            job["result"] = res
            if env.best is not None and res["best_beats_original_on_dev"]:
                with LOCK:
                    STATE["patch"] = {"edits": env.best["edits"], "explanation": env.best["explanation"],
                                      "setting": setting, "source": "live", "candidate": env.best["candidate"]}
        except Exception as e:  # shown in the UI log
            job["events"].append({"kind": "error", "text": f"{type(e).__name__}: {e}"})
        job["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return {"job": job["id"], "out_dir": os.path.relpath(out_dir, sim.HERE)}


def load_recorded():
    path = os.path.join(HERE, "recorded", "recorded.json")
    if not os.path.exists(path):
        return {"error": "no recorded run"}
    rec = json.load(open(path))
    with LOCK:
        STATE["patch"] = {"edits": rec["patch"]["edits"], "explanation": rec["patch"]["explanation"],
                          "setting": rec["setting"], "source": "recorded", "candidate": rec["patch"].get("candidate")}
    return {"setting": rec["setting"], "events": rec["events"], "recorded_at": rec.get("recorded_at"), "run": rec.get("run")}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(open(os.path.join(HERE, "index.html"), "rb").read(), ctype="text/html; charset=utf-8")
        if path == "/recorded.js":
            p = os.path.join(HERE, "recorded", "recorded.js")
            return self.send(open(p, "rb").read() if os.path.exists(p) else b"", ctype="application/javascript")
        if path == "/api/info":
            try:
                ripple_api.api_key(); key = True
            except Exception:
                key = False
            job, patch = STATE["job"], STATE["patch"]
            active = patch["setting"] if patch else job["setting"] if job and not job["done"] else None
            return self.send({"live": True, "api_key": key, "active_setting": active, "spent_usd": round(ripple_api.spent(), 4), "cap_usd": ripple_api.CAP,
                              "original_sha256": sim.params_hash(ORIGINAL), "config": sim.CFG,
                              "recorded": os.path.exists(os.path.join(HERE, "recorded", "recorded.json"))})
        if path == "/api/repair/status":
            job = STATE["job"]
            if not job:
                return self.send({"job": None})
            return self.send({"job": job["id"], "setting": job["setting"], "events": job["events"], "done": job["done"],
                              "result": job["result"]})
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/playback":
                return self.send(playback(body))
            if self.path == "/api/repair":
                return self.send(start_repair(body))
            if self.path == "/api/undo":
                with LOCK:
                    STATE["patch"] = None
                return self.send({"undone": True, "controller_sha256": sim.params_hash(ORIGINAL)})
            if self.path == "/api/recorded":
                return self.send(load_recorded())
        except Exception as e:
            return self.send({"error": f"{type(e).__name__}: {e}"}, 500)
        self.send({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print(f"Ripple Repair demo: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
