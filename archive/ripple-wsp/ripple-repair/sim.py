"""sim.py -- deterministic differential-drive rover simulator, the 4-16-16-2 tanh controller,
and the conventional teacher. numpy only. All settings come from config.json.

The disturbance (a dict such as {"left_gain": 0.5}) is an argument of the simulator; the repair
harness reads it from hidden/ and never shows it to the model.
"""
import hashlib, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
SHAPES = {"W1": (16, 4), "b1": (16,), "W2": (16, 16), "b2": (16,), "W3": (2, 16), "b3": (2,)}
INTACT = {"left_gain": 1.0, "right_gain": 1.0}


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_episode(seed):
    """Rover starts at the origin with a uniformly random heading; goal at U[dmin,dmax] in a random direction."""
    r = np.random.default_rng(int(seed))
    theta = float(r.uniform(-math.pi, math.pi))
    d = float(r.uniform(CFG["goal_dist_min"], CFG["goal_dist_max"]))
    phi = float(r.uniform(-math.pi, math.pi))
    return {"seed": int(seed), "x": 0.0, "y": 0.0, "theta": theta,
            "gx": d * math.cos(phi), "gy": d * math.sin(phi)}


def observe(x, y, theta, gx, gy, prev_l, prev_r):
    """Network inputs: goal offset in rover coordinates / dist_scale, previous commands."""
    ex, ey = gx - x, gy - y
    c, s = math.cos(theta), math.sin(theta)
    return np.array([(c * ex + s * ey) / CFG["dist_scale"], (-s * ex + c * ey) / CFG["dist_scale"],
                     prev_l, prev_r])


# ------------------------------------------------------------------ controllers
def mlp(params, obs):
    h1 = np.tanh(params["W1"] @ obs + params["b1"])
    h2 = np.tanh(params["W2"] @ h1 + params["b2"])
    return np.tanh(params["W3"] @ h2 + params["b3"])


def teacher(obs):
    """Conventional target seeker: steer proportionally to the bearing error, drive forward at cruise
    speed scaled by max(0, cos(bearing)) (so it turns in place when facing away), then low-pass with
    the previous command. Works on arrays of shape (..., 4)."""
    t = CFG["teacher"]
    obs = np.asarray(obs, dtype=float)
    dx, dy, pl, pr = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3]
    alpha = np.arctan2(dy, dx)
    v = t["cruise"] * np.maximum(0.0, np.cos(alpha))
    s = np.clip(t["steer_gain"] * alpha, -t["steer_max"], t["steer_max"])
    a = t["smooth_prev"]
    return np.stack([(1 - a) * (v - s) + a * pl, (1 - a) * (v + s) + a * pr], axis=-1)


def load_params(path):
    d = json.load(open(path))
    return {k: np.array(d[k], dtype=float) for k in SHAPES}


def save_params(params, path, extra=None):
    d = {k: np.asarray(params[k]).tolist() for k in SHAPES}
    d.update(extra or {})
    with open(path, "w") as f:
        json.dump(d, f)
    return params_hash(params)


def params_hash(params):
    h = hashlib.sha256()
    for k in SHAPES:
        h.update(np.asarray(params[k], dtype="<f8").tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ simulation
def run_episode(policy, ep, disturbance=None, record=False, action_filter=None):
    """policy(obs)->(left,right). action_filter(left,right)->(left,right) is an optional external
    action correction (used only by the gain-compensation baseline). Returns a result dict."""
    g = {**INTACT, **(disturbance or {})}
    dt, b, vmax = CFG["dt"], CFG["wheel_separation"], CFG["max_speed"]
    steps = int(round(CFG["duration"] / dt))
    x, y, th, gx, gy = ep["x"], ep["y"], ep["theta"], ep["gx"], ep["gy"]
    pl = pr = 0.0
    rows, success, t_done = [], False, CFG["duration"]
    for k in range(steps):
        o = observe(x, y, th, gx, gy, pl, pr)
        a = np.clip(np.asarray(policy(o), dtype=float), -1.0, 1.0)
        l, r = float(a[0]), float(a[1])
        if record:
            rows.append([round(k * dt, 4), x, y, th, o[0], o[1], o[2], o[3], l, r])
        pl, pr = l, r                      # the network sees its own previous outputs
        if action_filter is not None:
            l, r = action_filter(l, r)
            l, r = max(-1.0, min(1.0, l)), max(-1.0, min(1.0, r))
        vl = vmax * g["left_gain"] * l
        vr = vmax * g["right_gain"] * r
        v, w = (vl + vr) / 2, (vr - vl) / b
        x += dt * v * math.cos(th)
        y += dt * v * math.sin(th)
        th = wrap_angle(th + dt * w)
        if math.hypot(gx - x, gy - y) <= CFG["success_radius"]:
            success, t_done = True, round((k + 1) * dt, 4)
            break
    if record:  # final pose row (no action)
        o = observe(x, y, th, gx, gy, pl, pr)
        rows.append([round(t_done if success else steps * dt, 4), x, y, th, o[0], o[1], o[2], o[3], None, None])
    res = {"seed": ep["seed"], "success": success, "time": t_done,
           "final_dist": math.hypot(gx - x, gy - y), "goal": [gx, gy]}
    if record:
        res["rows"] = rows   # columns: t, x, y, theta, in_dx, in_dy, in_prev_l, in_prev_r, cmd_l, cmd_r
    return res


TRAJ_COLUMNS = ["t", "x", "y", "theta", "in_dx", "in_dy", "in_prev_l", "in_prev_r", "cmd_l", "cmd_r"]


def evaluate(policy, seeds, disturbance=None, action_filter=None):
    rs = [run_episode(policy, make_episode(s), disturbance, action_filter=action_filter) for s in seeds]
    return summarize(rs), rs


def summarize(rs):
    n = len(rs)
    return {"n": n,
            "success_rate": sum(r["success"] for r in rs) / n,
            "mean_capped_time": float(np.mean([r["time"] for r in rs])),
            "mean_final_dist": float(np.mean([r["final_dist"] for r in rs]))}


def nn_policy(params):
    return lambda o: mlp(params, o)


def rank_key(summary):
    """Ranking rule: maximize success, then minimize mean capped completion time. Smaller is better."""
    return (-summary["success_rate"], summary["mean_capped_time"])
