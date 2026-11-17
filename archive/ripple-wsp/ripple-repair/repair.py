"""repair.py -- Ripple repair harness: a restricted JSON-tool interface around the rover controller.

    export OPENROUTER_API_KEY=...
    python3 repair.py --setting A --out runs/A_r1          # setting names live in hidden/disturbances.json

Ripple gets five tools (inspect_weights, get_trajectory, submit_patch, evaluate_candidate,
restore_original) and nothing else: no shell, no files. The hidden disturbance is held in
RepairEnv._disturbance and is used only inside the simulator call; no tool output contains it.
The held-out seeds (hidden/heldout_seeds.json) are never read here.

No gradients anywhere in this file: a candidate is original weights + Ripple's numeric edits,
scored by running the fixed simulator on the 20 development episodes.
"""
import argparse, copy, json, math, os, re, sys, time
import numpy as np
import sim
import ripple_api

SPLITS = json.load(open(os.path.join(sim.HERE, "splits.json")))
MAX_EVALS = 10
MAX_ABS = 1000.0


# ------------------------------------------------------------------ patches
def apply_patch(original, edits):
    """Apply edits relative to the ORIGINAL weights. Returns (params, error)."""
    if not isinstance(edits, list) or not edits:
        return None, "patch.edits must be a non-empty list"
    if len(edits) > 64:
        return None, "at most 64 edits per patch"
    p = {k: v.copy() for k, v in original.items()}
    for n, e in enumerate(edits):
        where = f"edits[{n}]"
        if not isinstance(e, dict):
            return None, f"{where}: must be an object"
        unknown = set(e) - {"tensor", "op", "row", "col", "index", "value"}
        if unknown:
            return None, f"{where}: unknown keys {sorted(unknown)}"
        t = e.get("tensor")
        if t not in sim.SHAPES:
            return None, f"{where}: tensor must be one of {list(sim.SHAPES)} (the architecture is fixed)"
        op = e.get("op", "set")
        if op not in ("set", "add", "scale"):
            return None, f"{where}: op must be set, add or scale"
        shape = sim.SHAPES[t]
        sels = [k for k in ("row", "col", "index") if e.get(k) is not None]
        if len(sels) > 1:
            return None, f"{where}: give at most one of row, col, index"
        try:
            if not sels:
                sl = (slice(None),) * len(shape)
            elif sels[0] == "index":
                idx = e["index"] if isinstance(e["index"], list) else [e["index"]]
                if len(idx) != len(shape) or not all(isinstance(i, int) and not isinstance(i, bool) for i in idx):
                    return None, f"{where}: index must be {len(shape)} integer(s) for {t} with shape {list(shape)}"
                sl = tuple(idx)
            elif sels[0] == "row":
                if not isinstance(e["row"], int) or isinstance(e["row"], bool):
                    return None, f"{where}: row must be an integer"
                sl = (e["row"],) if len(shape) == 1 else (e["row"], slice(None))
            else:
                if len(shape) == 1:
                    return None, f"{where}: {t} is a bias vector; use index or row"
                if not isinstance(e["col"], int) or isinstance(e["col"], bool):
                    return None, f"{where}: col must be an integer"
                sl = (slice(None), e["col"])
            for i, s in zip(sl, shape):
                if isinstance(i, int) and not 0 <= i < s:
                    return None, f"{where}: index {i} out of range for {t} with shape {list(shape)}"
            target = p[t][sl]
            val = np.asarray(e.get("value"), dtype=float)
        except (TypeError, ValueError) as ex:
            return None, f"{where}: bad value ({ex})"
        if e.get("value") is None or not np.all(np.isfinite(val)):
            return None, f"{where}: value must be finite number(s)"
        if val.ndim and val.shape != np.shape(target):
            return None, f"{where}: value has shape {list(val.shape)} but the selection has shape {list(np.shape(target))}"
        new = val + 0 * target if op == "set" else target + val if op == "add" else target * val
        p[t][sl] = new
    for k in p:
        if p[k].shape != sim.SHAPES[k] or not np.all(np.isfinite(p[k])):
            return None, f"{k}: non-finite result"
        if np.max(np.abs(p[k])) > MAX_ABS:
            return None, f"{k}: |value| above {MAX_ABS} rejected"
    return p, None


def patch_delta(original, params):
    """Sparse list of changed parameters and summary stats."""
    changed, sq = [], 0.0
    for k in sim.SHAPES:
        d = params[k] - original[k]
        for idx in zip(*np.nonzero(d)):
            changed.append({"tensor": k, "index": [int(i) for i in idx],
                            "old": float(original[k][idx]), "new": float(params[k][idx])})
            sq += float(d[idx]) ** 2
    return changed, {"changed_parameters": len(changed), "patch_l2_norm": math.sqrt(sq)}


def rnd(x, n=4):
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), n)
    return None if x is None else round(float(x), n)


# ------------------------------------------------------------------ environment behind the tools
class RepairEnv:
    def __init__(self, original, disturbance, dev_seeds=None, obs_seeds=None, max_evals=MAX_EVALS):
        self.original = {k: v.copy() for k, v in original.items()}
        self._disturbance = dict(disturbance)          # hidden; never serialised into tool output
        self.dev_seeds = list(dev_seeds or SPLITS["repair_dev20"])
        self.obs_seeds = list(obs_seeds or SPLITS["observed5"])
        self.max_evals = max_evals
        self.pending = None          # {"params", "edits", "explanation"} submitted but maybe not evaluated
        self.pending_evaluated = False
        self.candidates = []         # evaluated candidates
        self.best = None
        self.observed = [sim.run_episode(sim.nn_policy(self.original), sim.make_episode(s), self._disturbance, record=True)
                         for s in self.obs_seeds]
        self.baseline, self.baseline_eps = sim.evaluate(sim.nn_policy(self.original), self.dev_seeds, self._disturbance)

    # -- tools
    def inspect_weights(self, which="original", tensor=None):
        if which == "candidate" and self.pending is None:
            return {"error": "no candidate submitted; which='original' is available"}
        p = self.pending["params"] if which == "candidate" else self.original
        names = [tensor] if tensor in sim.SHAPES else list(sim.SHAPES)
        return {"which": "candidate" if which == "candidate" else "original",
                "layout": "W[k] has shape [n_out, n_in]: W[i][j] connects input j to unit i. "
                          "h1=tanh(W1 x+b1), h2=tanh(W2 h1+b2), [left,right]=tanh(W3 h2+b3)",
                "shapes": {k: list(sim.SHAPES[k]) for k in names},
                "weights": {k: rnd(p[k]) for k in names}}

    def get_trajectory(self, episode_id=0, start_step=0, stride=2, max_rows=40):
        try:
            episode_id, start_step, stride, max_rows = int(episode_id), int(start_step), int(stride), int(max_rows)
        except (TypeError, ValueError):
            return {"error": "episode_id, start_step, stride, max_rows must be integers"}
        if not 0 <= episode_id < len(self.observed):
            return {"error": f"episode_id must be 0..{len(self.observed) - 1}"}
        r = self.observed[episode_id]
        stride, max_rows, start_step = max(1, stride), max(1, min(60, max_rows)), max(0, start_step)
        rows = r["rows"][start_step::stride][:max_rows]
        return {"episode_id": episode_id, "controller": "original", "environment": "changed (as observed)",
                "goal": rnd(r["goal"], 3), "reached_goal": r["success"], "end_time": r["time"],
                "final_distance": rnd(r["final_dist"], 3), "total_steps": len(r["rows"]) - 1,
                "dt": sim.CFG["dt"], "start_step": start_step, "stride": stride,
                "columns": sim.TRAJ_COLUMNS,
                "note": "row k is the state and inputs at time t, and the command the network issued at t (applied from t to t+dt)",
                "rows": [rnd(row, 4) for row in rows]}

    def submit_patch(self, patch):
        if isinstance(patch, str):
            try:
                patch = json.loads(patch)
            except json.JSONDecodeError as e:
                return {"accepted": False, "error": f"patch is not valid JSON: {e}"}
        if not isinstance(patch, dict):
            return {"accepted": False, "error": "patch must be an object {explanation, edits}"}
        params, err = apply_patch(self.original, patch.get("edits"))
        if err:
            return {"accepted": False, "error": err}
        changed, stats = patch_delta(self.original, params)
        if not changed:
            return {"accepted": False, "error": "patch changes nothing"}
        self.pending = {"params": params, "edits": patch["edits"], "explanation": str(patch.get("explanation", ""))[:2000]}
        self.pending_evaluated = False
        return {"accepted": True, **rnd_stats(stats), "applied_relative_to": "original checkpoint",
                "next": "call evaluate_candidate() to score it" if self.evals_left() else "no evaluations left"}

    def evals_left(self):
        return self.max_evals - len(self.candidates)

    def evaluate_candidate(self):
        if self.pending is None:
            return {"error": "no candidate: call submit_patch(patch) first"}
        if self.pending_evaluated:
            return {"error": "this candidate was already evaluated; submit a different patch first (no evaluation was used)"}
        if self.evals_left() <= 0:
            return {"error": f"evaluation budget of {self.max_evals} is used up"}
        pol = sim.nn_policy(self.pending["params"])
        eps = [sim.run_episode(pol, sim.make_episode(s), self._disturbance, record=True) for s in self.dev_seeds]
        summ = sim.summarize(eps)
        changed, stats = patch_delta(self.original, self.pending["params"])
        cand = {"candidate": len(self.candidates) + 1, "explanation": self.pending["explanation"],
                "edits": self.pending["edits"], "changed": changed, **stats, **summ,
                "episode_times": [e["time"] if e["success"] else None for e in eps],
                "t": time.time()}
        self.candidates.append(cand)
        self.pending_evaluated = True
        if self.best is None or sim.rank_key(summ) < sim.rank_key(self.best):
            self.best = cand
        # representative trajectories: worst, median, best by (failed, time, final distance)
        order = sorted(range(len(eps)), key=lambda i: (eps[i]["success"], -eps[i]["time"], -eps[i]["final_dist"]))
        picks = [order[0], order[len(order) // 2], order[-1]]
        trajs = []
        for i in dict.fromkeys(picks):
            e = eps[i]
            trajs.append({"dev_episode": i, "goal": rnd(e["goal"], 3), "reached_goal": e["success"],
                          "end_time": e["time"], "final_distance": rnd(e["final_dist"], 3),
                          "stride_steps": 10, "columns": sim.TRAJ_COLUMNS,
                          "rows": [rnd(row, 3) for row in e["rows"][::10][:15]]})
        return {"candidate": cand["candidate"], "evaluations_left": self.evals_left(),
                "dev_episodes": len(eps), "success_rate": rnd(summ["success_rate"], 3),
                "mean_capped_time": rnd(summ["mean_capped_time"], 3),
                "mean_final_distance": rnd(summ["mean_final_dist"], 3),
                "original_controller_same_episodes": self.baseline_public(),
                "best_candidate_so_far": {"candidate": self.best["candidate"],
                                          "success_rate": rnd(self.best["success_rate"], 3),
                                          "mean_capped_time": rnd(self.best["mean_capped_time"], 3)},
                "episode_times_null_means_failed": cand["episode_times"],
                "representative_trajectories_worst_median_best": trajs}

    def restore_original(self):
        self.pending, self.pending_evaluated = None, False
        return {"restored": True, "note": "no candidate is loaded; the controller is the original checkpoint"}

    def baseline_public(self):
        return {"success_rate": rnd(self.baseline["success_rate"], 3),
                "mean_capped_time": rnd(self.baseline["mean_capped_time"], 3),
                "mean_final_distance": rnd(self.baseline["mean_final_dist"], 3)}

    def call(self, name, args):
        if name == "inspect_weights":
            return self.inspect_weights(args.get("which", "original"), args.get("tensor"))
        if name == "get_trajectory":
            return self.get_trajectory(args.get("episode_id", 0), args.get("start_step", 0),
                                       args.get("stride", 2), args.get("max_rows", 40))
        if name == "submit_patch":
            return self.submit_patch(args.get("patch", args if "edits" in args else None))
        if name == "evaluate_candidate":
            return self.evaluate_candidate()
        if name == "restore_original":
            return self.restore_original()
        return {"error": f"unknown tool {name!r}; tools: inspect_weights, get_trajectory, submit_patch, evaluate_candidate, restore_original"}


def rnd_stats(s):
    return {"changed_parameters": s["changed_parameters"], "patch_l2_norm": rnd(s["patch_l2_norm"], 4)}


# ------------------------------------------------------------------ prompt and tool schemas
EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "tensor": {"type": "string", "enum": list(sim.SHAPES)},
        "op": {"type": "string", "enum": ["set", "add", "scale"],
               "description": "set: new value(s); add: original + value; scale: original * value"},
        "row": {"type": "integer", "description": "select a whole row i of a matrix (unit i's incoming weights), or element i of a bias"},
        "col": {"type": "integer", "description": "select a whole column j of a matrix (weights leaving input j)"},
        "index": {"type": "array", "items": {"type": "integer"}, "description": "[i, j] for one matrix element, [i] for one bias"},
        "value": {"description": "a number (applied to every selected element) or a list matching the selection's shape"},
    },
    "required": ["tensor", "op", "value"],
}
TOOLS = [
    {"type": "function", "function": {
        "name": "inspect_weights",
        "description": "Return controller weights and biases rounded to 4 decimals (W1 16x4, b1 16, W2 16x16, b2 16, W3 2x16, b3 2).",
        "parameters": {"type": "object", "properties": {
            "which": {"type": "string", "enum": ["original", "candidate"], "description": "default original"},
            "tensor": {"type": "string", "enum": list(sim.SHAPES), "description": "one tensor only; omit for all"}}}}},
    {"type": "function", "function": {
        "name": "get_trajectory",
        "description": "One of 5 recorded episodes (episode_id 0-4) of the ORIGINAL controller in the CHANGED environment: "
                       "time, x, y, heading, network inputs and commanded left/right actions per step, plus the goal position.",
        "parameters": {"type": "object", "properties": {
            "episode_id": {"type": "integer"}, "start_step": {"type": "integer", "description": "default 0"},
            "stride": {"type": "integer", "description": "every n-th step, default 2 (1 = every 0.05 s step)"},
            "max_rows": {"type": "integer", "description": "default 40, at most 60"}}, "required": ["episode_id"]}}},
    {"type": "function", "function": {
        "name": "submit_patch",
        "description": "Load a candidate = original checkpoint + your numeric edits (always relative to the original, not to an earlier candidate). "
                       "Only existing weights/biases can be edited. Does not evaluate.",
        "parameters": {"type": "object", "properties": {"patch": {"type": "object", "properties": {
            "explanation": {"type": "string", "description": "your diagnosis and why these edits compensate"},
            "edits": {"type": "array", "items": EDIT_SCHEMA}}, "required": ["explanation", "edits"]}},
            "required": ["patch"]}}},
    {"type": "function", "function": {
        "name": "evaluate_candidate",
        "description": f"Run the loaded candidate on 20 fixed development episodes in the changed environment. Uses 1 of your {MAX_EVALS} evaluations. "
                       "Returns success rate, mean capped completion time, mean final distance and 3 representative trajectories.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "restore_original",
        "description": "Discard the loaded candidate and return to the original checkpoint.",
        "parameters": {"type": "object", "properties": {}}}},
]


def system_prompt():
    c = sim.CFG
    return f"""You are repairing the neural controller of a simulated differential-drive rover by directly editing its weights.

THE ROVER (nominal simulator; fixed, you cannot change it)
  state: position (x, y), heading theta. Each step (dt = {c['dt']} s):
    left_velocity  = max_speed * left_gain  * left_command
    right_velocity = max_speed * right_gain * right_command
    linear_velocity  = (left_velocity + right_velocity) / 2
    angular_velocity = (right_velocity - left_velocity) / wheel_separation
    x += dt * linear_velocity * cos(theta);  y += dt * linear_velocity * sin(theta)
    theta = wrap_angle(theta + dt * angular_velocity)
  nominal values: max_speed = {c['max_speed']} m/s, wheel_separation = {c['wheel_separation']} m, left_gain = right_gain = 1.
  Commands are clipped to [-1, 1]. An episode succeeds when the rover is within {c['success_radius']} m of the goal;
  it fails if that has not happened after {c['duration']} s. Goals start {c['goal_dist_min']}-{c['goal_dist_max']} m away, heading is random.

THE CONTROLLER (the only thing you may edit)
  A fixed network: 4 inputs -> 16 tanh -> 16 tanh -> 2 tanh outputs.
    h1 = tanh(W1 x + b1);  h2 = tanh(W2 h1 + b2);  [left_command, right_command] = tanh(W3 h2 + b3)
  inputs x = [goal dx in rover coordinates / {c['dist_scale']},  goal dy in rover coordinates / {c['dist_scale']},
              previous left_command, previous right_command]   (dx is forward, dy is to the rover's left)
  outputs: index 0 = left wheel command, index 1 = right wheel command.
  It was trained once, by imitation of a conventional target-seeking controller, in the nominal environment,
  where it reaches the goal in every episode.

WHAT HAPPENED
  An actuator or sensor property of the rover has changed. You are not told which one or by how much.
  In the changed environment the original controller performs worse.

TOOLS (JSON only): inspect_weights, get_trajectory, submit_patch, evaluate_candidate, restore_original.
  A patch is {{"explanation": "...", "edits": [...]}}. Each edit selects part of one tensor
  (whole tensor, a row, a col, or one index) and applies op "set", "add" or "scale" with numeric value(s).
  Example: {{"tensor": "W3", "op": "scale", "row": 1, "value": 0.8}} or {{"tensor": "b3", "op": "add", "index": [0], "value": 0.05}}.
  Patches are always applied to the ORIGINAL checkpoint.

RULES
  You may evaluate at most {MAX_EVALS} candidate patches. Candidates are ranked by success rate first, then by lower
  mean capped completion time (a failed episode counts as {c['duration']} s). The best one is kept automatically.
  Work only through the tools: do the arithmetic yourself, call a tool, read the result, continue.
  Never report a result you did not get from evaluate_candidate."""


TASK = ("Diagnose the observed controller failure. Propose a concrete explanation, then directly edit the neural "
        "controller's weights or biases to compensate. You may evaluate at most 10 candidate patches. Do not train the "
        "network, compute gradients, or change the architecture or simulator. Prefer a small, explainable patch and "
        "preserve performance across varied starting states.\n\n"
        "Start by looking at the weights and at one or more trajectories. When you are finished, reply without a tool "
        "call: state your diagnosis, what your best patch changes and why, and its measured scores.")


# ------------------------------------------------------------------ text-form tool calls (fallback)
TEXT_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)


def first_json_objects(text):
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text[i:])
            out.append(obj)
            i += end
        except ValueError:
            i += 1


def text_tool_calls(content, step):
    """Tool calls written as text: <tool_call>{...}</tool_call>, or a bare JSON patch {explanation, edits}."""
    calls = []
    for raw in TEXT_CALL.findall(content or ""):
        for obj in first_json_objects(raw)[:1]:
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            if isinstance(fn, dict) and fn.get("name"):
                args = fn.get("arguments", fn.get("parameters", {}))
                calls.append((fn["name"], args))
    form = "text_tool_call" if calls else "none"
    if not calls:
        for obj in first_json_objects(content or ""):
            if isinstance(obj, dict) and isinstance(obj.get("edits"), list):
                calls.append(("submit_patch", {"patch": obj}))
                form = "text_json_patch"
                break
            if isinstance(obj, dict) and isinstance(obj.get("patch"), dict):
                calls.append(("submit_patch", {"patch": obj["patch"]}))
                form = "text_json_patch"
                break
    return [{"id": f"call_text_{step}_{i}", "type": "function",
             "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
            for i, (n, a) in enumerate(calls)], form


def continue_note(env, variant=0):
    """Neutral harness turn added after tool results (and re-worded after an empty reply). It carries no
    hint about the fault or the fix. Why it exists: with this model/provider, a turn that follows tool
    results often produces nothing visible up to the token limit unless the user turn explicitly asks
    for a plain-text answer (runs/aborted/A_attempt1_no_stream, runs/B_r1, runs/B_r2). The "nothing visible"
    replies are hidden reasoning that runs past max_tokens: the provider strips it and returns no content."""
    ask = ["First write in plain text, in at most 250 words, what you conclude so far and what you will do next; then make your tool call(s).",
           "Do not deliberate silently: start your visible reply immediately. Write in plain text, in at most 150 words, what you conclude so far about what changed in the rover. Then make a tool call.",
           "Reply now with a short plain-text paragraph (at most 100 words) stating your current hypothesis, followed by one tool call."][variant % 3]
    return (f"Continue. Evaluations left: {env.evals_left()} of {env.max_evals}. {ask} Keep a patch explanation under 60 words.")


# ------------------------------------------------------------------ the loop
def run_repair(env, out_dir, max_steps=60, max_tokens=4000, on_event=None, run_name=""):
    """Drive Ripple through the tool loop. on_event(dict) receives a concise live log (used by the demo)."""
    os.makedirs(out_dir, exist_ok=True)
    tfile = open(os.path.join(out_dir, "transcript.jsonl"), "a")
    t_start = time.time()
    tot = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}

    def log(rec):
        tfile.write(json.dumps({"t": round(time.time(), 2), **rec}, ensure_ascii=False) + "\n")
        tfile.flush()

    def emit(kind, **kw):
        ev = {"kind": kind, "elapsed_s": round(time.time() - t_start, 1), "cost_usd": round(tot["cost"], 5), **kw}
        log({"type": "event", **ev})
        if on_event:
            on_event(ev)
        else:
            print(f"[{ev['elapsed_s']:>6.1f}s ${ev['cost_usd']:.4f}] {kind}: " +
                  json.dumps({k: v for k, v in kw.items()}, ensure_ascii=False)[:400], flush=True)

    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": TASK + "\n\nOriginal controller on the 20 development episodes in the changed environment: "
                 + json.dumps(env.baseline_public())}]
    for m in messages:
        log({"type": "message", **m})
    emit("start", original_dev=env.baseline_public())

    stop, final, nudges, rejects, tool_forms = "max_steps", "", 0, 0, {}
    try:
        for step in range(1, max_steps + 1):
            budget_left = env.evals_left() > 0
            wire = [{k: v for k, v in m.items() if k != "_auto"} for m in messages]
            # After an empty reply (hidden reasoning ran past the limit) alternate a much larger budget,
            # which lets the hidden reasoning finish, with a re-roll at the normal budget.
            mt = max_tokens if nudges == 0 else [16000, max_tokens, 24000, max_tokens][(nudges - 1) % 4]
            data = ripple_api.chat(wire, tools=TOOLS if budget_left else None, max_tokens=mt, run=run_name or out_dir)
            ch = data["choices"][0]
            msg = ch.get("message") or {}
            u = data.get("usage") or {}
            tot["cost"] += data["_cost"]
            tot["prompt_tokens"] += u.get("prompt_tokens") or 0
            tot["completion_tokens"] += u.get("completion_tokens") or 0
            tot["api_calls"] += 1
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            thinking = msg.get("reasoning") or ""
            if "</think>" in content:      # the provider sometimes leaks the scratchpad into content
                head, content = content.rsplit("</think>", 1)
                thinking = (thinking + "\n" + head).strip()
            calls, form = msg.get("tool_calls") or [], "structured"
            if not calls:
                calls, form = text_tool_calls(content, step) if budget_left else ([], "none")
            tool_forms[form] = tool_forms.get(form, 0) + 1
            log({"type": "assistant", "step": step, "content": content, "tool_calls": calls, "tool_call_form": form,
                 "reasoning": thinking, "raw_content": msg.get("content"), "finish_reason": ch.get("finish_reason"), "usage": u,
                 "cost": data["_cost"], "provider": data.get("provider"), "latency_s": data["_latency_s"]})
            if content.strip():
                emit("ripple_says", text=content.strip()[:1500], tool_call_form=form)
            echo = {"role": "assistant", "content": content}
            if calls:
                echo["tool_calls"] = calls
            if calls or content.strip():
                messages.append(echo)

            if not calls:
                if not budget_left:
                    final, stop = content, "final_report"
                    break
                if ch.get("finish_reason") == "length" or not content.strip():
                    nudges += 1
                    if nudges > 8:
                        stop = "gave_up_empty_replies"
                        break
                    if not content.strip() and messages[-1]["role"] == "user" and messages[-1].get("_auto"):
                        messages.pop()      # replace the previous harness note instead of stacking user turns
                    note = ("Your last reply was cut off or empty. " if content.strip() else "") + continue_note(env, nudges)
                elif not env.candidates:
                    rejects += 1
                    if rejects > 5:
                        stop = "rejected_no_evaluation"
                        break
                    note = ("Not accepted: no candidate patch has been evaluated yet, so there is no result to report. "
                            "Call submit_patch with concrete numeric edits, then evaluate_candidate.")
                    emit("harness_rejected_report", reason="no patch was submitted and evaluated")
                elif env.best["success_rate"] < 1.0 and rejects < 2:
                    rejects += 1
                    note = (f"You still have {env.evals_left()} evaluations. Best so far: candidate {env.best['candidate']} with success "
                            f"{env.best['success_rate']:.2f}, mean capped time {env.best['mean_capped_time']:.2f} s (original: "
                            f"{env.baseline['success_rate']:.2f}, {env.baseline['mean_capped_time']:.2f} s). "
                            "Either continue with another candidate or repeat your final answer to stop.")
                    emit("harness_nudge", reason="budget left and success below 100%")
                else:
                    final, stop = content, "final_report"
                    break
                messages.append({"role": "user", "content": note, "_auto": True})
                log({"type": "message", "role": "user", "content": note, "auto": True})
                continue

            for tc in calls:
                fn = tc.get("function") or {}
                name = fn.get("name")
                try:
                    args = fn.get("arguments") or {}
                    args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    result = env.call(name, args) if isinstance(args, dict) else {"error": "arguments must be a JSON object"}
                except json.JSONDecodeError as e:
                    args, result = None, {"error": f"tool arguments were not valid JSON ({e}); nothing was run"}
                out = json.dumps(result, separators=(",", ":"))
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": out})
                log({"type": "tool", "step": step, "name": name, "args": args, "form": form, "result": result})
                if name == "submit_patch":
                    p = (args or {}).get("patch", args) or {}
                    emit("patch_submitted", explanation=(p.get("explanation") if isinstance(p, dict) else None),
                         edits=(p.get("edits") if isinstance(p, dict) else None),
                         accepted=result.get("accepted"), error=result.get("error"),
                         changed_parameters=result.get("changed_parameters"), tool_call_form=form)
                elif name == "evaluate_candidate" and "candidate" in result:
                    c = env.candidates[-1]
                    c.update({"api_cost_so_far": round(tot["cost"], 6), "elapsed_s": round(time.time() - t_start, 1),
                              "tool_call_form": form})
                    with open(os.path.join(out_dir, "candidates.jsonl"), "a") as f:
                        f.write(json.dumps(c) + "\n")
                    emit("candidate_evaluated", candidate=c["candidate"], success_rate=c["success_rate"],
                         mean_capped_time=round(c["mean_capped_time"], 3), mean_final_dist=round(c["mean_final_dist"], 3),
                         changed_parameters=c["changed_parameters"], is_best=env.best is c, evaluations_left=env.evals_left())
                else:
                    emit("tool", name=name, args=args, error=result.get("error"))
            if env.evals_left() <= 0:
                note = (f"All {env.max_evals} evaluations are used. Without calling tools, give your final answer: your diagnosis of "
                        "what changed in the rover, what the best candidate's patch changes and why it compensates, and its measured scores.")
            else:
                # Without a user turn after tool results this model/provider often returns nothing until the
                # token limit (observed in runs/aborted/A_attempt1_no_stream). The note carries no hint.
                note = continue_note(env)
            messages.append({"role": "user", "content": note, "_auto": True})
            log({"type": "message", "role": "user", "content": note, "auto": True})
            nudges = 0
    except ripple_api.SpendCapReached as e:
        stop = f"spend_cap: {e}"
    except (RuntimeError, KeyboardInterrupt) as e:
        stop = f"error: {type(e).__name__}: {e}"

    elapsed = round(time.time() - t_start, 1)
    improved = env.best is not None and sim.rank_key(env.best) < sim.rank_key(env.baseline)
    result = {"stop": stop, "elapsed_s": elapsed, **tot, "cost": round(tot["cost"], 6),
              "evaluations": len(env.candidates), "tool_call_forms": tool_forms,
              "original_dev": env.baseline,
              "best_candidate": None if env.best is None else {k: env.best[k] for k in
                  ("candidate", "success_rate", "mean_capped_time", "mean_final_dist", "changed_parameters", "patch_l2_norm")},
              "best_beats_original_on_dev": improved,
              "original_sha256": sim.params_hash(env.original)}
    if env.best is not None:
        params, _ = apply_patch(env.original, env.best["edits"])
        patch = {"relative_to_sha256": result["original_sha256"], "candidate": env.best["candidate"],
                 "explanation": env.best["explanation"], "edits": env.best["edits"], "changed": env.best["changed"]}
        json.dump(patch, open(os.path.join(out_dir, "best_patch.json"), "w"), indent=1)
        sim.save_params(params, os.path.join(out_dir, "patched.json"), {"note": "original + best_patch.json"})
        result["patched_sha256"] = sim.params_hash(params)
        result["patch_file_bytes"] = os.path.getsize(os.path.join(out_dir, "best_patch.json"))
    with open(os.path.join(out_dir, "final.md"), "w") as f:
        f.write(final or "")
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    emit("end", stop=stop, evaluations=len(env.candidates), best=result["best_candidate"],
         best_beats_original_on_dev=improved, final_report=(final or "")[:6000])
    log({"type": "end", **result})
    tfile.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, help="key in hidden/disturbances.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(sim.HERE, "checkpoints", "original.json"))
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4000)
    a = ap.parse_args()
    dist = json.load(open(os.path.join(sim.HERE, "hidden", "disturbances.json")))[a.setting]
    env = RepairEnv(sim.load_params(a.ckpt), dist)
    res = run_repair(env, a.out, a.max_steps, a.max_tokens, run_name=os.path.basename(a.out.rstrip("/")))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
